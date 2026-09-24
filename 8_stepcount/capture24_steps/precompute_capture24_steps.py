"""Precompute 10 s step counts for all Capture-24 participants.

Uses the RF StepCounter trained on OxWalk (see train_stepcounter_oxwalk.py).
Run from the 8_stepcount folder:

    cd 8_stepcount
    PYTHONPATH=. python capture24_steps/train_stepcounter_oxwalk.py      # once, ~1 min
    PYTHONPATH=. python capture24_steps/precompute_capture24_steps.py    # ~30-60 min

Options (environment variables):
    CAPTURE24_DIR  path to the Capture-24 folder (default: ../capture24)
    N_WORKERS      participants processed in parallel (default: 6; ~2-3 GB RAM each)
    PIDS           comma-separated subset, e.g. "P001,P002" (for testing)

Outputs (in capture24_steps/):
    per_pid/PXXX.csv.gz            cache, one file per participant (reruns skip done pids)
    capture24_steps_10s.csv.gz     one row per full 10 s window, all participants
    capture24_steps_daily.csv      one row per participant

Windowing: consecutive 10 s windows aligned like data.resample('10s') (i.e. floor
of the timestamp to 10 s). Only full windows (exactly 1000 samples) are kept.
Windows with NaN in x/y/z get steps = walk = NaN. The walk detector's HMM is run
separately on each contiguous run of valid windows. The annotation of a window is
its most frequent raw annotation (NaN if the window is entirely unannotated).
"""
import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so that `stepcount` is importable

C24_DIR = os.environ.get('CAPTURE24_DIR', os.path.join(HERE, '..', '..', 'capture24'))
MODEL_PATH = os.path.join(HERE, 'stepcounter_rf_oxwalk.joblib')
CACHE_DIR = os.path.join(HERE, 'per_pid')
OUT_10S = os.path.join(HERE, 'capture24_steps_10s.csv.gz')
OUT_DAILY = os.path.join(HERE, 'capture24_steps_daily.csv')

WINDOW_SEC = 10
SAMPLE_RATE = 100
WINDOW_LEN = WINDOW_SEC * SAMPLE_RATE
N_WORKERS = int(os.environ.get('N_WORKERS', 6))


def read_capture24(path):
    """Read in chunks (keeps memory low); annotation stored as a categorical."""
    chunks = []
    for c in pd.read_csv(
        path,
        usecols=['time', 'x', 'y', 'z', 'annotation'],
        dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': object},
        parse_dates=['time'],
        date_format='%Y-%m-%d %H:%M:%S.%f',
        chunksize=1_000_000,
    ):
        c['annotation'] = pd.Categorical(c['annotation'])
        chunks.append(c)
    cats = sorted(set().union(*(set(c['annotation'].cat.categories) for c in chunks)))
    for c in chunks:
        c['annotation'] = pd.Categorical(c['annotation'].astype(object), categories=cats)
    return pd.concat(chunks, ignore_index=True)


def windows_from_frame(data):
    """Split into 10 s windows (floor of timestamp). Return full windows only."""
    t = data['time'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    win = t // (WINDOW_SEC * 10**9)
    # data are time-sorted; find runs of identical window ids
    starts = np.flatnonzero(np.r_[True, win[1:] != win[:-1]])
    counts = np.diff(np.r_[starts, len(win)])
    full = counts == WINDOW_LEN
    starts = starts[full]

    idx = starts[:, None] + np.arange(WINDOW_LEN)[None, :]
    xyz = data[['x', 'y', 'z']].to_numpy()
    X = xyz[idx]  # (n, 1000, 3)

    codes = data['annotation'].cat.codes.to_numpy()[idx]  # -1 = NaN
    cats = data['annotation'].cat.categories
    anno = np.full(len(starts), None, dtype=object)
    for i, row in enumerate(codes):
        row = row[row >= 0]
        if len(row):
            anno[i] = cats[np.bincount(row).argmax()]

    tstart = pd.to_datetime(win[starts] * (WINDOW_SEC * 10**9))
    n_incomplete = int((~full).sum())
    return X, tstart, anno, n_incomplete


def process_pid(pid):
    out = os.path.join(CACHE_DIR, f'{pid}.csv.gz')
    if os.path.exists(out):
        return pid, 0.0, 'cached'
    t0 = time.time()
    import stepcount.models  # noqa: F401  (needed to unpickle the model)
    sc = joblib.load(MODEL_PATH)
    sc.n_jobs = 1
    sc.wd.n_jobs = 1
    sc.verbose = False

    data = read_capture24(os.path.join(C24_DIR, f'{pid}.csv.gz'))
    X, tstart, anno, n_incomplete = windows_from_frame(data)
    del data

    ok = ~np.isnan(X).any(axis=(1, 2))
    # contiguous segments of valid windows -> separate HMM sequences
    tsec = tstart.asi8[ok] // 10**9
    seg = np.cumsum(np.r_[True, np.diff(tsec) != WINDOW_SEC])

    steps = np.full(len(X), np.nan)
    walk = np.full(len(X), np.nan)
    if ok.any():
        s, w = sc.predict(X[ok], groups=seg, return_walk=True)
        steps[ok], walk[ok] = s, w

    df = pd.DataFrame({'pid': pid, 'time': tstart, 'steps': steps,
                       'walk': walk, 'annotation': anno})
    df.to_csv(out + '.tmp', index=False, compression='gzip',
              date_format='%Y-%m-%d %H:%M:%S')
    os.replace(out + '.tmp', out)
    return pid, time.time() - t0, f'{len(df)} windows, {n_incomplete} incomplete dropped'


def main():
    t0 = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        sys.exit(f'Model not found: {MODEL_PATH}. Run train_stepcounter_oxwalk.py first.')

    pids = os.environ.get('PIDS')
    if pids:
        pids = pids.split(',')
    else:
        pids = sorted(re.match(r'(P\d{3})', f).group(1)
                      for f in os.listdir(C24_DIR) if re.match(r'P\d{3}\.csv\.gz$', f))
    print(f'{len(pids)} participants, {N_WORKERS} workers', flush=True)

    from joblib import Parallel, delayed
    for pid, secs, msg in Parallel(n_jobs=N_WORKERS, return_as='generator_unordered')(
            delayed(process_pid)(p) for p in pids):
        print(f'[{time.time() - t0:6.0f} s] {pid}: {msg} ({secs:.0f} s)', flush=True)

    # ---- combine -----------------------------------------------------------
    df = pd.concat([pd.read_csv(os.path.join(CACHE_DIR, f'{p}.csv.gz'),
                                parse_dates=['time']) for p in pids],
                   ignore_index=True)

    lab = pd.read_csv(os.path.join(C24_DIR, 'annotation-label-dictionary.csv'),
                      index_col='annotation')
    df['label_walmsley'] = df['annotation'].map(lab['label:Walmsley2020'])
    df['label_willetts_specific'] = df['annotation'].map(lab['label:WillettsSpecific2018'])
    df['met'] = df['annotation'].str.extract(r'MET\s*([\d.]+)', expand=False).astype(float)

    meta = pd.read_csv(os.path.join(C24_DIR, 'metadata.csv'))
    df = df.merge(meta, on='pid', how='left')
    df = df[['pid', 'age', 'sex', 'time', 'steps', 'walk', 'annotation',
             'label_walmsley', 'label_willetts_specific', 'met']]
    df['walk'] = df['walk'].astype('Int8')
    df['steps'] = df['steps'].astype('Int16')
    df.to_csv(OUT_10S, index=False, compression='gzip',
              date_format='%Y-%m-%d %H:%M:%S')

    # ---- per-participant summary -------------------------------------------
    valid = df['steps'].notna()
    g = df.assign(valid=valid, annotated=df['annotation'].notna(),
                  walkmin=df['walk'].fillna(0) * WINDOW_SEC / 60).groupby('pid')
    daily = pd.DataFrame({
        'hours_recorded': g['valid'].sum() * WINDOW_SEC / 3600,
        'hours_annotated': g['annotated'].sum() * WINDOW_SEC / 3600,
        'total_steps': g['steps'].sum().astype(float),
        'walking_minutes': g['walkmin'].sum(),
    })
    daily['steps_per_day'] = daily['total_steps'] / daily['hours_recorded'] * 24
    daily['walking_minutes_per_day'] = daily['walking_minutes'] / daily['hours_recorded'] * 24
    daily = meta.set_index('pid').join(daily, how='right').reset_index()
    daily.round(2).to_csv(OUT_DAILY, index=False)

    # ---- sanity checks -----------------------------------------------------
    print('\n==== sanity checks ====')
    print(f'windows: {len(df):,}; participants: {df.pid.nunique()}')
    print(f'NaN-step windows: {(~valid).mean():.3%}; unannotated windows: '
          f'{df.annotation.isna().mean():.1%}')
    print('unmapped annotations:', df.loc[df.annotation.notna() & df.label_walmsley.isna(),
                                          'annotation'].nunique())
    sph = (df[valid].groupby('label_walmsley')['steps']
           .agg(hours=lambda s: len(s) * WINDOW_SEC / 3600, steps='sum'))
    sph['steps_per_hour'] = sph['steps'] / sph['hours']
    print('\nsteps per hour by Walmsley2020 label:\n', sph.round(1))
    print('\nsteps/day across participants:\n', daily['steps_per_day'].describe().round(0))
    for f in (OUT_10S, OUT_DAILY):
        print(f'{os.path.basename(f)}: {os.path.getsize(f) / 1e6:.1f} MB')
    print(f'total time: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
