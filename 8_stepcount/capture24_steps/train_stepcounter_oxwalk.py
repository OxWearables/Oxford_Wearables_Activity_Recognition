"""Train the final RF StepCounter on ALL 39 OxWalk wrist-100Hz participants.

Run from the 8_stepcount folder:
    cd 8_stepcount && PYTHONPATH=. python capture24_steps/train_stepcounter_oxwalk.py

Output: capture24_steps/stepcounter_rf_oxwalk.joblib
(windows are built exactly as in Tutorial.ipynb: 10 s, 100 Hz, keep full
1000-sample windows without NaNs; steps = number of annotated heel strikes)
"""
import os
import re
import json
import time
from glob import glob

import joblib
import numpy as np
import pandas as pd

from stepcount import models

HERE = os.path.dirname(os.path.abspath(__file__))
OXWALK_DIR = os.path.join(HERE, '..', 'OxWalk_Dec2022', 'Wrist_100Hz')
MODEL_PATH = os.path.join(HERE, 'stepcounter_rf_oxwalk.joblib')

WINDOW_SEC = 10
SAMPLE_RATE = 100
STEPTOL = 4
N_JOBS = int(os.environ.get('N_JOBS', 8))


def read_csv(filename):
    return pd.read_csv(
        filename, parse_dates=['timestamp'], index_col='timestamp',
        dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'Int64'},
    )


def make_windows(data, window_sec=WINDOW_SEC, sample_rate=SAMPLE_RATE):
    window_len = int(window_sec * sample_rate)
    X, Y = [], []
    for _, w in data.resample(f"{window_sec}s"):
        if len(w) < window_len or w.isna().any().any():
            continue
        X.append(w[['x', 'y', 'z']].to_numpy())
        Y.append(w['annotation'].sum())
    return np.stack(X), np.asarray(Y)


def main():
    t0 = time.time()
    X, Y, G = [], [], []
    files = sorted(glob(os.path.join(OXWALK_DIR, 'P*_wrist100.csv')))
    for f in files:
        pid = re.search(r'(P\d{2})', os.path.basename(f)).group(1).upper()
        _X, _Y = make_windows(read_csv(f))
        X.append(_X); Y.append(_Y); G.append(np.full(len(_Y), pid))
    X, Y, G = np.concatenate(X), np.concatenate(Y).astype(float), np.concatenate(G)
    print(f"{len(files)} participants, X shape {X.shape}, "
          f"walk windows (>= {STEPTOL} steps): {(Y >= STEPTOL).mean():.1%}")

    sc = models.StepCounter(wd_type='rf', cv=4, window_sec=WINDOW_SEC,
                            sample_rate=SAMPLE_RATE, steptol=STEPTOL,
                            n_jobs=N_JOBS, verbose=True)
    sc.fit(X, Y, groups=G)  # NB: 2nd positional arg is Y (steps), groups by keyword
    print("find_peaks_params:", sc.find_peaks_params)
    print("walk-detector threshold:", sc.wd.thresh)
    for k in ('walk_detector', 'step_counter'):
        print(k, 'CV scores:', sc.cv_results[k]['scores'])
    print('step_counter CV scores (true walk windows only):',
          sc.cv_results['step_counter']['scores_walk'])

    sc.verbose = False
    sc.wd.verbose = False
    joblib.dump(sc, MODEL_PATH, compress=3)
    with open(os.path.join(HERE, 'stepcounter_rf_oxwalk_params.json'), 'w') as fh:
        json.dump({'find_peaks_params': {k: float(v) for k, v in sc.find_peaks_params.items()},
                   'wd_thresh': float(sc.wd.thresh), 'n_windows': int(len(Y)),
                   'n_participants': int(len(np.unique(G))), 'window_sec': WINDOW_SEC,
                   'sample_rate': SAMPLE_RATE, 'steptol': STEPTOL}, fh, indent=2)
    print(f"Saved {MODEL_PATH} ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
