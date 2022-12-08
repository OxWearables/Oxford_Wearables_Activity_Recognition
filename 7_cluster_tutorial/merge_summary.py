"""
Merge the individual {eid}_summary.csv files together in 1 aggregated summary.csv file
"""

import os
import argparse
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='summary merger', usage='Merge individual _summary.csv files')
    parser.add_argument('path', type=str, help='root path (should contain the group folders)')
    args = parser.parse_args()

    files = glob(os.path.join(args.path, '**/*_summary.csv'))

    # hardcode some dtypes, the rest will be inferred
    df = dd.read_csv(files, assume_missing=False, header=0,
                     dtype={'eid': 'int',
                            'StartTime': 'string',
                            'EndTime': 'string',
                            'WearTime(days)': 'float',
                            'CalibrationOK': 'int'})

    # Aggregate all files in 1 large dataframe, and save to csv.
    # Can take a few minutes for 100k files.
    output = os.path.join(args.path, 'summary.csv')
    with ProgressBar():
        (
            df.compute(scheduler='processes', num_workers=3)
            .to_csv(output, index=False)
        )

    print(f'summary saved to {output}')


