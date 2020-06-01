import glob
import os
from typing import List

import pandas as  pd

all_splits = ["train", "test", "valid"]
all_langs = []


def get_paths(root_path: str, langs: List[str], splits: List[str]) -> List[str]:
    paths = []

    for lang in langs:
        for split in splits:
            path = os.path.join(root_path, lang, "final", "jsonl", split, "*.gz")
            paths.extend(glob.glob(path))

    return paths


def paths_to_df(paths: List[str]) -> pd.DataFrame:
    """Load a list of jsonl.gz files into a pandas DataFrame."""

    return pd.concat([pd.read_json(f,
                                   orient='records',
                                   compression='gzip',
                                   lines=True)
                      for f in paths], sort=False)


def get_data_df(root_path: str, langs: List[str], splits: List[str]) -> pd.DataFrame:
    return paths_to_df(get_paths(root_path, langs, splits))


if __name__ == '__main__':
    root_path = "../resources/data/"
