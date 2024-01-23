import os
from pathlib import Path
import gzip
import json
import pandas as pd
import numpy as np


class DatasetContainer:
    """
    This class should function as a container for a specific dataset. Its task is to contain the pandas dataframe
    as well as metadata that might be needed in certain use cases (such as validation).

    The DatasetContainer is initialized with either a pandas dataset (and metadata) or a path that links to a dataset.
    If both is provided the path will be ignored.
    """

    def __init__(self, dataset: pd.DataFrame = None, metadata: dict = None, path_of_dataset: str = None):
        super(DatasetContainer, self).__init__()

        if dataset is None and path_of_dataset is not None:
            self.load(path_of_dataset)
        else:
            self.data: pd.DataFrame = dataset
            self.meta: dict = metadata

    def save(self, path: str):
        if path is None:
            raise ValueError('No path was provided for the save function.')

        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data_json = self.data.to_json()
        meta_json = json.dumps(self.meta)

        container_dict = {
            'data': data_json,
            'meta': meta_json
        }

        # save file to ziped format
        with gzip.open(path, 'w') as fout:
            fout.write(json.dumps(container_dict).encode('utf-8'))

    def load(self, path: str):
        if path is None:
            raise ValueError('No path was provided for the load function.')

        with gzip.open(path, 'r') as fin:
            container_json = json.loads(fin.read().decode('utf-8'))

            self.meta = json.loads(container_json['meta'])
            
            if not self.meta is None and "dtypes" in self.meta:
                dtype = self.meta["dtypes"]
            else:
                dtype = True
            self.data = pd.read_json(container_json['data'], dtype=dtype)

            for column in self.data:
                if type(self.data[column].iloc[0]) == list:
                    self.data[column] = self.data[column].apply(lambda x: np.array(x))

        return self

