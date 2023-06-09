import torch
from functools import partial
from itertools import cycle
from packaging import version
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import IterableDataset, Dataset, DataLoader
from .cddataset import CDDataset
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, IterableDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)
'''
参数train, validation, test 要传入Dataset实例
'''
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, train=None, validation=None, test=None, predict=None):
        super().__init__()
        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

    def _train_dataloader(self):
        return CDDataset(self.dataset_configs["train"])

    def _val_dataloader(self):
        return CDDataset(self.dataset_configs["validation"])

    def _test_dataloader(self):
        return CDDataset(self.dataset_configs["test"])
    
    def _predict_dataloader(self):
        return CDDataset(self.dataset_configs["predict"])