import numpy as np
import os
import pickle

from torch.utils import data


class SpatialMNISTDataset(data.Dataset):
    def __init__(self, data_dir, nsamples=50, split='train'):
        splits = {
            'train': slice(0, 60000),
            'test': slice(60000, 70000)
        }
        self.nsamples = nsamples

        spatial_path = os.path.join(data_dir, 'spatial.pkl')
        with open(spatial_path, 'rb') as file:
            spatial = pickle.load(file)

        labels_path = os.path.join(data_dir, 'labels.pkl')
        with open(labels_path, 'rb') as file:
            labels = pickle.load(file)

        self._spatial = np.array(spatial[splits[split]]).astype(np.float32)[:10000]
        self._labels = np.array(labels[splits[split]])[:10000]

        ix = self._labels[:, 1] != 1
        self._spatial = self._spatial[ix]
        self._labels = self._labels[ix]

        assert len(self._spatial) == len(self._labels)
        self._n = len(self._spatial)

    def __getitem__(self, item):
        n = self._spatial[item].shape[0]
        idxs = np.random.permutation(self.nsamples) % n
        return self._spatial[item][idxs,:], self._labels[item]

    def __len__(self):
        return self._n

