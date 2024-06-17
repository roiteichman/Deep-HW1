import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        # ====== YOUR CODE: ======
        n = self.__len__()
        result = []
        for i in range((n+1)//2):
            result.append(i)
            if i != n - i - 1:
                result.append(n-i-1)
        return iter(result)

        # ========================

    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # ====== YOUR CODE: ======
    num_samples = len(dataset)
    dl_valid_size = int(num_samples * validation_ratio)
    dl_training_size = num_samples - dl_valid_size

    indices = list(range(num_samples))
    np.random.shuffle(indices)

    train_split = indices[:dl_training_size]
    valid_split = indices[dl_training_size:]
    dl_train_sampler = torch.utils.data.SubsetRandomSampler(train_split)
    dl_valid_sampler = torch.utils.data.SubsetRandomSampler(valid_split)

    dl_train = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=dl_train_sampler, num_workers=num_workers
    )
    dl_valid = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=dl_valid_sampler, num_workers=num_workers
    )

    return dl_train, dl_valid
    # ========================
