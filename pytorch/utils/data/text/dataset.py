from torch.utils.data import Sampler, Dataset
from collections import OrderedDict
from random import shuffle


class BucketDataset(Dataset):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        else:
            return self.inputs[index], self.targets[index]


class BucketBatchSampler(Sampler):

    # want inputs to be an array
    def __init__(self, batch_size, inputs, targets):
        # Throw an error if the number of inputs and targets don't match
        if targets is not None:
            if len(inputs) != len(targets):
                raise Exception("[BucketBatchSampler] inputs and targets have different sizes")
        # Remember batch size
        self.batch_size = batch_size
        # For each data item (it's index), keep track of combination of input and target lengths
        self.ind_n_len = []
        if targets is None:
            for i in range(0, len(inputs)):
                self.ind_n_len.append((i, (inputs[i].shape[0], 1)))
        else:
            for i in range(0, len(inputs)):
                self.ind_n_len.append((i, (inputs[i].shape[0], targets[i].shape[0])))

        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)


    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[(5,8)] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map() # <-- Could be a waste of performance
        # shuffle all the batches so they aren't ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i