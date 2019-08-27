# coding=utf-8
import random
import numpy as np
import torch
"""
Non-overlapping: take sets of 5 classes at a time. 12 splits of 5 classes each.
Intra-task shuffling: randomize class order every epoch.
Inter-task shuffling: randomly create sets of classes (should be the default). 
"""

class BatchSampler(object):
    '''
    BatchSampler: yield a batch of indexes at each iteration.

    The default in few-shot learning papers, where each iteration randomly selects classes_per_it (cpi).
    to synthesize a batch.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations, batch_size, intratask_shuffle=True):
        '''
        Initialize the BatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class
        - iterations: number of iterations (episodes) per epoch
        '''
        super(BatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        self.batch_size = batch_size
        self.intratask_shuffle = intratask_shuffle
        self.classes, self.counts = np.unique(self.labels, return_counts=True)

        self.idxs = range(len(self.labels))
        self.label_tens = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.label_lens = np.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label)[0, 0]
            self.label_tens[label_idx, np.where(np.isnan(self.label_tens[label_idx]))[0][0]] = idx
            self.label_lens[label_idx] += 1

    def sample_class_idxs(self):
        # Note, np.random.choice might have been a more efficient implementation than permuting the entire class list.
        # But leaving the original impl. unchanged.
        return np.random.permutation(len(self.classes))[:self.classes_per_it]

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class + 1 # To get that extra sample, which we throw away all but 1 of the class.
        cpi = self.classes_per_it # cpi-way
        num_samples = spc * cpi
        true_num_samples = (spc - 1) * cpi + 1 # the actual number of sampels in the batch (duplicate spc thrown away). == spc*cpi - (cpi-1)

        for it in range(self.iterations):
            total_batch = np.array([])
            for _ in range(self.batch_size):  # 'meta-batch' (across tasks)
                batch = np.empty(num_samples)  # initialize array. will contain multiple image indices of different classes.
                c_idxs = self.sample_class_idxs()
                # Example: for 5-way, 1-shot, we build up a list:
                # we build up a batch of [i1, i2, i3, i4, i5, j1, j2, j3, j4, j5]
                # slice objects indexes (0, 5). then (1, 6), (2, 7), etc.
                # choose the class by picking an offset between (0, num_classes), e.g. 3
                # then [3:3+6] = [i4, i5, j1, j2, j3, j4]
                # this is problematic b.c. it will always result in the target class j4 being the val class.
                for i, c in enumerate(self.classes[c_idxs]):
                    s = slice(i, i + num_samples, cpi)
                    label_idx = np.argwhere(self.classes == c)[0, 0]
                    if spc > self.label_lens[label_idx]:
                        raise AssertionError('More samples per class than exist in the dataset')
                    # Samples within the dataset's instances of the class c.
                    sample_idxs = np.random.permutation(self.label_lens[label_idx])[:spc]
                    # Select specific images from this label_idx via sample_idxs.
                    batch[s] = self.label_tens[label_idx][sample_idxs]
                if self.intratask_shuffle:
                    # This codebase has a weird 'circular wraparound' method of choosing validation examples, by selecting an offset.
                    # offset is used to select the class (last_layer_input) to classify.
                    offset = random.randint(0, cpi-1)
                    batch = batch[offset:offset + true_num_samples]
                    # Permuting the ordering of the training examples within the inner loop for the sequential meta-learner. The -1 causes
                    # it to ignore the last element in the batch, which is the one we classify.
                    batch[:true_num_samples - 1] = batch[:true_num_samples - 1][np.random.permutation(true_num_samples - 1)]
                else:
                    # keep training batch order fixed (since it informs the class logit ordering).
                    # Select inner validation example from the last `cpi` elements of the batch.
                    batch = batch[:true_num_samples] # 5 + 1
                    batch[-1] = np.random.choice(batch[-cpi:])
                total_batch = np.append(total_batch, batch)
            yield total_batch.astype(int)

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


class IntraTaskBatchSampler(BatchSampler):
    """A simple modification on top of existing BatchSampler: choose class idx from fixed set of non-overlapping sets.
    within a inner batch, tasks are still permuted.
    """
    def __init__(self, labels, classes_per_it, num_samples, iterations, batch_size):
        super(IntraTaskBatchSampler, self).__init__(labels, classes_per_it, num_samples, iterations, batch_size, intratask_shuffle=True)
        # Set up class sets.
        self._class_sets = [np.array(range(i*classes_per_it, i*classes_per_it+classes_per_it)) for i in range(len(self.classes)//classes_per_it)]

    def sample_class_idxs(self):
        # TODO - do we need to permute the classes here?
        return self._class_sets[np.random.choice(len(self._class_sets))]


class NonOverlappingTasksBatchSampler(BatchSampler):
    """Fixed set of non-overlapping sets, and no randomization within inner batch."""
    def __init__(self, labels, classes_per_it, num_samples, iterations, batch_size):
        super(NonOverlappingTasksBatchSampler, self).__init__(labels, classes_per_it, num_samples, iterations, batch_size, intratask_shuffle=False)
        # Set up class sets.
        self._class_sets = [np.array(range(i*classes_per_it, i*classes_per_it+classes_per_it)) for i in range(len(self.classes)//classes_per_it)]

    def sample_class_idxs(self):
        return self._class_sets[np.random.choice(len(self._class_sets))]