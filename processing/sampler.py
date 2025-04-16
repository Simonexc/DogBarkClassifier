import random

from torch.utils.data import Sampler
import torch


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = torch.tensor(dataset.labels)
        self.label_set = list(torch.unique(self.labels).numpy())
        self.label_to_indices = {
            label: torch.where(self.labels == label)[0]
            for label in self.label_set
        }
        for label in self.label_set:
            self.label_to_indices[label] = self.label_to_indices[label][torch.randperm(len(self.label_to_indices[label]))]
        self.used_label_indices_count = {label: 0 for label in self.label_set}
        self.count = 0
        self.num_samples = len(dataset)

    def __iter__(self):
        self.used_label_indices_count = {label: 0 for label in self.label_set}
        self.count = 0
        while self.count + self.batch_size <= self.num_samples:
            indices = []
            min_label_count = min([len(self.label_to_indices[label]) for label in self.label_set])
            label_batch_size = self.batch_size // len(self.label_set)
            if label_batch_size == 0:
                label_batch_size = 1 # Ensure at least one sample per class if batch size is smaller than num classes

            for label in self.label_set:
                label_indices = self.label_to_indices[label]
                start_index = self.used_label_indices_count[label]
                end_index = min(start_index + label_batch_size, min_label_count) # Ensure we don't go beyond available samples
                if end_index <= start_index: # Handle cases where class samples are exhausted
                    replace_indices_tensor = label_indices[torch.multinomial(torch.ones_like(label_indices, dtype=torch.float), num_samples=label_batch_size, replacement=True)]
                    replace_indices = replace_indices_tensor.tolist() # Convert back to list of indices
                    indices.extend(replace_indices)
                else:
                    current_indices_tensor = label_indices[start_index:end_index]
                    current_indices = current_indices_tensor.tolist() # Convert to list
                    indices.extend(current_indices)
                    self.used_label_indices_count[label] = end_index

            if len(indices) < self.batch_size: # Pad batch if not enough samples
                needed_padding = self.batch_size - len(indices)
                padding_indices = random.choices(indices, k=needed_padding) # Simple oversampling padding
                indices.extend(padding_indices)

            random.shuffle(indices)
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.num_samples // self.batch_size
