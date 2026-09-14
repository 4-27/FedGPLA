"""Dataset wrappers and labeled/unlabeled split helpers used by FedGPLA."""

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from .randaugment import RandAugmentMC


def classify_label(dataset, num_classes):
    label_indices = [[] for _ in range(num_classes)]
    for index, (_, label) in enumerate(dataset):
        label_indices[label].append(index)
    return label_indices


def show_clients_data_distribution(
        dataset, clients_indices_labeled, clients_indices_unlabeled,
        num_classes):
    per_client_labeled = []
    per_client_unlabeled = []

    for client, (labeled_indices, unlabeled_indices) in enumerate(zip(
            clients_indices_labeled, clients_indices_unlabeled)):
        labeled_counts = [0] * num_classes
        unlabeled_counts = [0] * num_classes
        for index in labeled_indices:
            labeled_counts[dataset[index][1]] += 1
        for index in unlabeled_indices:
            unlabeled_counts[dataset[index][1]] += 1

        per_client_labeled.append(labeled_counts)
        per_client_unlabeled.append(unlabeled_counts)
        print(f'client {client} labeled number per class : {labeled_counts}')
        print(f'client {client} unlabeled number per class  : {unlabeled_counts}')

    return per_client_labeled, per_client_unlabeled


def partition_train(list_label2indices, labeled_per_class):
    labeled = []
    unlabeled = []
    for indices in list_label2indices:
        shuffled = np.random.permutation(indices)
        labeled.append(shuffled[:labeled_per_class])
        unlabeled.append(shuffled[labeled_per_class:])
    return labeled, unlabeled


class Indices2Dataset_labeled(Dataset):
    """Apply the weak training transform to a client's labeled examples."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.client_dataset = []

    def load(self, indices):
        # Repeating the small labeled set avoids repeatedly rebuilding a loader
        # iterator during the fixed local-step budget used by the experiments.
        self.client_dataset = [self.dataset[index] for index in indices] * 2000

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32,
                padding=int(32 * 0.125),
                padding_mode='reflect',
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2471, 0.2435, 0.2616),
            ),
        ])
        image, label = self.client_dataset[index]
        return transform(image), label

    def __len__(self):
        return len(self.client_dataset)


class Indices2Dataset_unlabeled_fixmatch(Dataset):
    """Return weak/strong views and a label used only for diagnostics."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.client_dataset = []
        self.client_dataset_len = 0

    def load(self, indices):
        client_dataset = [self.dataset[index] for index in indices]
        self.client_dataset_len = len(client_dataset)
        self.client_dataset = client_dataset * 50

    @staticmethod
    def fixmatch(image):
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32,
                padding=int(32 * 0.125),
                padding_mode='reflect',
            ),
        ])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32,
                padding=int(32 * 0.125),
                padding_mode='reflect',
            ),
            RandAugmentMC(n=2, m=10),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2471, 0.2435, 0.2616),
            ),
        ])
        return normalize(weak(image)), normalize(strong(image))

    def __getitem__(self, index):
        image, label = self.client_dataset[index]
        weak, strong = self.fixmatch(image)
        return weak, strong, label

    def __len__(self):
        return self.client_dataset_len
