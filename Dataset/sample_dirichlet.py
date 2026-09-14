"""IID and Dirichlet non-IID client partitioning used in the experiments."""

import functools
import math

import numpy as np


def clients_indices_homo(list_label2indices, num_classes, num_clients):
    class_partitions = []
    for class_index in range(num_classes):
        class_indices = list_label2indices[class_index]
        class_partitions.append([
            class_indices[
                math.floor(client / num_clients * len(class_indices)):
                math.floor((client + 1) / num_clients * len(class_indices))
            ]
            for client in range(num_clients)
        ])

    client_partitions = []
    for client in range(num_clients):
        client_indices = []
        for class_index in range(num_classes):
            client_indices.extend(class_partitions[class_index][client])
        client_partitions.append(np.array(client_indices))
    return client_partitions


def clients_indices(
        list_label2indices, num_classes, num_clients, non_iid_alpha,
        seed=None):
    indices2targets = [
        (index, label)
        for label, indices in enumerate(list_label2indices)
        for index in indices
    ]
    batches = build_non_iid_by_dirichlet(
        seed=seed,
        indices2targets=indices2targets,
        non_iid_alpha=non_iid_alpha,
        num_classes=num_classes,
        num_indices=len(indices2targets),
        n_workers=num_clients,
    )
    flattened = functools.reduce(lambda left, right: left + right, batches)
    return partition_balance(flattened, num_clients)


def partition_balance(indices, num_splits):
    per_part, remainder = divmod(len(indices), num_splits)
    parts = []
    start = 0
    for part_index in range(num_splits):
        size = per_part + int(part_index < remainder)
        parts.append(indices[start:start + size])
        start += size
    return parts


def build_non_iid_by_dirichlet(
        seed, indices2targets, non_iid_alpha, num_classes, num_indices,
        n_workers):
    random_state = np.random.RandomState(seed)
    auxiliary_workers = 2
    if n_workers < auxiliary_workers:
        raise ValueError('Dirichlet partitioning requires at least two clients')
    random_state.shuffle(indices2targets)

    split_targets = []
    start = 0
    num_splits = math.ceil(n_workers / auxiliary_workers)
    workers_per_split = [
        auxiliary_workers if split_index < num_splits - 1
        else n_workers - auxiliary_workers * (num_splits - 1)
        for split_index in range(num_splits)
    ]
    original_num_workers = n_workers
    for split_index in range(num_splits):
        stop = start + int(
            auxiliary_workers / original_num_workers * num_indices
        )
        split_targets.append(indices2targets[
            start:num_indices if split_index == num_splits - 1 else stop
        ])
        start = stop

    batches = []
    remaining_workers = original_num_workers
    for targets, configured_workers in zip(split_targets, workers_per_split):
        targets = np.array(targets)
        target_count = len(targets)
        split_workers = min(auxiliary_workers, remaining_workers)
        if split_workers != configured_workers:
            raise RuntimeError('Unexpected client split configuration')
        remaining_workers -= auxiliary_workers

        min_size = 0
        split_batch = None
        while min_size < int(0.50 * target_count / split_workers):
            split_batch = [[] for _ in range(split_workers)]
            for class_index in range(num_classes):
                class_rows = np.where(targets[:, 1] == class_index)[0]
                class_indices = targets[class_rows, 0]
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, split_workers)
                    )
                    proportions = np.array([
                        proportion * (
                            len(worker_indices) < target_count / split_workers
                        )
                        for proportion, worker_indices
                        in zip(proportions, split_batch)
                    ])
                    proportions = proportions / proportions.sum()
                    boundaries = (
                        np.cumsum(proportions) * len(class_indices)
                    ).astype(int)[:-1]
                    split_batch = [
                        worker_indices + subset.tolist()
                        for worker_indices, subset in zip(
                            split_batch, np.split(class_indices, boundaries)
                        )
                    ]
                    min_size = min(len(part) for part in split_batch)
                except ZeroDivisionError:
                    pass
        if split_batch is not None:
            batches += split_batch
    return batches
