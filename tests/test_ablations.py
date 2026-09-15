"""CPU behavioral tests; no dataset download or GPU training is required."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import FedGPLA as training
from options import args_parser


MODES = ('full', 'all_hard', 'all_soft', 'uniform', 'without_memory')


def prior_server(mode):
    # Prior updates are independent of the CUDA model created by Global.__init__.
    server = object.__new__(training.Global)
    server.ablation_mode = mode
    server.num_classes = 3
    server.alpha_global_prior = 0.1
    server.evidence_memory = None
    server.global_prior = None
    return server


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = torch.tensor([
            [0.96, 0.03, 0.01],  # confident and agrees: hard in Full
            [0.02, 0.96, 0.02],  # confident but disagrees: soft in Full
            [0.30, 0.25, 0.45],  # agrees but not confident: soft in Full
        ])
        self.strong_predictions = torch.tensor([0, 0, 2])

    def test_hard_soft_and_hybrid_counts(self):
        hybrid = self.probabilities.clone()
        hybrid[0] = torch.tensor([1., 0., 0.])
        expected = {
            'full': hybrid, 'uniform': hybrid, 'without_memory': hybrid,
            'all_hard': torch.eye(3), 'all_soft': self.probabilities,
        }
        for mode in MODES:
            with self.subTest(mode=mode):
                actual = training.unlabeled_class_evidence(
                    self.probabilities, self.strong_predictions, 0.95, mode,
                )
                torch.testing.assert_close(actual, expected[mode])
                torch.testing.assert_close(actual.sum(-1), torch.ones(3))

    def test_local_estimation_keeps_labels_and_uses_all_eight_views(self):
        class ProbabilityModel(torch.nn.Module):
            def forward(self, images):
                probabilities = images.flatten(start_dim=1)
                return probabilities, probabilities.log()

        def cpu_loader(*args, **kwargs):
            kwargs.update(num_workers=0, pin_memory=False)
            return DataLoader(*args, **kwargs)

        weak = self.probabilities[:, None, :, None, None].repeat(1, 8, 1, 1, 1)
        # Perturb views without changing their mean; tests average, not one view.
        weak[:, 0, 0] += 0.005
        weak[:, 0, 1] -= 0.005
        weak[:, 1, 0] -= 0.005
        weak[:, 1, 1] += 0.005
        strong = torch.tensor([
            [0.98, 0.01, 0.01], [0.98, 0.01, 0.01], [0.01, 0.01, 0.98],
        ])[:, :, None, None]
        dataset = TensorDataset(weak, strong)
        local = object.__new__(training.Local)
        local.local_model = ProbabilityModel()
        local.num_classes = 3
        expected_counts = {
            'full': [3.32, 2.21, 0.47],
            'all_hard': [3., 2., 1.],
            'all_soft': [3.28, 2.24, 0.48],
            'uniform': [3.32, 2.21, 0.47],
            'without_memory': [3.32, 2.21, 0.47],
        }
        for mode in MODES:
            with self.subTest(mode=mode):
                args = SimpleNamespace(
                    gpu_id=0, batch_size_local_unlabeled=2,
                    reliability_threshold=0.95, alpha_local_prior=1.,
                    ablation_mode=mode,
                )
                # Keep real tensor operations and estimator; redirect only CUDA
                # transfers and worker spawning for this CPU-only fixture.
                with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor), \
                        patch.object(training, 'DataLoader', cpu_loader):
                    evidence, prior = local.estimate_local_evidence(
                        args, dataset, [2, 1, 0],
                    )
                np.testing.assert_allclose(evidence, expected_counts[mode], atol=1e-6)
                self.assertAlmostEqual(float(evidence.sum()), 6., places=6)
                np.testing.assert_allclose(
                    prior.numpy(), (np.array(expected_counts[mode]) + 1. / 3) / 7,
                    atol=1e-6,
                )
                # The returned statistics also feed the server, without a second
                # hard/soft conversion or normalizing each client's mass away.
                server = prior_server(mode)
                server.initialize_method_prior([[2, 1, 0], [0, 1, 2]])
                server.update_method_prior([0], [evidence])
                if mode != 'without_memory':
                    np.testing.assert_allclose(server.evidence_memory[0], evidence)


class ReferenceTests(unittest.TestCase):
    labeled = [[8, 1, 1], [0, 2, 8], [1, 4, 1]]

    def test_memory_retains_inactive_clients(self):
        server = prior_server('full')
        server.initialize_method_prior(self.labeled)
        server.update_method_prior([1], [[0, 5, 25]])
        torch.testing.assert_close(server.global_prior,
                                   training.smoothed_prior([9, 10, 27], 0.1, 3))
        server.update_method_prior([0], [[2, 0, 0]])
        torch.testing.assert_close(server.global_prior,
                                   training.smoothed_prior([3, 9, 26], 0.1, 3))

    def test_without_memory_uses_only_latest_participants(self):
        full = prior_server('full')
        server = prior_server('without_memory')
        for model in (full, server):
            model.initialize_method_prior(self.labeled)
        torch.testing.assert_close(server.global_prior, full.global_prior)
        self.assertIsNone(server.evidence_memory)
        server.update_method_prior([1], [[0, 5, 25]])
        torch.testing.assert_close(server.global_prior,
                                   training.smoothed_prior([0, 5, 25], 0.1, 3))
        server.update_method_prior([2, 0], [[0, 12, 0], [2, 0, 0]])
        torch.testing.assert_close(server.global_prior,
                                   training.smoothed_prior([2, 12, 0], 0.1, 3))
        self.assertIsNone(server.evidence_memory)

    def test_uniform_reference_stays_uniform(self):
        server = prior_server('uniform')
        server.initialize_method_prior(self.labeled)
        expected = torch.full((3,), 1. / 3, dtype=torch.float64)
        torch.testing.assert_close(server.global_prior, expected)
        server.update_method_prior([1], [[100, 0, 0]])
        torch.testing.assert_close(server.global_prior, expected)
        # Normal evidence construction and caching remain active.
        torch.testing.assert_close(server.evidence_memory[1],
                                   torch.tensor([100., 0., 0.], dtype=torch.float64))

    def test_full_participation_makes_memory_and_no_memory_agree(self):
        servers = [prior_server(mode) for mode in ('full', 'without_memory')]
        for server in servers:
            server.initialize_method_prior(self.labeled)
            server.update_method_prior([2, 0, 1], [[3, 1, 2], [5, 9, 3], [1, 0, 12]])
        torch.testing.assert_close(servers[0].global_prior, servers[1].global_prior)


class ExperimentTests(unittest.TestCase):
    def test_cli_defaults_and_run_isolation_do_not_change_rng(self):
        with tempfile.TemporaryDirectory() as root:
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.get_rng_state().clone()
            paths = []
            for mode in MODES + ('full',):
                args = args_parser(['--ablation_mode', mode, '--output_dir', root])
                self.assertEqual(args.my_method_prior_correction, 0.25)
                self.assertEqual(args.lambda_kl, 0.5)
                with patch.object(training.time, 'strftime', return_value='same_second'):
                    run = Path(training.create_run_directory(args))
                self.assertEqual(run.relative_to(root).parts[0], mode)
                self.assertTrue(run.is_dir())
                paths.append(run)
            self.assertEqual(len(set(paths)), len(paths))
            self.assertEqual(random.getstate(), python_state)
            after = np.random.get_state()
            np.testing.assert_array_equal(after[1], numpy_state[1])
            self.assertEqual(after[2:], numpy_state[2:])
            torch.testing.assert_close(torch.get_rng_state(), torch_state)

    def test_training_loop_initialization_sampling_and_output_files(self):
        seen_priors = {}

        class TinyGlobal(training.Global):
            def __init__(self, args):
                self.num_classes = args.num_classes
                self.ablation_mode = args.ablation_mode
                self.alpha_global_prior = args.alpha_global_prior
                self.evidence_memory = None
                self.global_prior = None
                self.params = {'avgpool.p': torch.tensor(2.)}

            def download_params(self):
                return self.params

            def fedavg_eval(self, params, data_test, batch_size_test):
                self.params = params
                return 0.5

        class TinyLocal:
            def __init__(self, args):
                pass

            def supervised_warmup_train(self, args, data, params, r, steps):
                return params

            def my_method_train(self, args, labeled, unlabeled, evidence_data,
                                labeled_counts, params, global_prior, r, client):
                seen_priors[args.ablation_mode].append(global_prior.clone())
                probabilities = torch.full((1, args.num_classes), 0.04 / 9)
                probabilities[0, 0] = 0.96
                counts = training.unlabeled_class_evidence(
                    probabilities, torch.tensor([0]), 0.95, args.ablation_mode,
                )[0].double() * len(evidence_data)
                counts += torch.tensor(labeled_counts, dtype=torch.float64)
                return params, [10, 8, 5], counts.numpy()

        dataset = [(Image.new('RGB', (32, 32)), c) for c in range(10) for _ in range(6)]
        original_partition = training.partition_train
        partitions, schedules = [], []
        with tempfile.TemporaryDirectory() as root:
            for mode in MODES:
                args = args_parser([
                    '--dataset', 'CIFAR10', '--alpha', '0.5', '--num_clients', '2',
                    '--num_online_clients', '1', '--ablation_mode', mode,
                    '--output_dir', root,
                ])
                np.random.seed(args.seed)
                random.seed(args.seed)
                torch.manual_seed(args.seed)
                seen_priors[mode] = []
                # Keep the real training orchestrator and prior updates, with
                # lightweight dataset/model fixtures for 10 warm-up + 3 SSL rounds.
                with patch.object(training.datasets, 'CIFAR10', return_value=dataset), \
                        patch.object(training, 'Global', TinyGlobal), \
                        patch.object(training, 'Local', TinyLocal), \
                        patch.object(training, 'partition_train',
                                     side_effect=lambda indices, n: original_partition(indices, 2)), \
                        patch.object(training, 'tqdm', return_value=range(1, 14)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    training.fixmatch(args)
                run = next((Path(root) / mode / 'CIFAR10' / 'alpha_0.5' / 'seed_7').iterdir())
                partitions.append((run / 'partition.json').read_text())
                schedules.append((run / 'clients.jsonl').read_text())
                config = json.loads((run / 'config.json').read_text())
                self.assertEqual(config['ablation_mode'], mode)
                self.assertEqual(config['evidence_views'], 8)
                self.assertTrue((run / 'logs/train.log').is_file())
                self.assertEqual(len(pd.read_csv(run / 'accuracy.csv')), 13)
                metrics = pd.read_csv(run / 'metrics.csv')
                self.assertTrue(metrics['pseudo_acc'].iloc[:10].isna().all())
                self.assertTrue((metrics['pseudo_acc'].iloc[10:] == 0.8).all())
            self.assertEqual(len(set(partitions)), 1)
            self.assertEqual(len(set(schedules)), 1)
            torch.testing.assert_close(seen_priors['full'][0], seen_priors['without_memory'][0])
            # With no persistent memory, the guard must still initialize only once.
            self.assertFalse(torch.allclose(seen_priors['without_memory'][0],
                                            seen_priors['without_memory'][1]))
            for prior in seen_priors['uniform']:
                torch.testing.assert_close(prior, torch.full((10,), 0.1, dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
