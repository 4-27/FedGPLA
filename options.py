"""Command-line options used by the minimal FedGPLA reproduction code."""

import argparse
import os


def args_parser():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Reproduce the main FedGPLA results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--method', default='FedGPLA')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument(
        '--dataset',
        choices=('CIFAR10', 'CIFAR100', 'SVHN', 'CINIC10'),
        default='CIFAR100',
    )
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--mu', type=int, default=2)
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Dirichlet concentration; use 0 for the IID split',
    )
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=128)
    parser.add_argument('--batch_size_local_labeled', type=int, default=128)
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=512)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)

    parser.add_argument('--reliability_threshold', type=float, default=0.95)
    parser.add_argument('--unsup_threshold', type=float, default=0.95)
    parser.add_argument('--alpha_local_prior', type=float, default=1.0)
    parser.add_argument('--alpha_global_prior', type=float, default=0.1)
    parser.add_argument('--my_method_prior_correction', type=float, default=0.25)
    parser.add_argument('--lambda_u', type=float, default=1.0)
    parser.add_argument('--lambda_kl', type=float, default=0.5)

    parser.add_argument(
        '--path_cifar10',
        default=os.path.join(project_dir, 'data', 'CIFAR10'),
    )
    parser.add_argument(
        '--path_cifar100',
        default=os.path.join(project_dir, 'data', 'CIFAR100'),
    )
    parser.add_argument(
        '--path_svhn',
        default=os.path.join(project_dir, 'data', 'SVHN'),
    )
    parser.add_argument(
        '--path_cinic10',
        default=os.path.join(project_dir, 'data', 'CINIC10'),
    )
    parser.add_argument(
        '--output_dir',
        default=os.path.join(project_dir, 'results'),
    )
    parser.add_argument(
        '--save_checkpoints',
        action='store_true',
        help='Save selected global-model checkpoints in addition to CSV metrics',
    )

    args = parser.parse_args()
    if args.gpu_id < 0:
        parser.error('--gpu_id must be non-negative')
    if args.alpha < 0:
        parser.error('--alpha must be non-negative')
    if args.num_clients < 2:
        parser.error('--num_clients must be at least 2')
    if not 1 <= args.num_online_clients <= args.num_clients:
        parser.error('--num_online_clients must be in [1, num_clients]')
    if not 0 <= args.reliability_threshold <= 1:
        parser.error('--reliability_threshold must be between 0 and 1')
    if not 0 <= args.unsup_threshold <= 1:
        parser.error('--unsup_threshold must be between 0 and 1')
    if args.alpha_local_prior <= 0 or args.alpha_global_prior <= 0:
        parser.error('prior smoothing coefficients must be greater than 0')
    if not 0 <= args.my_method_prior_correction <= 1:
        parser.error('--my_method_prior_correction must be between 0 and 1')
    if args.lambda_u < 0 or args.lambda_kl < 0:
        parser.error('loss weights must be non-negative')
    return args
