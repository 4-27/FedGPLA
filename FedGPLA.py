"""Minimal training entry point for reproducing the FedGPLA experiments."""

from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, \
    Indices2Dataset_unlabeled_fixmatch, partition_train
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
from Dataset.randaugment import RandAugmentMC  # [Modified for FedGPLA]
import numpy as np
import pandas as pd
from torch import max, eq, no_grad
from Dataset.CINIC10 import CINIC10
from torch.optim import SGD
import torch.nn.functional as F
from Model.resnet import ResNet_PC
from tqdm import tqdm
import copy
import torch
import random
from torch.utils.data import DataLoader, Dataset, RandomSampler  # [Modified for FedGPLA]
import logging
import os
import time


# ==================== [FedGPLA Start] ====================
# 以当前代码行为为准的固定实验常量。
FEDGPLA_WARMUP_ROUNDS = 10
FEDGPLA_EVIDENCE_VIEWS = 8
FEDGPLA_PRIOR_EPS = 1e-12


def smoothed_prior(class_counts, smoothing_alpha, num_classes):
    """按照 method 公式 (2)/(7)/(11) 对类别证据做 Dirichlet/Laplace 平滑。"""
    # 将输入复制为 CPU float64 张量，避免修改调用方保存的原始计数。
    counts = torch.as_tensor(class_counts, dtype=torch.float64).clone()
    # 计算公式分母：全部类别证据质量与平滑系数之和。
    denominator = counts.sum() + float(smoothing_alpha)
    # method 要求 alpha_l 和 alpha_g 大于零，因此这里显式拒绝非法分母。
    if denominator.item() <= 0:
        raise ValueError('Prior denominator must be positive')
    # 对每个类别加入 alpha/C，再除以公共分母得到归一化先验。
    return (counts + float(smoothing_alpha) / num_classes) / denominator


class Indices2Dataset_evidence(Dataset):
    """为每个真实无标签样本生成 V 个独立弱视图和 1 个强视图。"""

    def __init__(self, dataset):
        # 保存官方原始训练集对象；其中图像尚未应用 transform。
        self.dataset = dataset

    def load(self, indices):
        # 只缓存真正的无标签索引，不把 labeled 索引加入证据池。
        self.client_dataset = [self.dataset[i] for i in indices]

    def __getitem__(self, idx):
        # 读取一个真实无标签样本；标签只用于诊断，不参与任何方法计算。
        image, _ = self.client_dataset[idx]
        # 构造与官方弱增强完全相同的空间增强。
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32,
                padding=int(32 * 0.125),
                padding_mode='reflect',
            ),
        ])
        # 构造与官方强增强完全相同的空间增强和 RandAugment。
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32,
                padding=int(32 * 0.125),
                padding_mode='reflect',
            ),
            RandAugmentMC(n=2, m=10),
        ])
        # 真实图像仍使用官方 client training 的 ToTensor 与 normalization。
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2471, 0.2435, 0.2616),
            ),
        ])
        # 对同一图像独立调用八次弱增强，并堆叠为 [V,C,H,W]。
        weak_views = torch.stack([
            normalize(weak_transform(image))
            for _ in range(FEDGPLA_EVIDENCE_VIEWS)
        ])
        # 独立生成一张强增强视图，并应用官方 normalization。
        strong_view = normalize(strong_transform(image))
        return weak_views, strong_view

    def __len__(self):
        # 每个真实无标签样本在证据估计中恰好出现一次。
        return len(self.client_dataset)
# ===================== [FedGPLA End] =====================


class Global(object):
    def __init__(self, args):
        self.model = ResNet_PC(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes,
                            use_global_gem=True)

        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        self.gpu_id = args.gpu_id

        # ==================== [FedGPLA Start] ====================
        # [Modified for FedGPLA] 保存每个客户端最近一次上传的类别证据 A_{k,mem}。
        self.evidence_memory = None
        # [Modified for FedGPLA] 保存下一轮客户端使用的全局参考先验 pi_g。
        self.global_prior = None
        # [Modified for FedGPLA] 保存 method 中的全局平滑系数 alpha_g。
        self.alpha_global_prior = args.alpha_global_prior
        # ===================== [FedGPLA End] =====================

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images = images.cuda(self.gpu_id)
                labels = labels.cuda(self.gpu_id)
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.model.state_dict()

    # ==================== [FedGPLA Start] ====================
    def initialize_method_prior(self, all_clients_labeled_counts):
        """用所有客户端的真实 labeled count 初始化 memory bank 与 pi_g。"""
        # 将 K 个客户端的 labeled count 转成服务器端 float64 张量。
        memory = torch.as_tensor(all_clients_labeled_counts, dtype=torch.float64)
        # 检查第二维必须与当前任务类别数完全一致。
        if memory.dim() != 2 or memory.size(1) != self.num_classes:
            raise ValueError('Labeled-count memory must have shape [K, C]')
        # 复制计数作为 warm-up 结束后的初始 A_{k,mem}=n_k^l。
        self.evidence_memory = memory.clone()
        # 用所有缓存计数计算第一轮半监督训练所需的全局参考先验。
        self.refresh_method_prior()

    def refresh_method_prior(self):
        """按照 method 公式 (11) 从完整 client-wise cache 更新 pi_g。"""
        # memory bank 必须已由全部客户端 labeled count 初始化。
        if self.evidence_memory is None:
            raise RuntimeError('Evidence memory has not been initialized')
        # 对客户端维求和，得到服务器端全局类别证据。
        global_counts = self.evidence_memory.sum(dim=0)
        # 使用 alpha_g/C 平滑并归一化得到新的 pi_g。
        self.global_prior = smoothed_prior(
            global_counts,
            self.alpha_global_prior,
            self.num_classes,
        )

    def update_method_prior(self, client_ids, uploaded_evidence):
        """只替换本轮参与客户端的缓存，离线客户端缓存保持不变。"""
        # 每个参与客户端必须且只能上传一个 A_k。
        if len(client_ids) != len(uploaded_evidence):
            raise ValueError('Client ids and uploaded evidence must have equal length')
        # 逐客户端执行 A_{k,mem} <- A_k。
        for client_id, client_evidence in zip(client_ids, uploaded_evidence):
            # 将上传证据复制到 CPU float64，便于稳定地累计全局先验。
            evidence = torch.as_tensor(client_evidence, dtype=torch.float64)
            # 每个 A_k 必须包含恰好 C 个类别分量。
            if evidence.numel() != self.num_classes:
                raise ValueError('Client evidence has an invalid class dimension')
            # 只覆盖当前参与客户端对应的 memory bank 行。
            self.evidence_memory[int(client_id)] = evidence
        # 所有参与客户端完成替换后，再统一得到下一轮使用的 pi_g。
        self.refresh_method_prior()

    def initialize_for_method_fusion(self, list_dicts_local_params, list_nums_local_data):
        """按当前方法规定的聚合质量执行标准 FedAvg。"""
        # 从第一个客户端参数字典深拷贝出聚合结果容器。
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        # 对官方 state_dict 中的每一个张量执行完全相同的加权算术。
        for name_param in list_dicts_local_params[0]:
            # 保存当前参数来自各客户端的“参数乘聚合质量”。
            list_values_param = []
            # 同时遍历上传参数和该客户端在 method 中规定的聚合质量。
            for dict_local_params, num_local_data in zip(
                    list_dicts_local_params, list_nums_local_data):
                # 复用官方实现的乘法，不改变整型 buffer 的处理路径。
                list_values_param.append(
                    dict_local_params[name_param] * num_local_data
                )
            # 除以本轮参与客户端聚合质量总和，得到标准 FedAvg 参数。
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            # 将结果写回与官方 state_dict 同名的位置。
            fedavg_global_params[name_param] = value_global_param

        return fedavg_global_params

    # ===================== [FedGPLA End] =====================


class Local(object):
    def __init__(self, args):

        self.local_model = ResNet_PC(resnet_size=8, scaling=4,
                                  save_activations=False, group_norm_num_groups=None,
                                  freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes,
                                  use_global_gem=True)


        self.local_G = ResNet_PC(resnet_size=8, scaling=4,
                              save_activations=False, group_norm_num_groups=None,
                              freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes,
                              use_global_gem=True)

        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)

        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

        self.num_classes = args.num_classes

    # ==================== [FedGPLA Start] ====================
    def supervised_warmup_train(
            self, args, data_client_labeled, global_params, r,
            num_local_iterations):
        """前 10 轮只使用 labeled CE，并保留原本的本地 step 预算。"""
        # 使用官方 RandomSampler、batch size、drop_last、worker 和 pin_memory。
        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
        )
        # 每个客户端开始时仍从本轮下发的 global model 初始化。
        self.local_model.load_state_dict(global_params)
        # 使用训练模式更新参数和 BatchNorm running statistics。
        self.local_model.train()
        # 与官方 local_epochs 完全一致，不为 warm-up 引入新的 epoch 数。
        for local_epoch in range(args.local_epochs):
            # 每个本地 epoch 重新创建 labeled iterator，保持官方组织方式。
            labeled_iter = iter(self.labeled_trainloader)
            # [Modified for FedGPLA] step 数沿用官方由本地总训练池规模决定的预算。
            for epoch in range(num_local_iterations):
                # 尝试读取下一批 labeled 数据。
                try:
                    inputs_x, targets_x = labeled_iter.__next__()
                # 与官方代码一致，迭代器耗尽时立即从同一 loader 重新开始。
                except StopIteration:
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.__next__()
                # 将 labeled 图像和真值标签移动到官方指定 GPU。
                inputs_x = inputs_x.cuda(args.gpu_id)
                targets_x = targets_x.cuda(args.gpu_id)
                # 只对 labeled 弱增强图像执行前向传播。
                _, logits_x = self.local_model(inputs_x)
                # method 公式 (1)：计算平均 supervised cross-entropy。
                supervised_loss = F.cross_entropy(
                    logits_x,
                    targets_x,
                    reduction='mean',
                )
                # 记录当前 warm-up round、local epoch、batch 和 Ls。
                logging.info(
                    f'Round {r}, Local Epoch {local_epoch}, Batch {epoch}: '
                    f'Ls = {supervised_loss.item():.4f}'
                )
                # [Modified for FedGPLA] 复用 Local.__init__ 创建的官方 optimizer。
                self.optimizer.zero_grad()
                # 反向传播当前 labeled-only 损失。
                supervised_loss.backward()
                # 使用官方跨客户端、跨轮次复用的 SGD/momentum 状态更新参数。
                self.optimizer.step()
                # 对单个全局共享的 GeM 指数执行约束投影；p=1 对应原 AvgPool。
                self.local_model.avgpool.project_p_()
        # 上传当前客户端训练后的完整 state_dict。
        return copy.deepcopy(self.local_model.state_dict())

    def estimate_local_evidence(
            self, args, data_client_evidence, labeled_class_counts):
        """逐行实现 method 公式 (3)-(7)，得到 A_k 与固定的 pi_k。"""
        # 以本客户端真实 labeled class count n_k^l 初始化证据向量。
        local_evidence = torch.as_tensor(
            labeled_class_counts,
            dtype=torch.float64,
        ).cuda(args.gpu_id).clone()
        # 证据 loader 遍历真实无标签集合一次，不丢弃尾批次也不随机重排。
        evidence_loader = DataLoader(
            dataset=data_client_evidence,
            batch_size=args.batch_size_local_unlabeled,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=True,
        )
        # 证据由本轮下载模型在 inference mode 下重新估计。
        self.local_model.eval()
        # 证据估计不参与梯度计算，也不改变 optimizer 状态。
        with torch.no_grad():
            # 逐批读取八个弱视图和一个强视图。
            for weak_views, strong_view in evidence_loader:
                # 读取当前 batch size 和固定弱视图数 V。
                batch_size, num_views = weak_views.shape[:2]
                # 把 [B,V,C,H,W] 展平为模型可接收的 [B*V,C,H,W]。
                weak_views = weak_views.view(
                    batch_size * num_views,
                    *weak_views.shape[2:],
                ).cuda(args.gpu_id)
                # 将强增强视图移动到同一官方指定 GPU。
                strong_view = strong_view.cuda(args.gpu_id)
                # 对全部独立弱视图计算未校正 logits。
                _, weak_logits = self.local_model(weak_views)
                # 将弱视图 logits 转为类别概率。
                weak_probabilities = torch.softmax(weak_logits, dim=-1)
                # 恢复 [B,V,C] 并按 V 个视图求平均得到 p_bar。
                average_weak_probability = weak_probabilities.view(
                    batch_size,
                    num_views,
                    self.num_classes,
                ).mean(dim=1)
                # 对每个样本的单个强增强视图计算 logits。
                _, strong_logits = self.local_model(strong_view)
                # 取强增强预测类别，用于 method 的强弱一致性判断。
                strong_prediction = strong_logits.argmax(dim=-1)
                # 取得平均弱预测的最大概率和对应类别。
                weak_confidence, weak_prediction = (
                    average_weak_probability.max(dim=-1)
                )
                # 同时满足 tau_r 和强弱类别一致时，m_{k,j}=1。
                reliable_mask = (
                    weak_confidence.ge(args.reliability_threshold)
                    & weak_prediction.eq(strong_prediction)
                )
                # 为平均弱预测的 argmax 类别构造 one-hot hard count。
                hard_evidence = F.one_hot(
                    weak_prediction,
                    num_classes=self.num_classes,
                ).to(average_weak_probability.dtype)
                # 可靠样本贡献 hard count，其余样本贡献平均弱预测 soft count。
                sample_evidence = torch.where(
                    reliable_mask.unsqueeze(1),
                    hard_evidence,
                    average_weak_probability,
                )
                # 将当前 batch 每个样本恰好一次的证据累加到 A_k。
                local_evidence += sample_evidence.to(torch.float64).sum(dim=0)
        # 计算理论上应满足的证据总质量 N_k^l+N_k^u。
        expected_mass = float(
            np.sum(labeled_class_counts) + len(data_client_evidence)
        )
        # 检查 hard/soft 两类证据都确实每样本归一且只累计一次。
        if not torch.isclose(
                local_evidence.sum(),
                torch.tensor(
                    expected_mass,
                    dtype=torch.float64,
                    device=local_evidence.device,
                ),
                rtol=1e-5,
                atol=1e-5):
            # 证据质量不一致时立即停止，避免静默使用错误先验。
            raise RuntimeError('A_k mass is not equal to N_k^l + N_k^u')
        # 按 method 公式 (7) 使用 alpha_l/C 对 A_k 做平滑。
        local_prior = smoothed_prior(
            local_evidence.detach().cpu(),
            args.alpha_local_prior,
            self.num_classes,
        ).cuda(args.gpu_id)
        # 返回上传给服务器的 A_k，以及本轮本地训练固定使用的 pi_k。
        return local_evidence.detach().cpu().numpy(), local_prior

    def my_method_train(
            self, args, data_client_labeled, data_client_unlabeled,
            data_client_evidence, labeled_class_counts, global_params,
            global_prior, r, client_idx):
        """执行当前代码的先验校正、软伪标签学习和 local/global KL。"""
        # 使用官方 RandomSampler 构造 labeled loader。
        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
        )
        # 使用官方 RandomSampler 和 batch_size*mu 构造 unlabeled loader。
        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
        )
        # 每个客户端都从服务器本轮下发的同一个 global model 初始化。
        self.local_model.load_state_dict(global_params)
        # 保留一份本轮下发的冻结 global model，用于计算 p_g(alpha(u))。
        self.local_G.load_state_dict(global_params)
        self.local_G.eval()
        # 在任何本地梯度更新前，用下载模型估计 A_k 与 pi_k。
        local_evidence, local_prior = self.estimate_local_evidence(
            args,
            data_client_evidence,
            labeled_class_counts,
        )
        # 将服务器本轮下发的 pi_g 转成 GPU float64 张量。
        global_prior = torch.as_tensor(
            global_prior,
            dtype=torch.float64,
        ).cuda(args.gpu_id)
        # method 公式 (8)：逐类计算 log(pi_g+eps)-log(pi_k+eps)。
        prior_log_ratio = (
            torch.log(global_prior + FEDGPLA_PRIOR_EPS)
            - torch.log(local_prior + FEDGPLA_PRIOR_EPS)
        ).detach()
        # 证据估计结束后切回训练模式。
        self.local_model.train()
        # 沿用官方 local_epochs，不添加新的本地训练轮数。
        for local_epoch in range(args.local_epochs):
            # 每个 local epoch 都按官方方式重新创建两个 iterator。
            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)
            # [Modified for FedGPLA] 完全保留官方 local_iter 的计算公式。
            local_iter = int(
                len(data_client_unlabeled)
                / args.batch_size_local_labeled_fixmatch
            )
            # 只在最后一个 local epoch 汇总官方格式的伪标签诊断。
            if local_epoch + 1 == args.local_epochs:
                # 初始化全部 corrected pseudo-label 的正确数量。
                num_pseudo_corrects = 0
                # 初始化全部 corrected pseudo-label 的样本数量。
                num_pseudo_total = 0
                # 初始化通过 tau_u 的 corrected pseudo-label 数量。
                num_u_valid = 0
            # 按官方 step 预算执行当前客户端本地更新。
            for epoch in range(local_iter):
                # 读取 labeled batch。
                try:
                    inputs_x, targets_x = labeled_iter.__next__()
                # 与官方一致，耗尽时从相同 loader 重新开始。
                except StopIteration:
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.__next__()
                # 读取一组 weak/strong unlabeled batch。
                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = (
                        unlabeled_iter.__next__()
                    )
                # 与官方一致，耗尽时从相同 loader 重新开始。
                except StopIteration:
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = (
                        unlabeled_iter.__next__()
                    )
                # 将三组输入移动到官方指定 GPU。
                inputs_x = inputs_x.cuda(args.gpu_id)
                inputs_u_w = inputs_u_w.cuda(args.gpu_id)
                inputs_u_s = inputs_u_s.cuda(args.gpu_id)
                # 保存 labeled batch size，用于拆分 interleave 后的 logits。
                batch_size = inputs_x.shape[0]
                # [Modified for FedGPLA] 保留官方 interleave 的单次联合前向组织。
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                    2 * args.mu + 1,
                ).cuda(args.gpu_id)
                # 将 labeled target 移动到同一 GPU。
                targets_x = targets_x.cuda(args.gpu_id)
                # 隐藏真值只用于伪标签精度诊断。
                targets_u_groundtruth = targets_u_groundtruth.cuda(args.gpu_id)
                # 对 interleave 后的 labeled/weak/strong 输入执行一次官方模型前向。
                _, logits = self.local_model(inputs)
                # 恢复原 batch 顺序，保持官方 BatchNorm 行为。
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                # 前 batch_size 个 logits 对应 labeled 输入。
                logits_x = logits[:batch_size]
                # 其余 logits 等分为 weak unlabeled 与 strong unlabeled。
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                # method 公式 (1)：计算 labeled supervised loss。
                Ls = F.cross_entropy(logits_x, targets_x, reduction='mean')
                # 使用命令行记录的 prior log-ratio 校正强度。
                corrected_weak_logits = (
                    logits_u_w
                    + args.my_method_prior_correction
                    * prior_log_ratio.to(dtype=logits_u_w.dtype)
                )
                # method 公式 (8)：softmax 得到 corrected soft target q。
                corrected_soft_target = torch.softmax(
                    corrected_weak_logits,
                    dim=-1,
                ).detach()
                # 取得 corrected target 的最大概率及其类别。
                corrected_confidence, corrected_prediction = (
                    corrected_soft_target.max(dim=-1)
                )
                # method 公式 (9)：只保留最大 q 不低于 tau_u 的样本。
                valid_mask = corrected_confidence.ge(
                    args.unsup_threshold
                ).float()
                # 计算 strong prediction 的稳定 log-softmax。
                strong_log_probability = F.log_softmax(logits_u_s, dim=-1)
                # 逐样本计算 soft-target cross-entropy。
                per_sample_soft_ce = -(
                    corrected_soft_target * strong_log_probability
                ).sum(dim=-1)
                # 按 method 的 indicator mask 后对完整 unlabeled batch 取平均。
                Lu = (per_sample_soft_ce * valid_mask).mean()
                # 冻结 global model 在弱增强 alpha(u) 上计算 p_g(alpha(u))。
                with torch.no_grad():
                    _, global_weak_logits = self.local_G(inputs_u_w)
                    global_weak_log_probability = F.log_softmax(
                        global_weak_logits,
                        dim=-1,
                    )
                    global_weak_probability = (
                        global_weak_log_probability.exp()
                    )
                # p_l(A(u)) 是反向 KL 定义式右侧的 target。
                local_strong_log_probability_target = (
                    strong_log_probability
                )
                # 不使用 F.kl_div 的参数顺序推断，直接按定义计算：
                # KL(p_g || p_l) = sum_c p_g,c * (log p_g,c - log p_l,c)。
                Lkl = (
                    global_weak_probability
                    * (
                        global_weak_log_probability
                        - local_strong_log_probability_target
                    )
                ).sum(dim=-1).mean()
                # KL 仅在半监督路径执行，因此会在10轮预热结束后自动启用。
                loss = (
                    Ls
                    + args.lambda_u * Lu
                    + args.lambda_kl * Lkl
                )
                # 最后一个 local epoch 统计与官方输出兼容的伪标签指标。
                if local_epoch + 1 == args.local_epochs:
                    # 累计全部 corrected pseudo-label 的正确数。
                    num_pseudo_corrects += sum(
                        eq(
                            corrected_prediction.cpu(),
                            targets_u_groundtruth.cpu(),
                        )
                    ).item()
                    # 累计全部 corrected pseudo-label 的样本数。
                    num_pseudo_total += len(corrected_prediction)
                    # 累计通过 tau_u 的样本数。
                    num_u_valid += sum(valid_mask).item()
                # 记录当前 round、local epoch、batch、Ls、Lu 与 local/global KL。
                logging.info(
                    f'Round {r}, Local Epoch {local_epoch}, Batch {epoch}: '
                    f'Ls = {Ls.item():.4f}, Lu = {Lu.item():.4f}, '
                    f'Lkl = {Lkl.item():.4f}'
                )
                # [Modified for FedGPLA] 复用官方唯一 optimizer，不重建 SGD。
                self.optimizer.zero_grad()
                # 对联合损失执行反向传播。
                loss.backward()
                # 使用跨客户端、跨轮次保留的官方 momentum 状态更新模型。
                self.optimizer.step()
                # 保证客户端上传以及后续 FedAvg 使用的 p 始终位于 [1, 4]。
                self.local_model.avgpool.project_p_()
            # 最后一个 local epoch 结束后生成官方格式的客户端诊断。
            if local_epoch + 1 == args.local_epochs:
                # 计算全部 corrected pseudo-label 的精度。
                pseudo_client_acc = num_pseudo_corrects / num_pseudo_total
                # 计算通过 tau_u 的 corrected pseudo-label 比例。
                u_client_valid = num_u_valid / num_pseudo_total
                # 写入与官方日志字段兼容的客户端级指标。
                logging.info(
                    f'Round {r}, Local Epoch {local_epoch}, Client {client_idx}, '
                    f'pseudo_acc = {pseudo_client_acc: .4f}, '
                    f'pseudo_num_valid = {num_u_valid}, '
                    f'valid_ratio = {u_client_valid}'
                )
        pseudo_status = [
            num_pseudo_total,
            num_pseudo_corrects,
            num_u_valid,
        ]
        # 上传训练后模型、诊断和本轮训练前固定估计的 A_k。
        return (
            copy.deepcopy(self.local_model.state_dict()),
            pseudo_status,
            local_evidence,
        )
    # ===================== [FedGPLA End] =====================

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def fixmatch(args):
    alpha = args.alpha

    run_root = os.path.join(
        args.output_dir,
        args.dataset,
        f'alpha_{alpha}',
        f'seed_{args.seed}',
    )
    log_dir = os.path.join(run_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    cr_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(log_dir, '{method}_α={alpha}_{cr_time}.log'.format(method=args.method, alpha = alpha, cr_time=cr_time))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file
                        )

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100': # training:50k; testing:10k; for training, each class includes 500 images
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)

    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 150
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds = 400
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
        ])
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)

    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN.")
        exit(1)


    logging.info(
        'dataset:{dataset}\n'
        'num_classes:{num_classes}\n'
        'num_labeled:{num_labeled}\n'
        'non_iid:{alpha}\n'
        'mu:{mu}\n'
        'num_rounds:{num_rounds}\n'
        'batch_label:{batch_label}, batch_unlabel:{batch_unlabel}'.format(
            dataset=args.dataset,
            num_classes=args.num_classes,
            num_labeled=args.num_labeled,
            alpha=alpha,
            mu=args.mu,
            num_rounds=args.num_rounds,
            batch_label=args.batch_size_local_labeled,
            batch_unlabel=args.batch_size_local_unlabeled,
        ))

    random_state = np.random.RandomState(args.seed)

    list_label2indices = classify_label(data_local_training, args.num_classes)

    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

    # IID
    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(list_label2indices=list_label2indices_labeled,
                                                           num_classes=args.num_classes,
                                                           num_clients=args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices=list_label2indices_unlabeled,
                                                             num_classes=args.num_classes,
                                                             num_clients=args.num_clients)
    # Non-IID
    else:
        list_client2indices_labeled = clients_indices(list_label2indices=list_label2indices_labeled,
                                                      num_classes=args.num_classes,
                                                      num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices=list_label2indices_unlabeled,
                                                        num_classes=args.num_classes,
                                                        num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)

    # Show data distribubution
    # ==================== [FedGPLA Start] ====================
    # [Modified for FedGPLA] 保留官方打印行为，同时取得每个客户端 n_k^l。
    all_clients_labeled_counts, _ = show_clients_data_distribution(
        data_local_training,
        list_client2indices_labeled,
        list_client2indices_unlabeled,
        args.num_classes,
    )
    # 在官方把 labeled 索引加入 unlabeled 训练池之前，保存真正的无标签索引。
    list_client2indices_evidence = [
        copy.deepcopy(indices)
        for indices in list_client2indices_unlabeled
    ]
    # ===================== [FedGPLA End] =====================

    # add labeled samples without labels into the unlabeled dataset
    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    global_model = Global(args)
    local_model = Local(args)

    total_clients = list(range(args.num_clients))


    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)
    # ==================== [FedGPLA Start] ====================
    # [Modified for FedGPLA] 证据 wrapper 只加载与 labeled set 不相交的索引。
    indices2data_evidence = Indices2Dataset_evidence(data_local_training)
    # ===================== [FedGPLA End] =====================

    fedavg_acc = []
    fedavg_pseudo_acc = []
    fedavg_num_valid = []
    fedavg_valid_ratio = []
    # 记录服务器每轮 FedAvg 后的唯一全局 GeM 指数。
    fedavg_gem_p = []

    # FL training
    for r in tqdm(range(1, args.num_rounds + 1), desc='Server'):

        # ==================== [FedGPLA Start] ====================
        # 当前实现将前 10 轮计入总通信预算并用于监督预热。
        is_warmup_round = r <= FEDGPLA_WARMUP_ROUNDS
        # 在第一轮半监督训练开始前，用全部客户端 n_k^l 初始化 cache 与 pi_g。
        if not is_warmup_round and global_model.evidence_memory is None:
            global_model.initialize_method_prior(all_clients_labeled_counts)
        # 收集本轮参与客户端上传的 A_k，供聚合后更新 memory bank。
        list_local_evidence = []
        # ===================== [FedGPLA End] =====================

        dict_global_params = global_model.download_params()

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []

        # client training
        num_clients_u_total = 0
        num_clients_u_corrects = 0
        num_clients_u_valid = 0

        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled

            # ==================== [FedGPLA Start] ====================
            if is_warmup_round:
                # warm-up 不读取任何 unlabeled batch；此处只复用官方 step 预算。
                num_local_iterations = int(
                    len(data_client_unlabeled)
                    / args.batch_size_local_labeled_fixmatch
                )
                # 仅使用 labeled CE 执行本客户端更新。
                local_params = local_model.supervised_warmup_train(
                    args,
                    data_client_labeled,
                    copy.deepcopy(dict_global_params),
                    r,
                    num_local_iterations,
                )
                # method warm-up FedAvg 质量严格使用唯一 labeled 数量 N_k^l。
                list_nums_local_data.append(
                    len(list_client2indices_labeled[client])
                )
                # warm-up 不产生任何伪标签诊断。
                pseudo_status = None
            else:
                # 为当前客户端加载真正的无标签证据池。
                indices2data_evidence.load(
                    list_client2indices_evidence[client]
                )
                # 将证据 wrapper 传入 method 本地训练。
                data_client_evidence = indices2data_evidence
                # 使用固定 pi_g 估计 A_k、pi_k，并优化 Ls、Lu 和 Lkl。
                local_params, pseudo_status, local_evidence = (
                    local_model.my_method_train(
                        args,
                        data_client_labeled,
                        data_client_unlabeled,
                        data_client_evidence,
                        all_clients_labeled_counts[client],
                        copy.deepcopy(dict_global_params),
                        global_model.global_prior,
                        r,
                        client,
                    )
                )
                # method 半监督 FedAvg 质量使用唯一 N_k^l+N_k^u。
                list_nums_local_data.append(len(data_client_unlabeled))
                # 保存当前参与客户端上传的 A_k。
                list_local_evidence.append(local_evidence)
            # ===================== [FedGPLA End] =====================

            list_dicts_local_params.append(copy.deepcopy(local_params))
            # ==================== [FedGPLA Start] ====================
            # 半监督阶段沿用官方字段汇总 corrected pseudo-label 指标。
            if pseudo_status is not None:
                num_clients_u_total += pseudo_status[0]
                num_clients_u_corrects += pseudo_status[1]
                num_clients_u_valid += pseudo_status[2]
            # ===================== [FedGPLA End] =====================

        # ==================== [FedGPLA Start] ====================
        # warm-up 没有伪标签，使用 NaN 明确表示“不适用”而不是伪造零精度。
        if is_warmup_round:
            pseudo_acc = np.nan
            pseudo_valid_ratio = np.nan
        # 半监督阶段维持官方整体计算公式。
        else:
            pseudo_acc = num_clients_u_corrects / num_clients_u_total
            pseudo_valid_ratio = num_clients_u_valid / num_clients_u_total
        fedavg_pseudo_acc.append(pseudo_acc)
        fedavg_valid_ratio.append(pseudo_valid_ratio)
        fedavg_num_valid.append(num_clients_u_valid)

        fedavg_params = global_model.initialize_for_method_fusion(
            list_dicts_local_params,
            list_nums_local_data,
        )
        # avgpool.p 是 state_dict 中的单元素参数，已随其他模型参数完成 FedAvg。
        global_gem_p = float(fedavg_params['avgpool.p'].item())
        fedavg_gem_p.append(global_gem_p)
        # 半监督阶段在 FedAvg 后更新 cache 与下一轮 pi_g。
        if not is_warmup_round:
            global_model.update_method_prior(
                online_clients,
                list_local_evidence,
            )
        # ===================== [FedGPLA End] =====================

        global_acc = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test)
        fedavg_acc.append(global_acc)

        logging.info(
            f'Round {r}: global_gem_p = {global_gem_p:.6f}, '
            f'global_acc = {global_acc:.6f}'
        )
        print('round {round}, accuracy:{global_acc}, pseudo_acc:{fedavg_pseudo_acc}, num_valid:{fedavg_num_valid}, valid_ratio:{fedavg_valid_ratio}, gem_p:{global_gem_p:.6f}'.format(
            round = r,
            global_acc = global_acc,
            fedavg_pseudo_acc = fedavg_pseudo_acc[-1],
            fedavg_num_valid = fedavg_num_valid[-1],
            fedavg_valid_ratio = fedavg_valid_ratio[-1],
            global_gem_p = global_gem_p))

        result_dir = run_root
        os.makedirs(result_dir, exist_ok=True)

        # make specific result dir by cdw
        result_dir_spec = f'{result_dir}/{args.method}_α={alpha}_{cr_time}'
        os.makedirs(result_dir_spec, exist_ok=True)

        if args.save_checkpoints and (
                r == 1 or r == args.num_rounds
                or (r % 50 == 0 and r > 0.8 * args.num_rounds)):
            torch.save(fedavg_params, f'{result_dir_spec}/fedavg_params_round_{r}.pth')
            print(f"Saved model for round {r}")

        result_file = f'{result_dir}/{args.method}_α={alpha}_{cr_time}.csv'
        acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc)+1))
        acc_num_pseudo_label_csv_df = pd.DataFrame({'acc':fedavg_acc}, index = acc_num_pseudo_label_csv_index)
        # 保存文件
        acc_num_pseudo_label_csv_df.to_csv(result_file, encoding='utf8')

        result_pseudo_file = f'{result_dir}/{args.method}_α={alpha}_pseudo_{cr_time}.csv'
        min_length = min(
            len(fedavg_pseudo_acc),
            len(fedavg_valid_ratio),
            len(fedavg_num_valid),
            len(fedavg_gem_p),
            len(fedavg_acc),
        )

        # 创建 DataFrame，包含所有指标
        metrics_df = pd.DataFrame({
            'round': list(range(1, min_length + 1)),
            'acc': fedavg_acc[:min_length],
            'pseudo_acc': fedavg_pseudo_acc[:min_length],
            'valid_ratio': fedavg_valid_ratio[:min_length],
            'num_valid': fedavg_num_valid[:min_length],
            'gem_p': fedavg_gem_p[:min_length],
        })
        # 设置轮次为索引
        metrics_df.set_index('round', inplace=True)

        # 保存文件
        metrics_df.to_csv(result_pseudo_file, encoding='utf8')
        print(f"Metrics saved to {result_pseudo_file}")




if __name__ == '__main__':
    args = args_parser()
    if not torch.cuda.is_available():
        raise RuntimeError('FedGPLA reproduction requires a CUDA-capable GPU')
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed_all(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True  # cudnn
    fixmatch(args)
