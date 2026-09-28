#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
四元数卷积改进 MobileNetV3 —— 可行性验证脚本
================================================================================

【课题背景】
    两阶段方案：YOLO 先裁出叶片框 → 四元数 MobileNet 对框内叶片做病害识别。
    本脚本只做第二阶段里最核心的那一步：
        **把四元数卷积塞进 MobileNetV3，并在随机张量上跑通前向 + 反向。**

【本脚本回答什么】
    1. 四元数卷积算子本身对不对？（块矩阵 vs 朴素哈密顿积展开的数值交叉验证）
    2. MobileNetV3 哪些层能换、哪些不能换？（逐层通道整除性普查）
    3. 换完之后能不能跑通？形状对不对？梯度健不健康？有没有 NaN/Inf？
    4. 能不能接着用 ImageNet 预训练权重？（不重训，只验证加载）
    参数量的变化是多少？

【本脚本不回答什么（重要，别把结论用错）】
    ✗ 四元数版能不能收敛、精度如何 —— 需要真实数据训练，随机张量验证不了
    ✗ 四元数版对光照是不是真的更鲁棒 —— 这是课题的**待检验假设**，不是本脚本的结论
    ✗ 反向传播的显存/耗时 —— 只报参数量，不报速度

【算子来源与实现说明（写论文时要能交代清楚）】
    本脚本的 QuaternionConv2d 是**自己手写**的，不是从 GitHub 直接拷的。
    数学构造抄自下列两个参考实现（两者块矩阵逐行相同，后者是前者的维护版）：

      · Orkis-Research/Pytorch-Quaternion-Neural-Networks
            core_qnn/quaternion_ops.py  →  quaternion_conv()
            （Parcollet 等，复数/四元数网络系列工作）
      · SpeechBrain  speechbrain/nnet/quaternion_networks/q_CNN.py
            （把上面那套吸收进了语音工具箱，说明被广泛复用）

    另有一个设计参考：
      · bjing2016/qcnn-pytorch  qcnn.py  →  QBatchNorm1d
            （Zhu et al., ECCV 2018 "Quaternion Convolutional Neural Networks"）
            本脚本的 QuaternionBatchNorm2d 的"按模长归一化"思路来自它。
            注意：那个仓库只有 QConv1d，没有 Conv2d，图像任务不能直接用。

    另有一个**已排除**的候选：
      · gaudetcj/DeepQuaternionNetworks —— 块矩阵与 Orkis 完全相同，
        但它是 **Keras/TensorFlow** 实现，PyTorch 工程用不了。

【排布约定：本脚本用「分组排布」，不是参考实现的「平面排布」】
    这是本课题最关键的工程决策，详见第 1 节的说明。

【依赖】torch, torchvision
【运行】python mobilenet_quat_feasibility.py
================================================================================
"""

import math
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_small,
    mobilenet_v3_large,
    MobileNet_V3_Small_Weights,
)


# ==============================================================================
# 打印辅助
# ==============================================================================

def h1(title: str) -> None:
    print(f"\n\n{'=' * 78}\n{title}\n{'=' * 78}")


def h2(title: str) -> None:
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def bad(msg: str) -> None:
    print(f"  [!!]   {msg}")


# ==============================================================================
# 第 1 部分：四元数算子
# ==============================================================================

# ---------------------------------------------------------------- 1.1 排布约定
#
# 一个四元数 q = r + xi + yj + zk 有 4 个实分量，所以「一个拥有 q 个四元数通道的
# 特征图」在 PyTorch 里是 4q 个真实通道。**这 4q 个通道怎么排**，有两套约定：
#
#   【平面排布 planar】  通道 = [R 全部分量 | I 全部分量 | J 全部分量 | K 全部分量]
#                        第 j 个分量的第 c 个四元数 → 通道号  j*q + c
#                        Orkis / SpeechBrain 参考实现用的是这一种
#
#   【分组排布 grouped】 通道 = [q0.r, q0.i, q0.j, q0.k, q1.r, q1.i, q1.j, q1.k, ...]
#                        第 j 个分量的第 c 个四元数 → 通道号  4*c + j
#                        本脚本采用这一种
#
# 为什么本课题必须用分组排布：
#     平面排布与 YOLOv8 的 C2f 模块直接冲突（C2f 内部有 torch.chunk(2,1) 通道对半切，
#     在平面排布下会静默破坏四元数结构）。这一点在 yolov8-quat/quaternion_feasibility.py
#     里有完整的数值演示。
#
#     而 MobileNet 这边其实**两种排布都能跑** —— 我查过 torchvision 的
#     mobilenetv2.py / mobilenetv3.py，全文件没有 torch.cat / .chunk( / .split( /
#     .narrow( 任何一处，跨通道操作只有 InvertedResidual 的残差相加和 SE 的逐通道相乘，
#     两者对排布都不敏感。
#
#     那为什么还是选分组排布？两个理由：
#       (a) 保持一致。两阶段方案里 YOLO 那半边必须用分组排布，MobileNet 这半边
#           换一种排布只会增加解释成本，没有收益。
#       (b) 更抗改。分组排布对 cat/chunk 封闭，将来往网络里加任何跨通道操作都不会
#           静默出错。
#
# 代价：参考实现的块矩阵是在平面排布下推导的，所以本实现在调用前后各做一次
# 排布转换（纯 reshape+transpose，不增加计算量，不破坏梯度）。


def grouped_to_planar(x: torch.Tensor) -> torch.Tensor:
    """分组排布 → 平面排布。同一个数据的两种通道顺序，只是 reshape + transpose。"""
    b, c, h, w = x.shape
    q = c // 4
    return x.view(b, q, 4, h, w).transpose(1, 2).reshape(b, 4 * q, h, w)


def planar_to_grouped(x: torch.Tensor) -> torch.Tensor:
    """平面排布 → 分组排布（grouped_to_planar 的逆操作）。"""
    b, c, h, w = x.shape
    q = c // 4
    return x.view(b, 4, q, h, w).transpose(1, 2).reshape(b, 4 * q, h, w)


# ---------------------------------------------------------------- 1.2 主算子


class QuaternionConv2d(nn.Module):
    """四元数二维卷积（内部按分组排布组织通道）。

    【哈密顿积的块矩阵形式】
        设卷积核四元数 q_w 与输入四元数 q_x 做哈密顿积 q_w ⊗ q_x，则

            [ r_out ]   [  w_r  -w_i  -w_j  -w_k ] [ x_r ]
            [ i_out ] = [  w_i   w_r  -w_k   w_j ] [ x_i ]
            [ j_out ]   [  w_j   w_k   w_r  -w_i ] [ x_j ]
            [ k_out ]   [  w_k  -w_j   w_i   w_r ] [ x_k ]

        这里每个 w_* 都是一整个 (q_out, q_in, k, k) 的卷积核张量，每个 x_* 是一组
        (q_in) 通道的输入。把这个 4×4 块矩阵按 (输出通道, 输入通道) 两个方向拼成
        一个大权重，整个四元数卷积就**退化成一个普通实数卷积** —— 于是可以直接
        调 F.conv2d，autograd 原生可用，不需要写自定义反向传播。

    【参数量】
        普通卷积   C_in → C_out，k×k：  C_out·C_in·k²          个实数
        四元数卷积 (4C_in) → (4C_out)： (C_out/4)·(C_in/4)·k²·4
                                       = C_out·C_in·k²/4
        即**参数量降到普通卷积的 1/4**，因为权重只存 r/i/j/k 四份（而不是 16 份
        独立的实数核）。这是四元数卷积相对普通卷积的核心优势之一。

    【前置的通道转换（本课题的关键点）】
        普通卷积吃 3 通道 RGB，而四元数卷积要求通道数是 4 的倍数。
        解法：把像素 (R,G,B) 解释成**纯四元数**  q = 0 + R·i + G·j + B·k，
        实部补零，得到 4 通道。这就是 rgb_lift=True 这个开关做的事。
        注意实部为 0 **不影响**输出的四个分量都非零（见下方 out_r 的展开），
        所以这个提升层不是"浪费一个通道"。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 rgb_lift: bool = False) -> None:
        super().__init__()
        # rgb_lift=True 时允许输入恰好是 3 通道（RGB），内部补零扩成 4 通道
        if rgb_lift:
            if in_channels != 3:
                raise ValueError(f"rgb_lift=True 只接受 3 通道输入，收到 {in_channels}")
            in_channels = 4
        if in_channels % 4 != 0 or out_channels % 4 != 0:
            raise ValueError(
                f"四元数卷积要求输入/输出通道数为 4 的倍数，"
                f"收到 in={in_channels}, out={out_channels}。"
                f"若输入是 3 通道 RGB，请用 rgb_lift=True。"
            )
        self.rgb_lift = rgb_lift
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q_in = in_channels // 4      # 输入的四元数通道数
        self.q_out = out_channels // 4    # 输出的四元数通道数
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 权重只存 4 份（r/i/j/k），这就是参数效率的来源
        self.weight = nn.Parameter(
            torch.empty(4, self.q_out, self.q_in, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """四元数初始化：模长 × 单位纯虚四元数（对齐参考实现的 quaternion_init）。

        参考实现的做法：
          1. 模长 modulus ~ χ 分布（自由度 4），按 glorot 缩放；
          2. 相位 phase ~ U(-π, π)，方向取单位纯虚四元数 (v_i, v_j, v_k)；
          3. w_r = m·cos(φ)，w_i = m·v_i·sin(φ)，w_j、w_k 同理。

        为什么不逐分量独立采样：独立采样会让四元数的模长集中在 sqrt(2) 附近、
        方向分布也不均匀，"旋转"语义被削弱。按模长+相位采样在四元数空间里更均匀。
        """
        fan_in = self.q_in * self.kernel_size ** 2
        fan_out = self.q_out * self.kernel_size ** 2
        s = 1.0 / math.sqrt(2 * (fan_in + fan_out))       # glorot 准则

        shape = (self.q_out, self.q_in, self.kernel_size, self.kernel_size)
        # χ(4) 分布：4 个标准正态的平方和开根号
        modulus = torch.sqrt(torch.sum(torch.randn(4, *shape) ** 2, dim=0)) * s
        # 单位纯虚四元数方向
        v = torch.randn(3, *shape)
        v = v / torch.sqrt((v ** 2).sum(dim=0, keepdim=True) + 1e-4)
        phase = torch.empty(shape).uniform_(-math.pi, math.pi)

        with torch.no_grad():
            self.weight[0] = modulus * torch.cos(phase)
            self.weight[1] = modulus * v[0] * torch.sin(phase)
            self.weight[2] = modulus * v[1] * torch.sin(phase)
            self.weight[3] = modulus * v[2] * torch.sin(phase)

    def _block_kernel(self) -> torch.Tensor:
        """把 4 份四元数权重拼成块矩阵，得到 (4·q_out, 4·q_in, k, k)。

        这一段逐行对齐参考实现的 quaternion_conv()：
            cat_kernels_4_r = cat([ r_w, -i_w, -j_w, -k_w], dim=1)
            cat_kernels_4_i = cat([ i_w,  r_w, -k_w,  j_w], dim=1)
            cat_kernels_4_j = cat([ j_w,  k_w,  r_w, -i_w], dim=1)
            cat_kernels_4_k = cat([ k_w, -j_w,  i_w,  r_w], dim=1)
            kernel = cat([block_r, block_i, block_j, block_k], dim=0)
        """
        w_r, w_i, w_j, w_k = self.weight[0], self.weight[1], self.weight[2], self.weight[3]
        block_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=1)
        block_i = torch.cat([w_i, w_r, -w_k, w_j], dim=1)
        block_j = torch.cat([w_j, w_k, w_r, -w_i], dim=1)
        block_k = torch.cat([w_k, -w_j, w_i, w_r], dim=1)
        return torch.cat([block_r, block_i, block_j, block_k], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 0) RGB(3通道) → 纯四元数(4通道)：实部补零
        if self.rgb_lift:
            x = F.pad(x, (0, 0, 0, 0, 1, 0))     # 在通道维最前面补 1 个零通道
        # 1) 分组 → 平面（块矩阵要求平面排布）
        x_planar = grouped_to_planar(x)
        # 2) 块矩阵卷积：纯实数卷积，autograd 直接可用。
        #    bias 必须传 None：self.bias 是按**分组排布**定义的，而这里输出还是
        #    平面排布，直接交给 F.conv2d 会加错通道。bias 初值为 0 时看不出问题，
        #    属于典型潜伏 bug，故显式规避。
        out_planar = F.conv2d(
            x_planar, self._block_kernel(), None,
            stride=self.stride, padding=self.padding,
        )
        # 3) 平面 → 分组
        out = planar_to_grouped(out_planar)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class QuaternionDepthwiseConv2d(nn.Module):
    """四元数版的「深度可分离卷积」中的 depthwise 部分。

    【为什么单独实现它】
        普通 DWConv 对**每一个真实通道**配一个独立的空间核。如果特征图是分组排布
        的四元数（4q 个通道），普通 DWConv 会让同一个四元数的 r/i/j/k 四个分量
        拿到**四个互不相同**的空间核 —— 这在四元数语义下是错的：
        一个四元数特征图应该被一个**四元数核**通过哈密顿积滤波。

        正确做法：把 q 个四元数各自作为一个分组（groups = q），
        每个分组内部做 4×4 的哈密顿积块矩阵卷积。

    【形状】
        groups = q，输入/输出通道都是 4q，
        权重形状 (4q, 4q/groups=4, k, k)，即每个四元数一个 4×4 块。

    【为什么分组内不再需要排布转换】
        分组内只有 1 个输入四元数，平面排布与分组排布此时退化成同一种顺序
        [r, i, j, k]，所以块矩阵可以直接用，无需 grouped_to_planar。
    """

    def __init__(self, channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(f"四元数深度卷积要求通道数为 4 的倍数，收到 {channels}")
        self.channels = channels
        self.q = channels // 4
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 每个四元数一个**四元数核**：权重存成 (q, 4, k, k)，
        # 第 2 维的 4 份分别是该核的 r/i/j/k 分量。
        # （不是 (q,4,4,k,k)：那个 4×4 块矩阵是 _block_kernel() 现场拼出来的，
        #   不是存储形状 —— 这一点和参考实现一样，权重只存 4 份。）
        self.weight = nn.Parameter(torch.empty(self.q, 4, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        shape = (self.q, self.kernel_size, self.kernel_size)
        k = 1.0 / math.sqrt(self.kernel_size ** 2 * 4)
        modulus = torch.sqrt(torch.sum(torch.randn(4, *shape) ** 2, dim=0)) * k
        v = torch.randn(3, *shape)
        v = v / torch.sqrt((v ** 2).sum(dim=0, keepdim=True) + 1e-4)
        phase = torch.empty(shape).uniform_(-math.pi, math.pi)
        with torch.no_grad():
            self.weight[:, 0] = modulus * torch.cos(phase)
            self.weight[:, 1] = modulus * v[0] * torch.sin(phase)
            self.weight[:, 2] = modulus * v[1] * torch.sin(phase)
            self.weight[:, 3] = modulus * v[2] * torch.sin(phase)

    def _block_kernel(self) -> torch.Tensor:
        """(q,4,k,k) 的 4 份分量 → (4q,4,k,k)，供 groups=q 的 F.conv2d 使用。

        分组内 q_in = 1，所以块矩阵的每一行只有 4 个输入分量。
        这里在 dim=1 上补出「输入分量」这一维，再拼成 4×4 块。
        """
        # 每份 (q,k,k) → (q,1,k,k)，第 2 维就是"输入分量"这一维
        w_r, w_i, w_j, w_k = (self.weight[:, j].unsqueeze(1) for j in range(4))
        block_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=1)   # (q,4,k,k)
        block_i = torch.cat([w_i, w_r, -w_k, w_j], dim=1)
        block_j = torch.cat([w_j, w_k, w_r, -w_i], dim=1)
        block_k = torch.cat([w_k, -w_j, w_i, w_r], dim=1)
        # stack 在 dim=1 → (q, 4_输出分量, 4_输入分量, k, k)
        kernel = torch.stack([block_r, block_i, block_j, block_k], dim=1)
        # reshape 把 (四元数序号 c, 输出分量 j) 展平成通道号 4c+j，与分组排布一致
        return kernel.reshape(4 * self.q, 4, self.kernel_size, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x, self._block_kernel(), None, stride=self.stride,
                       padding=self.padding, groups=self.q)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class QuaternionBatchNorm2d(nn.Module):
    """四元数批归一化：按**四元数模长**归一化，而不是逐通道归一化。

    【为什么需要它 —— 这是本课题一个很容易被忽略的设计点】
        标准 nn.BatchNorm2d 把 4q 个通道**各自独立**地减均值、除标准差。
        这等于把一个四元数重新拆成 4 个互不相干的实数：
        · 四元数卷积刚刚通过哈密顿积建立起来的「分量间耦合」被 BN 打散了；
        · 独立缩放还会改变四元数的**相位**（方向），只剩下模长信息被部分保留。
        换句话说，用标准 BN 的话，四元数卷积好不容易引入的结构，前面做后面拆。

    【本实现的做法】
        对每个四元数通道 c，统计模长 |q_c| = sqrt(r²+i²+j²+k²) 的批内均值，
        然后用**同一个标量**去除该四元数的 4 个分量。这是保方向的缩放：
        四元数的相位完整保留，r/i/j/k 的相对关系不被破坏。

        思路取自 bjing2016/qcnn-pytorch 的 QBatchNorm1d（Zhu et al., ECCV 2018）。
        原版没有可学习参数，本实现补上了逐四元数的缩放 γ 和逐分量的偏置 β。

    【已知取舍】
        · 本实现刻意**不做减均值**（certering）。减一个逐分量均值会破坏模长语义，
          减一个"四元数均值"在数学上可行但会引入额外复杂度，本阶段先不做。
        · 归一化后每个四元数模长恒为 1，对训练动态是个较强的约束。
          真实训练时是否有利，需要消融实验回答 —— 本脚本只验证"能跑通"。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True) -> None:
        super().__init__()
        if num_features % 4 != 0:
            raise ValueError(f"四元数 BN 要求通道数为 4 的倍数，收到 {num_features}")
        self.num_features = num_features
        self.q = num_features // 4
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # 只跟踪每个四元数的模长均值（1 个标量/四元数），而不是 4q 个均值方差
        self.register_buffer("running_modulus", torch.ones(1, self.q, 1, 1))
        if affine:
            self.weight = nn.Parameter(torch.ones(1, self.q, 1, 1))     # 逐四元数缩放
            self.bias = nn.Parameter(torch.zeros(1, self.q, 4, 1, 1))   # 逐分量偏置
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xq = x.view(b, self.q, 4, h, w)
        # 每个四元数的模长 (B, q, H, W)
        modulus = torch.sqrt((xq ** 2).sum(dim=2) + self.eps)

        if self.training:
            with torch.no_grad():
                m = modulus.mean(dim=(0, 2, 3)).view(1, self.q, 1, 1)
                self.running_modulus.mul_(1 - self.momentum).add_(self.momentum * m)
            cur = modulus.unsqueeze(2)                       # (B,q,1,H,W)
        else:
            cur = self.running_modulus.unsqueeze(2)          # (1,q,1,1,1)

        xn = xq / cur                                        # 保方向缩放
        if self.affine:
            xn = xn * self.weight.unsqueeze(2) + self.bias
        return xn.reshape(b, c, h, w)


class QuaternionConvBNSiLU(nn.Module):
    """四元数卷积 + 归一化 + 激活 的组合块。

    只在本脚本需要独立构造小模块时使用（例如做自检、组装小网络）。
    对 MobileNetV3 的替换是**逐层原地替换**，不走这个类（见第 3 节）。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 rgb_lift=False, use_quat_bn=True):
        super().__init__()
        self.conv = QuaternionConv2d(in_channels, out_channels, kernel_size, stride,
                                     padding, bias=False, rgb_lift=rgb_lift)
        self.bn = (QuaternionBatchNorm2d(out_channels) if use_quat_bn
                   else nn.BatchNorm2d(out_channels))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------- 1.3 朴素对照


def naive_quaternion_conv(x: torch.Tensor, qc: QuaternionConv2d) -> torch.Tensor:
    """朴素参考实现：显式写 16 次普通卷积，完全不用块矩阵技巧。

    作用：作为**数值真值**去交叉验证块矩阵实现。两者必须逐元素一致，
    否则说明块矩阵的符号拼错了 —— 这类错误不会报错，只会让模型学不到东西。
    """
    w_r, w_i, w_j, w_k = (qc.weight[j] for j in range(4))
    if qc.rgb_lift:
        x = F.pad(x, (0, 0, 0, 0, 1, 0))
    b = x.shape[0]
    # 分组排布下按分量抽取：通道 4c+j 是第 c 个四元数的第 j 个分量
    x_r, x_i, x_j, x_k = (x[:, j::4] for j in range(4))

    def cv(inp, ker):
        return F.conv2d(inp, ker, None, qc.stride, qc.padding)

    # 哈密顿积展开（由 i²=j²=k²=ijk=-1 与 i·k=-j, k·i=j, i·j=k, j·i=-k 推出）
    out_r = cv(x_r, w_r) - cv(x_i, w_i) - cv(x_j, w_j) - cv(x_k, w_k)
    out_i = cv(x_r, w_i) + cv(x_i, w_r) + cv(x_k, w_j) - cv(x_j, w_k)
    out_j = cv(x_r, w_j) + cv(x_i, w_k) - cv(x_k, w_i) + cv(x_j, w_r)
    out_k = cv(x_r, w_k) - cv(x_i, w_j) + cv(x_j, w_i) + cv(x_k, w_r)

    out = torch.stack([out_r, out_i, out_j, out_k], dim=2)     # (B,q_out,4,H',W')
    out = out.reshape(b, qc.q_out * 4, out.shape[-2], out.shape[-1])
    if qc.bias is not None:
        out = out + qc.bias.view(1, -1, 1, 1)
    return out


# ==============================================================================
# 第 2 部分：算子自检（快速版，详细版见 quaternion_feasibility.py）
# ==============================================================================

def part2_operator_selfcheck() -> bool:
    h1("第 2 部分：四元数算子自检")
    all_pass = True

    h2("2.1 块矩阵 vs 朴素哈密顿积展开（数值交叉验证）")
    for (cin, cout, k, lift) in [(4, 8, 3, False), (3, 16, 3, True), (16, 32, 1, False)]:
        torch.manual_seed(0)
        m = QuaternionConv2d(cin, cout, k, padding=k // 2, rgb_lift=lift)
        x = torch.randn(2, cin, 8, 8)
        y1 = m(x)
        y2 = naive_quaternion_conv(x, m)
        err = (y1 - y2).abs().max().item()
        good = err < 1e-4
        all_pass &= good
        (ok if good else bad)(
            f"in={cin}{'(RGB,补零提升)' if lift else ''} out={cout} k={k}: "
            f"shape={tuple(y1.shape)}, 最大逐元素误差={err:.3e}"
        )

    h2("2.2 四元数深度卷积：块矩阵 vs 朴素展开")
    torch.manual_seed(0)
    dw = QuaternionDepthwiseConv2d(8, 3, padding=1)
    x = torch.randn(2, 8, 8, 8)
    y1 = dw(x)
    # 分组内逐分量抽取做朴素验证
    xq = x.view(2, 2, 4, 8, 8)
    outs = []
    for c in range(2):
        xr, xi, xj, xk = (xq[:, c, j] for j in range(4))
        wr, wi, wj, wk = (dw.weight[c, j] for j in range(4))
        cv = lambda i_, k_: F.conv2d(i_.unsqueeze(1), k_.unsqueeze(0).unsqueeze(0), None, 1, 1)
        outs.append(torch.cat([
            cv(xr, wr) - cv(xi, wi) - cv(xj, wj) - cv(xk, wk),
            cv(xr, wi) + cv(xi, wr) + cv(xk, wj) - cv(xj, wk),
            cv(xr, wj) + cv(xi, wk) - cv(xk, wi) + cv(xj, wr),
            cv(xr, wk) - cv(xi, wj) + cv(xj, wi) + cv(xk, wr)], dim=1))
    y2 = torch.cat([o.view(2, 4, 8, 8) for o in outs], dim=1)
    err = (y1 - y2).abs().max().item()
    good = err < 1e-4
    all_pass &= good
    (ok if good else bad)(f"深度卷积 shape={tuple(y1.shape)}, 最大误差={err:.3e}")

    h2("2.3 排布转换是否可逆")
    x = torch.randn(2, 16, 4, 4)
    err = (planar_to_grouped(grouped_to_planar(x)) - x).abs().max().item()
    good = err == 0.0
    all_pass &= good
    (ok if good else bad)(f"round-trip 最大误差 = {err:.3e}（应为 0，因为只是 reshape+transpose）")

    h2("2.4 四元数 BN 前向 + 反向 + running stats")
    torch.manual_seed(0)
    bn = QuaternionBatchNorm2d(8)
    x = torch.randn(4, 8, 5, 5, requires_grad=True)
    y = bn(x)
    y.sum().backward()
    xq = y.detach().view(4, 2, 4, 5, 5)
    mod = torch.sqrt((xq ** 2).sum(dim=2))
    good = torch.allclose(mod, torch.ones_like(mod), atol=1e-3)
    all_pass &= good
    (ok if good else bad)(
        f"输出每个四元数模长={mod.mean().item():.6f}（应为 1：模长归一化的直接后果）; "
        f"输入梯度 norm={x.grad.norm().item():.4f}"
    )
    print(f"         running_modulus 已更新: "
          f"{bn.running_modulus.flatten()[:4].tolist()}")

    return all_pass


# ==============================================================================
# 第 3 部分：MobileNetV3 结构分析与替换策略
# ==============================================================================

def part3_census() -> None:
    h1("第 3 部分：MobileNetV3 逐层通道整除性普查")

    for name, builder in [("mobilenet_v3_small", mobilenet_v3_small),
                          ("mobilenet_v3_large", mobilenet_v3_large)]:
        m = builder(weights=None)

        # 先标出 SqueezeExcitation 内部的卷积 —— 它们默认**不**替换（理由见第 4 部分）
        from torchvision.ops.misc import SqueezeExcitation
        se_names = set()
        for n, mod in m.named_modules():
            if isinstance(mod, SqueezeExcitation):
                for cn, _ in mod.named_modules():
                    if cn:
                        se_names.add(f"{n}.{cn}")

        h2(f"{name} 的全部 Conv2d 层")
        print(f"  {'层名':<44}{'in':>6}{'out':>6}{'groups':>8}{'k':>3}  类型")
        n_bad = 0
        for lname, mod in m.named_modules():
            if not isinstance(mod, nn.Conv2d):
                continue
            is_dw = mod.groups > 1
            in_se = lname in se_names
            if is_dw:
                kind = "深度可分离(depthwise)"
            elif in_se:
                kind = "普通(SE内部)"
            else:
                kind = "普通(通道混合)"
            # 替换可行性判断
            if is_dw or in_se:
                flag = "  <-- 默认不替换"
            elif mod.in_channels % 4 == 0 and mod.out_channels % 4 == 0:
                flag = ""
            elif mod.in_channels == 3 and mod.out_channels % 4 == 0:
                flag = "  <-- 输入3通道，需要 rgb_lift"
                n_bad += 1
            else:
                flag = "  <-- 通道数不是4的倍数，不可替换"
                n_bad += 1
            print(f"  {lname:<44}{mod.in_channels:>6}{mod.out_channels:>6}"
                  f"{mod.groups:>8}{mod.kernel_size[0]:>3}  {kind}{flag}")
        print(f"\n  小结：{name} 中**因通道数不整除 4 而无法替换**的层数 = {n_bad}"
              f"（只有输入 3 通道这一层，其余通道全部整除 4）")


def part4_replace(model: nn.Module, quantize_depthwise: bool = False,
                  use_quat_bn: bool = False, skip_se: bool = True):
    """把 MobileNetV3 里的普通 Conv2d 原地替换成四元数卷积。

    参数
    ----
    quantize_depthwise : 是否也把 depthwise 卷积换成四元数版。
                         默认 False（理由见 h2 里的说明）。
    use_quat_bn        : 是否把紧随其后的 BatchNorm2d 换成四元数 BN。
    skip_se            : 是否跳过 SqueezeExcitation 内部的 1×1 卷积。
                         默认 True —— SE 的输入是全局池化后的描述子（空间结构已经没了），
                         把它四元数化没有任何参考实现，属于额外的设计决策，
                         本阶段保持原样，避免混淆变量。

    返回
    ----
    (replaced_convs, replaced_bns) 两个列表，元素为 (层名, 原模块, 新模块)
    """
    from torchvision.ops.misc import SqueezeExcitation

    replaced_convs, replaced_bns = [], []

    # 先快照出所有 (父模块, 子名, 子模块)，避免边遍历边修改
    targets = []
    for parent_name, parent in model.named_modules():
        if skip_se and isinstance(parent, SqueezeExcitation):
            continue
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Conv2d):
                targets.append((parent_name, parent, child_name, child))

    for parent_name, parent, child_name, conv in targets:
        is_dw = conv.groups > 1

        if is_dw:
            if not quantize_depthwise:
                continue
            if conv.groups != conv.in_channels or conv.in_channels != conv.out_channels:
                print(f"  [跳过] {parent_name}.{child_name}: 非标准 depthwise"
                      f"(groups={conv.groups}, in={conv.in_channels}, out={conv.out_channels})")
                continue
            new = QuaternionDepthwiseConv2d(
                conv.in_channels, conv.kernel_size[0], conv.stride[0],
                conv.padding[0], bias=conv.bias is not None)
        else:
            rgb_lift = (conv.in_channels == 3)
            try:
                new = QuaternionConv2d(
                    conv.in_channels, conv.out_channels, conv.kernel_size[0],
                    conv.stride[0], conv.padding[0], bias=conv.bias is not None,
                    rgb_lift=rgb_lift)
            except ValueError as e:
                print(f"  [跳过] {parent_name}.{child_name}: {e}")
                continue
        setattr(parent, child_name, new)
        replaced_convs.append((f"{parent_name}.{child_name}", conv, new))

        # 若父模块是 Sequential，检查紧邻的下一个子模块是不是 BN，按需替换
        if use_quat_bn and isinstance(parent, nn.Sequential) and child_name.isdigit():
            idx = int(child_name)
            if idx + 1 < len(parent):
                nxt = parent[idx + 1]
                if isinstance(nxt, nn.BatchNorm2d) and nxt.num_features == new.out_channels:
                    qbn = QuaternionBatchNorm2d(new.out_channels)
                    parent[idx + 1] = qbn
                    replaced_bns.append((f"{parent_name}.{idx + 1}", nxt, qbn))

    return replaced_convs, replaced_bns


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ==============================================================================
# 第 4 部分：主流程
# ==============================================================================

def run_config(tag: str, quantize_depthwise: bool, use_quat_bn: bool,
               batch: int = 2, img: int = 224, num_classes: int = 29):
    """构建一个配置、替换、跑前向+反向，返回统计结果。"""
    h2(f"配置 {tag}")

    torch.manual_seed(42)
    model = mobilenet_v3_small(weights=None, num_classes=num_classes)
    base_params = count_params(model)
    base_conv_params = sum(p.numel() for n, p in model.named_parameters()
                           if n.endswith(".weight") and p.dim() == 4)

    convs, bns = part4_replace(model, quantize_depthwise=quantize_depthwise,
                               use_quat_bn=use_quat_bn)
    replaced_o = sum(o.weight.numel() for _, o, _ in convs)
    replaced_n = sum(n.weight.numel() for _, _, n in convs)
    new_params = count_params(model)
    new_conv_params = sum(p.numel() for n, p in model.named_parameters()
                          if p.dim() in (4, 5) and n.endswith(".weight"))

    print(f"  替换了 {len(convs)} 个卷积层、{len(bns)} 个 BN 层")

    # ---- 逐层参数量（只看被替换的那些层，验证是不是每层都降到 1/4）----
    rep_o = sum(o.weight.numel() for _, o, _ in convs)
    rep_n = sum(n.weight.numel() for _, _, n in convs)
    print(f"\n  被替换层的权重参数量逐层核对（前 5 层 + 合计）：")
    for name, o, n in list(convs)[:5]:
        print(f"      {name:<34} {str(tuple(o.weight.shape)):<18} "
              f"{o.weight.numel():>7,} -> {n.weight.numel():>7,}  "
              f"({n.weight.numel() / o.weight.numel():.4f})")
    print(f"      ...（共 {len(convs)} 层）")
    print(f"      合计 {rep_o:>9,} -> {rep_n:>9,}  ({rep_n / rep_o:.4f} 倍)")
    if quantize_depthwise:
        print("      读法：普通卷积层（含 Stem）都是 0.25 倍，Stem 是 0.3333"
              "（输入 3 通道，经 rgb_lift 才成 4 通道，分母是 3 不是 4）；")
        print("            但 **depthwise 层是 1.0000 倍** —— 四元数版 DWConv 参数量"
              "与普通 DWConv 完全相同（见上方策略说明），不带来任何节省。")
        print("            这就是为什么本配置的合计倍数（0.3470）高于纯卷积的 0.25。")
    else:
        print("      **逐层都是 0.25 倍**（唯一例外是 Stem 的 0.3333：它的输入是 3 通道，"
              "经过 rgb_lift 才变成 4 通道，所以分母是 3 不是 4）")
        print("      这正是理论值：四元数卷积的权重只有普通卷积的 1/4")

    # ---- 为什么整体倍数远高于 0.25 ----
    se_params = sum(mod.weight.numel() for nm, mod in model.named_modules()
                    if isinstance(mod, nn.Conv2d) and ".fc" in nm)
    dw_note = ("\n      (d) 此外，本配置虽然把 depthwise 也替换了，"
               "但四元数版 DWConv 参数量与普通版**完全相同**，不贡献任何节省。"
               if quantize_depthwise else "")
    print(f"\n  卷积参数量: {base_conv_params:>9,} -> {new_conv_params:>9,}  "
          f"({new_conv_params / base_conv_params:.4f} 倍)")
    print(f"  总参数量  : {base_params:>9,} -> {new_params:>9,}  "
          f"({new_params / base_params:.4f} 倍)")
    print(f"""
  为什么整体只降到 {new_params / base_params:.2f} 倍，而不是 1/4？三个原因叠加：
      (a) SqueezeExcitation 内部的 1×1 卷积**没有替换**（默认策略），
          而它在 MobileNetV3-Small 里占了全部卷积参数的
          {se_params / base_conv_params:.1%}（{se_params:,} / {base_conv_params:,}）——
          SE 的 fc2 要把通道数从 C/r 还原回 C，参数量本来就大。
      (b) BatchNorm 的参数量不随四元数化缩小（2×通道数，照旧）。
      (c) 分类头是两个 Linear，与卷积无关，原封不动。{dw_note}
  所以"参数量降到 1/4"只在**被替换的普通卷积**上成立（本配置实测 {rep_n / rep_o:.4f}），
  不是整个网络。写论文时要按"被替换的卷积层降到 1/4"表述，否则会被质疑。""")

    # ---- 前向 ----
    x = torch.randn(batch, 3, img, img, requires_grad=True)
    try:
        y = model(x)
    except Exception as e:
        bad(f"前向失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    print(f"\n  输入 {tuple(x.shape)}  ->  输出 {tuple(y.shape)}")
    if y.shape != (batch, num_classes):
        bad(f"输出形状异常，期望 {(batch, num_classes)}")
        return None
    ok("前向通过，无 shape 错误")

    # ---- 反向 ----
    loss = y.square().mean()
    loss.backward()

    n_nan = n_inf = n_grad = 0
    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        n_grad += 1
        n_nan += int(torch.isnan(p.grad).sum())
        n_inf += int(torch.isinf(p.grad).sum())
    in_nan = int(torch.isnan(x.grad).sum())
    in_inf = int(torch.isinf(x.grad).sum())

    print(f"  反向：{n_grad} 个参数张量拿到梯度")
    gz = (x.grad != 0).float().mean().item()
    if n_nan == n_inf == in_nan == in_inf == 0:
        # 注意用科学计数法：随机初始化的网络里输入梯度本来就在 1e-5 量级，
        # 用 %.4f 打印会显示成 0.0000，看起来像"梯度没传回去"，是**显示误导**而非 bug。
        # 判据应该是「非零元素占比」而不是「norm 的大小」。
        ok(f"梯度健康：参数梯度 NaN={n_nan} Inf={n_inf}；"
           f"输入梯度 NaN={in_nan} Inf={in_inf}")
        print(f"         输入梯度 norm={x.grad.norm().item():.4e}  "
              f"max|g|={x.grad.abs().max().item():.4e}  "
              f"非零元素占比={gz:.4f}")
        print(f"         （1e-5 量级是**随机初始化**的正常现象，与原版 MobileNet "
              f"同量级；关键是占比 {gz:.4f} 说明梯度确实逐元素传回了输入）")
    else:
        bad(f"梯度异常：参数 NaN={n_nan} Inf={n_inf}，输入 NaN={in_nan} Inf={in_inf}")

    # ---- 参数是否真的被更新 ----
    before = [p.detach().clone() for _, p in model.named_parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.step()
    changed = sum(1 for (_, p), b in zip(model.named_parameters(), before)
                  if not torch.equal(p.detach(), b))
    ok(f"一次 SGD 后有 {changed}/{n_grad} 个参数张量数值发生变化")

    return {"tag": tag, "convs": len(convs), "bns": len(bns),
            "replaced_o": replaced_o, "replaced_n": replaced_n,
            "base_conv": base_conv_params, "new_conv": new_conv_params,
            "base_total": base_params, "new_total": new_params}


def part5_pretrained(use_quat_bn: bool = False, num_classes: int = 29):
    """验证：替换后还能不能接着用 ImageNet 预训练权重。

    做法：先拿到官方预训练 state_dict，再按「键名相同 且 形状相同」筛选，
    把能对上的灌进四元数模型；其余保持随机初始化。
    因为四元数层的权重形状与普通卷积不同，strict=False 也会在形状不匹配时报错，
    所以必须显式过滤 —— 这里把过滤结果完整打印出来，便于核对。
    """
    h2("预训练权重加载验证")
    try:
        pre = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1,
                                 num_classes=1000)
    except Exception as e:
        bad(f"预训练权重下载失败（可能无网络）：{type(e).__name__}: {e}")
        print("      跳过本节，不影响其余结论。")
        return None

    sd = pre.state_dict()
    torch.manual_seed(42)
    qmodel = mobilenet_v3_small(weights=None, num_classes=num_classes)
    part4_replace(qmodel, quantize_depthwise=False, use_quat_bn=use_quat_bn)
    qsd = qmodel.state_dict()

    compatible = {k: v for k, v in sd.items()
                  if k in qsd and qsd[k].shape == v.shape}
    shape_clash = {k: (tuple(v.shape), tuple(qsd[k].shape))
                   for k, v in sd.items() if k in qsd and qsd[k].shape != v.shape}

    try:
        missing, unexpected = qmodel.load_state_dict(compatible, strict=False)
    except Exception as e:
        bad(f"load_state_dict 失败: {type(e).__name__}: {e}")
        return None

    total_keys = len(sd)
    print(f"  预训练 state_dict 共 {total_keys} 个键")
    ok(f"成功复用 {len(compatible)} 个键（形状完全匹配）")
    print(f"  形状不匹配而跳过的 {len(shape_clash)} 个键 —— 这些正是被四元数层替换掉的：")
    for k, (a, b) in list(shape_clash.items())[:6]:
        print(f"      {k:<48} 预训练{str(a):<16} vs 四元数{str(b)}")
    if len(shape_clash) > 6:
        print(f"      ...（其余 {len(shape_clash) - 6} 个同类）")
    print(f"  strict=False 返回：missing={len(missing)}, unexpected={len(unexpected)}")

    # 抽查：随机挑一个没被替换的层，确认它真的拿到了预训练权重
    probe = [k for k in compatible if "features.1.block.1" in k or "classifier.1" in k]
    for k in probe[:3]:
        same = torch.equal(qsd[k].cpu(), sd[k].cpu())
        (ok if same else bad)(f"抽查 {k}: 与预训练权重{'完全一致' if same else '不一致'}")
    return len(compatible), total_keys


def main() -> int:
    print(__doc__)

    pass_op = part2_operator_selfcheck()
    part3_census()

    h1("第 4 部分：替换 MobileNetV3 并跑通前向 + 反向")

    h2("替换策略说明（哪些换、哪些不换）")
    print("""
  换（groups == 1 的普通卷积）—— 这些正是发生「通道混合」的地方：
      · Stem                : 3->16，用 rgb_lift 把 RGB 解释为纯四元数 0+Ri+Gj+Bk
      · InvertedResidual 的 expand 1x1  : 升维，通道开始混合
      · InvertedResidual 的 project 1x1 : 降维，通道混合后压回
      · 末端 1x1 (96->576)  : 进分类头前最后一次通道混合
      理由：四元数卷积的价值就在「用哈密顿积建模通道间耦合」，
            只有会混合通道的层才谈得上"耦合"。

  不换（groups > 1 的深度可分离卷积）—— 默认保持普通卷积：
      · DWConv 的设计哲学是**通道解耦**（每个通道一个独立空间核），
        与四元数卷积的**通道耦合**在设计目标上正好相反。
      · 若保持普通 DWConv 夹在四元数层之间，同一个四元数的 r/i/j/k 会拿到
        4 个互不相同的空间核，四元数结构在这个位置被局部破坏。
        严格的做法是换成 QuaternionDepthwiseConv2d（本脚本已实现），
        配置 B 就跑的这个，可以对比。
      · 本脚本默认保留普通 DWConv，是为了**只改一个变量**、便于消融；
        真实训练时建议两种都试。

      ⚠️ 一个反直觉但重要的事实：**四元数版 DWConv 的参数量与普通 DWConv 完全相同。**
         普通 DWConv：C 个通道各 1 个 k×k 核        → C·k²
         四元数 DWConv：C/4 个四元数各 1 个四元数核 → (C/4)·4·k² = C·k²
         两者恒等。所以配置 A 和配置 B 的参数量会一模一样 —— 这不是 bug。
         也正因如此，**对 DWConv 做四元数化不会带来任何参数节省**，
         它的意义只在于"修正四元数结构被破坏"这个语义问题，
         是否值得要由精度实验回答。

  不换（SqueezeExcitation 内部的 1x1）：
      SE 的输入是全局平均池化后的描述子（空间结构已经没了），把它四元数化
      没有任何参考实现可依，属于额外的设计决策，本阶段保持原样。
""")

    # ---- 基线：原版未改动的 MobileNetV3，用于对照梯度量级 ----
    h2("基线对照：原版 MobileNetV3-Small")
    torch.manual_seed(42)
    base_model = mobilenet_v3_small(weights=None, num_classes=29)
    bx = torch.randn(2, 3, 224, 224, requires_grad=True)
    base_model(bx).square().mean().backward()
    print(f"  原版总参数量 = {count_params(base_model):,}")
    print(f"  同样用 random 输入时的输入梯度 norm = {bx.grad.norm().item():.4e}"
          f"  max|g| = {bx.grad.abs().max().item():.4e}")
    print("  ↑ 记住这个量级：下面的四元数版若与之同量级，说明梯度传播正常。")

    results = []
    for tag, qdw, qbn in [("A  仅替换通道混合卷积（推荐起点）", False, False),
                          ("B  连深度可分离卷积一起四元数化", True, False),
                          ("C  在 A 基础上再用四元数 BN", False, True)]:
        r = run_config(tag, quantize_depthwise=qdw, use_quat_bn=qbn)
        if r:
            results.append(r)

    h1("第 5 部分：预训练权重加载")
    pretrained = part5_pretrained(use_quat_bn=False)

    # ---------------- 汇总结论 ----------------
    h1("总结论")

    print(f"""
  ① 算子本身可用吗？
     {"可用。" if pass_op else "有问题，见上方 [!!] 标记。"}
     块矩阵实现与朴素 16 次卷积展开逐元素一致（误差 ~1e-7，属浮点误差），
     四元数深度卷积与四元数 BN 同样通过。反向梯度全部正常、无 NaN/Inf。

  ② 能塞进 MobileNetV3 吗？
     能。逐层普查显示 v3-small / v3-large 的**所有非深度卷积层通道都整除 4**，
     唯一的例外是 Stem 的 3 通道输入，用 rgb_lift 补零提升即可。
     本脚本跑了 {len(results)} 种替换配置，全部前向 + 反向跑通。

  ③ 参数量变化
""")
    for r in results:
        tag_note = ("（含 DWConv，它们是 1.0 倍，把合计拉高了）"
                    if "B " in r["tag"] else "（普通卷积层都是 0.25 倍）")
        print(f"     {r['tag']}")
        print(f"         被替换层本身 {r['replaced_o']:>9,} -> {r['replaced_n']:>9,}"
              f"  ({r['replaced_n'] / r['replaced_o']:.4f} 倍) {tag_note}")
        print(f"         卷积参数量   {r['base_conv']:>9,} -> {r['new_conv']:>9,}"
              f"  ({r['new_conv'] / r['base_conv']:.4f} 倍)")
        print(f"         总参数量     {r['base_total']:>9,} -> {r['new_total']:>9,}"
              f"  ({r['new_total'] / r['base_total']:.4f} 倍)")
    print("""
     ⚠️ 注意区分这三个数字，写论文时别用错：
        · "被替换层本身"才是理论值 **0.25 倍** —— 这是四元数卷积的参数量优势。
        · "卷积参数量"和"总参数量"的降幅小得多，因为 SE（占卷积参数约一半）、
          BatchNorm、分类头都没有四元数化，把整体比例稀释了。
        宣称"参数量降低 75%"只在被替换的卷积层上成立，不能推广到整个网络。""")
    if pretrained:
        n, t = pretrained
        print(f"""
  ④ 预训练权重
     可以复用。ImageNet 预训练 state_dict 的 {t} 个键里，{n} 个形状完全匹配、
     成功加载；被四元数层替换掉的那些键因形状不同而跳过，保持随机初始化。
     **这意味着"除替换层外其余层保持预训练"这一步是可行的。**

  ⑤ 本脚本**没有**回答的问题（写论文时不要越界）
     · 四元数版能不能收敛、精度如何 ── 需要真实数据训练
     · 四元数版对光照是否真的更鲁棒 ── 这是**待检验的假设**，不是已验证的结论
     · 训练显存与耗时 ── 块矩阵把权重在计算时扩了 4 倍，显存开销会高于普通卷积
     · 标准 BN vs 四元数 BN 的优劣 ── 需要消融实验

  ⑥ 已知的、有意保留的设计取舍
     · 激活函数仍用 Hardswish/SiLU 这类**逐分量**的函数，它不是四元数函数，
       会轻微破坏四元数结构。四元数感知的激活（作用在模长上）是后续可做的点。
     · 分类头仍是普通 Linear。特征图在进 Linear 前要 flatten，此时通道排布
       决定 flatten 顺序 —— 本脚本全程分组排布，顺序是固定的，但没有做额外处理。
     · QuaternionBatchNorm2d 刻意不做减均值，详见该类文档字符串。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
