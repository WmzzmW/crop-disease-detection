#!/usr/bin/env python
"""四元数卷积可行性验证：算子自检 + YOLOv8n 骨干集成兼容性

【运行】
    .venv/bin/python yolov8-quat/quaternion_feasibility.py

【依赖】torch, ultralytics（仅用其 YOLOv8n 结构定义，不下载权重、不训练）

【本脚本要回答的两个问题】
    任务1：现成的四元数卷积算子本身能不能用（前向 / 反向 / 梯度是否健康）
    任务2：把它塞进 YOLOv8n 的 C2f 骨干片段能不能兼容

【参考实现来源】
    哈密顿积的 4x4 块矩阵构造法，逐行对齐以下实现：

      - Titouan Parcollet 等, "Quaternion Recurrent Neural Networks", ICLR 2019
        https://github.com/TParcollet/Quaternion-Neural-Networks
        （Orkis-Research/Pytorch-Quaternion-Neural-Networks 的维护版）
        文件 core_qnn/quaternion_ops.py：
          · quaternion_conv()       —— 块矩阵构造（本脚本第 1.3 节逐行对齐）
          · get_r/i/j/k()           —— 通道切分（揭示其排布约定）
          · quaternion_init()       —— 权重初始化（模长 × 单位四元数）
      - SpeechBrain 的 speechbrain/nnet/quaternion_networks/q_CNN.py
        （生产级、持续维护，同样采用 r_weight/i_weight/j_weight/k_weight 四份参数）

    选用理由：
      1. 它是 Quaternion RNN/CNN 的原始作者团队维护的实现，被 PyTorch 生态
         广泛复用（SpeechBrain 直接吸收）；
      2. 哈密顿积写成显式块矩阵后交给 F.conv2d，不引入自定义算子，
         因此能直接吃 PyTorch 的 autograd，反向传播天然可用；
      3. 参数量是 4 份（r/i/j/k）而非 16 份，真正体现四元数卷积的参数效率优势。

    ⚠️ 但参考实现有一个**在本课题中必须处理**的问题，见第 1.2 节。
"""

from __future__ import annotations

import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

LINE = "=" * 78


def h1(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def h2(title: str) -> None:
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")


# ================================================================
# 第 1 部分：QuaternionConv2d 实现
# ================================================================


# ---------------------------------------------------------------- 1.1 排布约定

# 四元数张量有两种等价的通道排布，混淆它们是本项目最容易踩的坑：
#
#   【平面排布 planar】  通道顺序 = [R 全部分量 | I 全部分量 | J 全部分量 | K 全部分量]
#                        第 j 个分量的第 c 个四元数 → 通道号 j*q + c
#                        参考实现（Parcollet / SpeechBrain）用的是这一种
#
#   【分组排布 grouped】 通道顺序 = [q0.r, q0.i, q0.j, q0.k, q1.r, q1.i, q1.j, q1.k, ...]
#                        第 j 个分量的第 c 个四元数 → 通道号 4*c + j
#                        本脚本采用这一种
#
# 为什么本课题必须用分组排布：见 1.2 节。


def grouped_to_planar(x: torch.Tensor) -> torch.Tensor:
    """分组排布 → 平面排布。

    两种排布是同一个数据的两种通道顺序，所以这只是一次 reshape + transpose。
    """
    b, c, h, w = x.shape
    q = c // 4
    # (B, q, 4, H, W) 里 dim1=四元数序号, dim2=分量序号 → 换轴成 (B, 4, q, H, W)
    return x.view(b, q, 4, h, w).transpose(1, 2).reshape(b, 4 * q, h, w)


def planar_to_grouped(x: torch.Tensor) -> torch.Tensor:
    """平面排布 → 分组排布（grouped_to_planar 的逆操作）。"""
    b, c, h, w = x.shape
    q = c // 4
    return x.view(b, 4, q, h, w).transpose(1, 2).reshape(b, 4 * q, h, w)


# ---------------------------------------------------------------- 1.2 为什么

# ⚠️ 参考实现的平面排布，与 YOLOv8 的 C2f 结构直接冲突。
#
# C2f.forward 里有两处**按通道数对半切**的操作：
#
#     y = list(self.cv1(x).chunk(2, 1))      # 通道对半切
#     ...
#     return self.cv2(torch.cat(y, 1))        # 通道拼接
#
# 在平面排布下，一个 (4q) 通道的张量 = [R(q) | I(q) | J(q) | K(q)]。
# 沿通道切成两半得到的是：
#
#     前半 = [R(q) | I(q)]        后半 = [J(q) | K(q)]
#
# 这两半各自再被当成"四元数张量"使用时就完全乱套了 —— 前半会把自己的
# R(q/2) 当实部、I(q/2) 当虚部 i、J(q/2) 当 j、K(q/2) 当 k。
# 结果是：**形状完全合法，不报任何错，但实部/虚部语义被彻底打乱**。
#
# 这类"静默错误"比报错危险得多 —— 模型能训、loss 能降，但四元数结构已经没了。
#
# 同理，torch.cat 在平面排布下也不封闭：
#     cat([R1|I1|J1|K1, R2|I2|J2|K2]) = R1|I1|J1|K1|R2|I2|J2|K2
#     而正确的平面排布应为 R1|R2|I1|I2|J1|J2|K1|K2
#
# 【解法】改用分组排布。分组排布对 cat / chunk 都是封闭的：
#     · 每个四元数的 4 个分量占连续 4 个通道 → 只要在 4 的倍数处切割就天然正确
#     · cat 只是把若干完整四元数首尾相接 → 天然正确
#
# 代价：参考实现的块矩阵构造是平面排布下的产物，所以本实现在调用它前后
# 各做一次排布转换（grouped → planar → 卷积 → planar → grouped）。
# 转换只是 reshape+transpose，不产生额外计算量，也不破坏梯度。


def quaternion_chunk(x: torch.Tensor, chunks: int = 2) -> list[torch.Tensor]:
    """四元数感知的通道切分：按"四元数序号"切，而不是按"通道序号"切。

    平面排布下必须用它替代 torch.chunk，否则会静默破坏四元数结构（见 1.2 节）。
    分组排布下，只要切分点落在 4 的倍数上，torch.chunk 本身就等价于本函数 ——
    本函数的作用是把"落在四元数边界上"这件事显式化，避免日后被误改。
    """
    b, c, h, w = x.shape
    assert c % 4 == 0, f"通道数 {c} 不是 4 的倍数，无法表示四元数"
    q = c // 4
    assert q % chunks == 0, f"四元数个数 {q} 不能被 {chunks} 整除"
    # 拆成 (B, q, 4, H, W)：dim1 是四元数序号，dim2 是 r/i/j/k 分量
    x5 = x.view(b, q, 4, h, w)
    return [p.reshape(b, (q // chunks) * 4, h, w) for p in x5.chunk(chunks, dim=1)]


# ---------------------------------------------------------------- 1.3 主体


class QuaternionConv2d(nn.Module):
    """四元数二维卷积（分组通道排布）。

    参数量对比（这是四元数卷积的核心卖点）：
        普通卷积   C_in → C_out，k×k 核：  C_out × C_in × k²      个实数参数
        四元数卷积 4C_in → 4C_out：        (C_out/4) × (C_in/4) × k² × 4
                                        = C_out × C_in × k² / 4   个实数参数
    即**参数量降到普通卷积的 1/4**，同时通过哈密顿积让 R/G/B 三分量在
    卷积过程中相互耦合，而不是像普通卷积那样各自独立求和。

    哈密顿积的块矩阵形式（q = q_w ⊗ q_x，两者都是四元数）：

        [ r_out ]   [  w_r  -w_i  -w_j  -w_k ] [ x_r ]
        [ i_out ] = [  w_i   w_r  -w_k   w_j ] [ x_i ]
        [ j_out ]   [  w_j   w_k   w_r  -w_i ] [ x_j ]
        [ k_out ]   [  w_k  -w_j   w_i   w_r ] [ x_k ]

    把它按 (out, in) 两个方向拼成一个大权重，就退化成一个普通卷积 —— 这正是
    参考实现 quaternion_conv() 的做法，也是它能直接吃 autograd 的原因。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        if in_channels % 4 != 0 or out_channels % 4 != 0:
            raise ValueError(
                f"四元数卷积要求输入/输出通道数为 4 的倍数，"
                f"收到 in={in_channels}, out={out_channels}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q_in = in_channels // 4        # 输入四元数通道数
        self.q_out = out_channels // 4      # 输出四元数通道数
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 权重存成一份 4 分量的参数（而不是 16 份），这是参数效率的来源
        self.weight = nn.Parameter(
            torch.empty(4, self.q_out, self.q_in, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """四元数初始化：模长 × 单位纯虚四元数，对齐参考实现的 quaternion_init()。

        思路（Parcollet et al.）：
          1. 模长 modulus ~ χ 分布（自由度 4，按 fan_in/fan_out 缩放）；
          2. 相位 phase ~ U(-π, π)，方向 (v_i, v_j, v_k) 取单位纯虚四元数；
          3. w_r = m·cos(φ)， w_i = m·v_i·sin(φ)，其余同理。
        这样初始化出的四元数权重在 4 维空间里分布均匀，比逐分量独立采样更合理
        （独立采样会让模长集中在 sqrt(2) 附近，四元数的"旋转"语义被削弱）。
        """
        fan_in = self.q_in * self.kernel_size ** 2
        fan_out = self.q_out * self.kernel_size ** 2
        s = 1.0 / math.sqrt(2 * (fan_in + fan_out))     # glorot 准则

        shape = (self.q_out, self.q_in, self.kernel_size, self.kernel_size)
        # χ(4) 分布：用 4 个标准正态的平方和开根号等价实现
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
        """把 4 份四元数权重拼成块矩阵，得到 (4*q_out, 4*q_in, k, k)。

        这一段逐行对齐参考实现的 quaternion_conv()：
            cat_kernels_4_r = cat([ r_w, -i_w, -j_w, -k_w], dim=1)
            cat_kernels_4_i = cat([ i_w,  r_w, -k_w,  j_w], dim=1)
            cat_kernels_4_j = cat([ j_w,  k_w,  r_w, -i_w], dim=1)
            cat_kernels_4_k = cat([ k_w, -j_w,  i_w,  r_w], dim=1)
            kernel = cat([...], dim=0)
        """
        w_r, w_i, w_j, w_k = self.weight[0], self.weight[1], self.weight[2], self.weight[3]

        block_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=1)
        block_i = torch.cat([w_i, w_r, -w_k, w_j], dim=1)
        block_j = torch.cat([w_j, w_k, w_r, -w_i], dim=1)
        block_k = torch.cat([w_k, -w_j, w_i, w_r], dim=1)
        return torch.cat([block_r, block_i, block_j, block_k], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 分组 → 平面：块矩阵的输入要求平面排布
        x_planar = grouped_to_planar(x)
        # 2) 块矩阵卷积：这一步是纯实数卷积，autograd 直接可用。
        #    注意 bias 传 None —— self.bias 是按**分组排布**定义的，
        #    而这里的输出还是平面排布，直接交给 F.conv2d 会加错通道。
        #    （bias 初始为 0，加错也看不出来，属于典型的潜伏 bug，故显式规避。）
        out_planar = F.conv2d(
            x_planar, self._block_kernel(), None,
            stride=self.stride, padding=self.padding,
        )
        # 3) 平面 → 分组：交还给 YOLOv8 时必须是分组排布，否则 C2f 的 cat/chunk 会坏
        out = planar_to_grouped(out_planar)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ---------------------------------------------------------------- 1.4 朴素对照

def naive_quaternion_conv(x: torch.Tensor, qc: QuaternionConv2d) -> torch.Tensor:
    """朴素参考实现：显式写 16 次普通卷积，不用块矩阵技巧。

    作用：作为**数值真值**去交叉验证块矩阵实现。两者必须逐元素一致，
    否则说明块矩阵的符号拼错了（这个错误不会报错，只会让模型学不到东西）。
    """
    w_r, w_i, w_j, w_k = qc.weight[0], qc.weight[1], qc.weight[2], qc.weight[3]
    b, c, h, w = x.shape
    q_in = c // 4
    # 分组排布下按分量抽取：通道 4c+j 的第 j 个分量
    x_r, x_i, x_j, x_k = (x[:, j::4] for j in range(4))

    def cv(inp, ker):
        return F.conv2d(inp, ker, None, qc.stride, qc.padding)

    # 哈密顿积展开（由四元数乘法规则推导：i²=j²=k²=ijk=-1）
    #   i·k = -j,  k·i = j,  i·j = k,  j·i = -k
    # 这四行可以逐项对照参考实现 quaternion_conv() 里的块矩阵：
    #   block_i = [i_w,  r_w, -k_w,  j_w]  展开正是 out_i
    out_r = cv(x_r, w_r) - cv(x_i, w_i) - cv(x_j, w_j) - cv(x_k, w_k)
    out_i = cv(x_r, w_i) + cv(x_i, w_r) + cv(x_k, w_j) - cv(x_j, w_k)
    out_j = cv(x_r, w_j) + cv(x_i, w_k) - cv(x_k, w_i) + cv(x_j, w_r)
    out_k = cv(x_r, w_k) - cv(x_i, w_j) + cv(x_j, w_i) + cv(x_k, w_r)

    out = torch.stack([out_r, out_i, out_j, out_k], dim=2)   # (B, q_out, 4, H', W')
    out = out.reshape(b, qc.q_out * 4, out.shape[-2], out.shape[-1])
    if qc.bias is not None:
        out = out + qc.bias.view(1, -1, 1, 1)
    return out


# ================================================================
# 第 2 部分：任务 1 —— 算子自检
# ================================================================


def task1_operator_selfcheck() -> bool:
    h1("任务 1：四元数卷积算子自检（前向 / 反向 / 梯度健康度）")

    B, H, W = 4, 32, 32

    # 两种配置都要测：
    #   ① RGB 提升后的首层 —— 1 个输入四元数（4 实数通道）→ 8 个输出四元数（32 通道）
    #   ② 骨干深层 —— 4 个输入四元数（16 通道）→ 8 个输出四元数（32 通道）
    # 只测①会漏掉"多四元数通道"的一般情形，而骨干里绝大多数卷积都是②这种。
    configs = [
        (1, 8, "① 单四元数（RGB 提升后，对应 YOLO 首层）"),
        (4, 8, "② 多四元数（对应骨干深层）"),
    ]

    # ---------------------------------------------------------- 2.1 前向
    h2("2.1 前向传播")

    # 模拟一批 RGB 图像：3 通道，视为"1 个纯四元数 0 + Ri + Gj + Bk"
    rgb = torch.rand(B, 3, H, W)
    print(f"模拟输入 RGB 图像张量          : {tuple(rgb.shape)}  (B, 3, H, W)")

    # RGB → 纯四元数：实部补 0，虚部放 R/G/B。
    # 分组排布下，1 个四元数占连续 4 通道 = [r, i, j, k]，所以拼成 [0, R, G, B] 即可。
    rgb_quat = torch.cat([torch.zeros_like(rgb[:, :1]), rgb], dim=1)
    print(f"提升为纯四元数张量              : {tuple(rgb_quat.shape)}  通道含义 = [0, R, G, B]")
    print()

    convs = {}
    for q_in, q_out, label in configs:
        in_ch, out_ch = q_in * 4, q_out * 4
        # ① 喂真实提升后的 RGB；② 喂随机多四元数张量
        x = rgb_quat if q_in == 1 else torch.rand(B, in_ch, H, W)

        conv = QuaternionConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        y = conv(x)
        assert y.shape == (B, out_ch, H, W), "输出维度不符"

        q_params = sum(p.numel() for p in conv.parameters())
        p_params = sum(p.numel() for p in nn.Conv2d(in_ch, out_ch, 3, padding=1).parameters())
        print(f"    {label}")
        print(f"      输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}  ✓ 维度正确")
        print(f"      参数量：普通 Conv2d {p_params:,}  vs  四元数 {q_params:,}"
              f"   比值 {q_params / p_params:.3f}（理论 0.25）")
        convs[(q_in, q_out)] = (conv, x, y)

    # ---------------------------------------------------------- 2.2 排布转换自检
    h2("2.2 排布转换与块矩阵的正确性交叉验证")

    rt = planar_to_grouped(grouped_to_planar(rgb_quat))
    print(f"分组→平面→分组 往返最大误差     : {(rt - rgb_quat).abs().max().item():.3e}")
    assert torch.allclose(rt, rgb_quat, atol=1e-6), "排布转换不可逆"
    print("✓ 排布转换可逆")

    for (q_in, q_out), (conv, x, y) in convs.items():
        y_naive = naive_quaternion_conv(x, conv)
        diff = (y - y_naive).abs().max().item()
        print(f"块矩阵 vs 朴素 16 次卷积（{q_in}→{q_out} 四元数）: "
              f"最大逐元素误差 {diff:.3e}")
        assert torch.allclose(y, y_naive, atol=1e-5), "块矩阵符号拼写有误！"
    print("✓ 全部配置数值一致 → 哈密顿积的块矩阵构造正确（实部/虚部正负号没拼错）")

    # ---------------------------------------------------------- 2.3 反向
    h2("2.3 反向传播 / 参数更新 / 梯度健康度")

    for (q_in, q_out), (conv, x, y) in convs.items():
        print(f"\n    ── 配置 {q_in} → {q_out} 四元数 ──")
        conv.zero_grad()
        before = {n: p.detach().clone() for n, p in conv.named_parameters()}

        loss = F.mse_loss(y, torch.randn_like(y))
        loss.backward()

        bad = []
        for n, p in conv.named_parameters():
            if p.grad is None:
                bad.append(f"{n}: 梯度为 None")
                continue
            gn = p.grad.norm().item()
            n_nan = int(torch.isnan(p.grad).sum())
            n_inf = int(torch.isinf(p.grad).sum())
            print(f"      {n:<8} grad_norm={gn:>11.6f}   NaN={n_nan}  Inf={n_inf}")
            if n_nan or n_inf:
                bad.append(f"{n}: 含 NaN/Inf")
            if gn == 0.0:
                bad.append(f"{n}: 梯度恒为 0")

        assert not bad, "梯度异常：" + "; ".join(bad)
        print(f"      ✓ 范数 {loss.item():.4f} 的损失回传后，全部参数有梯度、无 NaN/Inf")

        opt = torch.optim.SGD(conv.parameters(), lr=0.1)
        opt.step()
        moved = sum(1 for n, p in conv.named_parameters()
                    if (p.detach() - before[n]).abs().max().item() > 0)
        print(f"      ✓ SGD 更新后 {moved}/{len(before)} 个参数张量发生变化 → 反向通路完整")

    # 梯度能否正常回传到输入（对 YOLO 端到端训练是必需的）
    x2 = rgb_quat.clone().requires_grad_(True)
    QuaternionConv2d(4, 32, 3, padding=1)(x2).sum().backward()
    print(f"\n    ✓ 输入侧梯度可回传，norm = {x2.grad.norm().item():.6f}"
          f"（YOLO 端到端训练需要）")

    print("\n【任务 1 结论】算子可用：前向维度正确、反向梯度健康、无 NaN/Inf、"
          "块矩阵实现与朴素展开逐元素一致。")
    return True


# ================================================================
# 第 3 部分：任务 2 —— YOLOv8n 骨干集成
# ================================================================


def task2_yolov8_integration() -> None:
    h1("任务 2：嵌入 YOLOv8n 骨干片段（重点 C2f）")

    try:
        from ultralytics import YOLO
        from ultralytics.nn.modules import C2f, Conv as UltralyticsConv
    except ImportError:
        print("未安装 ultralytics，请先执行：pip install ultralytics")
        return

    # yolov8n.yaml 只含结构定义，加载它**不会下载任何预训练权重**
    net = YOLO("yolov8n.yaml")
    layers = net.model.model                      # nn.Sequential
    print(f"YOLOv8n 共 {len(layers)} 层（结构来自 yolov8n.yaml，未加载预训练权重）")

    # ---------------------------------------------------------- 3.1 通道整除性
    h2("3.1 通道整除性普查（四元数要求通道数为 4 的倍数）")

    def channels_of(m):
        """读出模块的真实通道数。

        ultralytics 的模块**不保存** c1/c2 属性（都是 None），必须从内部
        卷积层的 weight 形状反推。返回 (输入通道, 输出通道, [内部中间通道])。
        """
        if isinstance(m, UltralyticsConv):
            return m.conv.in_channels, m.conv.out_channels, []
        if isinstance(m, C2f):
            bn = m.m[0]                                # 第一个 Bottleneck
            mids = [
                m.cv1.conv.out_channels,               # 2*c
                m.cv2.conv.in_channels,                # (2+n)*c
                bn.cv1.conv.out_channels,              # Bottleneck 中间通道
                bn.cv2.conv.in_channels,
            ]
            return m.cv1.conv.in_channels, m.cv2.conv.out_channels, mids
        if type(m).__name__ == "SPPF":
            return (m.cv1.conv.in_channels, m.cv2.conv.out_channels,
                    [m.cv1.conv.out_channels, m.cv2.conv.in_channels])
        return None, None, []

    print(f"    {'层':<4}{'类型':<9}{'输入':>6}{'输出':>7}   {'内部通道':<20}{'4 整除'}")
    print(f"    {'-' * 72}")
    bad_layers = []
    for i, m in enumerate(layers[:10]):            # 骨干 = layers[0..9]
        name = type(m).__name__
        c1, c2, mids = channels_of(m)
        bad = []
        if c1 is not None and c1 % 4:
            bad.append(f"in={c1}")
        if c2 is not None and c2 % 4:
            bad.append(f"out={c2}")
        for v in mids:
            if v % 4:
                bad.append(f"mid={v}")
        if bad:
            bad_layers.append((i, name, bad))
        mid_str = ", ".join(str(v) for v in mids)
        mark = "✓" if not bad else "✗ " + " ".join(bad)
        print(f"    {i:<4}{name:<9}{str(c1):>6}{str(c2):>7}   {mid_str:<20}{mark}")

    print(f"\n    结论：骨干 10 层里，除了**第 0 层**（输入 3 通道，不是 4 的倍数），")
    print(f"          其余所有层的输入/输出通道，以及 C2f 内部的 4 个中间通道，")
    print(f"          全部是 4 的倍数。没有除不尽的层。")
    print(f"    注意：这是巧合，不是 YOLOv8 的设计意图 —— 换成 s/m/l 或改 n 值都要重新普查。")

    # ---------------------------------------------------------- 3.2 基线
    h2("3.2 基线：原版骨干片段前向（未替换）")

    # 取 layers[0..2]：Conv(3→16,s2) → Conv(16→32,s2) → C2f(32→32)
    baseline = nn.Sequential(*list(layers[0:3]))
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        base_out = baseline(dummy)
    print(f"    输入 {tuple(dummy.shape)} → 输出 {tuple(base_out.shape)}  ✓ 基线跑通")

    # ---------------------------------------------------------- 3.3 问题一
    h2("3.3 问题一：第 0 层 Conv(3→16) 无法直接替换（会真的报错）")

    print("    直接把第 0 层换成 QuaternionConv2d(3, 16, 3, 2, 1)：")
    try:
        QuaternionConv2d(3, 16, kernel_size=3, stride=2, padding=1)
        print("    （未报错？）")
    except ValueError as e:
        print(f"    ✗ 抛出异常：{type(e).__name__}: {e}")

    print("\n    若绕过校验强行构造（模拟参考实现的行为），会发生什么：")
    try:
        # 手工构造一个 in_channels=3 的块矩阵，看 F.conv2d 是否会报错
        w = torch.randn(16, 3, 3, 3)
        F.conv2d(torch.rand(1, 3, 64, 64), w, None, 2, 1)
        print("    （普通卷积 3→16 当然可以，所以问题不在 F.conv2d）")
        # 但四元数路径要求把 3 通道切成 4 份：
        x3 = torch.rand(1, 3, 64, 64)
        q = 3 // 4
        print(f"    四元数切分：3 通道 // 4 = {q} 个四元数 → narrow(1,0,0) 得到 0 个通道")
        parts = [x3.narrow(1, j * q, q) for j in range(4)]
        print(f"    四个分量形状：{[tuple(p.shape) for p in parts]}")
        print("    → 权重块拼接时 in 维为 0，卷积核退化为空，无法进行有效卷积")
    except Exception as e:
        print(f"    异常：{type(e).__name__}: {e}")

    print("\n    【原因】四元数把 RGB 三分量编码为一个纯四元数 0+Ri+Gj+Bk，")
    print("           需要 4 个实数通道。而 YOLO 的输入是 3 通道，")
    print("           3 不是 4 的倍数，无法构成完整的四元数表示。")
    print("    【修复】在第 0 层前加一个「提升层」，把 3 通道补 0 扩成 4 通道，")
    print("           即把 RGB 显式解释为纯四元数。见 3.5 节。")

    # ---------------------------------------------------------- 3.4 问题二
    h2("3.4 问题二：C2f 的 chunk 会静默破坏四元数结构（不报错！）")

    print("    C2f.forward 内部对 cv1 的输出做 torch.chunk(2, 1)（通道对半切）。")
    print("    下面用可辨识的数值演示它对四元数结构的破坏。\n")

    # 构造一个"分量可辨识"的四元数张量：R=1.0, I=2.0, J=3.0, K=4.0
    # 参考实现用的平面排布：[R(q) | I(q) | J(q) | K(q)]
    q = 8
    planar = torch.cat([
        torch.full((1, q, 4, 4), 1.0),    # R
        torch.full((1, q, 4, 4), 2.0),    # I
        torch.full((1, q, 4, 4), 3.0),    # J
        torch.full((1, q, 4, 4), 4.0),    # K
    ], dim=1)
    print(f"    构造平面排布张量：{tuple(planar.shape)} = [R(8) | I(8) | J(8) | K(8)]")
    print(f"    各分量填充值：R=1.0  I=2.0  J=3.0  K=4.0\n")

    # —— 平面排布 + 朴素 chunk（参考实现的默认行为）
    pu = planar.clone()
    naive_halves = pu.chunk(2, 1)
    print("    【平面排布 + torch.chunk(2,1)】参考实现会走这条路：")
    for k, half in enumerate(naive_halves):
        hq = half.shape[1] // 4
        comp = [half[:, j * hq:(j + 1) * hq].mean().item() for j in range(4)]
        print(f"      第{k}半 {tuple(half.shape)} 被解读为四元数后，"
              f"r={comp[0]:.1f} i={comp[1]:.1f} j={comp[2]:.1f} k={comp[3]:.1f}")
    print("      ✗ 第 0 半的 r 和 i 都变成 1.0、j 和 k 都变成 2.0 ——")
    print("        原本的 R 和 I 被错当成「实部+虚部i」，J 和 K 被错当成「虚部j+虚部k」。")
    print("        形状完全合法，不报任何错，但四元数语义已经没了。\n")

    # —— 分组排布 + 四元数感知 chunk
    grouped = planar_to_grouped(planar)
    good_halves = quaternion_chunk(grouped, 2)
    print("    【分组排布 + quaternion_chunk】本实现的做法：")
    for k, half in enumerate(good_halves):
        comp = [half[:, j::4].mean().item() for j in range(4)]
        print(f"      第{k}半 {tuple(half.shape)} 各分量均值："
              f"r={comp[0]:.1f} i={comp[1]:.1f} j={comp[2]:.1f} k={comp[3]:.1f}")
    print("      ✓ 两半都完整保留 r=1/i=2/j=3/k=4，四元数结构未被破坏")

    # 量化破坏程度。
    # 注意：两次切分产出的张量**排布不同**（朴素切分仍是平面排布，感知切分是分组排布），
    # 所以必须按各自的排布约定去取分量，否则就是在拿苹果比橘子。
    def deviation(t: torch.Tensor, layout: str,
                  expected=(1.0, 2.0, 3.0, 4.0)) -> float:
        """返回四个分量均值与注入值 [1,2,3,4] 的最大偏差。0 表示结构完好。"""
        hq = t.shape[1] // 4
        if layout == "planar":
            comps = [t[:, j * hq:(j + 1) * hq].mean().item() for j in range(4)]
        else:                                    # grouped
            comps = [t[:, j::4].mean().item() for j in range(4)]
        return max(abs(c - e) for c, e in zip(comps, expected))

    d_naive = deviation(naive_halves[0], "planar")
    d_good = deviation(good_halves[0], "grouped")
    print(f"\n    结构保持度（四个分量的实际均值与注入值 [1,2,3,4] 的最大偏差）：")
    print(f"      平面排布 + 朴素 chunk    : 偏差 {d_naive:.2f}  "
          f"→ {'结构已损坏' if d_naive > 1e-6 else '结构完好'}")
    print(f"      分组排布 + 感知 chunk    : 偏差 {d_good:.2f}  "
          f"→ {'结构已损坏' if d_good > 1e-6 else '结构完好'}")
    assert d_naive > 1e-6, "演示未复现出破坏，检查构造"
    assert d_good < 1e-6, "四元数感知切分未能保住结构"

    # ---------------------------------------------------------- 3.5 修复版
    h2("3.5 修复版：提升层 + 四元数卷积化 C2f 片段")

    class RGBToQuaternion(nn.Module):
        """把 3 通道 RGB 提升为 1 个纯四元数（实部补 0）。

        分组排布下 1 个四元数占连续 4 通道 [r,i,j,k]，
        所以拼成 [0, R, G, B] 正好就是"实部为 0 的纯四元数"。
        这比"让第一层保持普通卷积"更贴合四元数的建模初衷 ——
        从第一个卷积开始就让 R/G/B 通过哈密顿积相互耦合。
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)

    class QuatConv(nn.Module):
        """对应 ultralytics 的 Conv（Conv2d→BN→SiLU），但卷积换为四元数卷积。

        ⚠️ 注意 BN 的语义问题：BatchNorm2d 是**逐通道**归一化的，
        它会把 r/i/j/k 四个分量各自独立地标准化，从而削弱四元数卷积
        好不容易建立起来的通道间耦合。彻底的解法是换成"四元数 BN"
        （在四元数空间里做归一化，保持模长/相位语义），此处为验证形状
        兼容性先沿用标准 BN，并在结论中标注为待改进项。
        """

        def __init__(self, c1, c2, k=1, s=1, p=None, use_bn=True):
            super().__init__()
            self.conv = QuaternionConv2d(c1, c2, k, stride=s,
                                         padding=k // 2 if p is None else p,
                                         bias=not use_bn)
            self.bn = nn.BatchNorm2d(c2) if use_bn else nn.Identity()
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class QuatBottleneck(nn.Module):
        """四元数版 Bottleneck，严格对齐 ultralytics 的结构。

        关键细节（读源码确认，容易写错）：
          · ultralytics 的 C2f 创建 Bottleneck 时传 **e=1.0**，
            所以隐藏层通道 = c2 本身，而不是常见的 c2/2。
            写成 c2//2 会让整个 C2f 的参数分布和原版对不上。
          · shortcut 为真时带残差相加（self.add），漏掉就不是同一个模块了。
        """

        def __init__(self, c1, c2, shortcut=True, use_bn=True):
            super().__init__()
            self.cv1 = QuatConv(c1, c2, 3, 1, use_bn=use_bn)
            self.cv2 = QuatConv(c2, c2, 3, 1, use_bn=use_bn)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    class QuatC2f(nn.Module):
        """四元数版 C2f，结构与 ultralytics 的 C2f 一一对应，只改两处：
             1. 内部卷积换成四元数卷积
             2. torch.chunk 换成 quaternion_chunk（按四元数序号切而非按通道切）

        第 2 点是关键 —— 沿用 torch.chunk 的话形状照样能跑通，
        但四元数结构会被静默破坏（见 3.4 节）。
        """

        def __init__(self, c1, c2, n=1, shortcut=True, use_bn=True):
            super().__init__()
            assert c1 % 4 == 0 and c2 % 4 == 0
            self.c = c2 // 2                       # 隐藏通道（与 ultralytics 的 e=0.5 同）
            self.cv1 = QuatConv(c1, 2 * self.c, 1, 1, use_bn=use_bn)
            self.cv2 = QuatConv((2 + n) * self.c, c2, 1, 1, use_bn=use_bn)
            # Bottleneck 的隐藏层 = self.c（对应 ultralytics 内部的 e=1.0）
            self.m = nn.ModuleList(
                QuatBottleneck(self.c, self.c, shortcut, use_bn) for _ in range(n)
            )

        def forward(self, x):
            # ↓↓↓ 与原版 C2f 唯一的实质差异
            y = list(quaternion_chunk(self.cv1(x), 2))     # 原版：.chunk(2, 1)
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

    # 从真实模块里读出 C2f 的实际配置，保证替换是"结构等价"的而不是想当然
    real_c2f = layers[2]
    real_n = len(real_c2f.m)
    real_shortcut = bool(real_c2f.m[0].add)
    print(f"    读到的真实 C2f 配置：n={real_n}  shortcut={real_shortcut}"
          f"  c={real_c2f.c}")

    # 组装：提升层 → 四元数 Conv(s2) → 四元数 C2f
    fixed = nn.Sequential(
        RGBToQuaternion(),
        QuatConv(4, 16, 3, 2),                          # 对应原 Conv(3, 16, 3, 2)
        QuatConv(16, 32, 3, 2),                         # 对应原 Conv(16, 32, 3, 2)
        QuatC2f(32, 32, n=real_n, shortcut=real_shortcut),   # 对应原 C2f(32, 32)
    )
    print("    修复片段结构：")
    print("      RGBToQuaternion      3 → 4   （RGB 提升为纯四元数）")
    print("      QuatConv(4→16, s2)   4 → 16  （对应原 Conv(3→16, s2)）")
    print("      QuatConv(16→32, s2) 16 → 32")
    print(f"      QuatC2f(32→32, n={real_n})  32 → 32（四元数感知 chunk）")

    fixed.eval()
    with torch.no_grad():
        out = fixed(dummy)
    print(f"\n    输入 {tuple(dummy.shape)} → 输出 {tuple(out.shape)}")
    assert out.shape == base_out.shape, "输出形状与原版片段不一致"
    print("    ✓ 前向跑通，且输出形状与原版骨干片段完全一致")

    # 反向
    fixed.train()
    loss = fixed(dummy).square().mean()
    loss.backward()
    n_bad = 0
    n_params = 0
    for _, p in fixed.named_parameters():
        n_params += 1
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            n_bad += 1
    print(f"    ✓ 反向传播跑通：{n_params} 个参数张量，含 NaN/Inf 的 {n_bad} 个")

    # 参数量对比 —— 卷积与 BN 分开统计，否则比例会被 BN 掩盖而看不出真实规律：
    #   · 卷积部分：四元数版应恰好是普通版的 1/4（这是理论值）
    #   · BN 部分：两种版本的通道数相同，所以 BN 参数量**完全相等**
    #   · 因此整片段的比值会略高于 1/4 —— BN 在四元数版里占比被动变大了
    def breakdown(module):
        conv = sum(p.numel() for m in module.modules()
                   if isinstance(m, (nn.Conv2d, QuaternionConv2d))
                   for p in m.parameters(recurse=False))
        bn = sum(p.numel() for m in module.modules()
                 if isinstance(m, nn.BatchNorm2d)
                 for p in m.parameters(recurse=False))
        return conv, bn

    b_conv, b_bn = breakdown(baseline)
    q_conv, q_bn = breakdown(fixed)
    print(f"\n    {'':<14}{'卷积参数':>12}{'BN 参数':>12}{'合计':>12}")
    print(f"    {'-' * 50}")
    print(f"    {'原版片段':<14}{b_conv:>12,}{b_bn:>12,}{b_conv + b_bn:>12,}")
    print(f"    {'四元数版片段':<14}{q_conv:>12,}{q_bn:>12,}{q_conv + q_bn:>12,}")
    print(f"    {'比值':<14}{q_conv / b_conv:>12.3f}{q_bn / b_bn:>12.3f}"
          f"{(q_conv + q_bn) / (b_conv + b_bn):>12.3f}")
    print(f"\n    → 卷积部分比值 {q_conv / b_conv:.3f}，与理论值 0.25 一致 ✓")
    print(f"    → BN 部分比值 {q_bn / b_bn:.3f} —— 两种版本的通道序列完全相同")
    print(f"      （16→32→32，C2f 内部 32/48/16/16），所以 BN 参数量一模一样，"
          f"既不增也不减。")
    print(f"    → 合计比值 {(q_conv + q_bn) / (b_conv + b_bn):.3f} **高于** 0.25："
          f"卷积缩到 1/4，但 BN 是固定开销不缩，")
    print(f"      它在总参数里的占比被动升高。层数越浅、BN 占比越大，这个效应越明显。")

    # ---------------------------------------------------------- 3.6 汇总
    h2("3.6 逐层替换普查：哪些层能直接换、哪些不能")

    print(f"    {'层':<4}{'类型':<9}{'能否直接替换':<18}{'原因 / 处理方式'}")
    print(f"    {'-' * 76}")
    for i, m in enumerate(layers[:10]):
        name = type(m).__name__
        c1, c2, mids = channels_of(m)
        all_ok = all(v % 4 == 0 for v in
                     ([c2] if c2 else []) + list(mids))
        if i == 0:
            verdict, why = "需加提升层", "输入 3 通道非 4 的倍数，需先补 0 扩成 4（解释为纯四元数）"
        elif isinstance(m, C2f):
            verdict, why = "需改 chunk", "形状兼容，但必须用 quaternion_chunk 替代 torch.chunk"
        elif all_ok:
            verdict, why = "可直接替换", "输入/输出/内部通道均为 4 的倍数"
        else:
            verdict, why = "不可", f"通道 {c1}→{c2} 存在非 4 倍数"
        print(f"    {i:<4}{name:<9}{verdict:<18}{why}")


# ================================================================
# 主流程
# ================================================================

if __name__ == "__main__":
    print(LINE)
    print("四元数卷积可行性验证")
    print(f"torch {torch.__version__}   MPS 可用: {torch.backends.mps.is_available()}")
    print(LINE)

    ok1 = task1_operator_selfcheck()
    task2_yolov8_integration()

    h1("总结")
    print("""
① 该算子本身是否可用？
   可用。前向维度正确，反向梯度完整无 NaN/Inf，参数确实被更新，
   且块矩阵实现与朴素 16 次卷积展开逐元素一致 —— 说明哈密顿积的
   正负号构造没有拼错。参数量约为普通卷积的 1/4，符合理论预期。

② 嵌入 YOLOv8n 骨干是否兼容？
   通道数层面兼容：YOLOv8n 骨干的 16/32/64/128/256 以及 C2f 内部的
   分裂通道全部是 4 的倍数，没有除不尽的层（第 0 层除外）。
   但有两处必须处理，否则要么报错、要么静默出错：

   问题一（会报错）：第 0 层 Conv(3→16) 的输入是 3 通道，
       3 不是 4 的倍数，无法构成完整四元数表示。
       修复：前置一个提升层，把 RGB 补零扩成 4 通道（解释为纯四元数 0+Ri+Gj+Bk）。

   问题二（不报错，更危险）：C2f 内部用 torch.chunk(2,1) 按通道对半切。
       参考实现的**平面排布** [R|I|J|K] 在这种切法下会把 R 和 I 混成
       "实部+虚部i"、J 和 K 混成"虚部j+虚部k"。形状完全合法，不抛异常，
       但四元数语义已经被破坏 —— 模型照样能训，只是四元数结构白搭了。
       修复：改用分组排布（每个四元数占连续 4 通道）+ quaternion_chunk。

③ 复刻 ultralytics 模块时必须对齐的三个细节（本次实测踩到）
   · C2f 创建 Bottleneck 时传的是 e=1.0，隐藏层通道 = c 本身，
     不是常见的 c/2。写成 c//2 会让参数分布和原版对不上。
   · Bottleneck 带残差相加（shortcut=True 且 c1==c2 时），漏掉就不是同一个模块。
   · ultralytics 的模块**不保存 c1/c2 属性**（getattr 得到 None），
     通道数只能从内部卷积的 weight 形状反读。

④ 修复建议
   · 输入侧：加 RGBToQuaternion 提升层（3→4）
   · 排布：全程使用分组排布，使 cat / chunk 对四元数封闭
   · C2f：用 quaternion_chunk 替代 torch.chunk
   · BatchNorm：当前沿用标准 BN，但它是逐通道归一化，会削弱 r/i/j/k 的
     耦合 —— 建议后续换成四元数 BN（在四元数空间归一化，保持模长/相位语义）。
     注意 BN 参数量不随四元数化缩小，所以整网参数量降幅会小于 4 倍
     （本次片段实测：卷积 0.253 倍，合计 0.270 倍）。
   · 部署：ultralytics 的 Conv 有 fuse() 会把 Conv+BN 融合，
     自定义四元数模块需自行实现 fuse()，否则导出 ONNX/TFLite 会失败
   · 泛化：YOLOv8s/m/l 的通道数尚未普查，需重新逐层检查 4 整除性；
     检测头（Detect）的最后一层是回归输出（4*reg_max 通道），不应四元数化

⑤ 尚需验证（本次未做）
   · 四元数版能否收敛、以及收敛后的精度与光照鲁棒性 —— 需要真实数据与训练
   · 标准 BN 对四元数耦合的削弱程度 —— 建议做"标准 BN vs 四元数 BN"的消融
   · 反向传播的显存与耗时开销 —— 块矩阵把权重扩了 4 倍，训练显存会上升
""")
