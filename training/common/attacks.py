"""对抗攻击实现：FGSM 与 PGD。

【为什么自己在像素空间实现，而不是调 Foolbox】
    1. 透明：ε 的物理含义（0~1 像素尺度，即文献里的 ε=0.03 ≈ 7.65/255）在代码里
       直接可见，答辩时可以逐行解释；
    2. 无 API 风险：Foolbox 3.x 各版本接口有变动，自己实现不会因库升级而跑不通；
    3. 与文献公式一一对应，见下面每个函数的注释。
    Foolbox 在本课题后续阶段仍会用到——检测器泛化测试需要 CW、MI-FGSM 这类更复杂
    的攻击，那部分直接调库更合适。

【最容易搞错的地方：归一化空间 vs 像素空间】
    送入模型的数据是**归一化后**的（减 ImageNet 均值、除标准差），取值范围大约是
    [-2.1, 2.6]，不是 [0, 1]。如果直接在这个空间里做攻击，会有两个错误：

      · **ε 的物理含义错乱**。归一化的链式法则会让像素空间的扰动变成 ε×std，
        而且三个通道的 std 不同（0.229/0.224/0.225），扰动幅度在通道之间还不一致。
        结果是"设了 ε=0.03，实际扰动是 0.0069"，比文献弱 4 倍多。

      · **clamp(0,1) 会把图弄坏**。归一化空间的合法范围是 [-2.1, 2.6]，
        用 clamp(0,1) 会把大量像素压到边界上，生成的根本不是合法图像。

    所以本模块统一在**像素空间 [0,1]** 里做攻击，只在送进模型前临时归一化。
    这样 ε 就是字面意思："每个像素最多改 ε"，和文献口径完全一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import IMAGENET_MEAN, IMAGENET_STD


class PixelSpace:
    """像素空间 <-> 归一化空间的转换，以及"给像素空间图像做预测"。

    把归一化统计量绑在对象上，避免每个调用点都手写一遍 tensor 构造
    （漏掉 .view(1,3,1,1) 会导致广播错误，而且报错信息很难懂）。
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def to_pixel(self, x_norm: torch.Tensor) -> torch.Tensor:
        """归一化空间 -> 像素空间 [0,1]"""
        return x_norm * self.std + self.mean

    def to_norm(self, x_pix: torch.Tensor) -> torch.Tensor:
        """像素空间 [0,1] -> 归一化空间"""
        return (x_pix - self.mean) / self.std

    def predict(self, x_pix: torch.Tensor) -> torch.Tensor:
        """对像素空间的图像做预测，返回 logits。"""
        return self.model(self.to_norm(x_pix))


# ================================================================ FGSM


def fgsm(ps: PixelSpace, x_pix: torch.Tensor, y: torch.Tensor, eps: float,
         criterion: nn.Module) -> torch.Tensor:
    """FGSM（Fast Gradient Sign Method）—— Goodfellow et al. 2015

        x_adv = clip( x + ε · sign( ∇ₓ J(x, y) ), 0, 1 )

    逐步解释：
      1. 算出损失 J 对**图像每个像素**的梯度 ∇ₓ J——它指出"每个像素往哪个方向调，
         损失会变大"（即模型更容易判错）；
      2. sign() 只保留方向（+1 / -1），不看大小；
      3. 每个像素朝这个方向挪 ε 这么多；
      4. clip 回 [0,1] 保证仍是合法图像。

    单步攻击：一算到位。快，但不是最强。
    """
    x_adv = x_pix.clone().detach().requires_grad_(True)
    loss = criterion(ps.predict(x_adv), y)
    # 用 autograd.grad 而不是 loss.backward()：只求对输入图像的梯度，
    # 不会往模型参数里累积梯度（避免污染模型状态）
    grad = torch.autograd.grad(loss, x_adv)[0]
    with torch.no_grad():
        x_adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0)
    return x_adv.detach()


# ================================================================ PGD


def pgd(ps: PixelSpace, x_pix: torch.Tensor, y: torch.Tensor, eps: float,
        alpha: float = 0.01, steps: int = 10, criterion: nn.Module | None = None,
        random_start: bool = True) -> torch.Tensor:
    """PGD（Projected Gradient Descent）—— Madry et al. 2018

    把 FGSM 迭代做多次，每次只挪一小步 α，挪完投影回 ε 允许的范围：

        x⁰   = x + Uniform(-ε, ε)                    （随机起点）
        xᵗ⁺¹ = Π_{ε}( xᵗ + α · sign(∇ₓ J(xᵗ, y)) )    （走一步 + 投影）

    投影 Π_ε 有两层含义，缺一不可：
      · 投影回 **ε 球**：max(min(x, x₀+ε), x₀-ε)，保证总扰动不超过 ε；
      · 投影回 **[0,1]**：保证仍是合法像素值。

    为什么比 FGSM 强：FGSM 只在原始点线性化一次；PGD 每走一步都在新位置重新
    线性化，能更准确地找到真正的攻击方向。文献里三篇独立工作的一致结论是
    「迭代攻击 >> 单步攻击」，本课题的实验会复现出同一趋势——这也是验证
    攻击实现正确的一个交叉检查点。
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    x_adv = x_pix.clone().detach()

    if random_start:
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = criterion(ps.predict(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_pix + eps), x_pix - eps)  # ε 球投影
            x_adv = x_adv.clamp(0.0, 1.0)                                  # 合法像素投影
    return x_adv.detach()


# ================================================================ 量化


def round_to_uint8(x_pix: torch.Tensor) -> torch.Tensor:
    """把像素空间图像按 8 位量化。

    攻击者能真正交付的是一张 8 位图片（PNG/JPG），不是浮点数组。浮点扰动
    经过 8 位取整后会有轻微损失（ε=0.03 时量化误差 ≤0.5/255，约为扰动幅度的 6.5%），
    少部分像素的扰动会被完全抹掉。所以报告"实际交付形态下的攻击成功率"比
    报告浮点版本更诚实。本模块同时给出两个数，差异本身就值得在论文里写一句。
    """
    return torch.round(x_pix * 255.0) / 255.0


# ================================================================ 工厂


ATTACKS = {
    "fgsm": lambda ps, x, y, eps, crit, **kw: fgsm(ps, x, y, eps, crit),
    "pgd": lambda ps, x, y, eps, crit, steps=10, alpha=0.01, **kw:
        pgd(ps, x, y, eps, alpha=alpha, steps=steps, criterion=crit),
}


def run_attack(name: str, ps: PixelSpace, x_pix: torch.Tensor, y: torch.Tensor,
               eps: float, steps: int = 10, alpha: float = 0.01) -> torch.Tensor:
    """按名字调用攻击。攻击一律用**不带标签平滑**的交叉熵。

    为什么不用训练时的 criterion：训练配了 label_smoothing=0.1，它会把目标
    从 one-hot 变成软标签，梯度方向因此偏离"让模型判错"这个真正的攻击目标。
    攻击必须用干净的交叉熵。
    """
    criterion = nn.CrossEntropyLoss()
    if name not in ATTACKS:
        raise ValueError(f"未知攻击：{name}，可选 {sorted(ATTACKS)}")
    return ATTACKS[name](ps, x_pix, y, eps, criterion, steps=steps, alpha=alpha)
