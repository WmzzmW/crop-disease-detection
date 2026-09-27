"""分类模型构建：MobileNetV3-Small + ImageNet 预训练 + 38 类分类头。

单独成文件的原因：训练脚本、对抗样本生成脚本、评估脚本都要拿到
**完全相同**的模型结构，否则权重加载会失败或静默错位。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES


def build_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                     dropout: float = 0.2) -> nn.Module:
    """构建 MobileNetV3-Small，换掉分类头为 num_classes 类。

    结构要点：
      - 骨干网络（features）输出 576 维特征向量
      - 分类头（classifier）= Linear(576,1024) -> Hardswish -> Dropout -> Linear(1024,38)
      - torchvision 的原生分类头本来就是一个 Dropout，这里把概率调成参数可控

    pretrained=True 时加载 ImageNet 权重——这是整个训练能收敛到 98%+ 的前提。
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # 把原生分类头里的 Dropout 换成指定概率，便于消融实验时统一调整
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Dropout):
            model.classifier[i] = nn.Dropout(p=dropout)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """冻结除分类头以外的全部参数（训练阶段一用）。

    为什么必须冻：分类头是随机初始化的，一上来就整体训练的话，
    它产生的大梯度会沿反向传播一路冲进骨干，把 ImageNet 学到的
    通用特征摧毁掉。先让分类头稳定下来，再整体微调。
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier.")


def unfreeze_all(model: nn.Module) -> None:
    """解冻全部参数（训练阶段二用）。"""
    for param in model.parameters():
        param.requires_grad = True


def trainable_parameters(model: nn.Module):
    """返回需要梯度的参数列表，交给优化器。

    只把 requires_grad=True 的参数传给优化器，避免 AdamW 的权重衰减
    作用在冻结参数上（虽然不会被更新，但语义上应该排除）。
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe_model(model: nn.Module) -> str:
    total, trainable = count_parameters(model)
    return f"MobileNetV3-Small   总参数 {total:,}（{total / 1e6:.2f}M）   可训练 {trainable:,}"


def save_checkpoint(path, model: nn.Module, meta: dict) -> None:
    """保存权重 + 元信息（类别映射、图像尺寸、指标等）。

    元信息必须一起存：将来加载权重时如果类别映射变了，
    预测结果会整体错位且不报错——这是最难排查的一类 bug。
    """
    from .config import get_class_names
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": get_class_names(),
        "num_classes": len(get_class_names()),
        **meta,
    }, path)


def load_checkpoint(path, device: torch.device, num_classes: int = NUM_CLASSES):
    """加载权重并返回 (模型, 元信息)。模型已置于 eval 模式。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier(num_classes=ckpt.get("num_classes", num_classes),
                             pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
