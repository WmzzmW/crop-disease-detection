"""数据加载：官方划分 + 从训练集分层切验证集 + 图像增强。

【划分策略——这是本项目最容易出错、也最要紧的一环】
    PlantVillage 里同一片叶子会被拍很多张（不同角度/光照）。如果按图片随机划分，
    同一片叶子的照片会一半进训练集、一半进测试集，模型在测试集上"见过"这片叶子，
    准确率会虚高好几个百分点——这叫数据泄露。

    官方划分文件 color_train.txt / color_test.txt 已经按叶片分组做好了，直接用。
    验证集从官方训练集里再切，因为是在已分组的集合内部切，不会引入泄露。

    因此三种划分的来源是：
        训练集 / 验证集  <-  color_train.txt（43,596 张）按类别分层切 90% / 10%
        测试集           <-  color_test.txt（10,709 张），全程不动，只在最终评估时用
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import (
    DEFAULT_IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TEST_SPLIT_FILE,
    TRAIN_SPLIT_FILE,
    get_class_names,
    read_split,
)


# ---------------------------------------------------------------- 变换


def build_transforms(img_size: int = DEFAULT_IMG_SIZE, train: bool = True):
    """按配方构造图像变换。

    训练集：随机裁剪 + 水平翻转 + 旋转 ±15° + 颜色抖动
            —— 每一项都对应一种拍摄差异（距离、朝向、角度、光照），
               目的是让模型别把"某一特定拍法"当成病害特征。
    验证/测试集：Resize 到稍大再中心裁剪。不做任何随机变换，
               否则同一个模型每次评估结果都不一样，无法比较。
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),   # 224 -> 256，再中心裁剪
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------- Dataset


class PlantVillageDataset(Dataset):
    """按 (图像路径, 类别编号) 列表读取图片。

    samples 与 class_names 暴露为属性，方便后续脚本（对抗样本生成、
    结果可视化）反查某张图来自哪个类别、对应哪个文件。
    """

    def __init__(self, samples: list[tuple[str, int]], transform=None) -> None:
        self.samples = samples
        self.transform = transform
        self.class_names = get_class_names()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")       # 防止混入灰度/带 alpha 的图
            if self.transform is not None:
                img = self.transform(img)
        return img, label

    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def class_counts(self) -> Counter:
        return Counter(self.labels())


# ---------------------------------------------------------------- 组装


def build_datasets(img_size: int = DEFAULT_IMG_SIZE, val_ratio: float = 0.1,
                   seed: int = 42, verbose: bool = True):
    """返回 (train_set, val_set, test_set)，划分逻辑集中在这里。

    val_ratio 是从官方训练集里切出来做验证的比例；测试集始终是官方测试集全量。
    """
    train_all = read_split(TRAIN_SPLIT_FILE)
    test_samples = read_split(TEST_SPLIT_FILE)

    paths = [p for p, _ in train_all]
    labels = [y for _, y in train_all]

    # stratify 保证每个类别在验证集里都按比例出现。小类别样本少，
    # 不分层的话可能整个类别都不出现在验证集里，验证指标就失去意义。
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=val_ratio, random_state=seed, stratify=labels,
    )

    train_set = PlantVillageDataset(
        list(zip(train_paths, train_labels)), build_transforms(img_size, train=True)
    )
    val_set = PlantVillageDataset(
        list(zip(val_paths, val_labels)), build_transforms(img_size, train=False)
    )
    test_set = PlantVillageDataset(test_samples, build_transforms(img_size, train=False))

    if verbose:
        names = get_class_names()
        counts = Counter(train_labels)
        per_class = [counts.get(i, 0) for i in range(len(names))]
        print(f"  训练集 {len(train_set):>6,} 张    验证集 {len(val_set):>6,} 张"
              f"    测试集 {len(test_set):>6,} 张")
        print(f"  类别数 {len(names)}   训练集每类样本："
              f"最少 {min(per_class)} / 中位 {sorted(per_class)[len(per_class) // 2]}"
              f" / 最多 {max(per_class)}")
        if min(per_class) < 100:
            smallest = [names[i] for i, c in enumerate(per_class) if c == min(per_class)]
            print(f"  ⚠️  样本最少的类别（{min(per_class)} 张）：{', '.join(smallest[:3])}")
            print("     样本不均衡，评估时务必看每类 F1，不能只看总准确率")

    return train_set, val_set, test_set


def build_loaders(train_set, val_set, test_set, batch_size: int = 64,
                  num_workers: int = 4, weighted_sampler: bool = False,
                  device: torch.device | None = None):
    """构造三个 DataLoader。

    weighted_sampler=True 时用类别倒频率采样，缓解样本不均衡；
    但它会改变每个 epoch 的有效样本分布，属于可选增强，默认关闭。
    """
    pin = device is not None and device.type == "cuda"
    common = dict(num_workers=num_workers, pin_memory=pin,
                  persistent_workers=num_workers > 0)

    if weighted_sampler:
        counts = train_set.class_counts()
        weights = [1.0 / counts[label] for label in train_set.labels()]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def stratified_subset(samples: list[tuple[str, int]], total: int) -> list[tuple[str, int]]:
    """每类取相同数量，凑够约 total 张。

    冒烟测试要截断数据时用这个，不要用 samples[:N]——划分文件的顺序是按类别
    聚在一起的，直接切前 N 张只会拿到两三个类别，跑出来的数字会变成
    「模型只学了 4 类」的结果，看起来像是代码有 bug，实际只是截断方式不对。
    """
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    n_per = max(1, total // max(len(by_class), 1))
    out: list[tuple[str, int]] = []
    for label in sorted(by_class):
        out.extend(by_class[label][:n_per])
    return out


def build_test_loader(test_set, batch_size: int = 64, num_workers: int = 4,
                      device: torch.device | None = None) -> DataLoader:
    """只构造测试集 DataLoader。

    评估脚本不需要训练集和验证集，走 build_loaders 就得传两个 None 进去，
    语义很别扭，所以单开一个。
    """
    pin = device is not None and device.type == "cuda"
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin,
                      persistent_workers=num_workers > 0)


def print_class_distribution(dataset: PlantVillageDataset, top: int = 5) -> None:
    """打印类别分布摘要（全量分布另有文件保存）。"""
    counts = dataset.class_counts()
    names = dataset.class_names
    ordered = sorted(((names[i], c) for i, c in counts.items()), key=lambda x: -x[1])
    print(f"  样本最多的 {top} 类：" + "，".join(f"{n} {c}" for n, c in ordered[:top]))
    print(f"  样本最少的 {top} 类：" + "，".join(f"{n} {c}" for n, c in ordered[-top:]))


def resolve_image_root() -> Path:
    """供其他脚本检查图像目录是否存在。"""
    from .config import IMAGE_ROOT
    return IMAGE_ROOT
