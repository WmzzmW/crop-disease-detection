"""全局路径与常量：所有训练/评估脚本的唯一配置来源。

【为什么单独一个文件】
    路径写错是这类项目最常见的低级错误，而且往往要等到跑了几十分钟才暴露。
    把路径和类别映射集中在这里，任何脚本 import 一下就拿到，不重复也不易错。

【关键路径关系】
    划分文件里的一行形如：  raw/color/Tomato___healthy/xxx.JPG
    图像实际所在位置：      data/raw/PlantVillage/raw/color/Tomato___healthy/xxx.JPG
    即：IMAGE_ROOT + <划分文件里的一行>
    （压缩包内层本来就叫 raw/，解压后路径里出现两个 raw，不是笔误）

【类别编号】
    38 个类别按名称排序后依次编号 0..37。这个顺序一旦定下就不能改——
    训练权重、混淆矩阵、对抗样本标签、后端知识库全部依赖它。
    首次运行会把它落盘到 data/splits/class_names.json，之后各脚本都读这份文件。
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------- 路径

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_ROOT = DATA_DIR / "raw" / "PlantVillage"        # 图像根目录（/ 划分文件里的相对路径）
SPLITS_DIR = DATA_DIR / "splits"                      # 官方划分与元数据
RUNS_DIR = PROJECT_ROOT / "training" / "runs"         # 训练输出（含权重、曲线、日志）
BACKEND_MODELS_DIR = PROJECT_ROOT / "backend" / "models"

TRAIN_SPLIT_FILE = SPLITS_DIR / "color_train.txt"
TEST_SPLIT_FILE = SPLITS_DIR / "color_test.txt"
CLASS_NAMES_FILE = SPLITS_DIR / "class_names.json"

# ---------------------------------------------------------------- 图像

# ImageNet 统计量。用预训练权重就必须用它，否则输入分布和预训练时不匹配，精度会掉。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMG_SIZE = 224
NUM_CLASSES = 38


# ---------------------------------------------------------------- 类别映射


def get_class_names() -> list[str]:
    """返回 38 个类别名（已排序），并在首次调用时落盘缓存。

    优先读缓存文件；没有缓存则扫描图像目录，排序后写盘。
    这样即使将来图像目录被删，只要 class_names.json 还在就能恢复映射。
    """
    if CLASS_NAMES_FILE.exists():
        names = json.loads(CLASS_NAMES_FILE.read_text(encoding="utf-8"))
        if len(names) == NUM_CLASSES:
            return names

    color_dir = IMAGE_ROOT / "raw" / "color"
    if not color_dir.is_dir():
        raise FileNotFoundError(
            f"找不到图像目录：{color_dir}\n"
            f"请先执行：.venv/bin/python training/dataset/prepare_dataset.py"
        )

    names = sorted(d.name for d in color_dir.iterdir() if d.is_dir())
    if len(names) != NUM_CLASSES:
        raise RuntimeError(f"预期 {NUM_CLASSES} 个类别，实际 {len(names)} 个：{color_dir}")

    CLASS_NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_FILE.write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return names


def read_split(path: Path) -> list[tuple[str, int]]:
    """读官方划分文件，返回 [(图像绝对路径, 类别编号), ...]。

    划分文件里的相对路径是相对 IMAGE_ROOT 的，且用 POSIX 分隔符（/），
    在 macOS 上直接拼接即可；Windows 需要额外转换，本课题不涉及。
    """
    class_to_idx = {name: i for i, name in enumerate(get_class_names())}
    samples: list[tuple[str, int]] = []
    missing_class: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        # rel = raw/color/<类别>/<文件名>，取第 3 段作为类别名
        parts = rel.split("/")
        if len(parts) < 4:
            continue
        class_name = parts[2]
        idx = class_to_idx.get(class_name)
        if idx is None:
            missing_class.add(class_name)
            continue
        samples.append((str(IMAGE_ROOT / rel), idx))

    if missing_class:
        raise RuntimeError(f"划分文件中出现未知类别：{sorted(missing_class)}")
    if not samples:
        raise RuntimeError(f"划分文件为空或格式不符：{path}")
    return samples
