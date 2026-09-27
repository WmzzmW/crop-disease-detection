"""通用工具：随机种子、设备选择、指标统计、终端+文件双写日志。

被 train_classifier.py 与 evaluate_classifier.py 共用。
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- 可复现


def set_seed(seed: int) -> None:
    """固定所有随机源，保证同一 seed 下结果可复现。

    注意：这不能保证跨设备（CPU / MPS / CUDA）完全一致，
    只能保证同一台机器上重复运行结果相同。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """选择计算设备。M1/M2 走 MPS，有 NVIDIA 卡走 CUDA，否则 CPU。"""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "mps":
        return "Apple GPU (MPS)"
    if device.type == "cuda":
        return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


# ---------------------------------------------------------------- 指标统计


class AverageMeter:
    """滑动平均累加器：按样本数加权，避免最后一个不满 batch 造成偏差。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 3)) -> list[float]:
    """返回 Top-k 准确率（百分数），顺序与 topk 参数一致。"""
    maxk = max(topk)
    batch_size = target.size(0)

    # 取分数最高的 maxk 个类别的下标
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result: list[float] = []
    for k in topk:
        k = min(k, maxk)
        correct_k = correct[:k].reshape(-1).float().sum().item()
        result.append(100.0 * correct_k / batch_size)
    return result


# ---------------------------------------------------------------- 计时


class Timer:
    """记录阶段耗时，并以人类可读格式输出。"""

    def __init__(self) -> None:
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def reset(self) -> None:
        self.t0 = time.time()


def human_time(seconds: float) -> str:
    """把秒数格式化成 1h23m 或 4m56s 这样的短串。"""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


# ---------------------------------------------------------------- 日志双写


class Tee:
    """把 stdout 同时写到终端和日志文件。

    tqdm 的进度条走 stderr，因此不会进日志文件——日志保持干净，
    终端该有的实时进度一样不少。
    """

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def start_logging(log_path: Path) -> None:
    """把 print 的输出同时写进 log_path，终端照常显示。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, fh)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 打印


def banner(title: str, char: str = "=", width: int = 70) -> None:
    print(char * width)
    print(title)
    print(char * width)


def section(title: str, width: int = 70) -> None:
    print(f"\n{'-' * width}\n{title}\n{'-' * width}")
