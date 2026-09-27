#!/usr/bin/env python
"""评估分类模型，产出开题报告「实验一」需要的全部指标与图。

对应开题报告 3.2 节的实验一要求：
    Top-1/Top-3 准确率、精确率、召回率、F1-Score、混淆矩阵、单张推理时间

【产出】
    <输出目录>/
        metrics.json           全部数值指标（写论文时直接取数）
        per_class_report.txt   每类的精确率/召回率/F1/支持数（文本表）
        confusion_matrix.png   38×38 混淆矩阵（原始计数 + 行归一化，两联图）
        confusion_matrix.json  混淆矩阵原始数值（可自行重绘）
        training_curves.png    训练曲线（需提供 --history，或自动从同目录读取）
        inference_time.json    推理耗时（GPU + CPU 单张）

【用法】
    # 自动找 training/runs/ 下最近一次训练的结果
    .venv/bin/python training/evaluate_classifier.py

    # 指定某次训练的权重
    .venv/bin/python training/evaluate_classifier.py \\
        --checkpoint training/runs/classifier_20260926_143012/best.pth

    # 只测推理速度，不做完整评估（几十秒）
    .venv/bin/python training/evaluate_classifier.py --time-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib
matplotlib.use("Agg")           # 无界面后端，避免弹窗、可在服务器上跑
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from common.config import NUM_CLASSES, RUNS_DIR, get_class_names
from common.data import build_datasets, build_test_loader
from common.model import build_classifier, load_checkpoint
from common.utils import (
    Timer,
    banner,
    describe_device,
    get_device,
    human_time,
    save_json,
    section,
)


# ================================================================ 绘图辅助


CJK_CANDIDATES = ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC",
                  "Arial Unicode MS", "Hiragino Sans GB", "Microsoft YaHei"]


def setup_font() -> bool:
    """尝试启用中文字体。返回是否成功。

    找不到中文字体时，图上中文会变成方块（豆腐块）。这里先探测，
    探测不到就让调用方改用英文标签，避免出一张废图。
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


# 中文可用与否，决定图上用哪套标签
HAS_CJK = setup_font()
L_CORRECT = "正确率" if HAS_CJK else "accuracy"
L_TRAIN = "训练" if HAS_CJK else "train"
L_VAL = "验证" if HAS_CJK else "val"
L_LOSS = "损失" if HAS_CJK else "loss"
L_EPOCH = "轮次" if HAS_CJK else "epoch"
L_TRUE = "真实类别" if HAS_CJK else "True label"
L_PRED = "预测类别" if HAS_CJK else "Predicted label"
L_COUNT = "样本数" if HAS_CJK else "count"
L_RATIO = "占该行比例" if HAS_CJK else "row-normalized"


# ================================================================ 指标


@torch.no_grad()
def collect_predictions(model, loader, device):
    """跑一遍数据集，收集全部预测结果。返回 (真实标签, 预测标签, 各类概率)。"""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    pbar = tqdm(loader, desc="  推理", ncols=110, leave=False, unit="batch")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.float().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())
    pbar.close()

    return (np.concatenate(all_targets), np.concatenate(all_preds),
            np.concatenate(all_probs))


def topk_accuracy(targets: np.ndarray, probs: np.ndarray, k: int) -> float:
    """Top-k 准确率（百分数）。"""
    topk = np.argsort(-probs, axis=1)[:, :k]
    hit = (topk == targets[:, None]).any(axis=1)
    return 100.0 * hit.mean()


# ================================================================ 绘图


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    """画 38×38 混淆矩阵：左图原始计数，右图行归一化（等价于每类召回率）。"""
    short = [n.replace("___", " / ").replace("_", " ") for n in class_names]
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float),
                        where=row_sum != 0)

    fig, axes = plt.subplots(1, 2, figsize=(24, 11))

    for ax, data, title, fmt in (
        (axes[0], cm, f"混淆矩阵 — {L_COUNT}" if not HAS_CJK else "混淆矩阵（样本数）", "d"),
        (axes[1], cm_norm, "混淆矩阵（行归一化 = 每类召回率）" if HAS_CJK
         else "Confusion matrix (row-normalized)", ".2f"),
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=data.max() if data.max() > 0 else 1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticklabels(short, fontsize=6)
        ax.set_xlabel(L_PRED, fontsize=10)
        ax.set_ylabel(L_TRUE, fontsize=10)
        ax.set_title(title, fontsize=12)

        # 只在数值够大时标注，否则 38×38 = 1444 个数字挤成一团没法看
        thresh = data.max() / 2 if data.max() > 0 else 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] > thresh and data[i, j] > 0:
                    ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                            color="white" if data[i, j] > thresh else "black",
                            fontsize=5)

    fig.suptitle("PlantVillage 测试集混淆矩阵（38 类）"
                 if HAS_CJK else "PlantVillage test set confusion matrix (38 classes)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: list[dict], out_path: Path):
    """画训练曲线：损失 + 准确率，标出两阶段分界。"""
    epochs = [h["epoch"] for h in history]
    stage2 = next((h["epoch"] for h in history if h["stage"] == "stage2"), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (tr_key, va_key), ylabel, title in (
        (axes[0], ("train_loss", "val_loss"), L_LOSS, "损失曲线" if HAS_CJK else "Loss"),
        (axes[1], ("train_top1", "val_top1"), L_CORRECT + " (%)",
         "准确率曲线" if HAS_CJK else "Accuracy"),
    ):
        ax.plot(epochs, [h[tr_key] for h in history], "o-", label=L_TRAIN, lw=2, ms=4)
        ax.plot(epochs, [h[va_key] for h in history], "s-", label=L_VAL, lw=2, ms=4)
        if stage2:
            ax.axvline(stage2 - 0.5, color="gray", ls="--", lw=1.2)
            ax.text(stage2 - 0.4, ax.get_ylim()[1], " 阶段二开始" if HAS_CJK
                    else " stage 2", fontsize=9, va="top", color="gray")
        ax.set_xlabel(L_EPOCH)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    # 标出最优验证轮次
    best = max(history, key=lambda h: h["val_top1"])
    axes[1].scatter([best["epoch"]], [best["val_top1"]], s=140, marker="*",
                    color="red", zorder=5,
                    label=f"best {best['val_top1']:.2f}%")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 推理耗时


@torch.no_grad()
def measure_inference_time(model, dataset, device: torch.device, n_warmup: int = 5,
                           n_measure: int = 50) -> dict:
    """测单张推理耗时。

    分两种模式测，因为它们的用途不同：
      batch=1   —— 对应 App 单张拍照上传的真实场景
      batch=32  —— 对应后端并发处理时的吞吐表现

    先跑 n_warmup 次预热（首次调用要编译算子、分配显存/缓存），
    不计入统计——否则第一个样本的耗时会严重拉高平均值。
    """
    from torch.utils.data import DataLoader

    results = {}
    for batch_size in (1, 32):
        if batch_size > len(dataset):
            continue
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for _ in range(max(1, n_warmup // batch_size)):
            for images, _ in loader:
                model(images.to(device))
                break

        total_images, total_sec = 0, 0.0
        for images, _ in loader:
            if total_images >= n_measure:
                break
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = Timer()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_sec += t.elapsed
            total_images += images.size(0)

        results[f"batch_{batch_size}"] = {
            "ms_per_image": 1000 * total_sec / max(total_images, 1),
            "images": total_images,
        }
    return results


# ================================================================ 主流程


def find_latest_checkpoint() -> Path | None:
    """在 training/runs/ 下找最近一次训练的 best.pth。"""
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(RUNS_DIR.glob("classifier_*/best.pth"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="评估分类模型并生成图表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="权重文件；默认自动找 training/runs 下最近一次")
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录；默认写到权重所在目录")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--time-only", action="store_true",
                   help="只测推理耗时，跳过完整评估")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        print("❌ 找不到权重文件。请先训练：")
        print("   .venv/bin/python training/train_classifier.py")
        print("   或指定：--checkpoint <路径>")
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("分类模型评估")
    print(f"权重：{ckpt_path}")
    print(f"输出：{out_dir}")
    if not HAS_CJK:
        print("提示：未找到中文字体，图表标签自动改用英文（数值不受影响）")

    device = get_device(args.device)
    model, meta = load_checkpoint(ckpt_path, device)
    img_size = meta.get("img_size", args.img_size)
    print(f"设备：{describe_device(device)}    图像尺寸：{img_size}")
    if "val_top1" in meta:
        print(f"该权重训练时：第 {meta.get('epoch', '?')} 轮，验证 Top-1 {meta['val_top1']:.2f}%")

    class_names = meta.get("class_names") or get_class_names()

    # ---------- 数据 ----------
    section("[1/4] 加载测试集")
    _, _, test_set = build_datasets(img_size=img_size, verbose=False)
    test_loader = build_test_loader(test_set, batch_size=args.batch_size,
                                    num_workers=args.num_workers, device=device)
    print(f"  测试集 {len(test_set):,} 张，{len(class_names)} 类")

    targets, probs = None, None
    metrics: dict = {}

    if not args.time_only:
        # ---------- 预测 ----------
        section("[2/4] 测试集推理")
        targets, preds, probs = collect_predictions(model, test_loader, device)

        # ---------- 指标 ----------
        section("[3/4] 分类指标")
        top1 = 100.0 * (preds == targets).mean()
        top3 = topk_accuracy(targets, probs, 3)
        top5 = topk_accuracy(targets, probs, 5)
        print(f"  Top-1 准确率  {top1:6.2f}%")
        print(f"  Top-3 准确率  {top3:6.2f}%")
        print(f"  Top-5 准确率  {top5:6.2f}%")

        # 宏平均 F1：每类先算 F1 再取平均。样本不均衡时它比总准确率更有意义，
        # 因为它不偏袒样本多的类。
        report_txt = classification_report(
            targets, preds, target_names=class_names, digits=4, zero_division=0)
        report_dict = classification_report(
            targets, preds, target_names=class_names, digits=4,
            zero_division=0, output_dict=True)
        macro_f1 = report_dict["macro avg"]["f1-score"]
        weighted_f1 = report_dict["weighted avg"]["f1-score"]
        print(f"  宏平均 F1     {macro_f1 * 100:6.2f}%   "
              f"（每类等权，不受样本量影响）")
        print(f"  加权平均 F1   {weighted_f1 * 100:6.2f}%")

        (out_dir / "per_class_report.txt").write_text(
            f"测试集分类报告（{len(targets):,} 张，{len(class_names)} 类）\n"
            f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"{'=' * 90}\n{report_txt}", encoding="utf-8")
        print(f"  每类明细已写入 per_class_report.txt")

        # 找出表现最差的类，答辩时会被问到
        per_class_f1 = [(class_names[i], report_dict[class_names[i]]["f1-score"],
                         report_dict[class_names[i]]["support"])
                        for i in range(len(class_names))
                        if class_names[i] in report_dict]
        worst = sorted(per_class_f1, key=lambda x: x[1])[:5]
        print(f"\n  F1 最低的 5 个类别（论文中需要分析原因）：")
        for name, f1, sup in worst:
            print(f"    {name:<34} F1 {f1 * 100:6.2f}%   样本 {int(sup):>4} 张")

        print(report_txt)

        # ---------- 混淆矩阵 ----------
        section("[4/4] 混淆矩阵")
        cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
        plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
        save_json({"labels": class_names, "matrix": cm.tolist()},
                  out_dir / "confusion_matrix.json")

        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        top_confusions = []
        for _ in range(5):
            i, j = np.unravel_index(off_diag.argmax(), off_diag.shape)
            if off_diag[i, j] == 0:
                break
            top_confusions.append({
                "true": class_names[i], "pred": class_names[j],
                "count": int(off_diag[i, j]),
                "pct_of_true_class": round(100 * off_diag[i, j] / cm[i].sum(), 2),
            })
            off_diag[i, j] = 0

        print("  混淆矩阵已保存：")
        print(f"    confusion_matrix.png   （38×38 热力图，原始计数 + 行归一化）")
        print(f"    confusion_matrix.json  （原始数值，可自行重绘）")
        print("\n  最常被混淆的类别对（论文中需重点分析）：")
        for c in top_confusions:
            print(f"    {c['true']:<32} 误判为 {c['pred']:<32} "
                  f"{c['count']:>4} 次（占该类 {c['pct_of_true_class']:.1f}%）")

        metrics = {
            "test_size": int(len(targets)),
            "top1": round(top1, 4), "top3": round(top3, 4), "top5": round(top5, 4),
            "macro_f1": round(macro_f1 * 100, 4),
            "weighted_f1": round(weighted_f1 * 100, 4),
            "worst_classes": [{"name": n, "f1": round(f * 100, 4), "support": int(s)}
                              for n, f, s in worst],
            "top_confusions": top_confusions,
            "per_class": {k: v for k, v in report_dict.items()
                          if k in class_names},
        }

    # ---------- 推理耗时 ----------
    section("推理耗时")
    print(f"  在 {describe_device(device)} 上测量（预热后取平均）")
    t_primary = measure_inference_time(model, test_set, device)
    timing = {f"{device.type}": t_primary}
    for k, v in t_primary.items():
        print(f"    {k:<12} {v['ms_per_image']:>7.1f} ms/张  （{v['images']} 张）")

    # CPU 单张耗时——开题报告的指标「单张推理时间 < 200ms（CPU）」用的是这个
    if device.type != "cpu":
        print(f"  再在 CPU 上测一次（对应开题报告的「单张推理时间(CPU)」指标）")
        cpu_model = build_classifier(num_classes=meta.get("num_classes", NUM_CLASSES),
                                     pretrained=False)
        # 直接复用已加载模型的权重，不要再去 meta 里找 state_dict——
        # load_checkpoint 故意把它从 meta 里剔除了，重新把整个权重文件读一遍也没必要。
        cpu_model.load_state_dict(model.state_dict())
        cpu_model.eval()
        t_cpu = measure_inference_time(cpu_model, test_set, torch.device("cpu"))
        timing["cpu"] = t_cpu
        for k, v in t_cpu.items():
            print(f"    cpu {k:<8} {v['ms_per_image']:>7.1f} ms/张  （{v['images']} 张）")
        ms = t_cpu["batch_1"]["ms_per_image"]
        target = 200
        print(f"\n  开题报告目标：单张 < {target} ms (CPU)  →  "
              f"实际 {ms:.1f} ms  {'✅ 达标' if ms < target else '❌ 未达标'}")

    save_json(timing, out_dir / "inference_time.json")
    metrics["inference_time"] = timing

    # ---------- 训练曲线 ----------
    history_path = out_dir / "history.json"
    if history_path.exists():
        hist = json.loads(history_path.read_text(encoding="utf-8"))
        if hist.get("history"):
            plot_training_curves(hist["history"], out_dir / "training_curves.png")
            print(f"\n  训练曲线已保存：training_curves.png")

    save_json(metrics, out_dir / "metrics.json")

    banner("评估完成")
    if "top1" in metrics:
        print(f"Top-1 {metrics['top1']:.2f}%   Top-3 {metrics['top3']:.2f}%   "
              f"宏平均 F1 {metrics['macro_f1']:.2f}%")
    else:
        print("（--time-only 模式，跳过了分类指标评估）")
    print(f"全部结果已写入：{out_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
