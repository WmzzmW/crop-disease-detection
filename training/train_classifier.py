#!/usr/bin/env python
"""训练农作物病虫害分类模型（MobileNetV3-Small + 两阶段迁移学习）。

【训练策略】
    阶段一（冻结骨干，只训分类头）
        分类头是随机初始化的，此时它的梯度很大。如果一上来就整体训练，
        这些梯度会沿反向传播冲进骨干网络，把 ImageNet 预训练学到的通用特征
        摧毁掉。所以先冻结骨干，让分类头单独收敛到合理位置。
        学习率可以大一些（1e-3），因为只调一个线性层。

    阶段二（解冻全部，低学习率微调）
        骨干特征已经"不错"了，只需要小幅调整去适配 PlantVillage 的病害特点。
        学习率必须小（1e-4），否则会过冲。配 cosine 衰减，越训越小。

【预期结果】
    测试集 Top-1 约 98%~99%。到这个量级就够了——PlantVillage 上 99%+ 是标配，
    不是成果。多花的时间应该投到对抗攻击与检测实验上。

【输出】
    training/runs/classifier_<时间戳>/
        best.pth          验证集最优权重
        last.pth          最后一轮权重
        history.json      每轮的 loss / 准确率，用于画训练曲线
        config.json       本次运行的全部参数（可复现性）
        log.txt           终端输出的完整副本

【用法】
    # 先跑冒烟测试，确认全流程没问题（约 2 分钟）
    .venv/bin/python training/train_classifier.py --smoke-test

    # 正式训练（M1 Pro 上约 1~1.5 小时）
    .venv/bin/python training/train_classifier.py

    # 后台跑，日志留在终端也留在文件里
    .venv/bin/python training/train_classifier.py 2>&1 | tee training/runs/latest.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# 让 `python training/train_classifier.py` 能 import 到 common 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

# MPS 上个别算子没有实现时自动回退到 CPU，避免训练中途报错中断
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common.config import DEFAULT_IMG_SIZE, RUNS_DIR, get_class_names
from common.data import build_datasets, build_loaders, stratified_subset
from common.model import (
    build_classifier,
    count_parameters,
    describe_model,
    freeze_backbone,
    save_checkpoint,
    trainable_parameters,
    unfreeze_all,
)
from common.utils import (
    AverageMeter,
    Timer,
    accuracy,
    banner,
    describe_device,
    get_device,
    human_time,
    save_json,
    section,
    set_seed,
    start_logging,
)


# ================================================================ 训练与评估


def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch: int, total_epochs: int, stage: str):
    """训练一轮，带实时进度条。返回 (loss, top1, top3)。"""
    model.train()
    loss_meter, top1_meter, top3_meter = AverageMeter(), AverageMeter(), AverageMeter()

    pbar = tqdm(
        loader,
        desc=f"  [{stage}] {epoch:>2}/{total_epochs} 训练",
        ncols=110, leave=False, unit="batch", dynamic_ncols=False,
    )
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc1, acc3 = accuracy(outputs, targets, topk=(1, 3))
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(acc1, n)
        top3_meter.update(acc3, n)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{top1_meter.avg:5.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )
    pbar.close()
    return loss_meter.avg, top1_meter.avg, top3_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "验证") -> tuple[float, float, float]:
    """在给定集合上评估，返回 (loss, top1, top3)。"""
    model.eval()
    loss_meter, top1_meter, top3_meter = AverageMeter(), AverageMeter(), AverageMeter()

    pbar = tqdm(loader, desc=f"  {desc:<12}", ncols=110, leave=False,
                unit="batch", dynamic_ncols=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc3 = accuracy(outputs, targets, topk=(1, 3))
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(acc1, n)
        top3_meter.update(acc3, n)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{top1_meter.avg:5.2f}%")
    pbar.close()
    return loss_meter.avg, top1_meter.avg, top3_meter.avg


# ================================================================ 主流程


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练 PlantVillage 病虫害分类模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 训练轮数与学习率
    p.add_argument("--epochs-stage1", type=int, default=5,
                   help="阶段一（冻结骨干，只训分类头）轮数")
    p.add_argument("--epochs-stage2", type=int, default=15,
                   help="阶段二（解冻全模型微调）轮数")
    p.add_argument("--lr-stage1", type=float, default=1e-3, help="阶段一学习率")
    p.add_argument("--lr-stage2", type=float, default=1e-4, help="阶段二学习率")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="标签平滑系数（0 表示关闭）")

    # 数据
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--val-ratio", type=float, default=0.1, help="从训练集切出的验证集比例")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 进程数；若卡住不动改成 0")
    p.add_argument("--weighted-sampler", action="store_true",
                   help="启用类别倒频率采样，缓解样本不均衡（默认关闭）")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto / mps / cuda / cpu")
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录，默认 training/runs/classifier_<时间戳>")
    p.add_argument("--tag", type=str, default="", help="给这次运行起个名字，会加在目录名后面")
    p.add_argument("--smoke-test", action="store_true",
                   help="冒烟测试：小数据量 + 各 1 轮，约 2 分钟，用于验证全流程能跑通")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 冒烟测试：把一切压到最小，只为验证"能不能跑通"
    if args.smoke_test:
        args.epochs_stage1, args.epochs_stage2 = 1, 1
        args.num_workers = 2
        print("⚠️  冒烟测试模式：只跑各 1 轮且仅用少量数据，结果无意义\n")

    # ---------- 输出目录与日志 ----------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"classifier_{stamp}" + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.output_dir) if args.output_dir else RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    start_logging(out_dir / "log.txt")

    banner("农作物病虫害分类模型训练")
    print(f"输出目录：{out_dir}")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(args.seed)
    device = get_device(args.device)

    # ---------- 数据 ----------
    section("[1/4] 准备数据")
    train_set, val_set, test_set = build_datasets(
        img_size=args.img_size, val_ratio=args.val_ratio, seed=args.seed,
    )
    if args.smoke_test:
        # 分层截断：每个类别取相同的张数，保证 38 类都出现。
        # 直接切 samples[:N] 会因为划分文件按类别聚集而只拿到两三个类别，
        # 跑出来的数字无法解释。
        train_set.samples = stratified_subset(train_set.samples, 38 * 40)
        val_set.samples = stratified_subset(val_set.samples, 38 * 8)
        test_set.samples = stratified_subset(test_set.samples, 38 * 8)
        print(f"  （冒烟测试：每类取相同张数，"
              f"{len(train_set)} / {len(val_set)} / {len(test_set)} 张，覆盖全部 38 类）")

    train_loader, val_loader, test_loader = build_loaders(
        train_set, val_set, test_set,
        batch_size=args.batch_size, num_workers=args.num_workers,
        weighted_sampler=args.weighted_sampler, device=device,
    )
    print(f"  每个 epoch {len(train_loader)} 个 batch"
          f"（batch size {args.batch_size}，图像 {args.img_size}×{args.img_size}）")
    if args.weighted_sampler:
        print("  已启用类别倒频率采样")

    # ---------- 模型 ----------
    section("[2/4] 构建模型")
    print(f"  计算设备：{describe_device(device)}")
    model = build_classifier().to(device)
    print(f"  {describe_model(model)}")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f"  损失函数：交叉熵（标签平滑 {args.label_smoothing}）")

    # ---------- 两阶段训练 ----------
    section("[3/4] 开始训练")
    history: list[dict] = []
    best_top1 = 0.0
    best_epoch = 0
    epoch_times: list[float] = []
    total_epochs = args.epochs_stage1 + args.epochs_stage2
    global_epoch = 0
    run_timer = Timer()

    def run_stage_1():
        nonlocal best_top1, best_epoch, global_epoch
        print(f"\n▶ 阶段一：冻结骨干，只训练分类头"
              f"（{args.epochs_stage1} 轮，lr={args.lr_stage1}）")
        freeze_backbone(model)
        _, trainable = count_parameters(model)
        print(f"  冻结后仅 {trainable:,} 个参数可训练\n")

        optimizer = AdamW(trainable_parameters(model), lr=args.lr_stage1,
                          weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage1, 1))

        for _ in range(args.epochs_stage1):
            global_epoch += 1
            t = Timer()
            tr_loss, tr_top1, tr_top3 = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                global_epoch, total_epochs, "阶段一")
            va_loss, va_top1, va_top3 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            _record_and_report(tr_loss, tr_top1, tr_top3, va_loss, va_top1, va_top3,
                               optimizer, "stage1", t.elapsed)

    def run_stage_2():
        nonlocal best_top1, best_epoch, global_epoch
        print(f"\n▶ 阶段二：解冻全模型微调"
              f"（{args.epochs_stage2} 轮，lr={args.lr_stage2}，cosine 衰减）")
        unfreeze_all(model)
        total, trainable = count_parameters(model)
        print(f"  解冻后 {trainable:,} 个参数可训练\n")

        optimizer = AdamW(trainable_parameters(model), lr=args.lr_stage2,
                          weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs_stage2, 1))

        for _ in range(args.epochs_stage2):
            global_epoch += 1
            t = Timer()
            tr_loss, tr_top1, tr_top3 = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                global_epoch, total_epochs, "阶段二")
            va_loss, va_top1, va_top3 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            _record_and_report(tr_loss, tr_top1, tr_top3, va_loss, va_top1, va_top3,
                               optimizer, "stage2", t.elapsed)

    def _record_and_report(tr_loss, tr_top1, tr_top3, va_loss, va_top1, va_top3,
                           optimizer, stage, elapsed):
        nonlocal best_top1, best_epoch
        epoch_times.append(elapsed)

        is_best = va_top1 > best_top1
        if is_best:
            best_top1, best_epoch = va_top1, global_epoch
            save_checkpoint(out_dir / "best.pth", model, {
                "epoch": global_epoch, "stage": stage,
                "val_top1": va_top1, "val_top3": va_top3,
                "img_size": args.img_size, "args": vars(args),
            })

        history.append({
            "epoch": global_epoch, "stage": stage,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr_loss, "train_top1": tr_top1, "train_top3": tr_top3,
            "val_loss": va_loss, "val_top1": va_top1, "val_top3": va_top3,
            "elapsed_sec": elapsed,
        })

        # ETA：按已完成轮次的平均耗时外推剩余轮数
        avg_t = sum(epoch_times) / len(epoch_times)
        remain = (total_epochs - global_epoch) * avg_t
        stamp_str = "★ 最优" if is_best else "      "
        print(f"轮 {global_epoch:>2}/{total_epochs} [{stage}] "
              f"训练 loss {tr_loss:.4f} acc {tr_top1:5.2f}%  |  "
              f"验证 loss {va_loss:.4f} acc {va_top1:5.2f}% top3 {va_top3:5.2f}%  |  "
              f"lr {optimizer.param_groups[0]['lr']:.1e}  |  "
              f"{human_time(elapsed)} 已用 {human_time(run_timer.elapsed)}"
              f" 剩余约 {human_time(remain)}  {stamp_str}")

    print(f"  共 {total_epochs} 轮（阶段一 {args.epochs_stage1} + 阶段二 {args.epochs_stage2}）")
    print("  进度条含义：loss/acc 为该轮截至当前的滑动平均，不是瞬时值\n")

    run_stage_1()
    run_stage_2()

    save_checkpoint(out_dir / "last.pth", model, {
        "epoch": global_epoch, "val_top1": history[-1]["val_top1"],
        "img_size": args.img_size, "args": vars(args),
    })

    # ---------- 测试集最终评估 ----------
    section("[4/4] 测试集最终评估")
    print(f"  加载验证集最优权重（第 {best_epoch} 轮，验证 Top-1 {best_top1:.2f}%）")
    best_model = build_classifier().to(device)
    ckpt = torch.load(out_dir / "best.pth", map_location=device, weights_only=False)
    best_model.load_state_dict(ckpt["state_dict"])

    t = Timer()
    te_loss, te_top1, te_top3 = evaluate(best_model, test_loader, criterion, device,
                                         desc="测试集")
    test_sec = t.elapsed

    print(f"\n  ┌─ 测试集结果 ─────────────────────────────")
    print(f"  │  Top-1 准确率   {te_top1:6.2f}%")
    print(f"  │  Top-3 准确率   {te_top3:6.2f}%")
    print(f"  │  交叉熵损失     {te_loss:.4f}")
    print(f"  │  推理耗时       {test_sec / len(test_set) * 1000:.1f} ms/张"
          f"（{describe_device(device)}，batch {args.batch_size}）")
    print(f"  └──────────────────────────────────────────")

    if not args.smoke_test:
        if te_top1 >= 98.0:
            print("\n  ✅ 达到预期（≥98%）。可以进入对抗攻击实验环节。")
        elif te_top1 >= 95.0:
            print("\n  ⚠️  达到开题报告目标（≥95%）但低于预期。可考虑延长阶段二轮数。")
        else:
            print("\n  ❌ 低于 95%。请检查：是否成功加载了预训练权重、"
                  "数据划分是否正确、学习率是否过大。")

    # ---------- 落盘 ----------
    save_json({"history": history,
               "best_epoch": best_epoch, "best_val_top1": best_top1,
               "test": {"loss": te_loss, "top1": te_top1, "top3": te_top3,
                        "ms_per_image": test_sec / len(test_set) * 1000},
               "total_seconds": run_timer.elapsed},
              out_dir / "history.json")
    save_json({"args": vars(args), "device": str(device),
               "test_top1": te_top1, "test_top3": te_top3,
               "finished_at": datetime.now().isoformat()},
              out_dir / "config.json")

    counts = Counter(train_set.labels())
    names = get_class_names()
    save_json({names[i]: counts.get(i, 0) for i in range(len(names))},
              out_dir / "train_class_counts.json")

    banner("训练完成")
    if args.smoke_test:
        print("⚠️  这是冒烟测试，每类只用了 40 张训练图、总共 2 轮。")
        print("    上面的准确率数字没有意义，本次运行只用于确认代码能跑通。")
        print("    要得到真实结果，请去掉 --smoke-test 重新运行。\n")
    print(f"总耗时：{human_time(run_timer.elapsed)}")
    print(f"测试集：Top-1 {te_top1:.2f}%   Top-3 {te_top3:.2f}%")
    print(f"\n输出文件：")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:<28} {f.stat().st_size / 1024:>9,.1f} KB")
    print(f"\n下一步：把 best.pth 拷到后端模型目录")
    print(f"  cp {out_dir / 'best.pth'} backend/models/classifier_mobilenet.pth")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
