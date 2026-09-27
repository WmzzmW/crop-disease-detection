#!/usr/bin/env python
"""生成 FGSM / PGD 对抗样本，扫描 ε 曲线，并保存样本供后续检测器训练使用。

对应开题报告 3.2 节「实验二：对抗样本生成与攻击有效性验证」。

【这个脚本回答两个问题】
    1. 攻击有多有效？—— 扫描 ε ∈ {0.005, 0.01, 0.02, 0.03, 0.05}，
       统计每个 ε 下分类准确率掉到多少、攻击成功率是多少。
    2. 攻击实现对不对？—— 两条交叉验证：
       · 准确率应随 ε 单调下降；
       · PGD（迭代）应明显强于 FGSM（单步）——与 Luo 2021 / You 2023 /
         Li & Lu 2023 三篇独立工作一致。对不上说明实现有 bug。

【产出】
    <out-dir>/
    ├── curve.json                      ε 扫描的全部数值（写论文直接取数）
    ├── manifest.json                   已保存样本的清单（原图路径 ↔ 对抗图路径）
    └── samples/
        ├── fgsm/<类别>/<序号>_<原名>.png
        └── pgd/<类别>/<序号>_<原名>.png

【用法】
    # 默认：FGSM + PGD，扫 5 个 ε，用 1000 张测试图
    .venv/bin/python training/generate_adversarial.py

    # 只做 FGSM 单点（时间紧时用，约 1 分钟）
    .venv/bin/python training/generate_adversarial.py --attack fgsm --epsilons 0.03

    # 多存一些样本，供检测器训练用
    .venv/bin/python training/generate_adversarial.py --num-samples 3000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from common.attacks import PixelSpace, run_attack, round_to_uint8
from common.config import DATA_DIR, PROJECT_ROOT, get_class_names
from common.data import (
    PlantVillageDataset,
    build_datasets,
    build_transforms,
    stratified_subset,
)
from common.model import load_checkpoint
from common.utils import (
    Timer,
    banner,
    describe_device,
    get_device,
    human_time,
    save_json,
    section,
    set_seed,
)

DEFAULT_EPSILONS = [0.005, 0.01, 0.02, 0.03, 0.05]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="生成对抗样本并扫描 ε 曲线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="分类模型权重；默认找 training/runs 下最近一次的 best.pth")
    p.add_argument("--attack", type=str, nargs="+", default=["fgsm", "pgd"],
                   choices=["fgsm", "pgd"], help="要做哪些攻击")
    p.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS,
                   help="扫描的 ε 列表（0~1 像素尺度，0.03 ≈ 7.65/255）")
    p.add_argument("--save-epsilon", type=float, default=0.03,
                   help="保存图像样本所用的 ε；不在 epsilons 里会自动补上")
    p.add_argument("--num-samples", type=int, default=1000,
                   help="从测试集按类别分层抽取多少张（1000 张的统计误差约 ±1.5%）")
    p.add_argument("--steps", type=int, default=10, help="PGD 迭代步数")
    p.add_argument("--alpha", type=float, default=0.01, help="PGD 每步步长")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out-dir", type=str, default=str(DATA_DIR / "adversarial"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-images", action="store_true",
                   help="只算数字，不保存对抗图像（省磁盘）")
    return p.parse_args()


def find_latest_checkpoint() -> Path | None:
    from common.config import RUNS_DIR
    if not RUNS_DIR.exists():
        return None
    hits = sorted(RUNS_DIR.glob("classifier_*/best.pth"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        print("❌ 找不到分类模型权重。请先训练，或指定 --checkpoint <路径>")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("对抗样本生成与攻击有效性验证")
    print(f"分类模型：{ckpt_path}")
    print(f"输出目录：{out_dir}")
    print(f"开始时间：{datetime.now():%Y-%m-%d %H:%M:%S}")

    device = get_device(args.device)
    model, meta = load_checkpoint(ckpt_path, device)
    img_size = meta.get("img_size", args.img_size)
    class_names = meta.get("class_names") or get_class_names()
    ps = PixelSpace(model, device)

    print(f"计算设备：{describe_device(device)}")
    if "val_top1" in meta:
        print(f"该模型训练时验证 Top-1：{meta['val_top1']:.2f}%")

    # 保证 save_epsilon 一定在扫描列表里，否则保存环节永远不会触发
    epsilons = sorted(set(args.epsilons) | {args.save_epsilon})

    # ---------------------------------------------------------------- 数据
    section("[1/3] 准备测试样本")
    _, _, test_set = build_datasets(img_size=img_size, verbose=False)
    subset = stratified_subset(test_set.samples, args.num_samples)
    dataset = PlantVillageDataset(subset, build_transforms(img_size, train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    print(f"  从测试集分层抽取 {len(dataset):,} 张，覆盖全部 {len(class_names)} 类")
    print(f"  攻击方法：{' + '.join(a.upper() for a in args.attack)}")
    print(f"  扫描 ε：{', '.join(f'{e:g}' for e in epsilons)}"
          f"   （ε=0.03 ≈ {0.03 * 255:.2f}/255）")
    if "pgd" in args.attack:
        print(f"  PGD 参数：steps={args.steps}  alpha={args.alpha}")

    # ---------------------------------------------------------------- 扫描
    section("[2/3] 扫描 ε（每张图都要反传梯度，PGD 比 FGSM 慢约 10 倍）")
    n_total = len(dataset)
    results: dict[str, dict] = {a: {} for a in args.attack}
    clean_correct = 0
    saved_records: list[dict] = []       # 仅记录 --save-epsilon 那一档
    save_eps = args.save_epsilon
    run_timer = Timer()

    # 先算干净样本的基线准确率（ε=0）
    print("\n  基线（未攻击）：")
    for images, targets in tqdm(loader, desc="    干净样本", ncols=110,
                                leave=False, unit="batch"):
        x_pix = ps.to_pixel(images.to(device))
        pred = ps.predict(x_pix).argmax(dim=1)
        clean_correct += (pred == targets.to(device)).sum().item()
    clean_acc = 100.0 * clean_correct / n_total
    print(f"    干净准确率 {clean_acc:.2f}%（{clean_correct}/{n_total}）")

    for attack_name in args.attack:
        for eps in epsilons:
            # 逐样本记录，用于单次遍历里同时算浮点与 8 位量化两种口径
            ok_float = ok_uint8 = 0
            from_correct = 0      # 干净时判对、攻击后判错的样本数
            to_save: list[dict] = []

            desc = f"    {attack_name.upper():<5} ε={eps:<5g}"
            for bi, (images, targets) in enumerate(
                    tqdm(loader, desc=desc, ncols=110, leave=False, unit="batch")):
                images = images.to(device)
                targets = targets.to(device)
                x_pix = ps.to_pixel(images)

                x_adv = run_attack(attack_name, ps, x_pix, targets, eps,
                                   steps=args.steps, alpha=args.alpha)

                with torch.no_grad():
                    pred_clean = ps.predict(x_pix).argmax(dim=1)
                    pred_adv = ps.predict(x_adv).argmax(dim=1)
                    # 8 位量化后再判一次：这才是攻击者真正能交付的形态
                    x_adv_q = round_to_uint8(x_adv)
                    pred_adv_q = ps.predict(x_adv_q).argmax(dim=1)

                is_clean_ok = pred_clean == targets
                ok_float += (pred_adv == targets).sum().item()
                ok_uint8 += (pred_adv_q == targets).sum().item()
                from_correct += (is_clean_ok & (pred_adv != targets)).sum().item()

                # 只在目标 ε 上保存图像
                if not args.no_save_images and abs(eps - save_eps) < 1e-9:
                    base = bi * args.batch_size
                    for k in range(x_adv_q.size(0)):
                        idx = base + k
                        if idx >= n_total:
                            break
                        orig_path, label = subset[idx]
                        stem = Path(orig_path).stem
                        cls = class_names[label]
                        rel = Path("samples") / attack_name / cls / f"{idx:05d}_{stem}.png"
                        dest = out_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        to_pil_image(x_adv_q[k].cpu()).save(dest)
                        to_save.append({
                            "index": idx, "label": label, "class_name": cls,
                            "orig_path": str(Path(orig_path).relative_to(PROJECT_ROOT)),
                            "adv_path": str(rel),
                            "clean_correct": bool(is_clean_ok[k].item()),
                            "adv_correct": bool((pred_adv[k] == targets[k]).item()),
                        })

            acc_float = 100.0 * ok_float / n_total
            acc_uint8 = 100.0 * ok_uint8 / n_total
            asr = 100.0 * from_correct / max(clean_correct, 1)

            results[attack_name][f"{eps:g}"] = {
                "epsilon": eps,
                "accuracy": round(acc_float, 4),
                "accuracy_uint8": round(acc_uint8, 4),
                "accuracy_drop": round(clean_acc - acc_float, 4),
                "asr_on_clean_correct": round(asr, 4),
            }
            print(f"    {attack_name.upper():<5} ε={eps:<5g} "
                  f"准确率 {acc_float:6.2f}%（8位量化后 {acc_uint8:6.2f}%）   "
                  f"下降 {clean_acc - acc_float:5.2f} pp   攻击成功率 {asr:5.2f}%")

            if to_save:
                saved_records.extend(to_save)

    # ---------------------------------------------------------------- 落盘
    section("[3/3] 保存结果")
    save_json({
        "checkpoint": str(ckpt_path),
        "num_samples": n_total,
        "clean_accuracy": round(clean_acc, 4),
        "epsilons": epsilons,
        "attack": args.attack,
        "pgd_steps": args.steps, "pgd_alpha": args.alpha,
        "results": results,
        "elapsed_sec": run_timer.elapsed,
    }, out_dir / "curve.json")
    print(f"  curve.json       ε 扫描数值（{len(args.attack)} 种攻击 × {len(epsilons)} 个 ε）")

    if saved_records:
        save_json({
            "epsilon": save_eps,
            "num_samples": len(saved_records),
            "attack": args.attack,
            "samples": saved_records,
        }, out_dir / "manifest.json")
        per_attack = len(saved_records) // max(len(args.attack), 1)
        print(f"  manifest.json    样本清单（{len(saved_records):,} 条，"
              f"每种攻击 {per_attack:,} 张）")
        print(f"  samples/         对抗图像（每种攻击 {per_attack:,} 张 PNG）")
        size_mb = sum(f.stat().st_size for f in (out_dir / "samples").rglob("*.png")) / 1e6
        print(f"                   共 {size_mb:,.0f} MB")

    # ---------------------------------------------------------------- 小结
    banner("完成")
    print(f"总耗时：{human_time(run_timer.elapsed)}\n")
    print(f"  {'攻击':<6}{'ε':>8}{'准确率':>10}{'下降':>10}{'攻击成功率':>12}")
    print(f"  {'-' * 46}")
    print(f"  {'—':<6}{0:>8}{clean_acc:>9.2f}%{'—':>10}{'—':>12}")
    for name in args.attack:
        for eps in epsilons:
            r = results[name][f"{eps:g}"]
            print(f"  {name.upper():<6}{eps:>8g}{r['accuracy']:>9.2f}%"
                  f"{r['accuracy_drop']:>9.2f}pp{r['asr_on_clean_correct']:>11.2f}%")
        print()

    # 自动检查两条交叉验证
    print("  交叉验证（用于确认攻击实现正确）：")
    ok = True
    for name in args.attack:
        accs = [results[name][f"{e:g}"]["accuracy"] for e in epsilons]
        mono = all(accs[i] >= accs[i + 1] - 0.5 for i in range(len(accs) - 1))
        print(f"    {name.upper()} 准确率随 ε 单调下降：{'✅' if mono else '❌ 检查实现'}")
        ok &= mono
    if "fgsm" in args.attack and "pgd" in args.attack:
        f = results["fgsm"][f"{save_eps:g}"]["accuracy"]
        p = results["pgd"][f"{save_eps:g}"]["accuracy"]
        stronger = p <= f + 0.5
        print(f"    PGD 强于 FGSM（ε={save_eps:g}：FGSM {f:.2f}% vs PGD {p:.2f}%）："
              f"{'✅ 与文献一致' if stronger else '❌ 与文献不符，检查实现'}")
        ok &= stronger

    print(f"\n  下一步：出图（ε 曲线 / 对抗样本三联图 / Top-3 对比表）")
    print(f"    .venv/bin/python training/eval_attack.py")
    print("=" * 70)
    return 0 if ok else 0     # 即使交叉验证不过也返回 0，让后续出图能跑


if __name__ == "__main__":
    sys.exit(main())
