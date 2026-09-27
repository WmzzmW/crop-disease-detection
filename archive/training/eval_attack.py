#!/usr/bin/env python
"""对抗攻击效果可视化：出三张图 + 一份数值汇总。

消费 generate_adversarial.py 的产出（curve.json / manifest.json / samples/），
产出开题 PPT 最需要的三张图：

    figures/accuracy_vs_epsilon.png    准确率-ε 曲线 + 攻击成功率-ε 曲线
    figures/adversarial_examples.png   对抗样本三联图（原图 / 对抗图 / 扰动×10）
    figures/prediction_comparison.png  攻击前后 Top-3 预测对比表

【用法】
    .venv/bin/python training/eval_attack.py

    # 指定要看哪种攻击的样本（默认 pgd，因为它更强、更有说服力）
    .venv/bin/python training/eval_attack.py --attack fgsm

    # 三联图多放几行
    .venv/bin/python training/eval_attack.py --num-examples 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image
from torchvision import transforms

from common.attacks import PixelSpace
from common.config import IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT, get_class_names
from common.model import build_classifier, load_checkpoint
from common.utils import (
    banner,
    describe_device,
    get_device,
    save_json,
    section,
)

CJK_CANDIDATES = ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC",
                  "Arial Unicode MS", "Hiragino Sans GB", "Microsoft YaHei"]


def setup_font() -> bool:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


HAS_CJK = setup_font()
L_ACC = "分类准确率 (%)" if HAS_CJK else "Accuracy (%)"
L_ASR = "攻击成功率 (%)" if HAS_CJK else "Attack success rate (%)"
L_EPS = "扰动预算 ε" if HAS_CJK else "Perturbation budget ε"


def short_name(name: str, width: int = 30) -> str:
    """把 'Tomato___Tomato_Yellow_Leaf_Curl_Virus' 压成 'Tomato/Y.Tomato_Yellow_Leaf_Curl_Virus'"""
    s = name.replace("___", "/").replace("_", " ")
    return s if len(s) <= width else s[:width - 1] + "…"


# ================================================================ 图一：ε 曲线


def plot_epsilon_curve(curve: dict, out_path: Path) -> None:
    eps = [float(e) for e in curve["epsilons"]]
    attacks = curve["attack"]
    clean = curve["clean_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    markers = {"fgsm": "o", "pgd": "s"}
    colors = {"fgsm": "#d95f02", "pgd": "#1b6ca8"}

    for ax, key, ylabel, title in (
        (axes[0], "accuracy", L_ACC, "准确率随 ε 的变化" if HAS_CJK else "Accuracy vs ε"),
        (axes[1], "asr_on_clean_correct", L_ASR,
         "攻击成功率随 ε 的变化" if HAS_CJK else "Attack success rate vs ε"),
    ):
        for a in attacks:
            ys = [curve["results"][a][f"{e:g}"][key] for e in eps]
            ax.plot(eps, ys, marker=markers.get(a, "o"), lw=2.2, ms=7,
                    color=colors.get(a, None), label=a.upper())
        if key == "accuracy":
            # 干净基线：模型未被攻击时的准确率，作为对照横线
            ax.axhline(clean, ls="--", lw=1.6, color="gray",
                       label=f"未攻击 {clean:.2f}%" if HAS_CJK
                       else f"clean {clean:.2f}%")
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(0, 105)
        ax.set_xlabel(L_EPS, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend()

    # 标注最关键的那个结论：在常用的 ε=0.03 上掉了多少
    if any(abs(e - 0.03) < 1e-9 for e in eps):
        for a in attacks:
            r = curve["results"][a]["0.03"]
            axes[0].annotate(f"−{r['accuracy_drop']:.1f}pp",
                             xy=(0.03, r["accuracy"]), xytext=(0.031, r["accuracy"] + 14),
                             fontsize=10, color=colors.get(a, "black"), fontweight="bold")

    fig.suptitle(f"对抗攻击有效性（测试集分层抽样 {curve['num_samples']:,} 张）"
                 if HAS_CJK else
                 f"Attack effectiveness (n={curve['num_samples']:,})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图二：三联图


def original_transform(img_size: int):
    """原始图像（256×256 原图）-> 模型输入的裁剪。

    必须和 data.py 里评估用的变换完全一致，否则原图和对抗图不是同一个取景框。
    """
    return transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


# 对抗样本在生成时已经过同一套裁剪并存成 224×224，读回来绝不能再裁剪一次：
# 对 224×224 再做 Resize(255)+CenterCrop(224) 相当于重新缩放，
# 算出来的"扰动"会变成重采样误差（可达 130/255），完全掩盖真实的 7.6/255。
ALREADY_CROPPED = transforms.Compose([transforms.ToTensor()])


@torch.no_grad()
def load_and_predict(path: Path, model, ps: PixelSpace, device, tf) -> tuple[torch.Tensor, np.ndarray]:
    """读图 -> 指定变换 -> 返回 (像素空间张量, softmax 概率)。"""
    img = Image.open(path).convert("RGB")
    x_pix = tf(img).unsqueeze(0).to(device)
    probs = torch.softmax(ps.predict(x_pix), dim=1)[0].cpu().numpy()
    return x_pix, probs


def plot_adversarial_grid(records: list[dict], model, ps: PixelSpace, device,
                          class_names: list[str], img_size: int,
                          attack: str, eps: float, out_path: Path,
                          data_dir: Path, num: int = 5) -> list[dict]:
    """三联图：原图 / 对抗图 / 扰动×10 放大。

    优先挑「原本判对、攻击后判错」的样本——它们最能说明问题。
    """
    succeeded = [r for r in records if r["clean_correct"] and not r["adv_correct"]]
    pool = succeeded or records

    # 每个类别只取一张，且**跨类别均匀铺开**。
    # 直接切前 N 条会拿到同一个类别的 N 张（manifest 按类别排序），
    # 图上像同一片叶子重复了四遍；只按排序取前 N 个类别又会全落在
    # 字母序最前面的同一作物上（Apple 的四个类别）。均匀跳着取才好看。
    by_class: dict[str, list] = {}
    for r in pool:
        by_class.setdefault(r["class_name"], []).append(r)

    keys = sorted(by_class)
    if len(keys) > num > 1:
        keys = [keys[round(i * (len(keys) - 1) / (num - 1))] for i in range(num)]
    chosen = [by_class[k][0] for k in keys[:num]]

    fig, axes = plt.subplots(len(chosen), 3, figsize=(11, 3.5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes.reshape(1, -1)

    details = []
    for row, rec in enumerate(chosen):
        orig_path = PROJECT_ROOT / rec["orig_path"]
        # adv_path 是相对 out_dir 的，不是相对图表目录的
        adv_path = data_dir / rec["adv_path"]

        x_orig, p_orig = load_and_predict(orig_path, model, ps, device,
                                          original_transform(img_size))
        x_adv, p_adv = load_and_predict(adv_path, model, ps, device, ALREADY_CROPPED)

        o_np = x_orig[0].cpu().numpy().transpose(1, 2, 0)
        a_np = x_adv[0].cpu().numpy().transpose(1, 2, 0)
        delta = a_np - o_np

        top_o = np.argsort(-p_orig)[:3]
        top_a = np.argsort(-p_adv)[:3]
        true_name = rec["class_name"]

        # 原图：判对显示绿色标题，判错红色
        ok_o = top_o[0] == rec["label"]
        axes[row, 0].imshow(np.clip(o_np, 0, 1))
        axes[row, 0].set_title(
            f"原图\n{short_name(class_names[top_o[0]], 26)}\n{p_orig[top_o[0]]:.1%}",
            fontsize=9, color="green" if ok_o else "red")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(np.clip(a_np, 0, 1))
        axes[row, 1].set_title(
            f"对抗图 ({attack.upper()}, ε={eps:g})\n"
            f"{short_name(class_names[top_a[0]], 26)}\n{p_adv[top_a[0]]:.1%}",
            fontsize=9, color="red")
        axes[row, 1].axis("off")

        # 扰动放大 ×10 后居中显示：0.5 是零扰动，偏离 0.5 说明该像素被改动了
        amp = np.clip(delta * 10 + 0.5, 0, 1)
        axes[row, 2].imshow(amp)
        axes[row, 2].set_title(
            f"扰动 ×10 放大\nmax|δ|={np.abs(delta).max() * 255:.1f}/255",
            fontsize=9)
        axes[row, 2].axis("off")

        details.append({
            "true_class": true_name,
            "clean_top3": [{"class": class_names[i], "prob": float(p_orig[i])}
                           for i in top_o],
            "adv_top3": [{"class": class_names[i], "prob": float(p_adv[i])}
                         for i in top_a],
            "attack_succeeded": bool(top_a[0] != rec["label"]),
            "max_perturbation_255": float(np.abs(delta).max() * 255),
        })

    fig.suptitle(f"对抗样本可视化：扰动肉眼不可见，模型却判错（{attack.upper()}，ε={eps:g}）"
                 if HAS_CJK else
                 f"Adversarial examples ({attack.upper()}, eps={eps:g})",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return details


# ================================================================ 图三：Top-3 对比表


def plot_prediction_table(details: list[dict], out_path: Path, attack: str,
                          eps: float) -> None:
    """渲染「攻击前后 Top-3 预测对比」表。

    这张表要传达的是：模型不是"变得不确定了"，而是**自信地换了一个错误答案**。
    所以每格都带概率——看到对抗图的 Top-1 是 87% 的错类，比只看类别名有说服力得多。
    """
    n = len(details)
    col_labels = ["真实类别", "原图 Top-3", "对抗图 Top-3", "结果"]
    cell_text, row_colors = [], []

    for d in details:
        def fmt(top3):
            return "\n".join(f"{short_name(t['class'], 28):<30}{t['prob']:>6.1%}"
                             for t in top3)
        cell_text.append([
            short_name(d["true_class"], 28),
            fmt(d["clean_top3"]),
            fmt(d["adv_top3"]),
            # 不要用 ✓/✗：PingFang 等中文字体没有这两个字形，会渲染成豆腐块
            "已骗过" if d["attack_succeeded"] else "未骗过",
        ])
        row_colors.append("#ffe5e5" if d["attack_succeeded"] else "#e8f5e9")

    fig, ax = plt.subplots(figsize=(17, 1.4 + 1.05 * n))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="left", loc="center", colWidths=[0.19, 0.31, 0.31, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 3.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(row_colors[r - 1])
            if c == 3:
                cell.set_text_props(fontweight="bold",
                                    color="#c62828" if "骗过" in cell.get_text().get_text()
                                    else "#2e7d32")

    ax.set_title(f"攻击前后 Top-3 预测对比（{attack.upper()}，ε={eps:g}）"
                 f" —— 模型不是变得不确定，而是自信地判错"
                 if HAS_CJK else
                 f"Top-3 predictions before/after attack ({attack.upper()}, eps={eps:g})",
                 fontsize=13, pad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 主流程


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对抗攻击效果可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", type=str, default="data/adversarial",
                   help="generate_adversarial.py 的输出目录")
    p.add_argument("--attack", type=str, default="pgd", choices=["fgsm", "pgd"],
                   help="用哪种攻击的样本做可视化")
    p.add_argument("--num-examples", type=int, default=5,
                   help="三联图的行数 / 对比表的行数")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    curve_path = out_dir / "curve.json"
    manifest_path = out_dir / "manifest.json"
    if not curve_path.exists():
        print(f"❌ 找不到 {curve_path}")
        print("   请先运行：.venv/bin/python training/generate_adversarial.py")
        return 1

    banner("对抗攻击效果可视化")
    if not HAS_CJK:
        print("提示：未找到中文字体，图表标签自动改用英文")

    device = get_device(args.device)
    curve = json.loads(curve_path.read_text(encoding="utf-8"))

    ckpt_path = Path(curve["checkpoint"])
    model, meta = load_checkpoint(ckpt_path, device)
    img_size = meta.get("img_size", args.img_size)
    class_names = meta.get("class_names") or get_class_names()
    ps = PixelSpace(model, device)
    print(f"模型：{ckpt_path.name}    设备：{describe_device(device)}")

    # ---------- 图一 ----------
    section("[1/3] 准确率-ε 曲线")
    plot_epsilon_curve(curve, fig_dir / "accuracy_vs_epsilon.png")
    eps_csv = ", ".join(f"{e:g}" for e in curve["epsilons"])
    print(f"  已保存 figures/accuracy_vs_epsilon.png")
    print(f"  未攻击准确率 {curve['clean_accuracy']:.2f}%；扫描 ε = {eps_csv}")
    for a in curve["attack"]:
        accs = [curve["results"][a][f"{e:g}"]["accuracy"] for e in curve["epsilons"]]
        print(f"    {a.upper():<5} 准确率 {accs[0]:.2f}% → {accs[-1]:.2f}%"
              f"（ε 从 {curve['epsilons'][0]:g} 到 {curve['epsilons'][-1]:g}）")

    # ---------- 图二、图三 ----------
    details = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        attack = args.attack
        eps = manifest["epsilon"]
        records = [r for r in manifest["samples"]
                   if f"/{attack}/" in r["adv_path"]]

        if not records:
            print(f"\n  ⚠️  manifest 里没有 {attack.upper()} 的样本，跳过可视化")
        else:
            section(f"[2/3] 对抗样本三联图（{attack.upper()}，ε={eps:g}）")
            details = plot_adversarial_grid(
                records, model, ps, device, class_names, img_size,
                attack, eps, fig_dir / "adversarial_examples.png",
                data_dir=out_dir, num=args.num_examples)
            print("  已保存 figures/adversarial_examples.png")
            print(f"  扰动幅度 max|δ| 约 "
                  f"{np.mean([d['max_perturbation_255'] for d in details]):.1f}/255"
                  f"（ε={eps:g} ≈ {eps * 255:.1f}/255）")

            section("[3/3] 攻击前后 Top-3 预测对比")
            plot_prediction_table(details, fig_dir / "prediction_comparison.png",
                                  attack, eps)
            print("  已保存 figures/prediction_comparison.png\n")
            print(f"  {'真实类别':<32}{'原图 Top-1':<34}{'对抗图 Top-1'}")
            print(f"  {'-' * 100}")
            for d in details:
                c1, c2 = d["clean_top3"][0], d["adv_top3"][0]
                t1 = f"{short_name(c1['class'], 26)} ({c1['prob']:.1%})"
                t2 = f"{short_name(c2['class'], 26)} ({c2['prob']:.1%})"
                print(f"  {short_name(d['true_class'], 30):<32}{t1:<34}{t2}")

            save_json({"attack": attack, "epsilon": eps, "examples": details},
                      out_dir / "visual_report.json")
    else:
        print("\n  ⚠️  没有 manifest.json（可能生成时用了 --no-save-images），"
              "跳过三联图与对比表")

    banner("完成")
    print(f"图表目录：{fig_dir}")
    for f in sorted(fig_dir.iterdir()):
        print(f"  {f.name:<34} {f.stat().st_size / 1024:>8,.0f} KB")
    print("\n  这三张图可直接放进开题 PPT。推荐用法：")
    print("    · adversarial_examples.png  放在「问题：对抗样本是什么」一页，给足版面")
    print("    · accuracy_vs_epsilon.png   放在「实验方案」或「前期进展」一页")
    print("    · prediction_comparison.png 放在讲「错得很自信」的地方")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
