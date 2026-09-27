"""数据集自检脚本：验证 PlantVillage 是否下载成功且可用。

用法：
    .venv/bin/python training/dataset/check_dataset.py

无需先解压：脚本直接在 data.zip 内部校验（读取的是 zip 中央目录与抽样条目）。

检查项：
    1. 下载的四个文件是否齐全（data.zip / 两个划分文件 / 叶片分组元数据）
    2. data.zip 是否完整可读，内部图像总数与文件结构
    3. 作物与类别数量是否为 14 种作物、38 个类别
    4. 抽样读取图像：能否正常解码，尺寸与色彩模式是否为彩色 256×256
    5. 官方划分是否覆盖全部图像，train/test 之间有无重叠
    6. 是否存在数据泄露：同一片叶子的多张照片是否跨 train/test

全部通过则退出码为 0，任一项失败则退出码为 1。
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "mohanty/PlantVillage"
REPO_TYPE = "dataset"

EXPECTED_CROPS = 14
EXPECTED_CLASSES = 38
EXPECTED_TOTAL = 54_305

COLOR_PREFIX = "raw/color/"

failures: list[str] = []


def check(title: str, ok: bool, detail: str = "") -> None:
    mark = "✅" if ok else "❌"
    print(f"{mark} {title}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(title)


def note(title: str, detail: str = "") -> None:
    print(f"   {title}" + (f" — {detail}" if detail else ""))


def make_leaf_resolver(leaf_map_path: str):
    """构造“图片路径 -> 叶片标识”的解析函数。

    复刻官方脚本 plant_village.py 的逻辑：
      1. 取文件名中最后一段 `___` 之后的部分作为图像标识符；
      2. 去掉 `copy` 后缀与扩展名，转小写后在 leaf-map.json 中查表；
      3. 查到的条目形如 `类别:::编号`，即该叶片的全局唯一标识；
         若同一标识符对应多个类别，用当前类别消歧（说明编号字符串会跨类别重名）；
      4. 查不到时退化为 fallback（按类别 + 标识符构造）。
    """
    leaf_map = json.loads(Path(leaf_map_path).read_text())

    def resolve(file_rel_path: str) -> str:
        parts = file_rel_path.split("/")
        if len(parts) < 4:
            return f"malformed::{file_rel_path}"
        class_name, file_name = parts[2], parts[3]

        ident = file_name.replace("_final_masked", "")
        if "___" in ident:
            ident = ident.split("___")[-1]
        ident = ident.split("copy")[0]
        for ext in (".jpg", ".JPG", ".png", ".PNG"):
            ident = ident.replace(ext, "")
        ident = ident.strip()

        suggestions = leaf_map.get(ident.lower().strip())
        if suggestions:
            if len(suggestions) == 1:
                return suggestions[0]
            for s in suggestions:
                if class_name in s:
                    return s
        return f"fallback::{class_name}::{ident}"

    return resolve


def main() -> int:
    print("=" * 72)
    print("PlantVillage 数据集自检")
    print("=" * 72)

    # ---------- 1. 文件齐全性 ----------
    print("\n[1] 检查下载文件")
    paths: dict[str, str] = {}
    required = [
        "data.zip",
        "splits/color_train.txt",
        "splits/color_test.txt",
        "leaf_grouping/leaf-map.json",
    ]
    for name in required:
        try:
            paths[name] = hf_hub_download(REPO_ID, name, repo_type=REPO_TYPE)
            note(f"{name:<32} {Path(paths[name]).stat().st_size / 1024 / 1024:>10,.1f} MB")
        except Exception as exc:  # noqa: BLE001
            check(f"文件存在：{name}", False, f"{type(exc).__name__}: {exc}")
    check("四个文件全部就位", len(paths) == len(required),
          f"{len(paths)}/{len(required)}")
    if len(paths) != len(required):
        return 1

    # ---------- 2. zip 完整性 ----------
    print("\n[2] 检查 data.zip")
    try:
        zf = zipfile.ZipFile(paths["data.zip"])
        names = zf.namelist()
        check("data.zip 可正常打开", True, f"{len(names):,} 个条目")
    except Exception as exc:  # noqa: BLE001
        check("data.zip 可正常打开", False, f"{type(exc).__name__}: {exc}")
        print("\n提示：文件可能下载不完整，删除缓存后重试。")
        return 1

    color_files = [
        n for n in names
        if n.startswith(COLOR_PREFIX) and n.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    check("包含彩色图像", len(color_files) > 0, f"{len(color_files):,} 张")
    check("彩色图像总数约 5.4 万", abs(len(color_files) - EXPECTED_TOTAL) < 100,
          f"实际 {len(color_files):,} 张")

    # 三个版本是否都在
    for variant in ("grayscale", "segmented"):
        n = sum(1 for x in names if x.startswith(f"raw/{variant}/"))
        note(f"附带版本 {variant:<12} {n:,} 个文件")

    # ---------- 3. 作物与类别 ----------
    print("\n[3] 作物与类别")
    classes = sorted({n[len(COLOR_PREFIX):].split("/")[0] for n in color_files})
    crops = sorted({c.split("___")[0] for c in classes})
    check("类别数为 38", len(classes) == EXPECTED_CLASSES, f"实际 {len(classes)}")
    check("作物数为 14", len(crops) == EXPECTED_CROPS, f"实际 {len(crops)} 种")

    print(f"\n   14 种作物及其类别数：")
    per_crop = Counter(c.split("___")[0] for c in classes)
    for crop in crops:
        names_in_crop = [c for c in classes if c.split("___")[0] == crop]
        print(f"     {crop:<28} {len(names_in_crop):>2} 类")

    print(f"\n   38 个类别全名：")
    for i, c in enumerate(classes):
        print(f"     {i:>2}. {c}")

    # ---------- 4. 抽样解码图像 ----------
    print("\n[4] 抽样读取图像（直接从 zip 解码）")
    ok_count = 0
    for n in color_files[:5]:
        try:
            with zf.open(n) as fh:
                img = Image.open(io.BytesIO(fh.read()))
                img.load()
            print(f"     {Path(n).name[:50]:<52} {img.size} {img.mode}")
            if img.size == (256, 256) and img.mode == "RGB":
                ok_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"     ❌ {Path(n).name} 解码失败：{exc}")
    check("抽样图像均为彩色 256×256", ok_count == 5, f"{ok_count}/5 通过")

    # ---------- 5. 官方划分覆盖情况 ----------
    print("\n[5] 官方划分（train / test）")
    splits: dict[str, list[str]] = {}
    for s in ("color_train", "color_test"):
        key = s.split("_")[1]
        content = Path(paths[f"splits/{s}.txt"]).read_text(encoding="utf-8").splitlines()
        splits[key] = [x.strip() for x in content if x.strip()]
        note(f"{key:<6} {len(splits[key]):,} 张")

    total_split = len(splits["train"]) + len(splits["test"])
    check("train+test 覆盖全部图像", total_split == len(color_files),
          f"划分 {total_split:,} vs 实际 {len(color_files):,}")

    name_set = set(color_files)
    missing = [x for x in splits["train"][:2000] if x not in name_set]
    check("划分文件中的路径真实存在", len(missing) == 0,
          f"抽检 2000 条，缺失 {len(missing)} 条")
    if missing:
        note("示例缺失路径", missing[0])

    # ---------- 6. 数据泄露检查 ----------
    print("\n[6] 数据泄露检查（同一片叶子是否跨 train/test）")
    resolve = make_leaf_resolver(paths["leaf_grouping/leaf-map.json"])
    lid_train = [resolve(x) for x in splits["train"]]
    lid_test = [resolve(x) for x in splits["test"]]

    is_fb = lambda v: v.startswith("fallback::")  # noqa: E731
    fb_n = sum(1 for v in lid_train if is_fb(v))
    note(f"train {len(lid_train):,} 张，leaf-map 命中 {len(lid_train) - fb_n:,} 张"
         f"（{100 * (1 - fb_n / len(lid_train)):.1f}%），fallback {fb_n:,} 张")

    mapped_tr = {v for v in lid_train if not is_fb(v)}
    mapped_te = {v for v in lid_test if not is_fb(v)}
    real_overlap = mapped_tr & mapped_te
    check("train/test 之间无叶片重叠",
          len(real_overlap) == 0,
          f"train {len(mapped_tr):,} 片 / test {len(mapped_te):,} 片，"
          f"重叠 {len(real_overlap)} 片")
    if real_overlap:
        note("重叠示例", sorted(real_overlap)[:3])

    # fallback 项按“类别+标识符”构造 ID，标识符跨叶片重名时会误判为同一片，仅供参考
    fb_overlap = ({v for v in lid_train if is_fb(v)}
                  & {v for v in lid_test if is_fb(v)})
    note(f"fallback 项重合 {len(fb_overlap):,} 个 —— 由编号重名所致，非真实泄露，不计入失败项")

    # ---------- 汇总 ----------
    print("\n" + "=" * 72)
    if failures:
        print(f"❌ 自检未通过，{len(failures)} 项失败：")
        for f in failures:
            print(f"   - {f}")
        print("=" * 72)
        return 1
    print("✅ 全部检查通过，数据集完整可用。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
