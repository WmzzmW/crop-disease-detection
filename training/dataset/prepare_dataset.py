"""从 data.zip 解压出可用的图像文件，并归置官方划分文件。

【做什么】
    1. 从 HuggingFace 缓存定位 data.zip（无需手动找路径）
    2. 只解压 `raw/color/` 下的 54,305 张彩色图像（约 0.79 GB）
       —— 另外两个版本 grayscale / segmented 暂不解压，需要时改 VARIANTS 再跑
    3. 把官方划分文件与叶片分组元数据复制到项目内，方便后续使用
    4. 校验解压结果（文件数、类别目录数）

【解压后的目录结构】
    data/raw/PlantVillage/raw/color/<类别>/<文件>.JPG     ← 54,305 张图像
    data/splits/color_train.txt                           ← 官方训练集划分
    data/splits/color_test.txt                            ← 官方测试集划分
    data/splits/leaf-map.json                             ← 叶片分组元数据

    注意：解压保持了压缩包内的原始结构，所以路径里有两个 raw
    （外层 data/raw/ 表示"原始数据"，内层是 zip 里的 raw/）。
    划分文件中的相对路径怎么用：
        data/raw/PlantVillage/ + <划分文件里的一行>
    例如 `raw/color/Tomato___healthy/xxx.JPG`
        -> data/raw/PlantVillage/raw/color/Tomato___healthy/xxx.JPG

【用法】
    .venv/bin/python training/dataset/prepare_dataset.py

    脚本可重复运行：已解压且大小一致的文件会自动跳过。
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "mohanty/PlantVillage"
REPO_TYPE = "dataset"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
ZIP_EXTRACT_ROOT = DATA_ROOT / "raw" / "PlantVillage"   # 图像解压到这里
SPLITS_DIR = DATA_ROOT / "splits"                        # 划分与元数据

# 要解压的图像版本；如需灰度/分割版，改成 ["color", "grayscale", "segmented"]
VARIANTS = ["color"]

# 需要复制到项目内的辅助文件：压缩包内路径 -> 项目内目标路径
AUX_FILES = {
    "splits/color_train.txt": SPLITS_DIR / "color_train.txt",
    "splits/color_test.txt": SPLITS_DIR / "color_test.txt",
    "leaf_grouping/leaf-map.json": SPLITS_DIR / "leaf-map.json",
}

PROGRESS_EVERY = 5_000

# 统计图像时按扩展名判断，注意大小写不敏感：
# 数据集里绝大多数是 .JPG，但也有少量 .jpg/.jpeg/.png，漏掉会少数 2 张
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def copy_aux_files() -> None:
    """把划分文件与元数据复制到项目内。"""
    print("[2] 复制官方划分文件")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for repo_path, dest in AUX_FILES.items():
        src = Path(hf_hub_download(REPO_ID, repo_path, repo_type=REPO_TYPE))
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            print(f"   {dest.name:<20} 已存在，跳过")
            continue
        shutil.copy2(src, dest)
        print(f"   {dest.name:<20} {src.stat().st_size / 1024:>8,.0f} KB  ->  {dest}")
    print()


def extract_images(zf: zipfile.ZipFile) -> tuple[int, int]:
    """解压指定版本的图像，返回 (新解压数, 跳过数)。"""
    infos = [
        i for i in zf.infolist()
        if not i.is_dir()
        and any(i.filename.startswith(f"raw/{v}/") for v in VARIANTS)
    ]
    print(f"[1] 解压图像（版本：{', '.join(VARIANTS)}，共 {len(infos):,} 张）")
    print(f"    目标目录：{ZIP_EXTRACT_ROOT}")
    print(f"    预计占用：约 {sum(i.file_size for i in infos) / 1024 ** 3:.2f} GB\n")

    done = skipped = 0
    for info in infos:
        dest = ZIP_EXTRACT_ROOT / info.filename
        if dest.exists() and dest.stat().st_size == info.file_size:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 1024 * 256)
        done += 1
        if done % PROGRESS_EVERY == 0:
            print(f"    已解压 {done:,} 张……", flush=True)

    print(f"\n    新解压 {done:,} 张，跳过已存在 {skipped:,} 张\n")
    return done, skipped


def verify() -> bool:
    """校验解压结果。"""
    print("[3] 校验解压结果")
    ok = True
    for variant in VARIANTS:
        root = ZIP_EXTRACT_ROOT / "raw" / variant
        if not root.exists():
            print(f"   ❌ 目录不存在：{root}")
            ok = False
            continue
        classes = [d for d in root.iterdir() if d.is_dir()]
        images = [p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        exts = Counter(p.suffix.lower() for p in images)
        print(f"   {variant:<12} {len(classes):>3} 个类别目录, {len(images):>7,} 张图像")
        print(f"      扩展名分布：{dict(sorted(exts.items()))}")
        if len(classes) != 38:
            print("      ⚠️  预期 38 个类别目录")
            ok = False
        if variant == "color" and len(images) != 54_305:
            print("      ⚠️  预期 54,305 张图像")
            ok = False
    for dest in AUX_FILES.values():
        if not dest.exists():
            print(f"   ❌ 缺少：{dest}")
            ok = False
    print()
    return ok


def main() -> int:
    print("=" * 72)
    print("准备 PlantVillage 数据集（解压 + 归置）")
    print("=" * 72 + "\n")

    zip_path = Path(hf_hub_download(REPO_ID, "data.zip", repo_type=REPO_TYPE))
    print(f"压缩包：{zip_path}")
    print(f"大小：{zip_path.stat().st_size / 1024 ** 3:.2f} GB\n")

    with zipfile.ZipFile(zip_path) as zf:
        extract_images(zf)

    copy_aux_files()

    if verify():
        total = sum(f.stat().st_size for f in ZIP_EXTRACT_ROOT.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        print(f"✅ 完成。图像共 {total / 1024 ** 3:.2f} GB，位于 {ZIP_EXTRACT_ROOT}")
        print("=" * 72)
        return 0

    print("❌ 校验未通过，请查看上面的提示。")
    print("=" * 72)
    return 1


if __name__ == "__main__":
    sys.exit(main())
