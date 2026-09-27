"""下载 PlantVillage 数据集。

【为什么不直接用 load_dataset】
    HF 上的 mohanty/PlantVillage 是"脚本型数据集"（plant_village.py），
    datasets 5.x 已不支持加载脚本，只能用它自动转换出来的 parquet 版本。
    而那个 parquet 只有 7 MB、仅包含图片路径字符串，不含真实图像数据
    （真正的图像包 data.zip 有 2 GB），加载后取不到 label / image 字段。
    因此改为直接下载仓库中的原始文件。

【下载内容】
    data.zip                      约 2.0 GB，全部图像（三个版本共用一个包）
    splits/color_train.txt        官方训练集文件列表（按叶片分组，无泄露）
    splits/color_test.txt         官方测试集文件列表
    leaf_grouping/leaf-map.json   叶片分组元数据（用于自定义划分时防泄露）

【文件落点】
    HuggingFace 缓存，默认 ~/.cache/huggingface/ 下，不在项目目录里。
    如需落到项目内，先设置：export HF_HOME=<项目>/data/cache

【用法】
    .venv/bin/python training/dataset/download_dataset.py
"""

from huggingface_hub import hf_hub_download

REPO_ID = "mohanty/PlantVillage"
REPO_TYPE = "dataset"

FILES = [
    "data.zip",                      # ~2.0 GB，图像主体
    "splits/color_train.txt",        # 官方彩色版训练集划分
    "splits/color_test.txt",         # 官方彩色版测试集划分
    "leaf_grouping/leaf-map.json",   # 叶片分组（防数据泄露）
]


def main() -> None:
    print(f"从 {REPO_ID} 下载 {len(FILES)} 个文件……\n")
    for name in FILES:
        print(f"[下载] {name}", flush=True)
        path = hf_hub_download(REPO_ID, name, repo_type=REPO_TYPE)
        print(f"       -> {path}\n", flush=True)
    print("全部下载完成。")


if __name__ == "__main__":
    main()
