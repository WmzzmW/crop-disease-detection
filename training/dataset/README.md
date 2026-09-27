# dataset —— 数据集获取与自检

三个一次性工具脚本。数据已就绪，**正常情况下不需要重跑**；换机器或数据损坏时才用得上。

---

## 三个脚本的分工

按执行顺序：

| 顺序 | 脚本 | 干什么 | 依赖上一个吗 |
|---|---|---|---|
| 1 | `download_dataset.py` | 从 HuggingFace 下载 4 个原始文件 | — |
| 2 | `prepare_dataset.py` | 解压 54,305 张图像 + 归置官方划分文件 | 是 |
| 3 | `check_dataset.py` | 完整性自检 + 数据泄露检查 | 否（直接查 zip） |

### 1. `download_dataset.py` —— 下载

下载 4 个文件到 HuggingFace 缓存（默认 `~/.cache/huggingface/`，**不在项目目录**）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `data.zip` | ~2.0 GB | 全部图像（彩色/灰度/分割三个版本共用一个包） |
| `splits/color_train.txt` | 4.0 MB | 官方训练集划分（43,596 张） |
| `splits/color_test.txt` | 1.0 MB | 官方测试集划分（10,709 张） |
| `leaf_grouping/leaf-map.json` | 2.4 MB | 叶片分组元数据（**防数据泄露的关键**） |

**为什么不直接用 `load_dataset`**：HF 上的 `mohanty/PlantVillage` 是"脚本型数据集"，`datasets` 5.x 已不支持加载脚本；它自动转换出的 parquet 版本只有 7 MB、仅含路径字符串，取不到真实图像。所以改为直接下载仓库原始文件。

想让缓存放进项目内：

```bash
export HF_HOME=$PWD/data/cache
.venv/bin/python training/dataset/download_dataset.py
```

### 2. `prepare_dataset.py` —— 解压与归置

- 只解压 `raw/color/` 下的 54,305 张彩色图像（约 0.79 GB）。灰度版与分割版暂不解压，需要时改脚本里的 `VARIANTS` 再跑。
- 把官方划分文件与叶片分组元数据复制到 `data/splits/`。
- 校验解压结果（文件数、类别目录数）。

**解压后的路径关系**（容易看混，务必记住）：

```
data/raw/PlantVillage/raw/color/<类别>/<文件>.JPG
└──── 项目内的"原始数据"目录 ────┘└─ 压缩包内的原始结构 ─┘
```

路径里出现两个 `raw` 不是笔误——外层 `data/raw/` 表示"原始数据"，内层 `raw/` 是 zip 里的原始结构。

因此，**划分文件里的一行**要这样用：

```python
图像绝对路径 = data/raw/PlantVillage/ + <划分文件里的一行>
# 例：raw/color/Tomato___healthy/xxx.JPG
#  -> data/raw/PlantVillage/raw/color/Tomato___healthy/xxx.JPG
```

脚本可重复运行，已解压且大小一致的文件会自动跳过。

### 3. `check_dataset.py` —— 自检

无需先解压，直接在 zip 内部校验。六项检查：

1. 4 个文件是否齐全
2. `data.zip` 是否完整可读、图像总数是否正确
3. 作物数（14）与类别数（38）是否正确
4. 抽样解码图像：能否正常读取、是否为彩色 256×256
5. 官方划分是否覆盖全部图像、train/test 有无重叠
6. **数据泄露检查**：同一片叶子的多张照片是否跨 train/test

全部通过退出码为 0，任一项失败为 1。

**第 6 项最重要**。PlantVillage 里同一片叶子会被拍很多张，如果划分时不按叶片分组，同一片叶子的照片会一半在训练集一半在测试集，测试准确率会虚高好几个百分点。官方划分已经做好了分组，这个检查就是验证这一点。

脚本还打印一些中间信息（fallback 项重合数等），那些是"编号跨类别重名"导致的假阳性，不计入失败。

---

## 全部重跑（换机器时）

```bash
export HF_HOME=$PWD/data/cache                         # 可选：缓存放项目内

.venv/bin/python training/dataset/download_dataset.py  # 下载（2 GB，看网速）
.venv/bin/python training/dataset/prepare_dataset.py   # 解压（约 1~2 分钟）
.venv/bin/python training/dataset/check_dataset.py     # 自检（约 30 秒）

# 三条命令应全部以 ✅ 结束。之后进入训练：
.venv/bin/python training/train_classifier.py --smoke-test
```
