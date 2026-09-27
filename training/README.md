# training —— 模型训练与评估（已归档）

> ## ⚠️ 存档说明：选题方向已变更
>
> 本目录是**方向变更前**的工作存档，原选题为
> 「抗对抗样本的农作物病虫害智能识别 App 设计与实现」。
>
> **新选题**：基于四元数卷积改进 YOLOv8 的大田作物叶片病害光照鲁棒性检测研究。
>
> 新方向下本目录代码**大部分不再使用**，保留原因有二：
> 1. `dataset/` 中的 PlantVillage 获取与自检脚本，后续有较小概率仍会用到；
> 2. `common/`、`train_classifier.py`、`evaluate_classifier.py` 可作为 PyTorch
>    训练/评估工程的写法参考。
>
> 存档时代码状态：分类模型已在 PlantVillage 上训练完成（测试集 Top-1 99.30%），
> FGSM/PGD 对抗攻击实验已完成（PGD ε=0.03 攻击成功率 100%）。
> 模型权重（`runs/`）与数据集（`data/`）按 `.gitignore` 未纳入版本库。

本目录存放 PlantVillage 病虫害分类模型的**数据准备、训练、评估**代码。

---

## 目录结构

```
training/
├── dataset/                    数据获取 · 解压 · 自检（只跑一次）
│   ├── download_dataset.py     从 HuggingFace 下载 PlantVillage
│   ├── prepare_dataset.py      解压图像 + 归置官方划分文件
│   └── check_dataset.py        数据集完整性 / 数据泄露自检
│
├── common/                     共享代码（被下面四个脚本 import，不单独运行）
│   ├── config.py               路径常量、38 类映射
│   ├── data.py                 数据加载、图像增强、划分
│   ├── model.py                模型构建、冻结/解冻、权重存取
│   ├── attacks.py              ★ FGSM / PGD 攻击实现（像素空间）
│   └── utils.py                随机种子、设备、指标、日志
│
├── train_classifier.py         ① 训练分类模型
├── evaluate_classifier.py      ② 评估 + 出图（混淆矩阵、训练曲线、推理耗时）
├── generate_adversarial.py     ③ 生成对抗样本 + 扫描 ε 曲线
├── eval_attack.py              ④ 攻击效果可视化（三联图、ε 曲线、Top-3 对比）
├── runs/                       训练输出（权重、曲线、日志），不进版本库
└── ../data/adversarial/        对抗样本与攻击实验的产出
```

**为什么这么分**：`dataset/` 的脚本是一次性工具（数据下好就不用再跑）；`common/` 是被复用的库；顶层两个脚本是你日常要反复执行的。对抗样本生成、检测器训练等后续脚本也会放在顶层，同样 import `common/`。

---

## 执行顺序

### 第 0 步：数据准备（已完成，无需重跑）

```bash
.venv/bin/python training/dataset/download_dataset.py    # 下载
.venv/bin/python training/dataset/prepare_dataset.py     # 解压 + 归置划分
.venv/bin/python training/dataset/check_dataset.py       # 自检
```

数据已就绪（38 类 / 54,305 张，位于 `data/raw/PlantVillage/`）。将来换了机器才需要重跑。

### 第 1 步：冒烟测试（约 2 分钟）

```bash
.venv/bin/python training/train_classifier.py --smoke-test
```

小数据量各跑 1 轮，只为验证「代码能不能跑通」。**结果无意义，但必须先跑一次**——否则可能等 1 小时后才发现路径写错。

### 第 2 步：正式训练（M1 Pro 上约 1~1.5 小时）

```bash
.venv/bin/python training/train_classifier.py
```

想让日志同时留在文件里：

```bash
.venv/bin/python training/train_classifier.py 2>&1 | tee training/runs/latest.log
```

### 第 3 步：评估与出图（约 3 分钟）

```bash
.venv/bin/python training/evaluate_classifier.py
```

自动找 `runs/` 下最近一次的 `best.pth`。也可指定：

```bash
.venv/bin/python training/evaluate_classifier.py \
    --checkpoint training/runs/classifier_20260926_143012/best.pth
```

### 第 4 步：交付给后端

```bash
mkdir -p backend/models
cp training/runs/classifier_*/best.pth backend/models/classifier_mobilenet.pth
```

### 第 5 步：对抗攻击实验（约 5~10 分钟）

```bash
# 生成 FGSM + PGD 对抗样本，扫描 ε ∈ {0.005 0.01 0.02 0.03 0.05}
.venv/bin/python training/generate_adversarial.py

# 出图
.venv/bin/python training/eval_attack.py
```

时间紧时的最小版本（只做 FGSM 单点，约 1 分钟）：

```bash
.venv/bin/python training/generate_adversarial.py --attack fgsm --epsilons 0.03
.venv/bin/python training/eval_attack.py --attack fgsm
```

---

## 各脚本详解

### `train_classifier.py` —— 训练分类模型

| 项目 | 设置 |
|---|---|
| 骨干网络 | MobileNetV3-Small（ImageNet 预训练） |
| 输入 | 224×224，ImageNet 归一化 |
| 阶段一 | 冻结骨干，只训分类头 — 5 轮，lr=1e-3 |
| 阶段二 | 解冻全模型微调 — 15 轮，lr=1e-4，cosine 衰减 |
| 优化器 | AdamW，weight_decay=1e-4 |
| 数据增强 | 随机裁剪(0.7~1.0) + 水平翻转 + 旋转±15° + 颜色抖动 |
| 正则化 | 标签平滑 0.1，分类头 Dropout 0.2 |
| batch size | 64 |
| 目标 | Top-1 ≥ 98% |

**两阶段为什么必须分开**：分类头是随机初始化的，梯度很大。如果一上来就整体训练，这些梯度会传进骨干，把 ImageNet 学到的通用特征摧毁掉。先让分类头单独收敛到合理位置，再整体低学习率微调。

**常用参数**：

```bash
--epochs-stage2 25      # 精度不够时延长阶段二
--batch-size 32         # 内存不够时调小
--num-workers 0         # 卡住不动时改成 0
--weighted-sampler      # 启用类别倒频率采样，缓解样本不均衡
--tag exp1              # 给本次运行起名，便于区分多次实验
--img-size 160          # 降分辨率加速（PlantVillage 上精度损失很小）
```

**输出**（`runs/classifier_<时间戳>/`）：

| 文件 | 内容 |
|---|---|
| `best.pth` | 验证集最优权重（下一步就用它） |
| `last.pth` | 最后一轮权重 |
| `history.json` | 每轮的 loss / 准确率，用于画训练曲线 |
| `config.json` | 本次运行的全部参数（论文里写"实验设置"时取这里） |
| `log.txt` | 终端输出的完整副本（中期检查、论文附录都用得上） |
| `train_class_counts.json` | 训练集每类样本数 |

### `evaluate_classifier.py` —— 评估与出图

对应开题报告 3.2 节「实验一」的全部指标要求。

| 产出 | 说明 |
|---|---|
| `metrics.json` | Top-1/Top-3/Top-5、宏平均 F1、加权 F1、F1 最低的类、最易混淆的类别对 |
| `per_class_report.txt` | 38 类的精确率 / 召回率 / F1 / 样本数 |
| `confusion_matrix.png` | 38×38 混淆矩阵（原始计数 + 行归一化） |
| `confusion_matrix.json` | 混淆矩阵数值，可自行重绘 |
| `training_curves.png` | 损失曲线 + 准确率曲线，标出两阶段分界与最优轮次 |
| `inference_time.json` | 推理耗时（MPS 与 CPU，batch=1 与 batch=32） |

**为什么同时报宏平均 F1 和总准确率**：PlantVillage 样本不均衡（有的类几百张、有的几千张），总准确率会被大类别主导。宏平均 F1 是每类先算 F1 再取平均，不受样本量影响。**只报准确率会被答辩评委问倒。**

**为什么专门测 CPU 推理耗时**：开题报告的指标是「单张推理时间 < 200ms（CPU）」。MPS 上的耗时好看但不能拿去对标这个指标，必须单独测 CPU。

参数：

```bash
--time-only     # 只测推理耗时，跳过完整评估（几十秒）
--batch-size 128
```

---

### `generate_adversarial.py` —— 生成对抗样本 + 扫 ε 曲线

对应开题报告 3.2 节「实验二：对抗样本生成与攻击有效性验证」。

| 项目 | 说明 |
|---|---|
| 攻击方法 | FGSM（单步）、PGD（迭代，默认 10 步、α=0.01） |
| 扫描 ε | 0.005 / 0.01 / 0.02 / 0.03 / 0.05（ε=0.03 ≈ 7.65/255） |
| 样本量 | 从测试集按类别分层抽 1000 张（统计误差约 ±1.5%） |
| 量化口径 | 同时报浮点与 **8 位量化后**的攻击成功率——8 位才是攻击者实际能交付的形态 |

**产出**：

```
data/adversarial/
├── curve.json       ε 扫描的全部数值（写论文直接取数）
├── manifest.json    样本清单（原图路径 ↔ 对抗图路径 ↔ 是否骗成功）
└── samples/
    ├── fgsm/<类别>/<序号>_<原名>.png
    └── pgd/<类别>/<序号>_<原名>.png
```

脚本结尾会做**两条交叉验证**，用来确认攻击实现没写错：

1. 准确率是否随 ε 单调下降；
2. **PGD 是否明显强于 FGSM**——与 Luo 2021 / You 2023 / Li & Lu 2023 三篇独立工作一致。
   对不上就说明实现有 bug。

**为什么在像素空间实现攻击**：模型输入是归一化过的（减均值除标准差），取值范围约 [-2.1, 2.6]。
如果直接在这个空间攻击并 `clamp(0,1)`，会有两个错误——ε 的物理含义变成 `ε×std`（弱 4 倍多，
且三通道不一致），而且 clamp 会把大量像素压到边界上、生成非法图像。所以统一在像素空间 [0,1]
做攻击，只在送进模型前临时归一化。详见 `common/attacks.py` 的模块注释。

### `eval_attack.py` —— 攻击效果可视化

| 产出 | 用途 |
|---|---|
| `figures/accuracy_vs_epsilon.png` | 准确率-ε + 攻击成功率-ε 双联图，标出未攻击基线 |
| `figures/adversarial_examples.png` | **三联图**：原图 / 对抗图 / 扰动×10。PPT 的王牌图 |
| `figures/prediction_comparison.png` | 攻击前后 Top-3 对比表，展示「错得很自信」 |
| `visual_report.json` | 图中样本的全部数值 |

三联图的样本是**跨类别均匀挑选**的，避免整张图都是同一作物的不同病害。

参数：

```bash
--attack pgd          # 用哪种攻击的样本做可视化（默认 pgd，它更强）
--num-examples 6      # 三联图的行数
```

**`adversarial_examples.png` 是 PPT 里最该给足版面的一张**：左中两图肉眼无法区分，
但模型从「Apple scab 82.8%」跳到「Tomato Septoria 58.2%」——连作物都换了。

---

## 常见问题

**训练时进度条卡住不动**
DataLoader 的多进程在 macOS 上偶发卡死。改成单进程：

```bash
.venv/bin/python training/train_classifier.py --num-workers 0
```

**显存/内存不足（进程被系统杀掉）**
调小 batch size：`--batch-size 32` 或 `16`。

**精度上不去（低于 95%）**
按顺序排查：① 确认用了预训练权重（脚本默认加载）；② 确认用的是官方划分而不是自己随机分（随机分会因为数据泄露而虚高，是很隐蔽的错）；③ 学习率是否过大；④ 延长阶段二轮数。

**别追 99.9%**
PlantVillage 上 99%+ 是标配，不是成果。到 98% 以上就该把时间投到对抗攻击与检测实验上。多花的每一小时在这里都是浪费。

**中文字体**
`evaluate_classifier.py` 画图时会自动探测中文字体（macOS 上通常能找到 PingFang SC）。探测不到就自动改用英文标签，不会出废图。
