# わかれーる Wakareeru

[![model tag](https://img.shields.io/github/v/tag/SniperPigeon/wakareeru?filter=v*&label=model)](https://github.com/SniperPigeon/wakareeru/tags)
[![inference tag](https://img.shields.io/github/v/tag/SniperPigeon/wakareeru-inference?filter=inference-v*&label=inference)](https://github.com/SniperPigeon/wakareeru-inference/tags)
[![last commit](https://img.shields.io/github/last-commit/SniperPigeon/wakareeru)](https://github.com/SniperPigeon/wakareeru/commits/main)
[![commit activity](https://img.shields.io/github/commit-activity/m/SniperPigeon/wakareeru)](https://github.com/SniperPigeon/wakareeru/pulse)
[![repo size](https://img.shields.io/github/repo-size/SniperPigeon/wakareeru)](https://github.com/SniperPigeon/wakareeru)
[![top language](https://img.shields.io/github/languages/top/SniperPigeon/wakareeru)](https://github.com/SniperPigeon/wakareeru)

> [!IMPORTANT]
> **此Repo为项目数据集构建与筛选主仓库，关于客户端及其功能反馈请前往App仓库[`wakareeru-app`](https://github.com/wakareeru-team/wakareeru-app) 仓库。**


Wakareeru 是一个面向日本铁路车辆的图像识别项目。用户只需上传或拍摄一张图片，即可获得画面中车辆可能所属的系列或细分车型。我们希望它不只是铁路爱好者的识别工具，也能成为更多人了解身边日本铁路车辆的入口。

本仓库是项目的数据与模型核心，负责从日文 Wikipedia 和 Wikimedia Commons 构建细粒度图像数据集，完成图片清洗、车辆主体检测、标签噪声复核、模型训练，并导出供推理服务使用的自包含模型 artifact。此仓库不负责模型推理服务和终端App.

> [!IMPORTANT]
> Wakareeru 目前处于 Alpha 阶段。数据集、标签和模型会持续更新，同一张图片在不同版本中可能得到不同结果。识别结果仅供参考，不应用于车辆运营、资产管理或安全相关判断。

## 当前进展


- 当前模型覆盖全国六家 JR 客运公司与 JR 货物的主要客运动车组、柴油动车组和机车。
- 已包含六家 JR 客运公司与 JR 货物；私铁和地下铁车辆尚未进入当前稳定范围。
- 已建立 SigLIP2 图片过滤、Grounding-DINO 主体检测和细粒度标签规则。
- 已建立 DINOv3 small-loss tracking、人工复核、Logistic Regression 预测与 crop 级标签纠正闭环。
- 可导出包含图片、标签、多语言 metadata 和生成清单的最终 crop 数据集。
- 可训练冻结 DINOv3 backbone 的线性分类头，记录评估报告、预测结果、checkpoint 与 latest-run 指针。
- 可导出包含 backbone、processor、分类头和本地化 metadata 的离线模型 artifact，由 `model_core` 和推理后端直接加载。

铁路车辆中存在仅凭普通照片难以可靠区分的近似型号。当前标签空间优先收录能够从整体外观学习稳定特征的车辆；货车、客车、旧型事业用车辆等类别会按现有规则排除。实际数据范围由配置与人工规则共同决定，不等同于 Wikipedia 列表中的全部车辆。


## 系统组成

```text
日文 Wikipedia + Wikimedia Commons
                  │
                  ▼
       车型解析、图片抓取与过滤
                  │
                  ▼
     LLM metadata + Grounding-DINO bbox
                  │
                  ▼
       SQLite manifest / crops / review
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 DINOv3 噪声筛查       人工复核与标签纠正
        └─────────┬─────────┘
                  ▼
     crop dataset + 多语言 label metadata
                  │
                  ▼
        DINOv3 backbone + linear head
                  │
                  ▼
      自包含模型 artifact → inference API → App
```

本仓库各部分的边界如下：

- `pipeline/`：稳定的数据管线，15 个可独立运行的逻辑阶段。
- `tools/`：人工复核、抽查、review CSV 导入导出与维护工具。
- `trainer/`：最终 crop 数据集的训练、验证、checkpoint 和模型导出。
- `model_core/`：训练与推理共用的模型结构、artifact loader、预处理和 crop 分类接口。
- `config/`：运行配置、人工标签规则、数据库基线 schema 和增量 migration。
- `src/crawler/`：探索性 notebook 与实验代码，不是稳定入口。
- `data/`：本地生成的数据库、原图、cache、review 输出、数据集和模型运行结果，通常不提交到 Git。

更完整的目录结构：

```text
pipeline_entry.py          # 管线总入口：全部、单阶段、范围或从某阶段继续
pipeline/
  stage_01_model_parsing.py
  stage_02_model_fixing.py
  stage_03_manifest_crawling.py
  stage_04_img_crawler.py
  stage_05_siglip_image_filtering.py
  stage_06_llm_metadata_labeling.py
  stage_07_gdino_bbox.py
  stage_08_fine_grain_series.py
  stage_09_DINOv3_feature_extraction.py
  stage_10_train_loss_tracking.py
  stage_11_loss_analysis.py
  stage_12_logistic_regression_filter.py
  stage_13_lr_prediction.py
  stage_13b_label_metadata_translation.py
  stage_14_store_crops.py
  constants.py
  utils.py
config/
  pipeline_config.yaml
  manual_series_overrides.csv
  manual_fine_grained_series.csv
  schema.sql
  migrations/
tools/
trainer/
model_core/
docs/
docker/
Dockerfile.basepod
```

## 快速开始

项目要求 Python 3.11 或更高版本，推荐使用仓库提供的 Python 3.12 Conda 环境。模型阶段建议在 CUDA GPU 上运行。

```bash
conda env create -f environment.yml
conda activate wakareeru
pip install -e ".[dev]"
cp .env.example .env
```

按需要在 `.env` 中配置 Hugging Face token：

```dotenv
HF_TOKEN=...
HUGGINGFACEHUB_API_TOKEN=...
```

- Stage 6 直接读取当前进程的 `OPENAI_API_KEY`，运行前需要在 shell 中导出：

  ```bash
  export OPENAI_API_KEY="..."
  ```

- Hugging Face token 用于需要鉴权或接受模型许可的模型下载。
- Wikipedia、Wikimedia Commons、OpenAI 与 Hugging Face 阶段都需要网络连接。

完整依赖见 [`environment.yml`](environment.yml)；[`pyproject.toml`](pyproject.toml) 主要定义可安装的 `model_core` 包、基础抓取依赖和开发工具。

## 数据目录与配置

所有稳定阶段默认读取 [`config/pipeline_config.yaml`](config/pipeline_config.yaml)。

- 代码、规则和配置文件相对项目根目录解析。
- SQLite、图片、cache、review 输出、数据集和模型结果相对 `path.data_root` 解析。
- `path.in_project_root: true` 时，`data_root` 相对仓库根目录解析。
- `path.in_project_root: false` 时，`data_root` 必须是绝对路径，适合 `/workspace/data` 等挂载卷。

默认生成状态位于 `data/`。不要在没有备份的情况下删除、覆盖或重建该目录；已有 SQLite 数据库会通过 [`config/migrations/`](config/migrations/) 按 `PRAGMA user_version` 增量迁移。

最常调整的配置组包括：

| 配置组 | 用途 |
| --- | --- |
| `path.*` | 数据根目录、数据库、原图、cache、dataset 与模型输出路径 |
| `crawler.*` | 运营方范围、测试/全量抓取、递归深度、下载并发与重试 |
| `image_filtering.*` | SigLIP2 模型、阈值、batch 与处理范围 |
| `llm_labeling.*` | OpenAI 模型、推理强度、web search 与重试 |
| `gdino.*` | Grounding-DINO 模型、检测阈值、NMS 与 batch |
| `fine_grain_series.*` | 细粒度系列人工规则 |
| `noise_detection.*` | DINOv3 特征、标签粒度、small-loss 训练输入与噪声过滤 |
| `logistic_regression_filter.*` | 人工 review 特征、训练阈值与 LR 模型 |
| `label_metadata_translation.*` | 新标签多语言 metadata 的人工翻译队列 |
| `crops_storage.*` | 最终 crop 选择策略、字段、图像格式与 dataset 输出 |
| `trainer.*` | backbone、输入尺寸、训练 phase、feature cache、评估与 artifact 导出 |

正式 pipeline 不为缺失的配置项静默使用代码默认值。修改配置结构时应同步更新 `pipeline_config.yaml`。

## 运行 Pipeline

运行配置中的全部逻辑阶段：

```bash
python pipeline_entry.py
```

只运行一个命名阶段，或从某阶段继续：

```bash
python pipeline_entry.py --only siglip_filter
python pipeline_entry.py --from manifest_crawling
```

按入口中的逻辑编号选择一个、多个或一段阶段：

```bash
python pipeline_entry.py --stages "5"
python pipeline_entry.py --stages "5 10 11"
python pipeline_entry.py --stages "9-13"
```

使用其他配置文件：

```bash
python pipeline_entry.py --config path/to/pipeline_config.yaml --only store_crops
```

完整管线并不一定一次无交互跑到底：

- `loss_analysis.request_manual_review: true` 时，Stage 11 会导出本轮分析并中断，等待人工复核。
- 人工 review 样本不足或无法满足 clean recall 条件时，Stage 12 会中断。
- Stage 13b 发现新的 label metadata 时，会导出翻译 CSV 并中断；填写后重跑同一阶段才会写回数据库。

### 阶段索引

这里的“编号”是 `--stages` 使用的逻辑顺序；文件名保留历史 stage 编号，因此 Stage 13b 是逻辑编号 14，`store_crops` 是逻辑编号 15。

| 编号 | Key | 脚本 | 主要职责 |
| ---: | --- | --- | --- |
| 1 | `model_parsing` | `stage_01_model_parsing.py` | 从日文 Wikipedia wikitext 解析系列、状态、类型与运营方，并应用基础排除规则 |
| 2 | `model_fixing` | `stage_02_model_fixing.py` | 应用人工修正，验证并生成 Commons 根分类映射 |
| 3 | `manifest_crawling` | `stage_03_manifest_crawling.py` | 遍历 Commons 分类树，写入分类、图片 manifest 与子树 checkpoint |
| 4 | `img_crawling` | `stage_04_img_crawler.py` | 下载图片，记录状态，并规范化文件名与数据库路径为 Unicode NFC |
| 5 | `siglip_filter` | `stage_05_siglip_image_filtering.py` | 用 SigLIP2 过滤内饰、局部细节和其他不适合训练的图片 |
| 6 | `llm_labeling` | `stage_06_llm_metadata_labeling.py` | 从分类路径抽取番台、子型号、运营方、特殊编成与涂装 |
| 7 | `gdino_bbox` | `stage_07_gdino_bbox.py` | 用 Grounding-DINO 检测车辆主体并写入 bbox crop 记录 |
| 8 | `fine_grain_series` | `stage_08_fine_grain_series.py` | 根据 LLM metadata 与人工规则构造 `fine_grained_series` |
| 9 | `feature_extraction` | `stage_09_DINOv3_feature_extraction.py` | 提取 crop 的 DINOv3 特征，只缓存 `features` 与 `crop_ids` |
| 10 | `loss_tracking` | `stage_10_train_loss_tracking.py` | 按当前标签空间训练线性头并记录逐样本 loss 与预测 |
| 11 | `loss_analysis` | `stage_11_loss_analysis.py` | 聚合 loss、错误率和预测一致性等噪声筛查特征 |
| 12 | `logistic_regression_filter` | `stage_12_logistic_regression_filter.py` | 用人工复核数据训练 Logistic Regression 噪声筛选器 |
| 13 | `lr_prediction` | `stage_13_lr_prediction.py` | 为未复核 crop 生成 LR 噪声预测，并可同步到数据库 |
| 14 | `label_metadata_translation` | `stage_13b_label_metadata_translation.py` | 增量补齐 label 翻译、三语运营方和日文 Wikipedia title |
| 15 | `store_crops` | `stage_14_store_crops.py` | 应用人工/LR 筛选与纠正，生成最终 crop 数据集 |

`pipeline/deprecated_stage_08_siglip_crop_filtering.py` 是弃用实验，不属于默认流程。

## 人工复核与维护工具

按噪声分数分层抽样、写入人工结论与正确标签：

```bash
python tools/noise_review_gradio.py
```

只读检查指定 loss round，支持 LR 高分、高错误率、预测不一致和按 label 均衡抽样：

```bash
python tools/spotcheck.py
```

检查 label 分布并进行跨 label 抽样和 crop 级纠正：

```bash
python tools/label_review_gradio.py
```

人工结论保存在 `crops.noise_review_*`。确认错标但仍属于已知标签空间的 crop，可写入 `manual_corrected_label`；训练和最终导出会优先使用纠正后的标签，但不会覆盖 `images` 中的原始来源标签。

跨机器迁移 review overlay 时，使用 stable key 与 bbox IoU 匹配，不依赖数据库自增 id：

```bash
python tools/export_noise_review_csv.py \
  --output-csv-path review/noise_review_labels.csv

python tools/import_noise_review_csv.py \
  --review-csv-path review/noise_review_labels.csv
```

跨平台迁移图片后若出现 macOS/Linux 的 NFC/NFD 文件名差异，先 dry-run，再显式应用：

```bash
python tools/normalize_image_paths.py
python tools/normalize_image_paths.py --apply
```

噪声闭环与重跑边界详见 [`docs/noise_review_loop.md`](docs/noise_review_loop.md)。

## 最终数据集

在多语言 metadata 完整后运行：

```bash
python pipeline_entry.py --only label_metadata_translation
python pipeline_entry.py --only store_crops
```

Stage 13b 只导出数据库中尚不存在的新 label；填写 `label_en`、`label_zh` 与三语 operator JSON 数组后重跑，会校验并事务写入 `label_metadata`。已有规范记录不会被自动覆盖。

默认数据集目录由 `path.dataset_dir` 控制，包含：

```text
dataset/
  images/               # 按 bbox 和 crop_pad_frac 生成的训练图片
  metadata.csv          # 样本路径、label id、人工复核标记与车辆 metadata
  labels.csv            # 连续的 label_id ↔ label 映射
  l10n_metadata.json    # 日/英/中 label、运营方与日文 Wikipedia title
  manifest.json         # 样本数、标签数、筛选策略和生成参数
```

`l10n_metadata.json` 只从数据库 `label_metadata` 规范表导出。缺少翻译、三语运营方数组不对齐、存在链接或语言污染时，导出会直接报错。

`crops_storage.selection_mode` 控制最终样本选择：

- `filtered`：应用人工噪声结论和数据库中的 LR 预测；人工 `ok` 与 `manual_corrected_label` 优先于模型预测。
- `all`：不按人工或 LR 结果排除 crop，但仍应用人工纠正标签。

启用 LR 预测过滤时，`lr_prediction.sync_to_db` 必须为 `true`。没有数据库预测的新 crop 会保留。

## 训练与模型导出

训练最终 crop 数据集：

```bash
python -m trainer.train
```

当前默认方案使用 DINOv3 ViT-S/16，拼接 CLS token 与排除 register tokens 后的 patch mean，并训练冻结 backbone 上的线性分类头。训练器支持：

- 固定随机种子的 train/validation 切分；
- 按 `image_path` 增量复用的线性头 feature cache；数据集缩小、重新切分或
  label id 变化不会重提特征，新增图片只补提取缺失项；
- AMP、top-k accuracy、macro/weighted F1；
- early stopping、逐 epoch 报告和验证集预测；
- 每个 phase 的 best checkpoint、`run_summary.json` 与 latest-run 指针。

缓存会保留当前 `metadata.csv` 已移除样本的特征，以便之后重新纳入时继续复用。
`trainer.image_size`、backbone、pooling 或特征维度变化后，应设置
`feature_cache_rebuild: true` 完整重建一次；正常训练保持为 `false`。缓存以路径
作为图片身份，若原路径的图片内容被覆盖，也需要显式重建。

导出供推理仓库使用的模型：

```bash
python -m trainer.export_inference_model
```

`trainer.export.checkpoint_path: "latest_best"` 会读取最新训练 run 最后一个 phase 的 best checkpoint，也可以指定相对 `path.data_root` 的 checkpoint。默认输出是一个可离线加载的自包含目录：

```text
model/<artifact>/
  backbone/
  processor/
  classifier.safetensors
  model_config.json
  labels.json
  l10n_metadata.json
  manifest.json
```

输入尺寸来自 checkpoint 中保存的训练配置，而不是导出时工作区中后来修改的值；导出器会同步 processor 的 `size` 与 `crop_size`。

仓库内的公共推理接口可直接加载 artifact：

```python
from model_core.loader import load_classifier
from model_core.predict import predict_crop

loaded = load_classifier("data/model/<artifact>", device="cpu")
predictions = predict_crop(
    loaded=loaded,
    image="example.jpg",
    top_k=5,
)
```

## RunPod / GPU 镜像

[`Dockerfile.basepod`](Dockerfile.basepod) 基于 RunPod CUDA/PyTorch 镜像，保留镜像自带的 `torch` / `torchvision`，并安装 [`requirements-runpod.txt`](requirements-runpod.txt)、`rclone` 与 `rsync`：

```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.basepod \
  -t wakareeru-basepod:local \
  --load \
  .
```

容器将 Hugging Face 与 pip cache 放在 `/workspace/.cache`，并通过 [`docker/entry.sh`](docker/entry.sh) 从运行时变量创建名为 `r2` 的 rclone remote：

```text
HF_TOKEN
R2_ACCESS_ID
R2_ACCESS_KEY
R2_ENDPOINT
```

建议把持久卷挂载到 `/workspace`，并配置：

```yaml
path:
  in_project_root: false
  data_root: /workspace/data
```

进入容器后可用 `rclone lsd r2:` 检查对象存储连接。共享数据库与快照约定见 [`docs/contributing_zh.md`](docs/contributing_zh.md)。

## 开发与贡献

```bash
ruff check .
pytest
```

数据库基线位于 [`config/schema.sql`](config/schema.sql)，既有数据库只能通过编号 migration 增量升级。涉及标签规则、schema、`model_core` 或 artifact 契约的改动，应说明重跑范围及其对 `wakareeru-inference` 的影响。

贡献前请阅读：

- [中文贡献规范](docs/contributing_zh.md)
- [English contributing guide](docs/contributing.md)
- [噪声复核与迭代训练闭环](docs/noise_review_loop.md)

## 应用、隐私与社区

Wakareeru App 当前无需注册账户，识别历史仅保存在用户设备上。图片会发送到服务器完成本次识别；当前不会把用户上传图片用于模型训练，也不会建立公开图片库。

服务目前免费且不包含广告。为控制运行成本，推理服务采用按需启动的计算方式，首次请求偶尔会有较长等待时间。

Wakareeru 是由日本铁路与摄影爱好者独立开发的项目，与 JR 集团及各铁路运营商不存在隶属、授权或合作关系。欢迎反馈 App 体验、数据呈现和模型识别问题：

- [加入 QQ 群](https://qm.qq.com/q/lN5cxE92bS)
- [加入 Discord](https://discord.gg/Y3KZtP7mGp)

## 数据来源说明

车型与标签来源主要是日文 Wikipedia，训练候选图片主要来自 Wikimedia Commons。Commons 文件各自具有独立的许可与署名要求；使用、再发布数据集或图片时，应读取并遵守原文件页面中的许可和 attribution 信息。
