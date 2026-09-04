# 人工车型目录添加说明

> 在向 `config/manual_series_catalog.csv` 手工添加任何车型前，维护者或 LLM
> agent 必须先完整阅读本说明，并按“Commons 真 root 核验”留下可复核证据。

该目录用于直接登记车型与 Wikimedia Commons root，适合车型列表页面分散、
Commons 前缀不规则的私铁和第三部门车辆。Stage 01 会把目录行追加到自动解析
结果；Stage 02 直接采用已核验的 root，不再执行自动 root 搜索。因此，填写者而非
Stage 02 承担 root 正确性的责任。

## 添加流程

1. 确认车辆谱系，决定使用 `new` 还是 `merge`。
2. 核对日文 Wikipedia title；别名或章节重定向也要确认最终指向的车型。
3. 通过 Commons API 核验候选 root 的内容、直接父分类和必要的子分类。
4. 在浏览器抽查 root 及其主要子分类中的图片，排除线路、车站、列车服务、内饰或
   泛运营商分类。
5. 填写 CSV；稳定且对整个 root 都成立的 Stage 06 metadata 可以同时填写并锁定。
6. 运行 Stage 01、02、03；检查输出的 series/root/operator，再运行 Stage 06。

```bash
python pipeline_entry.py --stages "1 2 3 6"
```

未来可以由带网页检索能力的 LLM agent 辅助第 1—4 步，但 agent 必须输出候选比较、
Commons API 的非空证据、父分类证据和抽查结论；不得仅按 category 名称猜测 root，
也不得跳过人工复核后直接批量写入目录。

## Commons 真 root 核验

这里的“真 root”是适合该目录行作为 Stage 03 起点的最小稳定车型/车族分类，不要求
它是 Commons 分类树的全局顶层。它必须同时满足：

- category 存在，且 `categoryinfo.files + categoryinfo.subcats > 0`；
- 语义是车型、车族或明确的运营商版本，而不是线路、车站、列车服务、活动、涂装、
  单车号或泛运营商车辆总类；
- 直接父分类至少能证明车型谱系或运营商车辆归属；继承车辆最好同时挂在原车型与
  新运营商车辆分类下；
- root 本身及计划递归的主要子分类没有明显跨车型污染；
- 与更宽和更窄的候选比较后，选择能完整覆盖目标车辆且不会无谓引入其他车型的
  最小稳定入口；
- 将核验日期、API 结果和关键父分类记录在本说明或对应 review 文档中。

文件数会变化，只能作为核验当日“分类非空”的证据。再次新增或修订 root 时必须重新
查询 Commons API，不能永久复用旧计数。

### 当前首批 root 核验台账

2026-08-01 使用 Commons MediaWiki API 的 `categoryinfo|categories` 重新核验。
以下 12 个候选分类都非空，且属于车型级或明确的运营商车型版本。结合已有数据库
覆盖情况后，本批实际采用其中 10 个作为新增 Stage 03 root：

| root | 当日 files/subcats | 关键证据/处理 |
| --- | ---: | --- |
| `Chizu Express HOT 3500 series` | 24 / 1 | `Diesel multiple units of Chizu Express Company` |
| `Chizu Express HOT 7000 series` | 40 / 1 | `Diesel multiple units of Chizu Express Company` |
| `Kitakinki Tango Railway KTR 001` | 16 / 0 | `Rolling stock of Kitakinki Tango Railway` |
| `Miyafuku Railway MF100/200` | 17 / 0 | `Rolling stock of Kitakinki Tango Railway` |
| `Kitakinki Tango Railway KTR 300` | 9 / 0 | `Rolling stock of Kitakinki Tango Railway` |
| `Kitakinki Tango Railway KTR 700/800` | 58 / 4 | `Rolling stock of Kitakinki Tango Railway` |
| `Kitakinki Tango Railway KTR 8000` | 14 / 2 | `Rolling stock of Kitakinki Tango Railway` |
| `Kitakinki Tango Railway KTR 8500` | 7 / 0 | 已有 `キハ85系` 图片覆盖，且规范 operator 已含北近畿タンゴ鉄道；不新增 root |
| `ET122` | 30 / 1 | `Rolling stock of Echigo Tokimeki Railway` |
| `ET127` | 49 / 0 | `JR East E127`；`Rolling stock of Echigo Tokimeki Railway` |
| `JNR 413 (Echigo Tokimeki Railway)` | 27 / 0 | 已有 `413系` / `455系` 图片与 operator 覆盖；不新增 root |
| `Hokuetsu Express 100 series` | 77 / 0 | `Rolling stock of Hokuetsu Express` |

这里验证的是“候选分类确为车型级入口”，不是声称它们没有更上层分类，也不代表
验证后必须重复抓取。`KTR 8500` 和越后心跳铁道 413/455 因现有图片及 operator
已经覆盖而从人工目录移除；`ET122`、`ET127` 保留独立 root，用于补充第三部门图片。

### 第二批 root 核验台账

2026-08-01 按同一 API 和父分类标准核验肥薩おれんじ鉄道、土佐くろしお鉄道、
鹿島臨海鉄道、三陸鉄道、青い森鉄道与 IGRいわて銀河鉄道。本批采用以下 13 个
车型级 root：

| root | 当日 files/subcats | 关键证据/处理 |
| --- | ---: | --- |
| `Hisatsu Orange Railway HSOR-100 series` | 38 / 1 | HSOR-100/150 同一车族；子分类继续由 Stage 03 递归 |
| `Tosa Kuroshio Railway TKT8000 series` | 12 / 0 | `Diesel multiple units of Tosa Kuroshio Railway` |
| `Tosa Kuroshio Railway 9640 series` | 7 / 3 | `Diesel multiple units of Tosa Kuroshio Railway` |
| `KRT 6000 series` | 45 / 0 | `Diesel multiple units of Kashima Seaside Railway` |
| `KRT 7000 series` | 4 / 0 | `Diesel multiple units of Kashima Seaside Railway` |
| `KRT 8000 series` | 7 / 0 | `Diesel multiple units of Kashima Seaside Railway` |
| `Sanriku Railway 36-100` | 11 / 0 | `Diesel multiple units of Sanriku Railway` |
| `Sanriku Railway 36-200` | 4 / 0 | 与 36-100 同谱系，基础 `series` merge 到 `36-100形` |
| `Sanriku Railway 36-R` | 5 / 0 | `Diesel multiple units of Sanriku Railway` |
| `Sanriku Railway 36-700` | 13 / 0 | `Diesel multiple units of Sanriku Railway` |
| `Aoimori 701 series` | 48 / 0 | merge 到 JR 东日本 `701系` |
| `Aoimori 703 series` | 12 / 0 | 按车辆谱系 merge 到 JR 东日本 `E721系` |
| `IGR 7000 series` | 14 / 0 | 按车辆谱系 merge 到 JR 东日本 `701系` |

以下车型不写入人工 crawl 行：

- 土佐くろしお鉄道 2000系与 2700系已位于现有 `JR Shikoku 2000` / `JR Shikoku 2700`
  通用 root；分类混有 JR 四国车辆，不能将整个 root 的 operator 锁为土佐くろしお鉄道。
- 三陸鉄道 36-Z、旧 36-300/400/500/1100/1200/2100，以及鹿島臨海鉄道旧
  キハ1000/2000形、KRD/KRD64 未找到独立非空车型 root；不得改用泛运营商分类。

### 第三批 root 核验台账：地下铁

2026-08-28 通过 Commons API 的 `categoryinfo`、直接父分类和主要子分类，核验东京
地下铁、都营地下铁与大阪地下铁。只采用非空的车型级 root；没有采用线路、运营商
总类、内饰类或 New Tram/AGT 车型。东京地下铁的现役车型另与其官方车辆介绍页交叉
核对，都营现役车型与东京都交通局官方车辆介绍页交叉核对。

东京地下铁采用以下 20 个 root：

| root | 当日 files/subcats | 关键证据/处理 |
| --- | ---: | --- |
| `Tokyo Metro 01 series` | 58 / 1 | `Rolling stock of Tokyo Metro`；排除熊本电铁转让子树 |
| `Tokyo Metro 02 series` | 108 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 03 series` | 57 / 1 | `Rolling stock of Tokyo Metro`；排除转让车子树 |
| `Tokyo Metro 05 series` | 175 / 2 | `Rolling stock of Tokyo Metro`；排除印尼 `Seri 05` 子树 |
| `Tokyo Metro 06 series` | 25 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 07 series` | 22 / 3 | 三个线路子分类均为东京地下铁同车型 |
| `Tokyo Metro 08 series` | 47 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 1000 series` | 86 / 1 | `Rolling stock of Tokyo Metro`；子分类为特别设计车 |
| `Tokyo Metro 2000 series` | 69 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 5000 series` | 45 / 1 | `Rolling stock of Tokyo Metro`；排除转让车子树 |
| `Tokyo Metro 6000 series` | 96 / 1 | `Rolling stock of Tokyo Metro`；排除印尼 `Seri 6000` 子树 |
| `Tokyo Metro 7000 series` | 145 / 2 | `Rolling stock of Tokyo Metro`；保留 7001 号车，排除印尼转让车子树 |
| `Tokyo Metro 8000 series` | 123 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 9000 series` | 93 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 10000 series` | 131 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 13000 series` | 52 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 15000 series` | 22 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 16000 series` | 69 / 1 | `Rolling stock of Tokyo Metro`；子分类为同车型编组 |
| `Tokyo Metro 17000 series` | 26 / 0 | `Rolling stock of Tokyo Metro` |
| `Tokyo Metro 18000 series` | 29 / 0 | `Rolling stock of Tokyo Metro` |

都营地下铁采用以下 11 个 root；直接父分类均包含
`Rolling stock of Tokyo Toei Subway`：

| root | 当日 files/subcats | 处理 |
| --- | ---: | --- |
| `Toei 5000 series` | 8 / 0 | 独立车型 root |
| `Toei 5200 series` | 7 / 0 | 独立车型 root；Wikipedia 使用 5000 形页面的 5200 形章节 |
| `Toei 5300 series` | 173 / 0 | 独立车型 root |
| `Toei 5500 series` | 21 / 13 | 子分类为同车型编组 |
| `Toei 6000 series` | 7 / 1 | 子分类继续由 Stage 03 递归 |
| `Toei 6300 series` | 29 / 35 | 子分类继续由 Stage 03 递归 |
| `Toei 6500 series` | 56 / 0 | 独立车型 root |
| `Toei 10-000 series` | 67 / 0 | 独立车型 root |
| `Toei 10-300 series` | 85 / 0 | 独立车型 root；不按外观来源并入 JR 基础系列 |
| `Toei 12-000 series` | 38 / 0 | 独立车型 root |
| `Toei 12-600 series` | 21 / 0 | 独立车型 root |

大阪地下铁采用以下 17 个 root；父分类均能回到 `Rolling stock of Osaka Metro`。
New 20 系列按 21--25 系分别抓取，避免把不同线路与外观标签混在一个基础类中：

| root | 当日 files/subcats | 处理 |
| --- | ---: | --- |
| `Osaka Subway 10 series` | 32 / 0 | 独立车型 root |
| `Osaka Subway Type 100` | 9 / 0 | 地下铁历史车辆；不是 New Tram 100 系 |
| `Osaka Subway 20 series` | 37 / 0 | 独立车型 root |
| `Osaka Subway 21 series` | 43 / 0 | New 20 系列；按线路车型拆分 |
| `Osaka Subway 22 series` | 26 / 0 | New 20 系列；按线路车型拆分 |
| `Osaka Subway 23 series` | 32 / 0 | New 20 系列；按线路车型拆分 |
| `Osaka Subway 24 series` | 22 / 0 | New 20 系列；已退出中央线 |
| `Osaka Subway 25 series` | 14 / 0 | New 20 系列；按线路车型拆分 |
| `Osaka Subway 30 series` | 19 / 0 | 独立车型 root |
| `Osaka Subway 30000 series` | 2 / 3 | 子分类继续由 Stage 03 递归 |
| `Osaka Metro 30000A series` | 30 / 1 | 30000A 独立车型级 root |
| `Osaka Metro 400 series` | 158 / 0 | 独立车型 root |
| `Osaka Subway 50 series` | 3 / 0 | 地下铁历史车辆 |
| `Osaka Subway 60 series` | 6 / 0 | 地下铁历史车辆 |
| `Osaka Subway 66 series` | 69 / 0 | 独立车型 root |
| `Osaka Subway 70 series` | 28 / 0 | 独立车型 root |
| `Osaka Subway 80 series` | 11 / 0 | 独立车型 root |

`Osaka Municipal Transportation Bureau 100 series` 与 `200 series` 属于 New Tram
AGT，不纳入本批地下铁目录。东京地下铁 root 内的海外/他社转让子树通过
`SERIES_CATEGORY_EXCLUDE_PATTERNS` 排除，防止人工锁定的 operator 污染转让车辆。

### 第四批 root 核验台账：地铁直通私铁及其现役车辆

2026-08-31 以小田急、东武、西武和东急官方现役车辆介绍为清单基线，通过
Commons API 核验 `categoryinfo`、直接父分类和主要子分类。所有采用的 root
都是非空车型级分类；子型 root 的父分类能回到对应车族。全部车型写入
`manual_series_catalog.csv` 并进入 `crawler.series_test_scope`。近似子型仍保留独立基础
series/root 以维持来源和递归边界，Stage 08 再合并为车族级训练标签。

小田急采用以下 9 个 root，直接父分类均包含
`Electric multiple units of Odakyu Electric Railway` 或 `Odakyu Romancecar`：

| root | 当日 files/subcats | 处理 |
| --- | ---: | --- |
| `Odakyu 1000 series` | 115 / 2 | 现役通勤车；保留历史千代田线直通资料 |
| `Odakyu 2000 series` | 41 / 0 | 现役通勤车 |
| `Odakyu 3000 series (II)` | 131 / 1 | 现役通勤车 |
| `Odakyu 4000 series (II)` | 56 / 4 | 千代田线与常磐缓行线直通；进 scope |
| `Odakyu 5000 series (II)` | 37 / 1 | 现役通勤车 |
| `Odakyu 8000 series` | 110 / 1 | 小田急原车谱 root；西武8000以 merge 追加 |
| `Odakyu 30000 series EXE` | 60 / 1 | Romancecar EXE / EXEα |
| `Odakyu 60000 series MSE` | 64 / 2 | 千代田线/有乐町线直通特急；进 scope |
| `Odakyu 70000 series GSE` | 21 / 1 | Romancecar GSE |

东武采用以下 23 个 root。50000、70000、9000、10000 和 20000 系列的父 root
含有独立子型；为了避免父 root 递归与子型 root 重复，父 series 通过
`SERIES_CATEGORY_EXCLUDE_PATTERNS` 跳过已单独登记的子树：

| root | 当日 files/subcats | 处理 |
| --- | ---: | --- |
| `Tobu N100 series` | 49 / 1 | SPACIA X |
| `Tobu 100 series` | 159 / 0 | SPACIA |
| `Tobu 200 series` | 67 / 1 | 特急车 |
| `Tobu 500 series` | 46 / 1 | Revaty |
| `Tobu 634 series` | 35 / 0 | 6050 系改造观光车 |
| `Tobu 6050 series` | 121 / 2 | 普通6050系；子分类为野岩鉄道/会津鉄道6050系，同一 root 登记三家 operator 且不锁定单一 operator |
| `Tobu 8000 series` | 243 / 1 | 800/850 型包含在同车族 |
| `Tobu 9000 series` | 43 / 1 | 有乐町/副都心线直通；排除 9050 子树；进 scope |
| `Tobu 9050 series` | 13 / 0 | `Tobu 9000 series` 子型；进 scope |
| `Tobu 10000 series` | 23 / 14 | 排除单独登记的 10030/10080 子树 |
| `Tobu 10030 series` | 28 / 14 | 10030/10050 车族 |
| `Tobu 10080 series` | 2 / 0 | merge 到东武10030型 |
| `Tobu 20000 series` | 22 / 4 | 历史日比谷线直通车；保留 20050/20070 子树，排除已单列的 20400 与 Alpico 20100 子树 |
| `Tobu 20400 series` | 10 / 4 | 20000 系改造车；子分类为 20410--20440 |
| `Tobu 30000 series` | 71 / 0 | 历史半藏门线直通车 |
| `Tobu 50000 series` | 41 / 3 | 地上基本型；排除 50050/50070/50090 子树 |
| `Tobu 50050 series` | 64 / 0 | 半藏门线/东急田园都市线直通；进 scope |
| `Tobu 50070 series` | 34 / 0 | 有乐町/副都心线直通；进 scope |
| `Tobu 50090 series` | 39 / 0 | TJ Liner |
| `Tobu 60000 series` | 33 / 7 | Urban Park Line |
| `Tobu 70000 series` | 25 / 1 | 日比谷线直通；排除 70090 子树；进 scope |
| `Tobu 70090 series` | 21 / 0 | TH Liner；日比谷线直通；进 scope |
| `Tobu 80000 series` | 24 / 0 | Urban Park Line |

东武90000系在核验日尚未开始营业，Commons 也没有非空车型 root，因此暂不写入
人工目录；待营业照片形成稳定分类后再核验。

2026-09-04 追加核验 `Alpico Kotsu 20100 series`，Commons API 返回
23 files / 0 subcats，直接父分类同时包含 `Rolling stock of Alpico Kotsu`
与 `Tobu 20000 series`。该 root 以 `merge` 并入基础 `東武20000系`，
Stage 08 再根据锁定的 `operator_en=Alpico Kotsu` 拆为独立训练标签
`アルピコ交通20100形`。东武父 root 递归时排除该子树，避免重复抓取。

西武采用以下 14 个 root，直接父分类均能回到
`Electric multiple units of Seibu Railway`：

| root | 当日 files/subcats | 处理 |
| --- | ---: | --- |
| `Seibu 001 series` | 82 / 2 | Laview |
| `Seibu 10000 series` | 70 / 2 | New Red Arrow |
| `Seibu 101 series` | 176 / 8 | 现役 101 车族 |
| `Seibu 2000 series` | 181 / 2 | 2000 / New 2000 车族 |
| `Seibu 20000 series` | 53 / 3 | 现役通勤车 |
| `Seibu 30000 series` | 85 / 1 | Smile Train |
| `Seibu 4000 series` | 45 / 2 | 近郊型 |
| `Seibu 40000 series` | 67 / 2 | 有乐町/副都心/东横线直通；进 scope |
| `Seibu 6000 series` | 130 / 1 | 有乐町/副都心/东横线直通；6000/6050 暂不拆分；进 scope |
| `Seibu 7000 series` | 62 / 1 | `Ex-Tōkyū 9000 series`；merge 到东急9000系，暂不做新基础标签 |
| `Seibu 8000 series` | 21 / 1 | `Ex-Odakyu 8000 series`；merge 到小田急8000形，暂不做新基础标签 |
| `Seibu 8500 series` | 30 / 1 | 山口线胶轮式车辆 |
| `Seibu 9000 series` | 35 / 5 | 现役通勤车 |
| `Seibu L00 series` | 5 / 1 | 山口线新型胶轮式车辆 |

东急现役官方车族采用以下 root，并为西武7000系保留已退役的东急9000系
原始谱系 root：

| root | 当日 files/subcats | 处理 |
| --- | ---: | --- |
| `Tōkyū 1000 series` | 70 / 1 | 现役池上/多摩川线；历史日比谷线直通 |
| `Tōkyū 3000 series (II)` | 50 / 0 | 南北线/都营三田线直通；进 scope |
| `Tōkyū 5000 series (II)` | 76 / 3 | 子分类为 5050、5050-4000、5080；全车族合并为东急5000系并进 scope |
| `Tōkyū 6000 series (II)` | 49 / 0 | 大井町线 |
| `Tōkyū 7000 series (II)` | 33 / 0 | 池上/多摩川线 |
| `Tokyu 2020 series` | 27 / 3 | 半藏门线直通；排除 3020/6020 子树；进 scope |
| `Tokyu 3020 series` | 20 / 0 | 南北线/都营三田线直通；进 scope |
| `Tokyu 6020 series` | 19 / 0 | 大井町线 |
| `Tōkyū 9000 series` | 30 / 20 | 东急已退役原车谱；西武7000以 merge 追加 |

扩展后共 52 个规范 series 均纳入 scope。Stage 08 将东武50000/50050/50070/50090、
东武9000/9050、东武70000/70090 以及东急2020/3020/6020 分别合并为车族级训练标签。
东武20000系中的 Alpico 来源则按 operator 拆为 `アルピコ交通20100形`。

## CSV 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `source_series` | 是 | 来源页面使用的车型名或别名，仅用于人工辨识和冲突报错 |
| `series` | 是 | 下游基础规范车型名，须按下节的车辆谱系规则填写 |
| `entry_kind` | 是 | `new` 新建规范车型；`merge` 并入已存在的规范车型 |
| `wiki_title` | 是 | 日文 Wikipedia 页面标题，可包含章节锚点 |
| `full_name` | 否 | 完整车型名；留空时使用 `wiki_title` |
| `status` | 否 | 例如 `現役`、`廃止`；人工行不再经过 Stage 01 状态过滤 |
| `type` | 是 | `pipeline.constants.POWER_TYPE_MAP` 支持的日文车辆类型 |
| `subtype` | 否 | 可选车辆子类型 |
| `operator_jp` | 是 | 日文运营公司；多个值用 `\|` 分隔 |
| `operator_en` | 是 | 英文运营公司；顺序和数量必须与日文列一致 |
| `commons_root_category` | 是 | 已核验的 Commons category 名，不含 `Category:` 前缀 |
| `submodel` | 否 | 整个 root 都稳定成立时才填 |
| `bandai` | 否 | 整个 root 都稳定成立时才填 |
| `special_formation` | 否 | 整个 root 都稳定成立时才填 |
| `special_livery` | 否 | 整个 root 都稳定成立时才填 |

Stage 06 的这些字段是单值字段。若一个人工 root 填了多个运营公司，它们仍作为
车型来源列表保留，但不会折叠成 operator 锁定值。需要锁定 `operator_jp` /
`operator_en` 时，每个 root 应只对应一个运营公司。

## `new`、`merge` 与第八阶段细分

全新且没有既有车辆谱系的车型使用 `entry_kind=new`，`series` 填期望的基础标签。
名称相同但实际无谱系关系的车辆也使用 `new`，并在 `series` 中加入运营公司限定，
避免同名异车冲突。

别名同车、从 JR/JNR 继承或与既有型号属于同一车辆谱系时，必须使用
`entry_kind=merge`，并把 `series` 写成已有基础标签。Stage 02 的处理方式是：

- 相同 `series + commons_root_category` 合并 operator 和手工 metadata；
- root 不同则保留多个抓取入口，但最终仍使用同一个基础 `series`。

即使第三部门版本的涂装、前脸或内装差异足以形成独立识别类，也不能在本目录拆出
新的基础 `series`。所有这种视觉细分只能在 Stage 08 根据 `operator_en`、
`submodel`、`bandai`、特殊编组或涂装 metadata 生成 `fine_grained_series`。
细分前必须保证 Stage 06 metadata 正确。

如果已有 JR root 已完整覆盖第三部门车辆，可以不新增重复入口；但需确认未来新增的
Commons 子分类不会被旧 checkpoint 漏掉，并确认 operator metadata 已经可靠。
最终三语 operator 规范信息仍以数据库 `label_metadata` 为准。

## Stage 06 手工 metadata 锁定

配置项 `llm_labeling.locked_manual_metadata_columns` 指定哪些非空人工值不能被 LLM
覆盖。当前配置为：

```yaml
llm_labeling:
  locked_manual_metadata_columns: ["operator_jp", "operator_en"]
```

数据流如下：

1. Stage 02 把人工目录行中的非空 Stage 06 字段保存为 `manual_metadata_json`；
2. Stage 03 将其写入每张图片的 `images.manual_metadata_json`；
3. Stage 06 仍按 category path 生成 LLM checkpoint；
4. 写回每张图片时，仅用配置列表中且该图片确有人工值的字段覆盖 LLM 结果；
5. Stage 06 每次运行还会对已处理图片重新强制应用锁定值，因此之后修改锁定配置也能
   生效；没有人工值的字段继续使用 LLM 结果。

允许配置的字段仅限 `submodel`、`bandai`、`operator_en`、`operator_jp`、
`special_formation`、`special_livery`。不要锁定会在同一 root 内变化的字段；例如
一个 root 混有多个番台时不能填写并锁定 `bandai`。

## 示例

```csv
source_series,series,entry_kind,wiki_title,full_name,status,type,subtype,operator_jp,operator_en,commons_root_category,submodel,bandai,special_formation,special_livery
架空鉄道2000形,架空鉄道2000形,new,架空鉄道2000形電車,,現役,電車,,架空鉄道,Example Railway,Example Railway 2000 series,,,,
第三部门701系,701系,merge,第三部门701系電車,,現役,電車,,第三部门鉄道,Third Sector Railway,Third Sector Railway 701 series,,,,
1000形,架空鉄道1000形,new,架空鉄道1000形電車,,現役,電車,,架空鉄道,Example Railway,Example Railway 1000 series,,,,
```
