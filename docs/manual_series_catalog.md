# 人工车型目录

`config/manual_series_catalog.csv` 用于直接登记车型与 Wikimedia Commons root，
适合车型列表页面分散、Commons 前缀不规则的私铁和第三部门车辆。目录中的行由
Stage 01 追加到自动解析结果，所填 root 由 Stage 02 直接采用，不执行 root 搜索。

## 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `source_series` | 是 | 来源页面使用的车型名或别名，仅用于人工辨识和冲突报错 |
| `series` | 是 | 下游数据库和训练标签使用的规范车型名，必须避免同名异车冲突 |
| `entry_kind` | 是 | `new` 新建规范车型；`merge` 并入此前已有的规范车型 |
| `wiki_title` | 是 | 日文 Wikipedia 页面标题，可指向分散的单车型页面 |
| `full_name` | 否 | 完整车型名；留空时使用 `wiki_title` |
| `status` | 否 | 例如 `现役`、`废止`；人工行不会再经过 Stage 01 状态过滤 |
| `type` | 是 | 必须是 `pipeline.constants.POWER_TYPE_MAP` 支持的日文车辆类型 |
| `subtype` | 否 | 可选的车辆子类型 |
| `operator_jp` | 是 | 日文运营公司；多个值用 `|` 分隔 |
| `operator_en` | 是 | 英文运营公司；多个值用 `|` 分隔，数量及顺序须与日文列一致 |
| `commons_root_category` | 是 | Commons category 名，不含 `Category:` 前缀 |

## 冲突处理

### 全新车型

使用 `entry_kind=new`，`series` 直接填写期望的规范标签。

### 同名异车

不要复用已有 `series`。在规范标签中加入运营公司限定，例如来源名均为
`1000形` 时，可将新车型的 `series` 写为 `某某鉄道1000形`。如果 `new` 行与
已有规范 `series` 冲突，Stage 01 会中断并提示修正。

### 别名同车或第三部门沿用 JR 车型

使用 `entry_kind=merge`，把 `series` 写成现有 JR 规范标签，
`source_series` 保留第三部门页面中的名称或别名。Stage 02 按以下方式处理：

- 与已有条目解析到相同 `series + commons_root_category`：合并 operator，避免重复抓取。
- root 不同：保留为同一规范车型的另一个 Commons 抓取入口，最终仍使用同一标签。

即使第三部门版本的涂装、前脸或内装差异足以形成独立识别类，也不能在本目录
新建基础 `series`。这类差异仍须 `merge` 到车辆谱系对应的 JR 基础车型，并在
Stage 08 依据 `operator_en`、`submodel`、`bandai`、特殊编组或涂装 metadata
生成 `fine_grained_series`。只有名称碰巧相同、实际不存在车辆谱系关系的同名异车，
才使用运营公司限定的新 `series`。

如果已有 JR 条目和 Commons 分类已经完整覆盖该第三部门运营公司及图片，不需要
再添加人工行。operator 的最终三语规范信息仍以数据库 `label_metadata` 为准。

## 示例

```csv
source_series,series,entry_kind,wiki_title,full_name,status,type,subtype,operator_jp,operator_en,commons_root_category
架空鉄道2000形,架空鉄道2000形,new,架空鉄道2000形電車,,现役,電車,,架空鉄道,Example Railway,Example Railway 2000 series
第三部门701系,701系,merge,第三部门701系電車,,现役,電車,,第三部门鉄道,Third Sector Railway,Third Sector Railway 701 series
1000形,架空鉄道1000形,new,架空鉄道1000形電車,,现役,電車,,架空鉄道,Example Railway,Example Railway 1000 series
```

编辑后运行：

```bash
python pipeline_entry.py --stages "1 2 3"
```
