import httpx
import re
import json
import os
import pandas as pd
import utils
import asyncio
import constants
logger = utils.get_logger("stage_01_model_parsing")


MANUAL_ENTRY_KINDS = {"new", "merge"}
MANUAL_SERIES_REQUIRED_COLUMNS = {
    "source_series",
    "series",
    "entry_kind",
    "wiki_title",
    "full_name",
    "status",
    "type",
    "subtype",
    "operator_jp",
    "operator_en",
    "commons_root_category",
}


def fetch_wikitext(page_title: str) -> str:
    """获取页面的原始Wikitext"""
    url = "https://ja.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content",     # 返回原始wikitext
        "rvslots": "main",
        "format": "json",
    }
    resp = httpx.get(url, params=params)
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page["revisions"][0]["slots"]["main"]["*"]

async def _fetch_one(
    client: httpx.AsyncClient, operator_jp: str, operator_en: str, page_title: str
) -> tuple[str, str, str, str]:
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    }
    resp = await client.get("https://ja.wikipedia.org/w/api.php", params=params)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    logger.info(f"页面：{page_title} 请求成功")
    return page_title, operator_jp, operator_en, page["revisions"][0]["slots"]["main"]["*"]

async def fetch_all(operators: list[tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
    async with httpx.AsyncClient(headers=constants.HEADERS, timeout=30) as client:
        results = await asyncio.gather(*[_fetch_one(client, jp, en, page) for jp, en, page in operators])
    # {page_title: (operator_jp, operator_en, wikitext)}
    return {page: (jp, en, wt) for page, jp, en, wt in results}


# ================== 解析车种信息 ==================

def parse_vehicle_wikitext(lines: list[str]) -> list[dict]:
    link_re = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
    definition_label_re = re.compile(r'^;\s*\[\[[^\]]+\]\]\s*（([^）]+)）')

    # 保留既有车型格式，并补充 0系及 N700S系等数字后的字母后缀车型。
    series_re = re.compile(
        r'^(?:'
        r'[A-Za-z゠-ヿ一-鿿\d][A-Za-z゠-ヿ一-鿿\-]*\d+(?:系|形)?'
        r'|[A-Za-z゠-ヿ一-鿿\-]*\d+[A-Za-z]+(?:系|形)'
        r'|\d(?:系|形)'
        r')$'
    )

    results = []
    current_h2 = ""
    current_h3 = ""
    current_h4 = ""
    current_subtype = ""
    # 跳过 概要，脚注，相关项等非车种列表部分
    skip_sections = constants.WIKI_PAGE_SKIP_SECTIONS

    for line in lines:
        line = line.rstrip('\n')

        m = re.match(r'^== (.+?) ==$', line)
        if m:
            current_h2 = m.group(1)
            current_h3 = ""
            current_h4 = ""
            current_subtype = ""
            continue

        m = re.match(r'^=== (.+?) ===$', line)
        if m:
            current_h3 = m.group(1)
            current_h4 = ""
            current_subtype = ""
            continue

        m = re.match(r'^==== (.+?) ====$', line)
        if m:
            current_h4 = m.group(1)
            continue

        if current_h2 in skip_sections:
            continue

        stripped_line = line.lstrip()
        is_bullet_item = stripped_line.startswith('*')
        is_definition_item = stripped_line.startswith(';')
        if not is_bullet_item and not is_definition_item:
            continue

        # 单星开头的粗体行（* '''xxx'''）才更新 subtype，** 及以上层级不更新
        subtype = re.match(r'^\*\s*\'\'\'(.+?)\'\'\'', line)
        if subtype:
            current_subtype = subtype.group(1).strip().strip('[]')

        status_heading = current_h2
        type_heading = current_h3
        if is_definition_item and current_h2 == "新幹線車両":
            status_heading = current_h3
            type_heading = (
                "新幹線電車"
                if current_h3 == "現有車両" or current_h4 == "電車"
                else "その他新幹線車両"
            )

        definition_label_match = (
            definition_label_re.match(stripped_line) if is_definition_item else None
        )

        for m in link_re.finditer(line):
            page  = m.group(1)
            label = m.group(2) or page
            label = label.split('・')[0].split('（')[0].strip()
            if definition_label_match:
                explicit_label = definition_label_match.group(1).strip()
                if series_re.match(explicit_label):
                    label = explicit_label

            if series_re.match(label):
                results.append({
                    "series":     label,
                    "wiki_title": page,
                    "status":     constants.STATUS_MAP.get(status_heading, status_heading),
                    "type":       type_heading,
                    "subtype":    current_subtype,
                })

    return results

def canonical_vehicle_key(entry: dict) -> tuple[str, str]:
    wiki_base = entry["wiki_title"].split("#", 1)[0]
    return entry["series"], wiki_base


def score_entry(entry: dict) -> int:
    return sum(bool(entry.get(field)) for field in ["type", "subtype", "status", "wiki_title"])


def add_unique(items: list, item):
    if item not in items:
        items.append(item)
        

def _as_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return [value]


def _extend_unique(items: list, values) -> None:
    for value in _as_list(values):
        if value not in items:
            items.append(value)


def _row_quality(row: pd.Series) -> tuple:
    title = row.get("wiki_title") or ""
    full_name = row.get("full_name") or ""
    return (
        title != row.get("series"),
        len(title),
        len(full_name),
        pd.notna(row.get("subtype")),
    )


def _split_manual_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split("|") if item.strip()]


def load_manual_series_catalog(path: str | os.PathLike) -> pd.DataFrame:
    """Load explicitly curated series rows that bypass Wikipedia/root discovery."""
    catalog_path = utils.join_project_root(path)
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"未找到人工车型目录：{catalog_path}")

    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    missing = MANUAL_SERIES_REQUIRED_COLUMNS - set(catalog.columns)
    if missing:
        raise ValueError(f"人工车型目录缺少列：{sorted(missing)}")
    if catalog.empty:
        return pd.DataFrame(
            columns=[
                "source_series",
                "series",
                "entry_kind",
                "wiki_title",
                "full_name",
                "status",
                "type",
                "subtype",
                "operator_page_title",
                "operator_jp",
                "operator_en",
                "commons_root_category",
            ]
        )

    catalog = catalog.copy()
    for col in MANUAL_SERIES_REQUIRED_COLUMNS:
        catalog[col] = catalog[col].str.strip()
    catalog["operator_page_title"] = pd.Series(
        [[] for _ in range(len(catalog))],
        index=catalog.index,
        dtype=object,
    )

    for row_number, row in catalog.iterrows():
        csv_row = row_number + 2
        required_values = [
            "source_series",
            "series",
            "entry_kind",
            "wiki_title",
            "type",
            "operator_jp",
            "operator_en",
            "commons_root_category",
        ]
        empty = [col for col in required_values if not row[col]]
        if empty:
            raise ValueError(f"人工车型目录第 {csv_row} 行存在空值：{empty}")
        if row["entry_kind"] not in MANUAL_ENTRY_KINDS:
            raise ValueError(
                f"人工车型目录第 {csv_row} 行 entry_kind 必须是 new 或 merge"
            )
        if row["type"] not in constants.POWER_TYPE_MAP:
            raise ValueError(
                f"人工车型目录第 {csv_row} 行 type 不受支持：{row['type']!r}"
            )
        if row["commons_root_category"].startswith("Category:"):
            raise ValueError(
                f"人工车型目录第 {csv_row} 行 commons_root_category 不应包含 Category: 前缀"
            )

        operators_jp = _split_manual_list(row["operator_jp"])
        operators_en = _split_manual_list(row["operator_en"])
        if len(operators_jp) != len(operators_en):
            raise ValueError(
                f"人工车型目录第 {csv_row} 行 operator_jp/operator_en 数量不一致"
            )
        catalog.at[row_number, "operator_jp"] = operators_jp
        catalog.at[row_number, "operator_en"] = operators_en
        if not row["full_name"]:
            catalog.at[row_number, "full_name"] = row["wiki_title"]

    duplicate_keys = catalog.duplicated(
        subset=["source_series", "wiki_title", "commons_root_category"],
        keep=False,
    )
    if duplicate_keys.any():
        duplicates = catalog.loc[
            duplicate_keys,
            ["source_series", "wiki_title", "commons_root_category"],
        ].to_dict(orient="records")
        raise ValueError(f"人工车型目录存在重复来源条目：{duplicates}")

    return catalog[
        [
            "source_series",
            "series",
            "entry_kind",
            "wiki_title",
            "full_name",
            "status",
            "type",
            "subtype",
            "operator_page_title",
            "operator_jp",
            "operator_en",
            "commons_root_category",
        ]
    ]


def append_manual_series_catalog(
    parsed_series: pd.DataFrame,
    manual_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Append curated roots while making canonical-label collisions explicit."""
    if manual_catalog.empty:
        return parsed_series

    rows = [parsed_series]
    known_series = set(parsed_series["series"].astype(str))
    for _, manual_row in manual_catalog.iterrows():
        series = manual_row["series"]
        entry_kind = manual_row["entry_kind"]
        if entry_kind == "new" and series in known_series:
            raise ValueError(
                f"人工车型 {manual_row['source_series']!r} 声明为 new，"
                f"但规范 series {series!r} 已存在；同车/别名请改用 merge，"
                "同名异车请为 series 添加运营公司限定"
            )
        if entry_kind == "merge" and series not in known_series:
            raise ValueError(
                f"人工车型 {manual_row['source_series']!r} 要 merge 到不存在的 "
                f"series {series!r}；请检查名称或先添加 new 条目"
            )
        rows.append(manual_row.to_frame().T)
        known_series.add(series)

    return pd.concat(rows, ignore_index=True, sort=False)




# ================== pipeline主函数 ==================

def main(config = None):
    
    # === 初始化 ===
    config = config or utils.load_pipeline_config()
    utils.init_db(config=config)
    logger = utils.get_logger("stage_01_model_parsing")
    active_operatos = config['crawler']['active_operators']
    
    
    # === 获取车型 ===
    operators = [op for op in constants.OPERATORS if op[0] in active_operatos]
    logger.info(f"正在处理的运营公司：{', '.join(op[0] for op in operators)}")
    wikitexts = asyncio.run(fetch_all(operators=operators))
    
    raw_series = []
    for page_title, (operator_jp, operator_en, wt) in wikitexts.items():
        entries = parse_vehicle_wikitext(wt.splitlines("\n"))
        for e in entries:
            e["operator_page_title"] = page_title
            e["operator_jp"] = operator_jp
            e["operator_en"] = operator_en
        raw_series.extend(entries)

    # 同一车型跨 JR 来源页合并；operator/page_title 保留为列表，不再因为重复而丢掉来源。
    merged: dict[tuple[str, str], dict] = {}
    for e in raw_series:
        key = canonical_vehicle_key(e)

        if key not in merged:
            merged[key] = {
                "series": e["series"],
                "wiki_title": e["wiki_title"],
                "status": e["status"],
                "type": e["type"],
                "subtype": e["subtype"],
                "operator_page_title": [e["operator_page_title"]],
                "operator_jp": [e["operator_jp"]],
                "operator_en": [e["operator_en"]],
                "full_name": e["wiki_title"],
            }
            continue

        current = merged[key]
        if score_entry(e) > score_entry(current):
            for field in ["wiki_title", "status", "type", "subtype"]:
                current[field] = e[field]
            current["full_name"] = e["wiki_title"]

        add_unique(current["operator_page_title"], e["operator_page_title"])
        add_unique(current["operator_jp"], e["operator_jp"])
        add_unique(current["operator_en"], e["operator_en"])

    all_series = pd.DataFrame(list(merged.values()))
    
    logger.info(f"共解析出 {len(all_series)} 个车型")
    logger.info('子车型:' + all_series['type'].value_counts().to_string())
    
    all_df = pd.DataFrame(all_series)
    #滤除对象：货车，因为多为货列连挂，难以找到单独的车辆照片，一阶段暂时跳过；客车，理由类似，一阶段保留
    excluding_types = constants.EXCLUDED_TYPES
    filtered_df = all_df[~all_df['type'].isin(excluding_types)]
    #现在滤除二级，对象为旧式营业车和事业用车
    exclduing_subtypes = constants.EXCLUDED_SUBTYPES
    filtered_df = filtered_df[~filtered_df['subtype'].isin(exclduing_subtypes)]
    excluding_statuses = constants.EXCLUDED_STATUSES
    excluded_status = filtered_df['status'].isin(excluding_statuses)
    status_exception = filtered_df['series'].isin(constants.EXCLUDED_STATUS_SERIES_EXCEPTIONS)
    final_df = filtered_df[~excluded_status | status_exception]
    final_df['type'].value_counts()
    final_df['subtype'].value_counts()


    # 按 series 去重：同一车型可能同时出现在多个 JR 来源页，保留一行并合并 operator 信息。

    duplicate_rows = final_df[final_df.duplicated(subset=["series"], keep=False)]

    merged_rows = []
    for _, group in final_df.groupby("series", sort=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0].copy())
            continue

        # 主行选信息量更高的标题；operator/page 信息从所有重复行合并。
        base = group.loc[max(group.index, key=lambda idx: _row_quality(group.loc[idx]))].copy()
        for col in ["operator_page_title", "operator_jp", "operator_en"]:
            merged = []
            for value in group[col]:
                _extend_unique(merged, value)
            base[col] = merged

        for col in ["status", "type", "subtype", "wiki_title", "full_name"]:
            if pd.isna(base[col]) or base[col] == "":
                first_valid = group[col].dropna()
                if not first_valid.empty:
                    base[col] = first_valid.iloc[0]

        merged_rows.append(base)

    final_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    logger.info(f"去重前重复行 {len(duplicate_rows)} 条；去重后剩余重复 series {final_df.duplicated(subset=['series']).sum()} 条")

    manual_catalog = load_manual_series_catalog(
        config["path"]["manual_series_catalog_path"]
    )
    final_df = append_manual_series_catalog(final_df, manual_catalog)
    logger.info(
        "已追加人工车型目录 %d 条；当前共有 %d 个抓取入口、%d 个规范 series",
        len(manual_catalog),
        len(final_df),
        final_df["series"].nunique(),
    )
    
    export_df = final_df.copy()
    # 对 operator这列列表进行json序列化
    logger.info("正在将 operator 列表进行 JSON 序列化")
    for col in ["operator_page_title", "operator_jp", "operator_en"]:
        export_df[col] = export_df[col].apply(lambda v: json.dumps(v, ensure_ascii=False))
    series_list_path = utils.join_data_root(config["path"]["series_list_path"], config=config)
    series_list_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(series_list_path, index=False, encoding="utf-8")
    logger.info(f"车型列表已保存到 {series_list_path},共 {len(export_df)} 条记录")
    
    
    
    
if __name__ == "__main__":
    main()
