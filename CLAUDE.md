# わかれーる — Japanese Train Image Classification Dataset

Fine-grained image classification dataset and model for Japanese rolling stock, targeting visually similar variants (e.g. 113系 vs 115系 commuter EMUs).

## Project Goal

Build a labeled image dataset from Wikimedia Commons, then train a fine-grained classifier using DINOv2-Large + SupCon Loss. New series should be addable via prototype-bank retrieval without retraining.

Coverage is phased:
1. **Phase 1** — JR東日本 (complete)
2. **Phase 2** — JR本州三社 (JR East + JR Central + JR West, in progress)
3. **Phase 3** — All JR companies nationwide
4. **Phase 4** — Private railways across Japan

## Architecture Decisions

- **Backbone**: DINOv2-Large (preferred over CLIP — stronger linear separability, patch tokens preserve local detail, better for fine-grained)
- **Head**: Linear probe or prototype retrieval
- **Loss**: CrossEntropy + SupCon Loss
- **Dataset format**: HuggingFace Dataset (feature caching, stratified splits, push_to_hub)
- **Incremental extension**: prototype-bank retrieval so new series don't require retraining

## Repository Layout

```
src/crawler/          # Data collection scripts and notebooks
  model_parse.ipynb   # Wikipedia wikitext parser → CSV label list
  img_crawler.ipynb   # Commons category lookup and image crawling
data/
  jr_honshu_series.csv  # Parsed label list for JR East/Central/West
                        # fields: series, wiki_title, full_name, status,
                        #         type, subtype, operator_jp, operator_en
```

## Label System

Labels parsed from Japanese Wikipedia vehicle list pages (H2/H3 headings + `[[link]]` extraction).

CSV fields: `series`, `wiki_title`, `full_name`, `status`, `type`, `subtype`, `operator_jp`, `operator_en`

- `full_name` copies `wiki_title` by default; overridden via `OVERRIDES` dict for special cases (e.g. E001形)
- `status` normalized across companies: 現役 / 廃止 / 導入予定
- Duplicates within the same operator removed; entry with more complete type/subtype info retained

Commons category naming pattern:
- 新幹線 → `Shinkansen`
- 国鉄 (inherited by JR) → `JNR`
- JR東日本 → `JR East`, JR東海 → `JR Central`, JR西日本 → `JR West`
- Katakana prefix converted to romaji with space before number/letter code (キハ40 → `Kiha 40`, キヤE195 → `Kiya E195`)
- Special Commons merges handled via `COMMONS_OVERRIDES` dict (e.g. 481系 → `JNR 485`)
- Series with no matching Commons category are dropped from the dataset (label boundary too ambiguous)

## Image Data Considerations

Commons images mix exterior and interior shots. Interior images are noise for an exterior-based classifier and need to be filtered. Candidate approach: ViT-based interior/exterior classifier as a pre-filter before the main crawl pipeline.

## Crawl Pipeline (planned)

1. **Commons category lookup** — `acprefix` search → top-level category per series; series returning empty results dropped
2. **Recursive image fetch** — `generator=categorymembers`, batch URL + metadata retrieval
3. **Async download** — `asyncio` + `httpx`, concurrency ≤ 5, delay 0.5 s, resume state in SQLite
4. **Interior/exterior filter** — ViT-based classifier to remove cabin/interior shots
5. **Quality filter** — short-edge ≥ 224 px, Laplacian variance (blur), perceptual hash dedup
6. **Subject detection** — DINOv2 attention map verification; RMBG foreground crop if needed; crop coords stored in DB, originals unchanged
7. **CLIP confidence scoring** — filter label noise; low-confidence images → Gradio manual review queue
8. **HuggingFace export** — stratified train/val/test split; features: `image`, `label`, `category`, `source`, `confidence`

## Known Issues & Decisions

| Issue | Resolution |
|-------|-----------|
| Digit-first series (113系 etc.) missing | Added `\d` to `series_re` first-char class |
| subtype leaking across H3 sections | Reset `current_subtype` on every H2/H3 change |
| status not normalized for JR Central | Replaced inline replace with `STATUS_MAP` dict |
| 国鉄-inherited cars using wrong operator prefix | Detect via `wiki_title.startswith("国鉄")` → `JNR` |
| Commons merges related series (481→485) | Drop empty-result series; log for manual review |
| Katakana+E-prefix ambiguity (キヤE195) | Generate two prefix candidates, try both, keep first hit |

## Dev Setup

```bash
conda activate wakareeru
pip install -e ".[dev]"
cp .env.example .env
```

Linting: `ruff check src/`  
Tests: `pytest`  
Python ≥ 3.11 required.

## Working Style Notes

- Only implement tasks explicitly requested — dataset specifics involve details that can't be fully specified in advance
- Do not add features, abstractions, or error handling beyond what the immediate task requires
