# わかれーる — Japanese Train Image Classification Dataset

Fine-grained image classification dataset and model for JR series rolling stock, targeting visually similar variants (e.g. 113系 vs 115系 commuter EMUs).

## Project Goal

Build a labeled image dataset from Wikimedia Commons, then train a fine-grained classifier using DINOv2-Large + SupCon Loss. New series should be addable via prototype-bank retrieval without retraining.

## Architecture Decisions

- **Backbone**: DINOv2-Large (preferred over CLIP — stronger linear separability, patch tokens preserve local detail, better for fine-grained)
- **Head**: Linear probe or prototype retrieval
- **Loss**: CrossEntropy + SupCon Loss
- **Dataset format**: HuggingFace Dataset (feature caching, stratified splits, push_to_hub)
- **Incremental extension**: prototype-bank retrieval so new series don't require retraining

## Repository Layout

```
src/crawler/          # Data collection scripts and notebooks
  model_parse.py      # Wikipedia wikitext parser → CSV label list
  model_parse.ipynb   # Notebook version
  img_crawler.ipynb   # Image crawling (WIP)
data/
  jr_east_series.csv  # Parsed JR East series label list (fields: series, wiki_title, status, type, subtype)
```

## Label System

Labels parsed from Japanese Wikipedia vehicle list pages (H2/H3 headings + `[[link]]` extraction).

CSV fields: `series`, `wiki_title`, `status`, `type`, `subtype`

Commons category naming pattern:
- 新幹線 → `Shinkansen`
- 国鉄 → `JNR`
- JR東日本 → `JR East`
- Top-level category example: `JR East E231`
- Sub-category example: `JR East E231-500 (Yamanote Line)`

`wiki_title_to_commons_prefix()` converts `wiki_title` to a Commons search prefix automatically.

## Crawl Pipeline (planned)

1. **Commons category lookup** — `acprefix` search → top-level category per series, exclude `Interiors of …` and line-specific sub-categories
2. **Recursive image fetch** — `generator=categorymembers`, batch URL + metadata retrieval
3. **Async download** — `asyncio` + `httpx`, concurrency ≤ 5, delay 0.5 s, resume state in SQLite
4. **Quality filter** — short-edge ≥ 224 px, Laplacian variance (blur), perceptual hash dedup
5. **Subject detection** — DINOv2 attention map verification; RMBG foreground crop if needed; crop coords stored in DB, originals unchanged
6. **CLIP confidence scoring** — filter label noise; low-confidence images → Gradio manual review queue
7. **HuggingFace export** — stratified train/val/test split; features: `image`, `label`, `category`, `source`, `confidence`

## Known Gaps

- Only JR East vehicles are parsed so far (~35 active series); JR West, Tokai, Kyushu, Hokkaido, Shikoku, Freight pending
- Katakana series codes (キハ40 etc.) need a katakana → romaji mapping table for Commons search
- Sublabel strategy for variants with significant visual differences (e.g. E231 commuter / suburban / subway) not yet decided

## Dev Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in any required API keys
```

Linting: `ruff check src/`  
Tests: `pytest`  
Python ≥ 3.11 required.

## Working Style Notes

- Only implement tasks explicitly requested — dataset specifics involve details that can't be fully specified in advance
- Do not add features, abstractions, or error handling beyond what the immediate task requires
