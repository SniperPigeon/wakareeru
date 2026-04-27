## 解析日文wikipedia并保存列车种类为JSON格式，为后续爬虫阶段使用
import httpx
from bs4 import BeautifulSoup
import re
import json

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

# 标题-运营者格式
TARGET_PAGES = [
    ("JR東日本の車両形式", "JR東日本"),
    #("JR東海の車両形式", "JR東海"),
]

all_series = []


for page_title, operator in TARGET_PAGES:
    text = fetch_wikitext(page_title)
    # 使用正则表达式提取车种信息
    