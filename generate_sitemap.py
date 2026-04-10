#!/usr/bin/env python3
"""
Generate a clean sitemap.xml for brand-genome.github.io
Run this script from your project root after Hugo builds the site,
OR run it standalone - it reads process_data_v2.py's slugify logic
to generate brand slugs directly from your source data.

Usage: python3 generate_sitemap.py
Output: static/sitemap.xml  (place this in your Hugo static/ folder)
"""

import json
import re
import unicodedata
from datetime import date
from pathlib import Path

BASE_URL = "https://brand-genome.github.io"
TODAY = date.today().isoformat()

# ── same slugify as your process_data scripts ──────────────────────────────
def slugify(text):
    if not text:
        return "unknown"
    text = unicodedata.normalize('NFKD', str(text))
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text or "unknown"

urls = []

def add(path, priority="0.5", changefreq="monthly"):
    urls.append({
        "loc": f"{BASE_URL}{path}",
        "lastmod": TODAY,
        "changefreq": changefreq,
        "priority": priority
    })

# ── 1. Static pages ────────────────────────────────────────────────────────
add("/", priority="1.0", changefreq="weekly")
add("/about/", priority="0.8")

# ── 2. Brand pages ─────────────────────────────────────────────────────────
# Try to read brand names from hugo_stats or reconstruct from data files
brand_slugs = set()

# Method 1: read from hugo_stats.json classes/ids (won't have brands)
# Method 2: scan content/brands/ directory if it exists
content_brands = Path("content/brands")
if content_brands.exists():
    for md_file in content_brands.glob("*.md"):
        slug = md_file.stem
        if slug != "_index":
            brand_slugs.add(slug)
    print(f"Found {len(brand_slugs)} brands from content/brands/")

# Method 3: read from process_data output / data files
if not brand_slugs:
    # Try reading from any brands json data file
    for data_file in ["data/brands.json", "data/summary.json"]:
        p = Path(data_file)
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            if isinstance(d, list):
                for item in d:
                    name = item.get("name") or item.get("title") or item.get("slug")
                    if name:
                        brand_slugs.add(slugify(name))
                print(f"Found {len(brand_slugs)} brands from {data_file}")
                break

if brand_slugs:
    for slug in sorted(brand_slugs):
        add(f"/brands/{slug}/", priority="0.8", changefreq="monthly")
else:
    print("WARNING: No brand pages found. Make sure to run this from your Hugo project root.")
    print("Expected: content/brands/*.md files to exist")

# ── 3. Taxonomy list pages ─────────────────────────────────────────────────
taxonomies = [
    "sectors", "regions", "years", "languages", "tags",
    "industries", "countries",
    "revenue_buckets", "operating_income_buckets", "net_profit_buckets",
    "employees_buckets", "total_assets_buckets", "total_equity_buckets",
    "market_cap_buckets",
    "products_or_materials_produced", "products",
    "headquarters_locations", "subsidiaries",
    "foundation_dates", "foundation_year_buckets",
    "dominant_colors", "color_tones",
    "lightings", "perspectives", "image_backgrounds",
    "color_schemes", "photography_genres", "concepts",
    "depths", "image_effects",
    "hair_styles", "facial_expressions", "clothing_styles",
    "clothing_colors", "posings", "gazes", "body_sections",
    "logo_elements", "brand_colors", "typographies", "imagery_styles",
]

for tax in taxonomies:
    add(f"/{tax}/", priority="0.6", changefreq="weekly")

# ── 4. Write sitemap.xml ───────────────────────────────────────────────────
out_path = Path("static/sitemap.xml")
out_path.parent.mkdir(exist_ok=True)

lines = ['<?xml version="1.0" encoding="UTF-8"?>']
lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

for u in urls:
    lines.append("  <url>")
    lines.append(f"    <loc>{u['loc']}</loc>")
    lines.append(f"    <lastmod>{u['lastmod']}</lastmod>")
    lines.append(f"    <changefreq>{u['changefreq']}</changefreq>")
    lines.append(f"    <priority>{u['priority']}</priority>")
    lines.append("  </url>")

lines.append("</urlset>")

sitemap_content = "\n".join(lines)
out_path.write_text(sitemap_content)

print(f"\n✅ Generated static/sitemap.xml")
print(f"   Total URLs: {len(urls)}")
print(f"   Brand pages: {len(brand_slugs)}")
print(f"   Taxonomy pages: {len(taxonomies)}")
print(f"\nNext step: commit and push static/sitemap.xml to your repo")