#!/usr/bin/env python3
import re
import unicodedata
from datetime import date
from pathlib import Path

BASE_URL = "https://brand-genome.github.io"
TODAY = date.today().isoformat()

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
    urls.append((f"{BASE_URL}{path}", TODAY, changefreq, priority))

# 1. Static pages
add("/", priority="1.0", changefreq="weekly")
add("/about/", priority="0.8")

# 2. Brand pages
brand_slugs = set()
content_brands = Path("content/brands")
if content_brands.exists():
    for md_file in content_brands.glob("*.md"):
        slug = md_file.stem
        if slug != "_index":
            brand_slugs.add(slug)
    print(f"Found {len(brand_slugs)} brands from content/brands/")
else:
    print("WARNING: content/brands/ not found. Run from Hugo project root.")

for slug in sorted(brand_slugs):
    add(f"/brands/{slug}/", priority="0.8", changefreq="monthly")

# 3. Taxonomy list pages
taxonomies = [
    "sectors", "regions", "years", "languages", "tags",
    "industries", "countries",
    "revenue_buckets", "operating_income_buckets", "net_profit_buckets",
    "employees_buckets", "total_assets_buckets", "total_equity_buckets",
    "market_cap_buckets", "products_or_materials_produced", "products",
    "headquarters_locations", "subsidiaries", "foundation_dates",
    "foundation_year_buckets", "dominant_colors", "color_tones",
    "lightings", "perspectives", "image_backgrounds", "color_schemes",
    "photography_genres", "concepts", "depths", "image_effects",
    "hair_styles", "facial_expressions", "clothing_styles",
    "clothing_colors", "posings", "gazes", "body_sections",
    "logo_elements", "brand_colors", "typographies", "imagery_styles",
]
for tax in taxonomies:
    add(f"/{tax}/", priority="0.6", changefreq="weekly")

# 4. Build XML manually (no extra whitespace, no BOM, no shebang in output)
xml_lines = []
xml_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
xml_lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
for loc, lastmod, changefreq, priority in urls:
    xml_lines.append('<url>')
    xml_lines.append(f'<loc>{loc}</loc>')
    xml_lines.append(f'<lastmod>{lastmod}</lastmod>')
    xml_lines.append(f'<changefreq>{changefreq}</changefreq>')
    xml_lines.append(f'<priority>{priority}</priority>')
    xml_lines.append('</url>')
xml_lines.append('</urlset>')

sitemap_content = '\n'.join(xml_lines)

# Write WITHOUT BOM, WITHOUT trailing newline issues
out_path = Path("static/sitemap_new.xml")
out_path.parent.mkdir(exist_ok=True)
out_path.write_bytes(sitemap_content.encode('utf-8'))

# Verify first bytes
with open(out_path, 'rb') as f:
    first_bytes = f.read(20)
print(f"\nFirst bytes (hex): {first_bytes.hex()}")
print(f"First chars: {repr(first_bytes.decode('utf-8'))}")

if first_bytes.startswith(b'<?xml'):
    print("✅ File starts correctly with <?xml")
else:
    print("❌ ERROR: File does not start with <?xml !")

print(f"\n✅ Generated static/sitemap_new.xml")
print(f"   Total URLs: {len(urls)}")
print(f"   Brand pages: {len(brand_slugs)}")
print(f"   Taxonomy pages: {len(taxonomies)}")