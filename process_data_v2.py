#!/usr/bin/env python3
"""
Brand Identity Knowledge Graph - Data Processor v2
Merges Twitter data, Brand Guidelines data, Wikidata, and GPT extractions
into Hugo-compatible Markdown files for the Knowledge Graph website.

Updates in v2:
- Added color data (dominant colors, color tones) from Twitter images
- Added revenue buckets taxonomy
- Removed instance_of, official_website from display
- Renamed Twitter -> Promotion terminology
"""

import json
import os
import re
import csv
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import unicodedata

# Try to import spaCy for NER
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Text entity extraction will be skipped.")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/Users/vashu/Desktop/Brand Identity")
OUTPUT_DIR = BASE_DIR / "github_pages_building" / "brand-identity-kg" / "content" / "brands"
DATA_OUTPUT_DIR = BASE_DIR / "github_pages_building" / "brand-identity-kg" / "data"

# Input files
TWITTER_ANNO_FILE = BASE_DIR / "modified_twitter_anno.jsonl"
LLAVA_IMG_FILE = BASE_DIR / "llava_outputs_img2 (1).json"
LLAVA_HUMAN_FILE = BASE_DIR / "llava_outputs_human3 (1).json"
COMPANY_SECTOR_FILE = BASE_DIR / "company_sector.csv"
WIKIDATA_FILE = BASE_DIR / "wikidata_extracted_data_v3.json"
BG_DATASET_FILE = BASE_DIR / "final_brand_dataset_with_pdf_urls.json"
GPT_OUTPUT_DIR = BASE_DIR / "gpt_guideline_extraction" / "gpt4o_output"

# Wikidata properties to EXCLUDE completely
WIKIDATA_EXCLUDE_PROPERTIES = {
    'geonames_id', 'commons_category', 'topic_main_category',
    'instance_of',  # Remove per user request
    'official_website',  # Remove per user request  
    'logo_image',  # Remove from Additional Properties
}

# Financial bucket definitions (in USD) - used for revenue, operating income, net profit
FINANCIAL_BUCKETS = [
    (float('-inf'), -1_000_000_000, "LOSS-OVER-1B"),
    (-1_000_000_000, -100_000_000, "LOSS-100M-1B"),
    (-100_000_000, -10_000_000, "LOSS-10M-100M"),
    (-10_000_000, 0, "LOSS-UNDER-10M"),
    (0, 1_000_000, "UNDER-1M"),
    (1_000_000, 10_000_000, "1M-10M"),
    (10_000_000, 100_000_000, "10M-100M"),
    (100_000_000, 500_000_000, "100M-500M"),
    (500_000_000, 1_000_000_000, "500M-1B"),
    (1_000_000_000, 10_000_000_000, "1B-10B"),
    (10_000_000_000, 50_000_000_000, "10B-50B"),
    (50_000_000_000, 100_000_000_000, "50B-100B"),
    (100_000_000_000, 500_000_000_000, "100B-500B"),
    (500_000_000_000, 1_000_000_000_000, "500B-1T"),
    (1_000_000_000_000, float('inf'), "OVER-1T"),
]

EMPLOYEE_BUCKETS = [
    (0, 100, "UNDER-100"),
    (100, 1_000, "100-1K"),
    (1_000, 10_000, "1K-10K"),
    (10_000, 50_000, "10K-50K"),
    (50_000, 100_000, "50K-100K"),
    (100_000, 500_000, "100K-500K"),
    (500_000, float('inf'), "OVER-500K"),
]

# Alias for backwards compatibility
REVENUE_BUCKETS = FINANCIAL_BUCKETS

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def slugify(text):
    """Convert text to URL-friendly slug."""
    if not text:
        return "unknown"
    text = unicodedata.normalize('NFKD', str(text))
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text or "unknown"

def normalize_company_name(name):
    """Normalize company name for matching."""
    if not name:
        return ""
    return re.sub(r'[^\w]', '', str(name).lower())

def safe_value(val, default="UNKNOWN"):
    """Return value or default if None/empty."""
    if val is None or val == "" or val == []:
        return default
    return val

def clean_list_values(values):
    """Clean list values, removing 'Not applicable' and duplicates."""
    if not values:
        return []
    cleaned = []
    for v in values:
        if v and str(v).lower() not in ['not applicable', 'n/a', 'none', '']:
            cleaned.append(slugify(str(v)))
    return list(set(cleaned))

def extract_media_url(media_str):
    """Extract full URL from media string."""
    if not media_str:
        return None
    match = re.search(r"fullUrl='([^']+)'", str(media_str))
    if match:
        return match.group(1)
    return None

def format_wikidata_value(value):
    """Format Wikidata property values for display."""
    if isinstance(value, dict):
        main_val = value.get('value', str(value))
        qualifiers = []
        if 'point_in_time' in value:
            qualifiers.append(f"as of {value['point_in_time']}")
        if 'start_time' in value:
            qualifiers.append(f"from {value['start_time']}")
        if 'end_time' in value:
            qualifiers.append(f"until {value['end_time']}")
        if qualifiers:
            return f"{main_val} ({', '.join(qualifiers)})"
        return str(main_val)
    elif isinstance(value, list):
        return [format_wikidata_value(v) for v in value]
    return str(value)

def escape_yaml_string(s):
    """Escape string for YAML."""
    if not s:
        return '""'
    s = str(s)
    if any(c in s for c in [':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`', '"', "'"]):
        s = s.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{s}"'
    return s

def parse_revenue(revenue_str):
    """Parse revenue string to numeric value."""
    if not revenue_str:
        return None
    
    revenue_str = str(revenue_str).lower().replace(',', '').replace('$', '').strip()
    
    # Extract number and multiplier
    multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'million': 1_000_000,
        'b': 1_000_000_000,
        'billion': 1_000_000_000,
        't': 1_000_000_000_000,
        'trillion': 1_000_000_000_000,
    }
    
    # Try to find number
    match = re.search(r'([\d.]+)\s*([kmbt]|million|billion|trillion)?', revenue_str)
    if match:
        try:
            num = float(match.group(1))
            mult_key = match.group(2) or ''
            multiplier = multipliers.get(mult_key, 1)
            return num * multiplier
        except:
            pass
    return None

def get_revenue_bucket(revenue_value):
    """Get revenue/financial bucket for a given value."""
    if revenue_value is None:
        return None
    
    for low, high, bucket in FINANCIAL_BUCKETS:
        if low <= revenue_value < high:
            return bucket
    return None

def get_financial_bucket(value):
    """Alias for get_revenue_bucket - works for any financial metric."""
    return get_revenue_bucket(value)

def get_employee_bucket(employee_value):
    """Get employee count bucket for a given value."""
    if employee_value is None:
        return None
    for low, high, bucket in EMPLOYEE_BUCKETS:
        if low <= employee_value < high:
            return bucket
    return None

def format_revenue_display(revenue_value):
    """Format revenue for display."""
    if revenue_value is None:
        return None
    
    if revenue_value >= 1_000_000_000_000:
        return f"${revenue_value / 1_000_000_000_000:.2f}T"
    elif revenue_value >= 1_000_000_000:
        return f"${revenue_value / 1_000_000_000:.2f}B"
    elif revenue_value >= 1_000_000:
        return f"${revenue_value / 1_000_000:.2f}M"
    elif revenue_value >= 1_000:
        return f"${revenue_value / 1_000:.2f}K"
    else:
        return f"${revenue_value:.2f}"

def format_count_display(value):
    """Format count values for display."""
    if value is None:
        return None
    return f"{int(round(value)):,}"

def extract_year_info(value):
    """Extract year/time qualifiers from a Wikidata value."""
    if isinstance(value, dict):
        year_parts = []
        if value.get('point_in_time'):
            year_parts.append(f"as of {value['point_in_time']}")
        if value.get('start_time'):
            year_parts.append(f"from {value['start_time']}")
        if value.get('end_time'):
            year_parts.append(f"until {value['end_time']}")
        return ', '.join(year_parts) if year_parts else '-'

    text = str(value)
    years = re.findall(r'\b\d{4}(?:-\d{2}-\d{2})?\b', text)
    return ', '.join(years[:3]) if years else '-'

def clean_display_value(value):
    """Normalize display values for properties table."""
    text = str(value)
    return text.replace('mailto:', '').replace('|', '\\|')

def build_history_entry(value, bucket_fn, display_fn):
    """Build a normalized history entry with year info and bucket."""
    numeric_source = value.get('value') if isinstance(value, dict) else value
    numeric_value = parse_revenue(numeric_source)
    if numeric_value is None:
        return None

    return {
        'formatted': display_fn(numeric_value),
        'year_info': extract_year_info(value),
        'bucket': bucket_fn(numeric_value),
        'raw': format_wikidata_value(value),
    }

def to_value_list(value):
    """Normalize value into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def extract_connection_values(value):
    """Extract plain string connection values from Wikidata value(s)."""
    extracted = []
    for item in to_value_list(value):
        if isinstance(item, dict):
            item_val = item.get('value')
            if item_val:
                extracted.append(str(item_val))
        elif item:
            extracted.append(str(item))
    return extracted

# =============================================================================
# TEXT ENTITY EXTRACTION (spaCy)
# =============================================================================

def extract_brand_entities(text):
    """Extract brand-related entities from text using spaCy and pattern matching."""
    entities = {
        'logo_elements': [],
        'brand_colors': [],
        'typographies': [],
        'imagery_styles': [],
    }
    
    if not text or not SPACY_AVAILABLE:
        return entities
    
    text = str(text)
    
    # Color patterns
    color_pattern = r'\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|brown|gold|silver|navy|teal|cyan|magenta|maroon|beige|coral|crimson|indigo|violet|turquoise|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3}|rgb\([^)]+\)|pantone\s+\d+|cmyk\([^)]+\))\b'
    colors = re.findall(color_pattern, text.lower())
    entities['brand_colors'] = list(set(slugify(c) for c in colors))
    
    # Typography patterns
    font_pattern = r'\b(arial|helvetica|times|georgia|verdana|roboto|open\s*sans|montserrat|lato|oswald|raleway|poppins|playfair|merriweather|nunito|ubuntu|futura|garamond|bodoni|avenir|gotham|proxima\s*nova|brandon|din|akzidenz|univers|frutiger|gill\s*sans|optima|century\s*gothic|trebuchet|impact|comic\s*sans|courier|monaco|consolas|source\s*sans|inter|manrope|work\s*sans|space\s*grotesk|serif|sans-serif|monospace|display|script|slab)\b'
    fonts = re.findall(font_pattern, text.lower())
    entities['typographies'] = list(set(slugify(f) for f in fonts))
    
    # Logo element patterns
    logo_pattern = r'\b(wordmark|logomark|symbol|icon|emblem|monogram|lettermark|combination\s*mark|abstract|mascot|signature|badge|crest|seal|lockup|stacked|horizontal|vertical|primary|secondary|alternate|favicon|app\s*icon)\b'
    logos = re.findall(logo_pattern, text.lower())
    entities['logo_elements'] = list(set(slugify(l) for l in logos))
    
    # Imagery style patterns
    imagery_pattern = r'\b(photography|illustration|graphic|minimal|bold|clean|modern|classic|vintage|retro|contemporary|professional|playful|serious|dynamic|static|geometric|organic|abstract|realistic|flat|gradient|duotone|monochrome|colorful|muted|vibrant|subtle|dramatic|natural|artificial|candid|staged|lifestyle|product|portrait|landscape|macro|aerial)\b'
    imagery = re.findall(imagery_pattern, text.lower())
    entities['imagery_styles'] = list(set(slugify(i) for i in imagery))
    
    return entities

# =============================================================================
# WEBSITE EXTRACTION
# =============================================================================

def extract_all_websites(item):
    """
    Extract all websites from gpt_web_search, pdf_extracted_url, 
    and official_website_wiki_web_search fields.
    Returns a deduplicated list of website URLs.
    """
    websites = set()
    
    # 1. From gpt_web_search -> official_website
    gpt_web_search = item.get('gpt_web_search', {})
    if gpt_web_search and isinstance(gpt_web_search, dict):
        official_websites = gpt_web_search.get('official_website', [])
        if official_websites:
            if isinstance(official_websites, list):
                for url in official_websites:
                    if url and isinstance(url, str):
                        websites.add(url.strip())
            elif isinstance(official_websites, str):
                websites.add(official_websites.strip())
    
    # 2. From pdf_extracted_url
    pdf_urls = item.get('pdf_extracted_url', [])
    if pdf_urls:
        if isinstance(pdf_urls, list):
            for url in pdf_urls:
                if url and isinstance(url, str):
                    websites.add(url.strip())
        elif isinstance(pdf_urls, str):
            websites.add(pdf_urls.strip())
    
    # 3. From official_website_wiki_web_search
    wiki_web_search = item.get('official_website_wiki_web_search', {})
    if wiki_web_search and isinstance(wiki_web_search, dict):
        # Check for official_website field
        wiki_official = wiki_web_search.get('official_website', [])
        if wiki_official:
            if isinstance(wiki_official, list):
                for url in wiki_official:
                    if url and isinstance(url, str):
                        websites.add(url.strip())
            elif isinstance(wiki_official, str):
                websites.add(wiki_official.strip())
        
        # Also check for any url/urls fields
        for key in ['url', 'urls', 'website', 'websites']:
            val = wiki_web_search.get(key)
            if val:
                if isinstance(val, list):
                    for url in val:
                        if url and isinstance(url, str):
                            websites.add(url.strip())
                elif isinstance(val, str):
                    websites.add(val.strip())
    
    # Filter out empty strings and invalid URLs
    valid_websites = []
    for url in websites:
        if url and len(url) > 5:  # Basic validation
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            valid_websites.append(url)
    
    return sorted(valid_websites)

# =============================================================================
# COLOR DATA PROCESSING
# =============================================================================

def compute_statistics(values):
    """Compute comprehensive statistics for a list of values."""
    if not values:
        return None
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    # Mean
    mean = sum(sorted_vals) / n
    
    # Median
    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        median = sorted_vals[n//2]
    
    # Percentiles (25th, 75th, 90th)
    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (data[c] - data[f]) * (k - f) if c < len(data) else data[f]
    
    p25 = percentile(sorted_vals, 25)
    p75 = percentile(sorted_vals, 75)
    p90 = percentile(sorted_vals, 90)
    
    # Min and Max
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    
    # Standard deviation
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    std_dev = variance ** 0.5
    
    return {
        'mean': round(mean * 100, 1),
        'median': round(median * 100, 1),
        'p25': round(p25 * 100, 1),
        'p75': round(p75 * 100, 1),
        'p90': round(p90 * 100, 1),
        'min': round(min_val * 100, 1),
        'max': round(max_val * 100, 1),
        'std_dev': round(std_dev * 100, 1),
        'count': n
    }

def process_color_data(posts):
    """Process color data from all Twitter posts with comprehensive statistics."""
    # Collect per-image data for each color
    color_coverages = defaultdict(list)  # color -> list of coverage values per image
    tone_values = defaultdict(list)  # tone -> list of values per image
    total_images = 0
    
    for post in posts:
        colour_data = post.get('colour', {})
        if not colour_data:
            continue
        
        total_images += 1
        
        # Collect color coverages for this image
        colors = colour_data.get('colors', {})
        for color_name, color_info in colors.items():
            coverage = color_info.get('coverage', 0) if isinstance(color_info, dict) else 0
            normalized_color = color_name.lower().replace('_', '-')
            color_coverages[normalized_color].append(coverage)
        
        # Collect tone values for this image
        tones = colour_data.get('tones', {})
        for tone_name, tone_value in tones.items():
            tone_values[tone_name.lower()].append(tone_value)
    
    # Compute statistics for each color
    color_stats = {}
    dominant_colors = []
    color_palette = []
    
    if total_images > 0:
        # Calculate stats for colors that appear in at least 10% of images
        for color, coverages in color_coverages.items():
            appearance_rate = len(coverages) / total_images
            if appearance_rate >= 0.1:  # Appears in at least 10% of images
                stats = compute_statistics(coverages)
                if stats and stats['mean'] >= 3:  # At least 3% mean coverage
                    color_stats[color] = {
                        **stats,
                        'appearance_rate': round(appearance_rate * 100, 1)
                    }
        
        # Sort by mean coverage to get dominant colors
        sorted_colors = sorted(color_stats.items(), key=lambda x: -x[1]['mean'])
        
        for color, stats in sorted_colors[:8]:  # Top 8 colors
            dominant_colors.append(slugify(color))
            color_palette.append({
                'color': color.replace('-', ' ').title(),
                'mean': stats['mean'],
                'median': stats['median'],
                'p25': stats['p25'],
                'p75': stats['p75'],
                'p90': stats['p90'],
                'min': stats['min'],
                'max': stats['max'],
                'std_dev': stats['std_dev'],
                'appearance_rate': stats['appearance_rate'],
                'image_count': stats['count']
            })
    
    # Compute tone statistics
    tone_stats = {}
    dominant_tone = None
    
    if total_images > 0:
        for tone, values in tone_values.items():
            stats = compute_statistics(values)
            if stats:
                tone_stats[tone] = stats
        
        # Dominant tone is the one with highest mean
        if tone_stats:
            dominant_tone = max(tone_stats.items(), key=lambda x: x[1]['mean'])[0]
    
    return {
        'dominant_colors': dominant_colors,
        'color_palette': color_palette,
        'color_stats': color_stats,
        'dominant_tone': slugify(dominant_tone) if dominant_tone else None,
        'tone_stats': tone_stats,
        'total_images_analyzed': total_images
    }

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_twitter_data():
    """Load Twitter annotations."""
    print("Loading Twitter annotations...")
    twitter_by_company = defaultdict(list)
    
    if not TWITTER_ANNO_FILE.exists():
        print(f"Warning: {TWITTER_ANNO_FILE} not found")
        return twitter_by_company
    
    with open(TWITTER_ANNO_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    company = normalize_company_name(data.get('new_company', ''))
                    if company:
                        twitter_by_company[company].append(data)
                except json.JSONDecodeError:
                    continue
    
    print(f"  Loaded {len(twitter_by_company)} companies with Twitter data")
    return twitter_by_company

def load_llava_data():
    """Load LLaVa visual annotations."""
    print("Loading LLaVa visual annotations...")
    llava_img = {}
    llava_human = {}
    
    if LLAVA_IMG_FILE.exists():
        with open(LLAVA_IMG_FILE, 'r', encoding='utf-8') as f:
            llava_img = json.load(f)
        print(f"  Loaded {len(llava_img)} image annotations")
    
    if LLAVA_HUMAN_FILE.exists():
        with open(LLAVA_HUMAN_FILE, 'r', encoding='utf-8') as f:
            llava_human = json.load(f)
        print(f"  Loaded {len(llava_human)} human annotations")
    
    return llava_img, llava_human

def load_company_sectors():
    """Load company to sector mapping."""
    print("Loading company sectors...")
    sectors = {}
    
    if not COMPANY_SECTOR_FILE.exists():
        print(f"Warning: {COMPANY_SECTOR_FILE} not found")
        return sectors
    
    with open(COMPANY_SECTOR_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            company = normalize_company_name(row.get('company', ''))
            sector = row.get('Sector', row.get('Short sector', ''))
            if company and sector:
                sectors[company] = [s.strip() for s in sector.split(',')]
    
    print(f"  Loaded {len(sectors)} company-sector mappings")
    return sectors

def load_wikidata():
    """Load Wikidata extractions."""
    print("Loading Wikidata...")
    wikidata_by_company = {}
    
    if not WIKIDATA_FILE.exists():
        print(f"Warning: {WIKIDATA_FILE} not found")
        return wikidata_by_company
    
    with open(WIKIDATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            company = normalize_company_name(item.get('company', ''))
            if company:
                wikidata_by_company[company] = item
    
    print(f"  Loaded {len(wikidata_by_company)} Wikidata entries")
    return wikidata_by_company

def load_bg_dataset():
    """Load Brand Guidelines dataset."""
    print("Loading Brand Guidelines dataset...")
    
    if not BG_DATASET_FILE.exists():
        print(f"Warning: {BG_DATASET_FILE} not found")
        return []
    
    with open(BG_DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} brand entries")
    return data

def load_gpt_guidelines():
    """Load GPT guideline extractions."""
    print("Loading GPT guideline extractions...")
    gpt_by_filename = {}
    
    if not GPT_OUTPUT_DIR.exists():
        print(f"Warning: {GPT_OUTPUT_DIR} not found")
        return gpt_by_filename
    
    for file in GPT_OUTPUT_DIR.glob("*_analysis.json"):
        stem = file.stem.replace('_analysis', '')
        try:
            with open(file, 'r', encoding='utf-8') as f:
                gpt_by_filename[stem.lower()] = json.load(f)
        except:
            continue
    
    print(f"  Loaded {len(gpt_by_filename)} GPT analyses")
    return gpt_by_filename

# =============================================================================
# DATA MERGING
# =============================================================================

def match_gpt_guidelines(pdf_filename, gpt_data):
    """Match PDF filename to GPT analysis."""
    if not pdf_filename:
        return None
    
    stem = Path(pdf_filename).stem.lower()
    
    if stem in gpt_data:
        return gpt_data[stem]
    
    for suffix in ['-en', '-eng', '-english', '_en', '_eng']:
        test_stem = stem.replace(suffix, '')
        if test_stem in gpt_data:
            return gpt_data[test_stem]
    
    for key in gpt_data:
        if stem in key or key in stem:
            return gpt_data[key]
    
    return None

def aggregate_visual_attributes(posts, llava_img, llava_human):
    """Aggregate visual attributes from all posts/images."""
    attributes = {
        'lightings': [],
        'perspectives': [],
        'image_backgrounds': [],
        'color_schemes': [],
        'photography_genres': [],
        'concepts': [],
        'depths': [],
        'image_effects': [],
        'hair_styles': [],
        'facial_expressions': [],
        'clothing_styles': [],
        'clothing_colors': [],
        'posings': [],
        'gazes': [],
        'body_sections': [],
    }
    
    for post in posts:
        media_key = post.get('media_keys', '')
        if not media_key:
            continue
        
        if media_key in llava_img:
            img_data = llava_img[media_key]
            attributes['lightings'].extend(clean_list_values(img_data.get('image_lighting', [])))
            attributes['perspectives'].extend(clean_list_values(img_data.get('perspective', [])))
            attributes['image_backgrounds'].extend(clean_list_values(img_data.get('image_background', [])))
            attributes['color_schemes'].extend(clean_list_values(img_data.get('colors', [])))
            attributes['photography_genres'].extend(clean_list_values(img_data.get('photography_genre', [])))
            attributes['concepts'].extend(clean_list_values(img_data.get('concept', [])))
            attributes['depths'].extend(clean_list_values(img_data.get('depth', [])))
            attributes['image_effects'].extend(clean_list_values(img_data.get('image_effects', [])))
        
        if media_key in llava_human:
            human_data = llava_human[media_key]
            attributes['hair_styles'].extend(clean_list_values(human_data.get('hair_style', [])))
            attributes['facial_expressions'].extend(clean_list_values(human_data.get('facial_expression', [])))
            attributes['clothing_styles'].extend(clean_list_values(human_data.get('clothing_style', [])))
            attributes['clothing_colors'].extend(clean_list_values(human_data.get('clothing_color_palette', [])))
            attributes['posings'].extend(clean_list_values(human_data.get('posing', [])))
            attributes['gazes'].extend(clean_list_values(human_data.get('gaze', [])))
            attributes['body_sections'].extend(clean_list_values(human_data.get('visible_body_section', [])))
    
    for key in attributes:
        attributes[key] = list(set(attributes[key]))
    
    return attributes

def process_wikidata_properties(wikidata):
    """Process Wikidata properties, filtering out excluded ones."""
    if not wikidata or 'properties' not in wikidata:
        return {}, {}, {}, {}
    
    processed = {}
    primary_buckets = {
        'revenue_bucket': None,
        'operating_income_bucket': None,
        'net_profit_bucket': None,
        'employees_bucket': None,
        'total_assets_bucket': None,
        'total_equity_bucket': None,
        'market_cap_bucket': None,
    }

    histories = {
        'revenue_history': [],
        'operating_income_history': [],
        'net_profit_history': [],
        'employees_history': [],
        'total_assets_history': [],
        'total_equity_history': [],
        'market_cap_history': [],
    }

    one_degree = {
        'products_or_materials_produced': [],
        'products': [],
        'headquarters_locations': [],
        'subsidiaries': [],
        'foundation_dates': [],
    }
    
    props = wikidata.get('properties', {})
    
    history_configs = {
        'total_revenue': ('revenue_history', 'revenue_bucket', get_financial_bucket, format_revenue_display),
        'operating_income': ('operating_income_history', 'operating_income_bucket', get_financial_bucket, format_revenue_display),
        'net_profit': ('net_profit_history', 'net_profit_bucket', get_financial_bucket, format_revenue_display),
        'employees': ('employees_history', 'employees_bucket', get_employee_bucket, format_count_display),
        'total_assets': ('total_assets_history', 'total_assets_bucket', get_financial_bucket, format_revenue_display),
        'total_equity': ('total_equity_history', 'total_equity_bucket', get_financial_bucket, format_revenue_display),
        'market_capitalization': ('market_cap_history', 'market_cap_bucket', get_financial_bucket, format_revenue_display),
        'market_cap': ('market_cap_history', 'market_cap_bucket', get_financial_bucket, format_revenue_display),
    }
    
    for key, value in props.items():
        if key in WIKIDATA_EXCLUDE_PROPERTIES:
            continue

        if key in history_configs:
            history_key, bucket_key, bucket_fn, display_fn = history_configs[key]
            entries = []
            for item in to_value_list(value):
                entry = build_history_entry(item, bucket_fn, display_fn)
                if entry and entry.get('bucket'):
                    entries.append(entry)

            if entries:
                histories[history_key].extend(entries)
                if not primary_buckets[bucket_key]:
                    primary_buckets[bucket_key] = entries[0]['bucket']
            continue

        if key == 'product_or_material_produced':
            one_degree['products_or_materials_produced'].extend(extract_connection_values(value))
        elif key == 'product':
            one_degree['products'].extend(extract_connection_values(value))
        elif key == 'headquarters_location':
            one_degree['headquarters_locations'].extend(extract_connection_values(value))
        elif key == 'subsidiary':
            one_degree['subsidiaries'].extend(extract_connection_values(value))
        elif key == 'inception':
            for item in extract_connection_values(value):
                year_match = re.search(r'\b\d{4}\b', item)
                one_degree['foundation_dates'].append(year_match.group(0) if year_match else item)

        formatted = format_wikidata_value(value)
        if formatted:
            processed[key] = formatted

    for key in histories:
        unique = []
        seen = set()
        for entry in histories[key]:
            sig = f"{entry.get('formatted')}|{entry.get('year_info')}|{entry.get('bucket')}"
            if sig not in seen:
                seen.add(sig)
                unique.append(entry)
        histories[key] = unique

    for key in one_degree:
        deduped = []
        seen = set()
        for item in one_degree[key]:
            normalized = slugify(item)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        one_degree[key] = deduped

    return processed, primary_buckets, histories, one_degree

# =============================================================================
# MARKDOWN GENERATION
# =============================================================================

def generate_brand_markdown(brand_info):
    """Generate Hugo-compatible Markdown for a brand."""
    
    frontmatter = {
        'title': brand_info.get('name', 'Unknown Brand'),
        'slug': brand_info.get('slug', 'unknown'),
        'description': brand_info.get('description', ''),
        'date': datetime.now().isoformat(),
        'draft': False,
        
        # Core taxonomies
        'sectors': brand_info.get('sectors', []),
        'regions': brand_info.get('regions', []),
        'years': brand_info.get('years', []),
        'languages': brand_info.get('languages', []),
        'tags': brand_info.get('tags', []),
        
        # Wikidata taxonomies
        'industries': brand_info.get('industries', []),
        'countries': brand_info.get('countries', []),
        
        # Financial bucket taxonomies
        'revenue_buckets': [brand_info.get('revenue_bucket')] if brand_info.get('revenue_bucket') else [],
        'operating_income_buckets': [brand_info.get('operating_income_bucket')] if brand_info.get('operating_income_bucket') else [],
        'net_profit_buckets': [brand_info.get('net_profit_bucket')] if brand_info.get('net_profit_bucket') else [],
        'employees_buckets': [brand_info.get('employees_bucket')] if brand_info.get('employees_bucket') else [],
        'total_assets_buckets': [brand_info.get('total_assets_bucket')] if brand_info.get('total_assets_bucket') else [],
        'total_equity_buckets': [brand_info.get('total_equity_bucket')] if brand_info.get('total_equity_bucket') else [],
        'market_cap_buckets': [brand_info.get('market_cap_bucket')] if brand_info.get('market_cap_bucket') else [],

        # One-degree connection taxonomies
        'products_or_materials_produced': brand_info.get('products_or_materials_produced', []),
        'products': brand_info.get('products', []),
        'headquarters_locations': brand_info.get('headquarters_locations', []),
        'subsidiaries': brand_info.get('subsidiaries', []),
        'foundation_dates': brand_info.get('foundation_dates', []),
        
        # Visual taxonomies
        'lightings': brand_info.get('lightings', []),
        'perspectives': brand_info.get('perspectives', []),
        'image_backgrounds': brand_info.get('image_backgrounds', []),
        'color_schemes': brand_info.get('color_schemes', []),
        'photography_genres': brand_info.get('photography_genres', []),
        'concepts': brand_info.get('concepts', []),
        'depths': brand_info.get('depths', []),
        'image_effects': brand_info.get('image_effects', []),
        
        # Color taxonomies (from image color analysis)
        'dominant_colors': brand_info.get('dominant_colors', []),
        'color_tones': [brand_info.get('dominant_tone')] if brand_info.get('dominant_tone') else [],
        
        # Human visual taxonomies
        'hair_styles': brand_info.get('hair_styles', []),
        'facial_expressions': brand_info.get('facial_expressions', []),
        'clothing_styles': brand_info.get('clothing_styles', []),
        'clothing_colors': brand_info.get('clothing_colors', []),
        'posings': brand_info.get('posings', []),
        'gazes': brand_info.get('gazes', []),
        'body_sections': brand_info.get('body_sections', []),
        
        # Extracted entities from guidelines (spaCy)
        'logo_elements': brand_info.get('logo_elements', []),
        'brand_colors': brand_info.get('brand_colors', []),
        'typographies': brand_info.get('typographies', []),
        'imagery_styles': brand_info.get('imagery_styles', []),
        
        # Metadata
        'wikidata_description': brand_info.get('wikidata_description', ''),
        'wikidata_url': brand_info.get('wikidata_url', ''),
        'has_twitter': brand_info.get('has_twitter', False),
        'has_guidelines': brand_info.get('has_guidelines', False),
        'promotion_image_count': brand_info.get('twitter_post_count', 0),
        'guideline_count': brand_info.get('guideline_count', 0),
        'sample_image_urls': brand_info.get('sample_image_urls', []),
    }
    
    # Build YAML frontmatter
    lines = ['---']
    
    for key, value in frontmatter.items():
        if isinstance(value, list):
            if value:
                lines.append(f'{key}:')
                for item in value:
                    if item:  # Skip None/empty items
                        lines.append(f'  - {escape_yaml_string(item)}')
            else:
                lines.append(f'{key}: []')
        elif isinstance(value, bool):
            lines.append(f'{key}: {str(value).lower()}')
        elif isinstance(value, (int, float)):
            lines.append(f'{key}: {value}')
        else:
            lines.append(f'{key}: {escape_yaml_string(value)}')
    
    lines.append('---')
    lines.append('')
    
    # Build content body
    content_lines = []
    
    # Websites section (union of all sources)
    if brand_info.get('websites'):
        content_lines.append('## Official Websites')
        content_lines.append('')
        for url in brand_info['websites']:
            content_lines.append(f'- [{url}]({url})')
        content_lines.append('')
    
    # Guidelines section (moved before Wikidata per user request)
    if brand_info.get('guidelines'):
        content_lines.append('## Brand Guidelines')
        content_lines.append('')
        for guideline in brand_info['guidelines']:
            year = guideline.get('year', 'UNKNOWN')
            content_lines.append(f'### {year}')
            content_lines.append('')
            
            analysis = guideline.get('analysis', {})
            if isinstance(analysis, dict) and analysis.get('status') == 'success':
                anal_data = analysis.get('analysis', {})
                
                for section in ['logo_information', 'color_information', 'typography_information', 
                               'imagery_photography', 'spacing_layout', 'brand_voice', 'notes']:
                    section_data = anal_data.get(section, [])
                    if section_data:
                        section_title = section.replace('_', ' ').title()
                        content_lines.append(f'**{section_title}:**')
                        content_lines.append('')
                        for item in section_data:
                            if isinstance(item, dict):
                                desc = item.get('description', '')
                                if desc:
                                    content_lines.append(f'- {desc}')
                            else:
                                content_lines.append(f'- {item}')
                        content_lines.append('')
            else:
                content_lines.append('*Guidelines data not available*')
                content_lines.append('')
    
    # Promotion Insights section (renamed from Twitter Insights)
    if brand_info.get('twitter_posts'):
        content_lines.append('## Promotion Insights')
        content_lines.append('')
        
        # Comprehensive Color Analysis
        if brand_info.get('color_palette'):
            total_images = brand_info.get('total_images_analyzed', 0)
            content_lines.append('### Color Analysis')
            content_lines.append(f'*Statistics computed across {total_images} images*')
            content_lines.append('')
            
            # Main color statistics table
            content_lines.append('| Color | Mean |')
            content_lines.append('|-------|------|')
            for color_info in brand_info['color_palette'][:8]:
                content_lines.append(
                    f"| {color_info['color']} | "
                    f"{color_info.get('mean', 0)}% |"
                )
            content_lines.append('')
            
            # Tone statistics
            if brand_info.get('tone_stats'):
                content_lines.append('### Tone Distribution')
                content_lines.append('')
                content_lines.append('| Tone | Mean |')
                content_lines.append('|------|------|')
                for tone, stats in brand_info['tone_stats'].items():
                    content_lines.append(
                        f"| {tone.title()} | "
                        f"{stats.get('mean', 0)}% |"
                    )
                content_lines.append('')
        
        # Show sample posts
        content_lines.append('### Sample Images')
        content_lines.append('')
        
        for i, post in enumerate(brand_info['twitter_posts'][:10]):
            content_lines.append(f'#### Image {i+1}')
            content_lines.append('')
            
            # Image link (using HTML for smaller size)
            media_url = extract_media_url(post.get('media', ''))
            media_key = post.get('media_keys', '')
            if media_url:
                content_lines.append(f'<img src="{media_url}" alt="Image {i+1}" style="max-width: 300px; max-height: 300px; border-radius: 8px;">')
                content_lines.append('')
            elif media_key:
                content_lines.append(f'**Image ID:** `{media_key}`')
                content_lines.append('')
            
            # Caption/content
            content = post.get('content', '').replace('\n', ' ').strip()
            if content:
                content_lines.append(f'> {content[:300]}{"..." if len(content) > 300 else ""}')
                content_lines.append('')
            
            # Visual attributes
            captions = post.get('short_captions', '')
            keywords = post.get('keywords', '')
            if captions or keywords:
                content_lines.append('**Attributes:**')
                if captions:
                    content_lines.append(f'- Caption: {captions}')
                if keywords:
                    content_lines.append(f'- Keywords: {keywords}')
                content_lines.append('')
        
        if len(brand_info['twitter_posts']) > 10:
            content_lines.append(f'*... and {len(brand_info["twitter_posts"]) - 10} more images*')
            content_lines.append('')
    
    # Wikidata properties section (moved to end)
    if brand_info.get('wikidata_properties'):
        content_lines.append('## Additional Properties')
        content_lines.append('')
        if brand_info.get('wikidata_url'):
            content_lines.append('| Property | Value |')
            content_lines.append('|----------|-------|')
            content_lines.append(f"| Wikidata Link | [Open Wikidata]({brand_info['wikidata_url']}) |")
            content_lines.append('')

        content_lines.append('| Property | Value |')
        content_lines.append('|----------|-------|')

        one_degree_rows = [
            ('Product Or Material Produced', brand_info.get('products_or_materials_produced', []), '/products_or_materials_produced/'),
            ('Product', brand_info.get('products', []), '/products/'),
            ('Headquarters Location', brand_info.get('headquarters_locations', []), '/headquarters_locations/'),
            ('Subsidiary', brand_info.get('subsidiaries', []), '/subsidiaries/'),
            ('Foundation Date', brand_info.get('foundation_dates', []), '/foundation_dates/'),
        ]
        for label, values, base_path in one_degree_rows:
            if values:
                links = [f"[{v.replace('-', ' ').title()}]({base_path}{slugify(v)}/)" for v in values[:12]]
                if len(values) > 12:
                    links.append(f"+{len(values)-12} more")
                content_lines.append(f"| {label} | {'; '.join(links)} |")

        for key, value in brand_info['wikidata_properties'].items():
            display_key = key.replace('_', ' ').title()
            if isinstance(value, list):
                display_val = ', '.join(str(v) for v in value[:5])
                if len(value) > 5:
                    display_val += f' (+{len(value)-5} more)'
            else:
                display_val = str(value)
            display_val = clean_display_value(display_val)
            content_lines.append(f'| {display_key} | {display_val} |')
        content_lines.append('')

    def add_history_table(title, amount_col, entries, taxonomy_base):
        if not entries:
            return
        content_lines.append(f'### {title}')
        content_lines.append('')
        content_lines.append(f'| {amount_col} | Year information | Bucket |')
        content_lines.append('|---|---|---|')
        for entry in entries:
            bucket = entry.get('bucket', '')
            bucket_link = f"[{bucket}]({taxonomy_base}{slugify(bucket)}/)" if bucket else '-'
            amount = entry.get('formatted') or '-'
            year_info = (entry.get('year_info') or '-').replace('|', '\\|')
            content_lines.append(f"| {amount} | {year_info} | {bucket_link} |")
        content_lines.append('')
    
    add_history_table('Revenue History', 'Revenue ($)', brand_info.get('revenue_history', []), '/revenue_buckets/')
    add_history_table('Operating Income History', 'Operating Income ($)', brand_info.get('operating_income_history', []), '/operating_income_buckets/')
    add_history_table('Net Profit History', 'Net Profit ($)', brand_info.get('net_profit_history', []), '/net_profit_buckets/')
    add_history_table('Employees History', 'Employees', brand_info.get('employees_history', []), '/employees_buckets/')
    add_history_table('Total Assets History', 'Total Assets ($)', brand_info.get('total_assets_history', []), '/total_assets_buckets/')
    add_history_table('Total Equity History', 'Total Equity ($)', brand_info.get('total_equity_history', []), '/total_equity_buckets/')
    add_history_table('Market Capitalization History', 'Market Capitalization ($)', brand_info.get('market_cap_history', []), '/market_cap_buckets/')
    
    return '\n'.join(lines) + '\n'.join(content_lines)

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    print("=" * 60)
    print("Brand Identity Knowledge Graph - Data Processor v2")
    print("=" * 60)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    twitter_data = load_twitter_data()
    llava_img, llava_human = load_llava_data()
    company_sectors = load_company_sectors()
    wikidata = load_wikidata()
    bg_dataset = load_bg_dataset()
    gpt_guidelines = load_gpt_guidelines()
    
    print()
    print("Processing brands...")
    
    all_brands = {}
    
    # ==========================================================================
    # STEP 1: Process Brand Guidelines dataset
    # ==========================================================================
    print("  Processing Brand Guidelines dataset...")
    
    for item in bg_dataset:
        name = item.get('name', '')
        if not name:
            continue
        
        normalized = normalize_company_name(name)
        slug = slugify(name)
        
        if normalized not in all_brands:
            all_brands[normalized] = {
                'name': name,
                'slug': slug,
                'description': item.get('description', '') or '',
                'sectors': [],
                'regions': [],
                'years': [],
                'languages': [],
                'tags': [],
                'industries': [],
                'countries': [],
                'wikidata_description': '',
                'wikidata_url': '',
                'wikidata_properties': {},
                'guidelines': [],
                'twitter_posts': [],
                'has_twitter': False,
                'has_guidelines': False,
                'twitter_post_count': 0,
                'guideline_count': 0,
                'revenue_bucket': None,
                'operating_income_bucket': None,
                'net_profit_bucket': None,
                'employees_bucket': None,
                'total_assets_bucket': None,
                'total_equity_bucket': None,
                'market_cap_bucket': None,
                'revenue_history': [],
                'operating_income_history': [],
                'net_profit_history': [],
                'employees_history': [],
                'total_assets_history': [],
                'total_equity_history': [],
                'market_cap_history': [],
                'products_or_materials_produced': [],
                'products': [],
                'headquarters_locations': [],
                'subsidiaries': [],
                'foundation_dates': [],
                # Visual attributes
                'lightings': [],
                'perspectives': [],
                'image_backgrounds': [],
                'color_schemes': [],
                'photography_genres': [],
                'concepts': [],
                'depths': [],
                'image_effects': [],
                'hair_styles': [],
                'facial_expressions': [],
                'clothing_styles': [],
                'clothing_colors': [],
                'posings': [],
                'gazes': [],
                'body_sections': [],
                # Color analysis
                'dominant_colors': [],
                'dominant_tone': None,
                'color_palette': [],
                'color_stats': {},
                'tone_stats': {},
                'total_images_analyzed': 0,
                # Extracted entities
                'logo_elements': [],
                'brand_colors': [],
                'typographies': [],
                'imagery_styles': [],
                # Websites (union of all sources)
                'websites': [],
                'sample_image_urls': [],
            }
        
        brand = all_brands[normalized]
        
        # Extract websites from all sources
        websites = extract_all_websites(item)
        if websites:
            brand['websites'].extend(websites)
        
        # Update basic info
        year = item.get('year', '')
        if year:
            brand['years'].append(slugify(str(year)))
        
        region = item.get('region', '')
        if region:
            brand['regions'].append(slugify(region))
        
        language = item.get('language', '')
        if language:
            brand['languages'].append(slugify(language))
        
        tags = item.get('tags', [])
        if tags:
            brand['tags'].extend(slugify(t) for t in tags)
        
        sector = item.get('sector', [])
        if sector:
            brand['sectors'].extend(slugify(s) for s in sector)
        
        # Process Wikidata from BG dataset
        item_wikidata = item.get('wikidata', {})
        if item_wikidata:
            if not brand['wikidata_description']:
                brand['wikidata_description'] = item_wikidata.get('description', '')
            if not brand['wikidata_url']:
                brand['wikidata_url'] = item_wikidata.get('wikidata_url', '')
            
            # Process properties (excluding instance_of, official_website)
            props, primary_buckets, histories, one_degree = process_wikidata_properties(item_wikidata)
            brand['wikidata_properties'].update(props)

            for bucket_key, bucket_val in primary_buckets.items():
                if bucket_val and not brand.get(bucket_key):
                    brand[bucket_key] = bucket_val

            for hist_key, hist_values in histories.items():
                brand[hist_key].extend(hist_values)

            for conn_key, conn_values in one_degree.items():
                brand[conn_key].extend(conn_values)
            
            # Extract taxonomies from wikidata
            wd_props = item_wikidata.get('properties', {})
            
            industry = wd_props.get('industry', [])
            if industry:
                brand['industries'].extend(slugify(str(i)) for i in industry if isinstance(i, str))
            
            country = wd_props.get('country', [])
            if country:
                brand['countries'].extend(slugify(str(c)) for c in country if isinstance(c, str))
        
        # Match GPT guidelines
        pdf_filename = item.get('pdf_filename', '')
        gpt_analysis = match_gpt_guidelines(pdf_filename, gpt_guidelines)
        
        guideline_entry = {
            'year': year or 'UNKNOWN',
            'analysis': gpt_analysis,
        }
        brand['guidelines'].append(guideline_entry)
        brand['has_guidelines'] = True
        
        # Extract entities from guidelines using spaCy patterns
        if gpt_analysis and isinstance(gpt_analysis.get('analysis'), dict):
            anal = gpt_analysis['analysis']
            all_text = []
            for section in ['logo_information', 'color_information', 'typography_information',
                           'imagery_photography', 'spacing_layout', 'brand_voice', 'notes']:
                for item_data in anal.get(section, []):
                    if isinstance(item_data, dict):
                        all_text.append(item_data.get('description', ''))
            
            combined_text = ' '.join(all_text)
            entities = extract_brand_entities(combined_text)
            brand['logo_elements'].extend(entities['logo_elements'])
            brand['brand_colors'].extend(entities['brand_colors'])
            brand['typographies'].extend(entities['typographies'])
            brand['imagery_styles'].extend(entities['imagery_styles'])
    
    # ==========================================================================
    # STEP 2: Process Twitter dataset
    # ==========================================================================
    print("  Processing Twitter dataset...")
    
    for company_norm, posts in twitter_data.items():
        matched_brand = None
        
        if company_norm in all_brands:
            matched_brand = company_norm
        else:
            best_match = None
            best_score = 0
            for brand_norm in all_brands:
                shorter = min(len(company_norm), len(brand_norm))
                longer = max(len(company_norm), len(brand_norm))
                
                if shorter < 4:
                    continue
                    
                if company_norm in brand_norm or brand_norm in company_norm:
                    overlap_ratio = shorter / longer
                    if overlap_ratio > 0.7 and overlap_ratio > best_score:
                        best_score = overlap_ratio
                        best_match = brand_norm
            
            matched_brand = best_match
        
        if not matched_brand:
            display_name = posts[0].get('new_company', company_norm) if posts else company_norm
            slug = slugify(display_name)
            
            all_brands[company_norm] = {
                'name': display_name.title(),
                'slug': slug,
                'description': '',
                'sectors': [],
                'regions': [],
                'years': [],
                'languages': [],
                'tags': [],
                'industries': [],
                'countries': [],
                'wikidata_description': '',
                'wikidata_url': '',
                'wikidata_properties': {},
                'guidelines': [],
                'twitter_posts': [],
                'has_twitter': False,
                'has_guidelines': False,
                'twitter_post_count': 0,
                'guideline_count': 0,
                'revenue_bucket': None,
                'operating_income_bucket': None,
                'net_profit_bucket': None,
                'employees_bucket': None,
                'total_assets_bucket': None,
                'total_equity_bucket': None,
                'market_cap_bucket': None,
                'revenue_history': [],
                'operating_income_history': [],
                'net_profit_history': [],
                'employees_history': [],
                'total_assets_history': [],
                'total_equity_history': [],
                'market_cap_history': [],
                'products_or_materials_produced': [],
                'products': [],
                'headquarters_locations': [],
                'subsidiaries': [],
                'foundation_dates': [],
                'lightings': [],
                'perspectives': [],
                'image_backgrounds': [],
                'color_schemes': [],
                'photography_genres': [],
                'concepts': [],
                'depths': [],
                'image_effects': [],
                'hair_styles': [],
                'facial_expressions': [],
                'clothing_styles': [],
                'clothing_colors': [],
                'posings': [],
                'gazes': [],
                'body_sections': [],
                'dominant_colors': [],
                'dominant_tone': None,
                'color_palette': [],
                'color_stats': {},
                'tone_stats': {},
                'total_images_analyzed': 0,
                'logo_elements': [],
                'brand_colors': [],
                'typographies': [],
                'imagery_styles': [],
                'websites': [],
                'sample_image_urls': [],
            }
            matched_brand = company_norm
        
        brand = all_brands[matched_brand]
        brand['twitter_posts'].extend(posts)
        brand['has_twitter'] = True

        for post in posts:
            media_url = extract_media_url(post.get('media', ''))
            if media_url:
                brand['sample_image_urls'].append(media_url)
        
        # Add sectors from company_sector.csv
        if company_norm in company_sectors:
            brand['sectors'].extend(slugify(s) for s in company_sectors[company_norm])
        
        # Aggregate visual attributes
        visual_attrs = aggregate_visual_attributes(posts, llava_img, llava_human)
        for key, values in visual_attrs.items():
            brand[key].extend(values)
        
        # Process color data
        color_data = process_color_data(posts)
        brand['dominant_colors'].extend(color_data['dominant_colors'])
        brand['dominant_tone'] = color_data['dominant_tone'] or brand['dominant_tone']
        brand['color_palette'] = color_data['color_palette']
        brand['color_stats'] = color_data['color_stats']
        brand['tone_stats'] = color_data['tone_stats']
        brand['total_images_analyzed'] = color_data['total_images_analyzed']
    
    # ==========================================================================
    # STEP 3: Add Wikidata for Twitter companies
    # ==========================================================================
    print("  Adding Wikidata for Twitter companies...")
    
    for company_norm, wd in wikidata.items():
        if company_norm in all_brands:
            brand = all_brands[company_norm]
            
            if not brand['wikidata_description']:
                brand['wikidata_description'] = wd.get('description', '')
            if not brand['wikidata_url']:
                brand['wikidata_url'] = wd.get('wikidata_url', '')
            
            # Process properties
            props, primary_buckets, histories, one_degree = process_wikidata_properties(wd)
            brand['wikidata_properties'].update(props)

            for bucket_key, bucket_val in primary_buckets.items():
                if bucket_val and not brand.get(bucket_key):
                    brand[bucket_key] = bucket_val

            for hist_key, hist_values in histories.items():
                brand[hist_key].extend(hist_values)

            for conn_key, conn_values in one_degree.items():
                brand[conn_key].extend(conn_values)
            
            # Extract taxonomies
            wd_props = wd.get('properties', {})
            
            industry = wd_props.get('industry', [])
            if industry:
                brand['industries'].extend(slugify(str(i)) for i in industry if isinstance(i, str))
            
            country = wd_props.get('country', [])
            if country:
                brand['countries'].extend(slugify(str(c)) for c in country if isinstance(c, str))
    
    # ==========================================================================
    # STEP 4: Deduplicate and finalize
    # ==========================================================================
    print("  Finalizing brand data...")
    
    for brand in all_brands.values():
        # Deduplicate all lists
        for key in ['sectors', 'regions', 'years', 'languages', 'tags', 'industries',
                    'countries', 'lightings', 'perspectives', 'image_backgrounds', 
                    'color_schemes', 'photography_genres', 'concepts', 'depths', 
                    'image_effects', 'hair_styles', 'facial_expressions', 'clothing_styles', 
                    'clothing_colors', 'posings', 'gazes', 'body_sections', 
                    'logo_elements', 'brand_colors', 'typographies', 'imagery_styles',
                    'dominant_colors', 'products_or_materials_produced', 'products',
                    'headquarters_locations', 'subsidiaries', 'foundation_dates']:
            brand[key] = list(set(brand.get(key, [])))
            brand[key] = [v for v in brand[key] if v]

        brand['sample_image_urls'] = list(dict.fromkeys(brand.get('sample_image_urls', [])))

        for history_key in [
            'revenue_history', 'operating_income_history', 'net_profit_history',
            'employees_history', 'total_assets_history', 'total_equity_history', 'market_cap_history'
        ]:
            seen = set()
            unique_values = []
            for item in brand.get(history_key, []):
                sig = f"{item.get('formatted')}|{item.get('year_info')}|{item.get('bucket')}"
                if sig not in seen:
                    seen.add(sig)
                    unique_values.append(item)
            brand[history_key] = unique_values
        
        # Deduplicate websites
        brand['websites'] = sorted(list(set(brand.get('websites', []))))
        
        # Update counts
        brand['twitter_post_count'] = len(brand['twitter_posts'])
        brand['guideline_count'] = len(brand['guidelines'])
        
        # Set defaults
        if not brand['sectors']:
            brand['sectors'] = ['unknown']
        if not brand['regions']:
            brand['regions'] = ['unknown']
    
    # ==========================================================================
    # STEP 5: Generate Markdown files
    # ==========================================================================
    print()
    print(f"Generating {len(all_brands)} brand markdown files...")
    
    for normalized, brand in all_brands.items():
        slug = brand['slug']
        filename = OUTPUT_DIR / f"{slug}.md"
        
        content = generate_brand_markdown(brand)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # ==========================================================================
    # STEP 6: Generate summary data
    # ==========================================================================
    print("Generating summary data...")
    
    summary = {
        'total_brands': len(all_brands),
        'brands_with_twitter': sum(1 for b in all_brands.values() if b['has_twitter']),
        'brands_with_guidelines': sum(1 for b in all_brands.values() if b['has_guidelines']),
        'total_promotion_images': sum(b['twitter_post_count'] for b in all_brands.values()),
        'total_guidelines': sum(b['guideline_count'] for b in all_brands.values()),
    }
    
    with open(DATA_OUTPUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"  Total brands: {summary['total_brands']}")
    print(f"  Brands with promotion images: {summary['brands_with_twitter']}")
    print(f"  Brands with guidelines: {summary['brands_with_guidelines']}")
    print(f"  Total promotion images: {summary['total_promotion_images']}")
    print(f"  Total guidelines: {summary['total_guidelines']}")
    print()
    print(f"Output written to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
