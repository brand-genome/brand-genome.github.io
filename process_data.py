#!/usr/bin/env python3
"""
Brand Identity Knowledge Graph - Data Processor
Merges Twitter data, Brand Guidelines data, Wikidata, and GPT extractions
into Hugo-compatible Markdown files for the Knowledge Graph website.
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

# Wikidata properties to EXCLUDE (pure IDs with no brand meaning)
WIKIDATA_ID_PROPERTIES = {
    'geonames_id',  # Geographic database ID
    'commons_category',  # Wikimedia commons category
    'topic_main_category',  # Wikipedia category
}

# Wikidata properties to INCLUDE (even though they have 'id' in name, they're brand-relevant)
WIKIDATA_INCLUDE_PROPERTIES = {
    'facebook_id', 'instagram_username', 'twitter_username', 'youtube_channel_id',
    'linkedin_id', 'ISIN', 'official_website', 'logo_image', 'image',
    'country', 'headquarters_location', 'inception', 'industry', 'employees',
    'total_revenue', 'net_profit', 'total_assets', 'founder', 'founded_by',
    'chief_executive_officer', 'chairperson', 'board_member', 'legal_form',
    'stock_exchange', 'parent_organization', 'subsidiary', 'owned_by', 'owner_of',
    'brand', 'product', 'instance_of', 'field_of_work', 'operating_area',
    'member_of', 'part_of', 'partnership', 'business_division', 'manufacturer',
    'coordinates', 'location', 'located_in_admin', 'email', 'use'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def slugify(text):
    """Convert text to URL-friendly slug."""
    if not text:
        return "unknown"
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', str(text))
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Convert to lowercase and replace spaces/special chars with hyphens
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
            # Slugify for taxonomy use
            cleaned.append(slugify(str(v)))
    return list(set(cleaned))

def extract_media_url(media_str):
    """Extract full URL from media string."""
    if not media_str:
        return None
    # Look for fullUrl in the media string
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
    # If contains special chars, quote it
    if any(c in s for c in [':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`', '"', "'"]):
        # Escape quotes and wrap in quotes
        s = s.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{s}"'
    return s

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
    
    # Extract filename without path and extension
    stem = Path(pdf_filename).stem.lower()
    
    # Direct match
    if stem in gpt_data:
        return gpt_data[stem]
    
    # Try without common suffixes
    for suffix in ['-en', '-eng', '-english', '_en', '_eng']:
        test_stem = stem.replace(suffix, '')
        if test_stem in gpt_data:
            return gpt_data[test_stem]
    
    # Fuzzy match - find closest
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
        
        # Get image attributes
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
        
        # Get human attributes
        if media_key in llava_human:
            human_data = llava_human[media_key]
            attributes['hair_styles'].extend(clean_list_values(human_data.get('hair_style', [])))
            attributes['facial_expressions'].extend(clean_list_values(human_data.get('facial_expression', [])))
            attributes['clothing_styles'].extend(clean_list_values(human_data.get('clothing_style', [])))
            attributes['clothing_colors'].extend(clean_list_values(human_data.get('clothing_color_palette', [])))
            attributes['posings'].extend(clean_list_values(human_data.get('posing', [])))
            attributes['gazes'].extend(clean_list_values(human_data.get('gaze', [])))
            attributes['body_sections'].extend(clean_list_values(human_data.get('visible_body_section', [])))
    
    # Deduplicate
    for key in attributes:
        attributes[key] = list(set(attributes[key]))
    
    return attributes

def collect_official_websites(brand_data):
    """Collect and deduplicate official websites from multiple sources."""
    urls = set()
    
    # From gpt_web_search
    gpt_search = brand_data.get('gpt_web_search', {}) or {}
    official_sites = gpt_search.get('official_website', []) or []
    for url in official_sites:
        if url:
            urls.add(url)
    
    # From wikidata
    wikidata = brand_data.get('wikidata', {}) or {}
    props = wikidata.get('properties', {}) or {}
    wiki_urls = props.get('official_website', []) or []
    for url in wiki_urls:
        if isinstance(url, str):
            urls.add(url)
        elif isinstance(url, dict):
            urls.add(url.get('value', ''))
    
    # From pdf_extracted_url (this might be the guidelines URL, skip if so)
    # We're not showing PDF URLs per requirement
    
    return list(urls) if urls else ["UNKNOWN"]

def process_wikidata_properties(wikidata):
    """Process Wikidata properties, filtering out pure IDs."""
    if not wikidata or 'properties' not in wikidata:
        return {}
    
    processed = {}
    props = wikidata.get('properties', {})
    
    for key, value in props.items():
        # Skip excluded ID properties
        if key in WIKIDATA_ID_PROPERTIES:
            continue
        
        # Format the value
        formatted = format_wikidata_value(value)
        if formatted:
            processed[key] = formatted
    
    return processed

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
        'instance_types': brand_info.get('instance_types', []),
        
        # Visual taxonomies
        'lightings': brand_info.get('lightings', []),
        'perspectives': brand_info.get('perspectives', []),
        'image_backgrounds': brand_info.get('image_backgrounds', []),
        'color_schemes': brand_info.get('color_schemes', []),
        'photography_genres': brand_info.get('photography_genres', []),
        'concepts': brand_info.get('concepts', []),
        'depths': brand_info.get('depths', []),
        'image_effects': brand_info.get('image_effects', []),
        
        # Human visual taxonomies
        'hair_styles': brand_info.get('hair_styles', []),
        'facial_expressions': brand_info.get('facial_expressions', []),
        'clothing_styles': brand_info.get('clothing_styles', []),
        'clothing_colors': brand_info.get('clothing_colors', []),
        'posings': brand_info.get('posings', []),
        'gazes': brand_info.get('gazes', []),
        'body_sections': brand_info.get('body_sections', []),
        
        # Extracted entities
        'logo_elements': brand_info.get('logo_elements', []),
        'brand_colors': brand_info.get('brand_colors', []),
        'typographies': brand_info.get('typographies', []),
        'imagery_styles': brand_info.get('imagery_styles', []),
        
        # Complex data (as params)
        'official_websites': brand_info.get('official_websites', []),
        'wikidata_url': brand_info.get('wikidata_url', ''),
        'wikidata_label': brand_info.get('wikidata_label', ''),
        'wikidata_description': brand_info.get('wikidata_description', ''),
        'wikidata_aliases': brand_info.get('wikidata_aliases', []),
        'has_twitter': brand_info.get('has_twitter', False),
        'has_guidelines': brand_info.get('has_guidelines', False),
        'twitter_post_count': brand_info.get('twitter_post_count', 0),
        'guideline_count': brand_info.get('guideline_count', 0),
    }
    
    # Build YAML frontmatter
    lines = ['---']
    
    for key, value in frontmatter.items():
        if isinstance(value, list):
            if value:
                lines.append(f'{key}:')
                for item in value:
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
    
    # Wikidata properties section
    if brand_info.get('wikidata_properties'):
        content_lines.append('## Properties')
        content_lines.append('')
        content_lines.append('| Property | Value |')
        content_lines.append('|----------|-------|')
        for key, value in brand_info['wikidata_properties'].items():
            display_key = key.replace('_', ' ').title()
            if isinstance(value, list):
                display_val = ', '.join(str(v) for v in value[:5])
                if len(value) > 5:
                    display_val += f' (+{len(value)-5} more)'
            else:
                display_val = str(value)
            # Escape pipes in values
            display_val = display_val.replace('|', '\\|')
            content_lines.append(f'| {display_key} | {display_val} |')
        content_lines.append('')
    
    # Guidelines section
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
    
    # Twitter posts section
    if brand_info.get('twitter_posts'):
        content_lines.append('## Twitter Insights')
        content_lines.append('')
        
        # Show up to 10 posts
        for i, post in enumerate(brand_info['twitter_posts'][:10]):
            content_lines.append(f'### Post {i+1}')
            content_lines.append('')
            
            # Content
            content = post.get('content', '').replace('\n', ' ').strip()
            if content:
                content_lines.append(f'> {content[:500]}{"..." if len(content) > 500 else ""}')
                content_lines.append('')
            
            # Metadata
            date = post.get('date_processed', 'UNKNOWN')
            likes = post.get('likes', 0)
            content_lines.append(f'**Date:** {date} | **Likes:** {likes}')
            content_lines.append('')
            
            # Image link
            media_url = extract_media_url(post.get('media', ''))
            media_key = post.get('media_keys', '')
            if media_key:
                content_lines.append(f'**Image:** `{media_key}`')
                if media_url:
                    content_lines.append(f' [View Image ↗]({media_url})')
                content_lines.append('')
            
            # Visual attributes for this post
            if media_key:
                content_lines.append('**Visual Attributes:**')
                # These are aggregated at brand level, but we note them here
                captions = post.get('short_captions', '')
                if captions:
                    content_lines.append(f'- Caption: {captions}')
                keywords = post.get('keywords', '')
                if keywords:
                    content_lines.append(f'- Keywords: {keywords}')
                content_lines.append('')
        
        if len(brand_info['twitter_posts']) > 10:
            content_lines.append(f'*... and {len(brand_info["twitter_posts"]) - 10} more posts*')
            content_lines.append('')
    
    return '\n'.join(lines) + '\n'.join(content_lines)

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    print("=" * 60)
    print("Brand Identity Knowledge Graph - Data Processor")
    print("=" * 60)
    print()
    
    # Create output directories
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
    
    # Track all brands
    all_brands = {}  # normalized_name -> brand_info
    
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
                'instance_types': [],
                'official_websites': [],
                'wikidata_url': '',
                'wikidata_label': '',
                'wikidata_description': '',
                'wikidata_aliases': [],
                'wikidata_properties': {},
                'guidelines': [],
                'twitter_posts': [],
                'has_twitter': False,
                'has_guidelines': False,
                'twitter_post_count': 0,
                'guideline_count': 0,
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
                # Extracted entities
                'logo_elements': [],
                'brand_colors': [],
                'typographies': [],
                'imagery_styles': [],
            }
        
        brand = all_brands[normalized]
        
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
        
        # Collect websites
        websites = collect_official_websites(item)
        brand['official_websites'].extend(websites)
        
        # Process Wikidata from BG dataset
        item_wikidata = item.get('wikidata', {})
        if item_wikidata:
            if not brand['wikidata_url']:
                brand['wikidata_url'] = item_wikidata.get('wikidata_url', '')
                brand['wikidata_label'] = item_wikidata.get('label', '')
                brand['wikidata_description'] = item_wikidata.get('description', '')
                brand['wikidata_aliases'] = item_wikidata.get('aliases', [])
            
            # Process properties
            props = process_wikidata_properties(item_wikidata)
            brand['wikidata_properties'].update(props)
            
            # Extract taxonomies from wikidata
            wd_props = item_wikidata.get('properties', {})
            
            instance_of = wd_props.get('instance_of', [])
            if instance_of:
                brand['instance_types'].extend(slugify(str(i)) for i in instance_of if isinstance(i, str))
            
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
        
        # Extract entities from guidelines
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
        # Find matching brand or create new
        matched_brand = None
        
        # Try exact match first
        if company_norm in all_brands:
            matched_brand = company_norm
        else:
            # Try fuzzy match - but require significant overlap
            # Only match if one is a substantial part of the other (>70% of shorter string)
            best_match = None
            best_score = 0
            for brand_norm in all_brands:
                shorter = min(len(company_norm), len(brand_norm))
                longer = max(len(company_norm), len(brand_norm))
                
                # Skip if too short (avoid matching "art" to everything)
                if shorter < 4:
                    continue
                    
                # Check if one contains the other
                if company_norm in brand_norm or brand_norm in company_norm:
                    # Calculate overlap ratio
                    overlap_ratio = shorter / longer
                    if overlap_ratio > 0.7 and overlap_ratio > best_score:
                        best_score = overlap_ratio
                        best_match = brand_norm
            
            matched_brand = best_match
        
        # Create new brand if no match
        if not matched_brand:
            # Get display name from first post
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
                'instance_types': [],
                'official_websites': [],
                'wikidata_url': '',
                'wikidata_label': '',
                'wikidata_description': '',
                'wikidata_aliases': [],
                'wikidata_properties': {},
                'guidelines': [],
                'twitter_posts': [],
                'has_twitter': False,
                'has_guidelines': False,
                'twitter_post_count': 0,
                'guideline_count': 0,
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
                'logo_elements': [],
                'brand_colors': [],
                'typographies': [],
                'imagery_styles': [],
            }
            matched_brand = company_norm
        
        brand = all_brands[matched_brand]
        brand['twitter_posts'].extend(posts)
        brand['has_twitter'] = True
        
        # Add sectors from company_sector.csv
        if company_norm in company_sectors:
            brand['sectors'].extend(slugify(s) for s in company_sectors[company_norm])
        
        # Aggregate visual attributes
        visual_attrs = aggregate_visual_attributes(posts, llava_img, llava_human)
        for key, values in visual_attrs.items():
            brand[key].extend(values)
    
    # ==========================================================================
    # STEP 3: Add Wikidata for Twitter companies
    # ==========================================================================
    print("  Adding Wikidata for Twitter companies...")
    
    for company_norm, wd in wikidata.items():
        if company_norm in all_brands:
            brand = all_brands[company_norm]
            
            if not brand['wikidata_url']:
                brand['wikidata_url'] = wd.get('wikidata_url', '')
                brand['wikidata_label'] = wd.get('label', '')
                brand['wikidata_description'] = wd.get('description', '')
                brand['wikidata_aliases'] = wd.get('aliases', [])
            
            # Process properties
            props = {}
            wd_props = wd.get('properties', {})
            for key, value in wd_props.items():
                if key not in WIKIDATA_ID_PROPERTIES:
                    props[key] = format_wikidata_value(value)
            brand['wikidata_properties'].update(props)
            
            # Extract taxonomies
            instance_of = wd_props.get('instance_of', [])
            if instance_of:
                brand['instance_types'].extend(slugify(str(i)) for i in instance_of if isinstance(i, str))
            
            industry = wd_props.get('industry', [])
            if industry:
                brand['industries'].extend(slugify(str(i)) for i in industry if isinstance(i, str))
            
            country = wd_props.get('country', [])
            if country:
                brand['countries'].extend(slugify(str(c)) for c in country if isinstance(c, str))
            
            # Official website
            official_site = wd_props.get('official_website', [])
            if official_site:
                for site in official_site:
                    if isinstance(site, str):
                        brand['official_websites'].append(site)
                    elif isinstance(site, dict):
                        brand['official_websites'].append(site.get('value', ''))
    
    # ==========================================================================
    # STEP 4: Deduplicate and finalize
    # ==========================================================================
    print("  Finalizing brand data...")
    
    for brand in all_brands.values():
        # Deduplicate all lists
        for key in ['sectors', 'regions', 'years', 'languages', 'tags', 'industries',
                    'countries', 'instance_types', 'official_websites', 'wikidata_aliases',
                    'lightings', 'perspectives', 'image_backgrounds', 'color_schemes',
                    'photography_genres', 'concepts', 'depths', 'image_effects',
                    'hair_styles', 'facial_expressions', 'clothing_styles', 'clothing_colors',
                    'posings', 'gazes', 'body_sections', 'logo_elements', 'brand_colors',
                    'typographies', 'imagery_styles']:
            brand[key] = list(set(brand.get(key, [])))
            # Remove empty strings
            brand[key] = [v for v in brand[key] if v]
        
        # Update counts
        brand['twitter_post_count'] = len(brand['twitter_posts'])
        brand['guideline_count'] = len(brand['guidelines'])
        
        # Set defaults for empty required fields
        if not brand['sectors']:
            brand['sectors'] = ['unknown']
        if not brand['regions']:
            brand['regions'] = ['unknown']
        if not brand['official_websites'] or brand['official_websites'] == ['UNKNOWN']:
            brand['official_websites'] = []
    
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
        'total_twitter_posts': sum(b['twitter_post_count'] for b in all_brands.values()),
        'total_guidelines': sum(b['guideline_count'] for b in all_brands.values()),
    }
    
    with open(DATA_OUTPUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"  Total brands: {summary['total_brands']}")
    print(f"  Brands with Twitter data: {summary['brands_with_twitter']}")
    print(f"  Brands with guidelines: {summary['brands_with_guidelines']}")
    print(f"  Total Twitter posts: {summary['total_twitter_posts']}")
    print(f"  Total guidelines: {summary['total_guidelines']}")
    print()
    print(f"Output written to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
