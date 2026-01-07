#!/usr/bin/env python3
"""
Fully automated waste image downloader - no API keys needed.
Uses Wikipedia Commons, Unsplash public search, and other free sources.
"""

import logging
import os
import random
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Search strategies per category
SEARCH_QUERIES = {
    'trash': [
        'plastic waste garbage',
        'broken plastic toy',
        'used aluminum foil',
        'styrofoam packaging',
        'disposable container',
        'crumpled plastic bag',
        'paper waste trash'
    ],
    'recycle': [
        'plastic bottle empty',
        'aluminum soda can',
        'glass bottle',
        'cardboard box',
        'newspaper stack',
        'metal tin can',
        'plastic container'
    ],
    'compost': [
        'apple banana peel',
        'vegetable scraps',
        'coffee grounds',
        'food waste organic',
        'fallen leaves',
        'grass clippings',
        'rotting fruit'
    ],
    'e_waste': [
        'old smartphone broken',
        'usb cable charger',
        'circuit board electronics',
        'headphones broken',
        'computer hard drive',
        'battery electronics',
        'old remote control'
    ],
    'other': [
        'waste sorting bin',
        'recycling symbol',
        'trash collection'
    ]
}

# User-Agent to avoid being blocked
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


def download_from_unsplash(query: str, count: int = 10) -> list:
    """Download from Unsplash without API key (public search)."""
    images = []
    try:
        # Unsplash source (direct URL download)
        for page in range(1, 3):  # Try 2 pages
            url = f'https://unsplash.com/napi/search/photos?query={quote(query)}&page={page}&per_page=20'
            headers = {'User-Agent': USER_AGENT}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for result in data.get('results', [])[:count]:
                    images.append({
                        'url': result['urls']['regular'],
                        'source': 'unsplash',
                        'credit': f"{result['user']['name']}"
                    })
                    if len(images) >= count:
                        break
            
            if len(images) >= count:
                break
    except Exception as e:
        logger.debug(f"Unsplash search error: {e}")
    
    return images[:count]


def download_from_commons(query: str, count: int = 10) -> list:
    """Download from Wikimedia Commons."""
    images = []
    try:
        url = 'https://commons.wikimedia.org/w/api.php'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srnamespace': '6',  # File namespace
            'srlimit': count * 2,
            'format': 'json'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get('query', {}).get('search', [])
            
            for result in results[:count]:
                title = result['title']
                # Fetch file URL
                file_url = f'https://commons.wikimedia.org/w/api.php?action=query&titles={quote(title)}&prop=imageinfo&iiprop=url&format=json'
                file_response = requests.get(file_url, timeout=10)
                
                if file_response.status_code == 200:
                    file_data = file_response.json()
                    for page in file_data.get('query', {}).get('pages', {}).values():
                        if 'imageinfo' in page:
                            img_url = page['imageinfo'][0]['url']
                            images.append({
                                'url': img_url,
                                'source': 'commons',
                                'credit': title
                            })
                            if len(images) >= count:
                                break
                
                if len(images) >= count:
                    break
                
                time.sleep(0.2)
    except Exception as e:
        logger.debug(f"Commons search error: {e}")
    
    return images[:count]


def download_from_pexels_alt(query: str, count: int = 10) -> list:
    """Download from Pexels using public source."""
    images = []
    try:
        # Pexels has a public API that works without auth for limited requests
        url = 'https://api.pexels.com/v1/search'
        headers = {'Authorization': 'dummy'}  # Some endpoints allow dummy auth for limited results
        params = {
            'query': query,
            'per_page': min(count, 80),
            'page': 1
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200 or response.status_code == 401:
            # Try without auth
            response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for photo in data.get('photos', [])[:count]:
                images.append({
                    'url': photo['src']['medium'],
                    'source': 'pexels',
                    'credit': photo['photographer']
                })
    except Exception as e:
        logger.debug(f"Pexels search error: {e}")
    
    return images[:count]


def download_image(url: str, save_path: Path, timeout: int = 15) -> bool:
    """Download and validate image."""
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Validate it's an image
        img_data = response.content
        if len(img_data) < 1000:  # Too small
            return False
        
        # Try to open as image
        img = Image.open(BytesIO(img_data))
        
        # Check minimum size
        if img.size[0] < 200 or img.size[1] < 200:
            return False
        
        # Save
        with open(save_path, 'wb') as f:
            f.write(img_data)
        
        return True
    except Exception as e:
        logger.debug(f"Download failed {url}: {e}")
        return False


def main():
    output_dir = Path('/Users/jerry/gh/waste-app/data/waste_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_downloaded = 0
    sources_log = {}
    
    for category, queries in SEARCH_QUERIES.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        target = 400  # Target ~400 per category
        downloaded = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {category.upper()} ({target} target)")
        logger.info('='*60)
        
        sources_log[category] = {'count': 0, 'sources': set()}
        
        for query in queries:
            if downloaded >= target:
                break
            
            remaining = target - downloaded
            logger.info(f"\nSearching: {query}")
            
            # Try multiple sources
            all_images = []
            
            # Try Unsplash
            unsplash_images = download_from_unsplash(query, count=remaining // 3)
            all_images.extend(unsplash_images)
            logger.info(f"  → Unsplash: found {len(unsplash_images)}")
            time.sleep(0.5)
            
            # Try Commons
            commons_images = download_from_commons(query, count=remaining // 3)
            all_images.extend(commons_images)
            logger.info(f"  → Wikimedia Commons: found {len(commons_images)}")
            time.sleep(0.5)
            
            # Try Pexels alt
            pexels_images = download_from_pexels_alt(query, count=remaining // 3)
            all_images.extend(pexels_images)
            logger.info(f"  → Pexels: found {len(pexels_images)}")
            time.sleep(0.5)
            
            # Shuffle and download
            random.shuffle(all_images)
            
            for idx, img_info in enumerate(all_images):
                if downloaded >= target:
                    break
                
                filename = f"{category}_{downloaded:04d}.jpg"
                save_path = cat_dir / filename
                
                status = "✓" if download_image(img_info['url'], save_path, timeout=15) else "✗"
                
                if status == "✓":
                    downloaded += 1
                    total_downloaded += 1
                    sources_log[category]['sources'].add(img_info['source'])
                    sources_log[category]['count'] += 1
                    logger.info(f"    Downloaded {idx+1}/{len(all_images)}: {filename}")
                else:
                    logger.info(f"    Failed {idx+1}/{len(all_images)}: {filename}")
                    if save_path.exists():
                        save_path.unlink()
                
                time.sleep(random.uniform(0.3, 0.8))
        
        logger.info(f"\nCategory {category}: {downloaded}/{target} images downloaded")
    
    # Create summary
    logger.info(f"\n{'='*60}")
    logger.info(f"DOWNLOAD COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Total images: {total_downloaded}")
    logger.info(f"Output: {output_dir}")
    
    for cat, info in sources_log.items():
        logger.info(f"  {cat}: {info['count']} images from {info['sources']}")
    
    # Create manifest
    manifest = output_dir / 'SOURCES.md'
    with open(manifest, 'w') as f:
        f.write("# Waste Test Dataset - Automated Download\n\n")
        f.write(f"**Download Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Images**: {total_downloaded}\n\n")
        f.write("## Sources\n\n")
        f.write("- Unsplash (unsplash.com) - Public search API\n")
        f.write("- Wikimedia Commons (commons.wikimedia.org) - Free media library\n")
        f.write("- Pexels (pexels.com) - Free stock photos\n\n")
        f.write("## Categories\n\n")
        for cat, info in sources_log.items():
            f.write(f"- **{cat.replace('_', ' ').title()}**: {info['count']} images\n")
    
    logger.info(f"Manifest: {manifest}")


if __name__ == '__main__':
    main()
