#!/usr/bin/env python3
"""
Download waste classification training/test images from free online sources.

Supports:
- Unsplash API (requires UNSPLASH_ACCESS_KEY environment variable)
- Pexels API (requires PEXELS_API_KEY environment variable)  
- Pixabay API (requires PIXABAY_API_KEY environment variable)
- Manual file organization

Usage:
    python3 scripts/download_waste_images.py --help
    python3 scripts/download_waste_images.py --source unsplash --category recycle --count 100
    python3 scripts/download_waste_images.py --organize-only   # Rename and organize existing files
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Categories and target counts
WASTE_CATEGORIES = {
    'trash': {
        'target': 450,
        'search_terms': [
            'trash garbage waste',
            'plastic waste',
            'styrofoam foam packaging',
            'broken plastic',
            'used wrapper',
            'disposable packaging'
        ]
    },
    'recycle': {
        'target': 550,
        'search_terms': [
            'plastic bottle',
            'aluminum can',
            'glass bottle',
            'cardboard box',
            'newspaper',
            'metal can',
            'recycling bin',
            'recyclable waste'
        ]
    },
    'compost': {
        'target': 350,
        'search_terms': [
            'food waste scraps',
            'apple banana orange peel',
            'vegetable waste',
            'coffee grounds',
            'leaves compost',
            'organic waste'
        ]
    },
    'e_waste': {
        'target': 350,
        'search_terms': [
            'old smartphone broken phone',
            'usb cable charging cable',
            'circuit board electronics',
            'headphones earbuds',
            'computer battery',
            'electronic waste e-waste'
        ]
    },
    'other': {
        'target': 200,
        'search_terms': [
            'waste management',
            'trash sorting',
            'recycling symbol'
        ]
    }
}


class UnsplashDownloader:
    """Download from Unsplash API."""
    
    BASE_URL = 'https://api.unsplash.com'
    
    def __init__(self, access_key: Optional[str] = None):
        self.access_key = access_key or os.getenv('UNSPLASH_ACCESS_KEY')
        if not self.access_key:
            logger.warning("UNSPLASH_ACCESS_KEY not set. Unsplash downloads disabled.")
            self.enabled = False
        else:
            self.enabled = True
    
    def search(self, query: str, count: int = 30) -> list:
        """Search Unsplash for images."""
        if not self.enabled:
            return []
        
        url = f'{self.BASE_URL}/search/photos'
        params = {
            'query': query,
            'per_page': min(count, 30),  # API limit
            'order_by': 'relevant',
            'client_id': self.access_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get('results', []):
                results.append({
                    'source': 'unsplash',
                    'url': result['urls']['regular'],
                    'thumb_url': result['urls']['thumb'],
                    'credit': f"Photo by {result['user']['name']} on Unsplash",
                    'license': 'Unsplash License'
                })
            return results
        except Exception as e:
            logger.error(f"Unsplash search error: {e}")
            return []


class PexelsDownloader:
    """Download from Pexels API."""
    
    BASE_URL = 'https://api.pexels.com/v1'
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PEXELS_API_KEY')
        if not self.api_key:
            logger.warning("PEXELS_API_KEY not set. Pexels downloads disabled.")
            self.enabled = False
        else:
            self.enabled = True
    
    def search(self, query: str, count: int = 30) -> list:
        """Search Pexels for images."""
        if not self.enabled:
            return []
        
        url = f'{self.BASE_URL}/search'
        params = {
            'query': query,
            'per_page': min(count, 80),
            'page': 1
        }
        headers = {'Authorization': self.api_key}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for photo in data.get('photos', []):
                results.append({
                    'source': 'pexels',
                    'url': photo['src']['medium'],
                    'thumb_url': photo['src']['small'],
                    'credit': f"Photo by {photo['photographer']} on Pexels",
                    'license': 'Pexels License (CC0)'
                })
            return results
        except Exception as e:
            logger.error(f"Pexels search error: {e}")
            return []


class PixabayDownloader:
    """Download from Pixabay API."""
    
    BASE_URL = 'https://pixabay.com/api'
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PIXABAY_API_KEY')
        if not self.api_key:
            logger.warning("PIXABAY_API_KEY not set. Pixabay downloads disabled.")
            self.enabled = False
        else:
            self.enabled = True
    
    def search(self, query: str, count: int = 30) -> list:
        """Search Pixabay for images."""
        if not self.enabled:
            return []
        
        params = {
            'key': self.api_key,
            'q': query,
            'per_page': min(count, 200),
            'image_type': 'photo'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for hit in data.get('hits', []):
                results.append({
                    'source': 'pixabay',
                    'url': hit['webformatURL'],
                    'thumb_url': hit['previewURL'],
                    'credit': f"Photo by {hit['user']} on Pixabay",
                    'license': 'Pixabay License (CC0)'
                })
            return results
        except Exception as e:
            logger.error(f"Pixabay search error: {e}")
            return []


def download_image(url: str, save_path: Path, timeout: int = 10) -> bool:
    """Download a single image."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def create_sources_manifest(output_dir: Path, images_info: dict):
    """Create SOURCES.md manifest."""
    manifest_path = output_dir / 'SOURCES.md'
    
    with open(manifest_path, 'w') as f:
        f.write("# Waste Dataset Image Sources\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        
        for category, info in images_info.items():
            f.write(f"## {category.replace('_', ' ').title()}\n\n")
            f.write(f"**Count**: {len(info['images'])}\n\n")
            
            if info['images']:
                f.write("**Sources**:\n")
                sources_set = set()
                for img in info['images']:
                    if 'credit' in img:
                        sources_set.add(img['credit'])
                
                for source in sorted(sources_set):
                    f.write(f"- {source}\n")
            
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download waste classification images from free online sources'
    )
    parser.add_argument(
        '--output-dir',
        default='/Users/jerry/gh/waste-app/data/waste_test',
        help='Output directory for images'
    )
    parser.add_argument(
        '--category',
        choices=list(WASTE_CATEGORIES.keys()) + ['all'],
        default='all',
        help='Category to download (default: all)'
    )
    parser.add_argument(
        '--count',
        type=int,
        help='Override target count for category'
    )
    parser.add_argument(
        '--source',
        choices=['unsplash', 'pexels', 'pixabay', 'all'],
        default='all',
        help='Download source (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without downloading'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip download, only organize existing files'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between downloads (seconds)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloaders
    downloaders = {
        'unsplash': UnsplashDownloader(),
        'pexels': PexelsDownloader(),
        'pixabay': PixabayDownloader()
    }
    
    # Collect which sources to use
    sources_to_use = [args.source] if args.source != 'all' else list(downloaders.keys())
    
    # Collect which categories to download
    categories_to_download = [args.category] if args.category != 'all' else list(WASTE_CATEGORIES.keys())
    
    images_info = {cat: {'images': []} for cat in WASTE_CATEGORIES.keys()}
    total_downloaded = 0
    
    if not args.no_download:
        logger.info(f"Starting downloads to {output_dir}")
        
        for category in categories_to_download:
            cat_dir = output_dir / category
            cat_dir.mkdir(exist_ok=True)
            
            target_count = args.count if args.count else WASTE_CATEGORIES[category]['target']
            search_terms = WASTE_CATEGORIES[category]['search_terms']
            
            logger.info(f"\nCategory: {category.upper()} (target: {target_count} images)")
            
            downloaded_in_category = 0
            
            for search_term in search_terms:
                if downloaded_in_category >= target_count:
                    break
                
                logger.info(f"  Searching: {search_term}")
                
                for source_name in sources_to_use:
                    if downloaded_in_category >= target_count:
                        break
                    
                    downloader = downloaders[source_name]
                    if not downloader.enabled:
                        continue
                    
                    remaining = target_count - downloaded_in_category
                    results = downloader.search(search_term, count=remaining)
                    
                    for idx, result in enumerate(results):
                        if downloaded_in_category >= target_count:
                            break
                        
                        filename = f"{category}_{source_name}_{search_term.replace(' ', '_')}_{idx:03d}.jpg"
                        save_path = cat_dir / filename
                        
                        if args.dry_run:
                            logger.info(f"    [DRY RUN] Would download: {result['credit']}")
                        else:
                            if download_image(result['url'], save_path):
                                logger.info(f"    ✓ Downloaded: {filename}")
                                images_info[category]['images'].append(result)
                                downloaded_in_category += 1
                                total_downloaded += 1
                                time.sleep(args.delay)
                            else:
                                logger.warning(f"    ✗ Failed: {filename}")
    
    # Create manifest
    if not args.dry_run:
        create_sources_manifest(output_dir, images_info)
        logger.info(f"\n{'='*60}")
        logger.info(f"Download complete!")
        logger.info(f"Total images downloaded: {total_downloaded}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Manifest: {output_dir / 'SOURCES.md'}")
    else:
        logger.info(f"\nDry run complete. {total_downloaded} images would be downloaded.")


if __name__ == '__main__':
    main()
