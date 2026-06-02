#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch YouTube downloader for Ref4D reference collection.

Default paths are anchored to the repository layout:
- Videos: ref4d_build/ref_collect/downloaded_videos/<category>/
- Runtime state: ref4d_build/ref_collect/state/
- Logs: ref4d_build/ref_collect/logs/
"""

import json
import csv
import argparse
import os
import subprocess
import sys
import random
import time
from pathlib import Path
import logging
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
REFVIDEO_DIR = SCRIPT_DIR / 'downloaded_videos'
STATE_DIR = SCRIPT_DIR / 'state'
LOG_DIR = SCRIPT_DIR / 'logs'

for _dir in (REFVIDEO_DIR, STATE_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / 'video_download.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REQUIRED_VIDEO_LICENSE = 'creativeCommon'


class VideoDownloader:
    """VideoDownloader implementation."""
    
    def __init__(self, output_dir=str(REFVIDEO_DIR), state_dir=str(STATE_DIR), proxy=None, quality='best'):
        """  init   routine."""
        self.output_dir = Path(output_dir)
        self.state_dir = Path(state_dir)
        self.proxy = proxy
        self.quality = quality
        self.downloaded_log = self.state_dir / 'downloaded_videos.txt'
        self.download_record = self.state_dir / 'download_record.json'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloaded_ids = self._load_downloaded_ids()
        self.download_records = self._load_download_records()
        
        self._verify_downloaded_files()
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Downloaded videos: {len(self.downloaded_ids)}")
        logger.info("Resume support is enabled")
    
    def _load_downloaded_ids(self):
        """ load downloaded ids routine."""
        downloaded_ids = set()
        if self.downloaded_log.exists():
            with open(self.downloaded_log, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        downloaded_ids.add(line)
        return downloaded_ids
    
    def _load_download_records(self):
        """ load download records routine."""
        if self.download_record.exists():
            try:
                with open(self.download_record, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load download records: {e}")
                return {}
        return {}
    
    def _save_download_record(self, video_id, video_info, file_path):
        """ save download record routine."""
        self.download_records[video_id] = {
            'video_id': video_id,
            'title': video_info.get('title', ''),
            'category': video_info.get('category', ''),
            'file_path': str(file_path),
            'download_time': datetime.now().isoformat(),
            'url': video_info.get('url', ''),
            'duration_seconds': video_info.get('duration_seconds', 0),
            'license': video_info.get('license', ''),
            'license_checked_at': video_info.get('license_checked_at', '')
        }
        
        try:
            with open(self.download_record, 'w', encoding='utf-8') as f:
                json.dump(self.download_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save download records: {e}")
    
    def _verify_downloaded_files(self):
        """ verify downloaded files routine."""
        missing_files = []
        
        for video_id in list(self.downloaded_ids):
            if video_id in self.download_records:
                file_path = Path(self.download_records[video_id].get('file_path', ''))
                if not file_path.exists():
                    missing_files.append(video_id)
                    logger.warning(f"Missing file: {video_id} - {file_path}")
                    self.downloaded_ids.remove(video_id)
        
        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing files; they will be downloaded again")
            self._save_all_downloaded_ids()

    def _is_license_verified(self, video):
        return video.get('license', '') == REQUIRED_VIDEO_LICENSE
    
    def _save_all_downloaded_ids(self):
        """ save all downloaded ids routine."""
        with open(self.downloaded_log, 'w', encoding='utf-8') as f:
            for video_id in sorted(self.downloaded_ids):
                f.write(f"{video_id}\n")
    
    def _save_downloaded_id(self, video_id):
        """ save downloaded id routine."""
        with open(self.downloaded_log, 'a', encoding='utf-8') as f:
            f.write(f"{video_id}\n")
        self.downloaded_ids.add(video_id)
    
    def _find_downloaded_file(self, video_id, category):
        """ find downloaded file routine."""
        category_dir = self.output_dir / category
        if not category_dir.exists():
            return None
        
        patterns = [
            f"{category}_{video_id}.mp4",
            f"{category}_{video_id}.mkv",
            f"{category}_{video_id}.webm",
        ]
        
        for pattern in patterns:
            file_path = category_dir / pattern
            if file_path.exists():
                return file_path
        
        return None
    
    def _cleanup_fragments(self, video_id, category):
        """ cleanup fragments routine."""
        category_dir = self.output_dir / category
        if not category_dir.exists():
            return
        
        patterns = [
            f"{category}_{video_id}.mp4.frag*",
            f"{category}_{video_id}.mp4-frag*",
            f"{category}_{video_id}.webm.frag*",
            f"{category}_{video_id}.webm-frag*",
            f"{category}_{video_id}.mkv.frag*",
            f"{category}_{video_id}.mkv-frag*",
            f"{category}_{video_id}.f*",
            f"{category}_{video_id}.part",
            f"{category}_{video_id}.ytdl",
        ]
        
        cleaned_count = 0
        for pattern in patterns:
            for file_path in category_dir.glob(pattern):
                try:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned up: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Cleanup failed: {file_path.name} - {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
    
    def cleanup_all_fragments(self):
        """Cleanup all fragments routine."""
        logger.info("Scanning and cleaning all remaining temporary files...")
        
        total_cleaned = 0
        patterns = [
            "*.frag",
            "*.frag*",
            "*-frag*",
            "*.part",
            "*.ytdl",
            "*.f[0-9]*",
        ]
        
        for category_dir in self.output_dir.iterdir():
            if category_dir.is_dir() and category_dir.name not in ['downloaded_videos.txt', 'download_record.json']:
                for pattern in patterns:
                    for file_path in category_dir.glob(pattern):
                        try:
                            file_path.unlink()
                            total_cleaned += 1
                            logger.debug(f"Cleaned up: {file_path}")
                        except Exception as e:
                            logger.warning(f"Cleanup failed: {file_path} - {e}")
        
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} temporary files in total")
        else:
            logger.info("No temporary files need cleanup")
    
    def _check_ytdlp_installed(self):
        """ check ytdlp installed routine."""
        try:
            result = subprocess.run(
                ['yt-dlp', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"yt-dlp version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Error: yt-dlp was not found")
            logger.error("Install it first: pip install yt-dlp")
            return False
    
    def download_video(
        self,
        video_id,
        url,
        title,
        category,
        duration_seconds=0,
        license_value='',
        license_checked_at='',
    ):
        """Download video routine."""
        if video_id in self.downloaded_ids:
            existing_file = self._find_downloaded_file(video_id, category)
            if existing_file and existing_file.exists():
                file_size_mb = existing_file.stat().st_size / (1024 * 1024)
                logger.info(f"Skipping already downloaded video: [{video_id}] {title} ({file_size_mb:.1f} MB)")
                return False
            else:
                logger.warning(f"Record exists but file is missing; downloading again: [{video_id}] {title}")
                self.downloaded_ids.remove(video_id)
        
        category_dir = self.output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        output_template = str(category_dir / f"{category}_{video_id}.%(ext)s")
        
        cmd = [
            'yt-dlp',
            url,
            '-o', output_template,
            '--no-playlist',
            '--no-warnings',
            '--progress',
        ]
        
        if self.quality == 'best':
            cmd.extend(['-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'])
        elif self.quality in ['1080p', '720p', '480p']:
            height = self.quality.replace('p', '')
            cmd.extend(['-f', f'bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'])
        
        if self.proxy:
            cmd.extend(['--proxy', self.proxy])
        
        cmd.extend([
            '--continue',
            '--no-check-certificate',
            '--retries', '10',
            '--fragment-retries', '10',
            '--concurrent-fragments', '1',
            '--sleep-interval', '2',
            '--max-sleep-interval', '5',
            '--sleep-requests', '1',
            '--extractor-retries', '5',
            '--file-access-retries', '5',
            '--hls-prefer-native',
            '--no-hls-use-mpegts',
        ])
        
        cmd.extend([
            '--merge-output-format', 'mp4',
            '--fixup', 'force',
            '--postprocessor-args', 'ffmpeg:-c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -ar 44100 -r 30 -vsync cfr -movflags +faststart',
            '--recode-video', 'mp4',
        ])
        
        logger.info("=" * 60)
        logger.info(f"Downloading: [{video_id}] {title}")
        logger.info(f"Theme: {category}")
        if duration_seconds:
            logger.info(f"Duration: {duration_seconds}s ({duration_seconds//60}m {duration_seconds%60}s)")
        logger.info(f"License: {license_value or 'missing'}")
        logger.info(f"URL: {url}")
        logger.info("-" * 60)
        logger.info("DOVER compatibility processing is enabled; videos will be re-encoded after download")
        logger.info("  - Constant frame rate (CFR) at 30 fps")
        logger.info("  - Standard H.264 encoding")
        logger.info("  - Fixed moov atom placement")
        logger.info("-" * 60)
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            downloaded_file = self._find_downloaded_file(video_id, category)
            
            if downloaded_file and downloaded_file.exists():
                file_size_mb = downloaded_file.stat().st_size / (1024 * 1024)
                
                self._save_downloaded_id(video_id)
                
                video_info = {
                    'video_id': video_id,
                    'title': title,
                    'category': category,
                    'url': url,
                    'duration_seconds': duration_seconds,
                    'license': license_value,
                    'license_checked_at': license_checked_at
                }
                self._save_download_record(video_id, video_info, downloaded_file)
                
                logger.info(f"Download succeeded: [{video_id}] {title}")
                logger.info(f"  File size: {file_size_mb:.1f} MB")
                logger.info(f"  Saved path: {downloaded_file}")
                logger.info("=" * 60)
                return True
            else:
                logger.error(f"Download finished but output file was not found: [{video_id}] {title}")
                logger.error("=" * 60)
                return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: [{video_id}] {title}")
            logger.error(f"Exit code: {e.returncode}")
            
            self._cleanup_fragments(video_id, category)
            
            logger.error("=" * 60)
            return False
        
        except KeyboardInterrupt:
            logger.warning("\nDownload interrupted by user")
            logger.warning("Tip: the next run will resume from this video automatically")
            raise
    
    def download_from_json(self, json_file, categories=None, limit=None):
        """Download from json routine."""
        logger.info(f"Reading: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        if 'manifest' in data:
            manifest = data['manifest']
        elif 'categories' in data:
            manifest = data['categories']
        else:
            logger.error("Error: unrecognized JSON format")
            return
        
        if categories:
            manifest = {k: v for k, v in manifest.items() if k in categories}
        
        total_videos = sum(len(videos) for videos in manifest.values())
        logger.info("=" * 60)
        logger.info("Download plan:")
        logger.info(f"  Themes: {len(manifest)}")
        logger.info(f"  Total videos: {total_videos}")
        if limit:
            logger.info(f"  Per-theme limit: {limit}")
        logger.info("=" * 60)
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        for category, videos in manifest.items():
            logger.info(f"\nTheme: {category} ({len(videos)} videos)")
            logger.info("-" * 60)
            
            videos_to_download = videos[:limit] if limit else videos
            
            for i, video in enumerate(videos_to_download, 1):
                video_id = video.get('video_id', '')
                url = video.get('url', '')
                title = video.get('title', '')
                duration = video.get('duration_seconds', 0)
                
                if not video_id or not url:
                    logger.warning(f"Skipping invalid video entry: {video}")
                    continue
                if not self._is_license_verified(video):
                    skip_count += 1
                    logger.warning(
                        f"Skipping unverified/non-{REQUIRED_VIDEO_LICENSE} video: "
                        f"[{video_id}] {title} (license={video.get('license', '') or 'missing'})"
                    )
                    continue
                
                logger.info(f"[{i}/{len(videos_to_download)}]")
                
                if video_id in self.downloaded_ids:
                    skip_count += 1
                    logger.info(f"Skipping already downloaded video: [{video_id}] {title}")
                    continue
                
                result = self.download_video(
                    video_id=video_id,
                    url=url,
                    title=title,
                    category=category,
                    duration_seconds=duration,
                    license_value=video.get('license', ''),
                    license_checked_at=video.get('license_checked_at', '')
                )
                
                if result:
                    success_count += 1
                    delay = random.uniform(5, 15)
                    logger.info(f"Waiting {delay:.1f}s before the next download...")
                    time.sleep(delay)
                else:
                    fail_count += 1
                    delay = random.uniform(10, 30)
                    logger.info(f"Download failed; waiting {delay:.1f}s before continuing...")
                    time.sleep(delay)
        
        logger.info("\n" + "=" * 60)
        logger.info("Download complete!")
        logger.info("=" * 60)
        logger.info(f"Succeeded: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Skipped: {skip_count}")
        logger.info(f"Total: {success_count + fail_count + skip_count}")
        logger.info("=" * 60)
    
    def download_from_csv(self, csv_file, categories=None, limit=None):
        """Download from csv routine."""
        logger.info(f"Reading: {csv_file}")
        
        videos_by_category = {}
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = row.get('category', 'unknown')
                if categories and category not in categories:
                    continue
                
                if category not in videos_by_category:
                    videos_by_category[category] = []
                
                videos_by_category[category].append({
                    'video_id': row.get('video_id', ''),
                    'url': row.get('url', ''),
                    'title': row.get('title', ''),
                    'duration_seconds': int(row.get('duration_seconds', 0)),
                    'license': row.get('license', ''),
                    'license_checked_at': row.get('license_checked_at', '')
                })
        
        manifest = videos_by_category
        
        total_videos = sum(len(videos) for videos in manifest.values())
        logger.info("=" * 60)
        logger.info("Download plan:")
        logger.info(f"  Themes: {len(manifest)}")
        logger.info(f"  Total videos: {total_videos}")
        if limit:
            logger.info(f"  Per-theme limit: {limit}")
        logger.info("=" * 60)
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        for category, videos in manifest.items():
            logger.info(f"\nTheme: {category} ({len(videos)} videos)")
            logger.info("-" * 60)
            
            videos_to_download = videos[:limit] if limit else videos
            
            for i, video in enumerate(videos_to_download, 1):
                video_id = video.get('video_id', '')
                url = video.get('url', '')
                title = video.get('title', '')
                duration = video.get('duration_seconds', 0)
                
                if not video_id or not url:
                    continue
                if not self._is_license_verified(video):
                    skip_count += 1
                    logger.warning(
                        f"Skipping unverified/non-{REQUIRED_VIDEO_LICENSE} video: "
                        f"[{video_id}] {title} (license={video.get('license', '') or 'missing'})"
                    )
                    continue
                
                logger.info(f"[{i}/{len(videos_to_download)}]")
                
                if video_id in self.downloaded_ids:
                    skip_count += 1
                    logger.info(f"Skipping already downloaded video: [{video_id}] {title}")
                    continue
                
                result = self.download_video(
                    video_id=video_id,
                    url=url,
                    title=title,
                    category=category,
                    duration_seconds=duration,
                    license_value=video.get('license', ''),
                    license_checked_at=video.get('license_checked_at', '')
                )
                
                if result:
                    success_count += 1
                else:
                    fail_count += 1
        
        logger.info("\n" + "=" * 60)
        logger.info("Download complete!")
        logger.info("=" * 60)
        logger.info(f"Succeeded: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Skipped: {skip_count}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Batch YouTube video downloader with per-theme support'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input file path (JSON or CSV)'
    )
    parser.add_argument(
        '--output-dir',
        default=str(REFVIDEO_DIR),
        help='Output root directory (default: ref4d_build/ref_collect/downloaded_videos)'
    )
    parser.add_argument(
        '--category',
        action='append',
        help='Theme(s) to download. Can be passed multiple times; omit to download all themes'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of videos to download per theme (default: all)'
    )
    parser.add_argument(
        '--state-dir',
        default=str(STATE_DIR),
        help='Directory for resume state files'
    )
    parser.add_argument(
        '--proxy',
        default=None,
        help='Proxy URL, for example: http://127.0.0.1:7897'
    )
    parser.add_argument(
        '--quality',
        default='best',
        choices=['best', '1080p', '720p', '480p'],
        help='Video quality (default: best)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean all remaining temporary files (.frag, .part, etc.)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}")
        return
    
    downloader = VideoDownloader(
        output_dir=args.output_dir,
        state_dir=args.state_dir,
        proxy=args.proxy,
        quality=args.quality
    )
    
    if not downloader._check_ytdlp_installed():
        return
    
    if args.cleanup:
        downloader.cleanup_all_fragments()
        logger.info("\nCleanup complete!")
        return
    
    if args.input.endswith('.json'):
        downloader.download_from_json(
            args.input,
            categories=args.category,
            limit=args.limit
        )
    elif args.input.endswith('.csv'):
        downloader.download_from_csv(
            args.input,
            categories=args.category,
            limit=args.limit
        )
    else:
        print(f"Error: unsupported file format: {args.input}")
        print("Supported formats: .json, .csv")


if __name__ == '__main__':
    main()

