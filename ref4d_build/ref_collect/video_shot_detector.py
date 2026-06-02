#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shot-change detection and clip extraction (CPU path).

Input videos are expected under ref4d_build/ref_collect/downloaded_videos/<category>/*.mp4.
Output clips are written under ref4d_build/ref_collect/downloaded_clip.
"""

import argparse
import random
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
import json
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
REFVIDEO_DIR = SCRIPT_DIR / 'downloaded_videos'
CPU_CLIP_DIR = SCRIPT_DIR / 'downloaded_clip'
STATE_DIR = SCRIPT_DIR / 'state'
LOG_DIR = SCRIPT_DIR / 'logs'

for _dir in (REFVIDEO_DIR, CPU_CLIP_DIR, STATE_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
except ImportError:
    print("Error: scenedetect is not installed. Install it with: pip install scenedetect")
    exit(1)


class VideoShotDetector:
    def __init__(self, 
                 input_dir: str = str(REFVIDEO_DIR),
                 output_dir: str = str(CPU_CLIP_DIR),
                 ffmpeg_path: str = 'ffmpeg',
                 progress_file: str = str(STATE_DIR / 'shot_detection_progress.json')):
        """  init   routine."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.ffmpeg_path = ffmpeg_path
        self.progress_file = Path(progress_file)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = STATE_DIR / 'temp_clips'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(LOG_DIR / 'shot_detection.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.shot_threshold = 35.0
        
        self.processed_videos: Set[str] = self._load_progress()
        
        self.themes = [
            'animals_and_ecology',
            'architecture', 
            'commercial_marketing',
            'food',
            'industrial_activity',
            'landscape',
            'people_daily',
            'sports_competition',
            'transportation'
        ]
        
    def _load_progress(self) -> Set[str]:
        """ load progress routine."""
        if not self.progress_file.exists():
            return set()
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed = set(data.get('processed_videos', []))
                self.logger.info(f"Resumed from progress file: {len(processed)} videos already processed")
                return processed
        except Exception as e:
            self.logger.warning(f"Failed to load progress file: {e}; starting from scratch")
            return set()
    
    def _save_progress(self, video_path: Path):
        """ save progress routine."""
        try:
            relative_path = str(video_path.relative_to(self.input_dir))
            self.processed_videos.add(relative_path)
            
            data = {
                'last_update': datetime.now().isoformat(),
                'processed_count': len(self.processed_videos),
                'processed_videos': sorted(list(self.processed_videos))
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def _is_processed(self, video_path: Path) -> bool:
        """ is processed routine."""
        try:
            relative_path = str(video_path.relative_to(self.input_dir))
            return relative_path in self.processed_videos
        except:
            return False
    
    def check_ffmpeg(self) -> bool:
        """Check ffmpeg routine."""
        try:
            result = subprocess.run([self.ffmpeg_path, '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def get_available_themes(self) -> List[str]:
        """Get available themes routine."""
        available = []
        for theme in self.themes:
            theme_dir = self.input_dir / theme
            if theme_dir.exists() and theme_dir.is_dir():
                available.append(theme)
        return available
    
    def get_theme_videos(self, theme: str) -> List[Path]:
        """Get theme videos routine."""
        theme_dir = self.input_dir / theme
        if not theme_dir.exists():
            self.logger.warning(f"Theme folder does not exist: {theme_dir}")
            return []
        
        videos = list(theme_dir.glob("*.mp4"))
        self.logger.info(f"Theme '{theme}' has {len(videos)} video files")
        return sorted(videos)
    
    def detect_shot_cuts(self, video_path: Path) -> Tuple[bool, List[float]]:
        """Detect shot cuts routine."""
        try:
            video_manager = VideoManager([str(video_path)])
            scene_manager = SceneManager()
            
            scene_manager.add_detector(ContentDetector(threshold=self.shot_threshold))
            video_manager.set_downscale_factor()
            
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            video_manager.release()
            
            if len(scene_list) <= 1:
                self.logger.info(f"Video {video_path.name} has no shot changes")
                return False, []
            
            valid_shots = []
            for shot in scene_list:
                duration = shot[1].get_seconds() - shot[0].get_seconds()
                if duration >= 3.0:
                    valid_shots.append(shot)
            
            if len(valid_shots) <= 1:
                self.logger.info(f"Video {video_path.name} has no valid shot changes")
                return False, []
            
            cut_points = []
            for i in range(1, len(valid_shots)):
                cut_time = valid_shots[i][0].get_seconds()
                cut_points.append(cut_time)
            
            self.logger.info(f"Video {video_path.name} has {len(cut_points)} detected shot-change points")
            return True, cut_points
            
        except Exception as e:
            self.logger.error(f"Shot detection failed: {video_path.name} - {str(e)}")
            return False, []
    
    def extract_clip(self, video_path: Path, start_time: float, duration: float, 
                    output_path: Path) -> bool:
        """Extract clip routine."""
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-profile:v', 'high',      # H.264 profile
                '-level', '4.0',           # H.264 level
                '-pix_fmt', 'yuv420p',
                '-r', '30',
                '-vsync', 'cfr',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                '-movflags', '+faststart',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"FFmpeg error: {result.stderr.decode('utf-8', errors='ignore')}")
                return False
            
            return output_path.exists() and output_path.stat().st_size > 0
            
        except Exception as e:
            self.logger.error(f"Clip extraction failed: {str(e)}")
            return False
    
    def create_multi_shot_clip(self, video_path: Path, cut_points: List[float], 
                              output_path: Path) -> bool:
        """Create multi shot clip routine."""
        if not cut_points:
            return False
        
        video_duration = self.get_video_duration(video_path)
        if video_duration == 0:
            self.logger.error(f"Unable to read video duration; skipping multi-shot processing: {video_path.name}")
            return False
        
        cut_point = cut_points[len(cut_points)//2]
        
        if cut_point >= video_duration - 1:
            self.logger.warning(f"Cut point ({cut_point:.1f}s) is close to the video end; using the first cut point")
            cut_point = cut_points[0] if cut_points[0] < video_duration - 5 else video_duration / 2
        
        total_duration = random.uniform(10, 20)
        
        before_duration = total_duration / 2
        after_duration = total_duration / 2
        
        before_start = max(0, cut_point - before_duration)
        actual_before_duration = cut_point - before_start
        
        available_after = video_duration - cut_point
        if after_duration > available_after:
            after_duration = max(0, available_after - 0.5)
            self.logger.warning(f"Adjusted post-cut duration: {after_duration:.1f}s (remaining video: {available_after:.1f}s)")
        
        temp_dir = self.temp_dir
        temp_dir.mkdir(exist_ok=True)
        
        temp_before = temp_dir / f"before_{output_path.stem}.mp4"
        temp_after = temp_dir / f"after_{output_path.stem}.mp4"
        
        try:
            if not self.extract_clip(video_path, before_start, actual_before_duration, temp_before):
                return False
            
            if not self.extract_clip(video_path, cut_point, after_duration, temp_after):
                return False
            
            concat_list = temp_dir / f"concat_{output_path.stem}.txt"
            with open(concat_list, 'w', encoding='utf-8') as f:
                f.write(f"file '{temp_before.absolute()}'\n")
                f.write(f"file '{temp_after.absolute()}'\n")
            
            concat_cmd = [
                self.ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list),
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-profile:v', 'high',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p',
                '-r', '30',
                '-vsync', 'cfr',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                '-movflags', '+faststart',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(concat_cmd, capture_output=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Concatenation failed: {result.stderr.decode('utf-8', errors='ignore')}")
                success = False
            else:
                success = output_path.exists() and output_path.stat().st_size > 0
            
            for temp_file in [temp_before, temp_after, concat_list]:
                if temp_file.exists():
                    temp_file.unlink()
            
            try:
                if temp_dir.exists() and not list(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except:
                pass
            
            if success:
                actual_total = actual_before_duration + after_duration
                self.logger.info(f"Multi-shot clip generated: {output_path.name} (cut point: {cut_point:.1f}s, total duration: {actual_total:.1f}s)")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-shot clip: {str(e)}")
            try:
                temp_dir = self.temp_dir
                if temp_dir.exists():
                    for temp_file in temp_dir.glob(f"*{output_path.stem}*"):
                        temp_file.unlink()
            except:
                pass
            return False
    
    def get_video_duration(self, video_path: Path) -> float:
        """Get video duration routine."""
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', str(video_path),
                '-f', 'null',
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stderr
            
            import re
            match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', output)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = float(match.group(3))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return total_seconds
            else:
                self.logger.warning(f"Unable to read video duration: {video_path.name}")
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to get video duration: {str(e)}")
            return 0
    
    def create_single_shot_clip(self, video_path: Path, output_path: Path) -> bool:
        """Create single shot clip routine."""
        video_duration = self.get_video_duration(video_path)
        
        if video_duration == 0:
            self.logger.error(f"Unable to read video duration; skipping: {video_path.name}")
            return False
        
        clip_duration = random.uniform(5, 10)
        
        min_start = 5.0
        max_end = video_duration - 5.0
        max_start = max_end - clip_duration
        
        if max_start <= min_start:
            if video_duration <= 10:
                start_time = 0
                clip_duration = video_duration
                self.logger.warning(f"Video is too short ({video_duration:.1f}s); extracting the full video: {video_path.name}")
            else:
                start_time = (video_duration - clip_duration) / 2
                self.logger.warning(f"Video is short ({video_duration:.1f}s); extracting from the middle: {video_path.name}")
        else:
            start_time = random.uniform(min_start, max_start)
        
        success = self.extract_clip(video_path, start_time, clip_duration, output_path)
        
        if success:
            self.logger.info(f"Single-shot clip generated: {output_path.name} ({start_time:.1f}s-{start_time+clip_duration:.1f}s, source duration: {video_duration:.1f}s)")
        
        return success
    
    def process_theme(self, theme: str) -> Dict[str, int]:
        """Process theme routine."""
        videos = self.get_theme_videos(theme)
        if not videos:
            self.logger.warning(f"No video files found for theme '{theme}'")
            return {"single": 0, "multi": 0, "failed": 0, "skipped": 0}
        
        theme_dir = self.output_dir / theme
        single_dir = theme_dir / "single"
        multi_dir = theme_dir / "multi"
        
        single_dir.mkdir(parents=True, exist_ok=True)
        multi_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Processing {len(videos)} videos for theme '{theme}'...")
        self.logger.info("Output directories:")
        self.logger.info(f"  single-shot: {single_dir}")
        self.logger.info(f"  multi-shot: {multi_dir}")
        self.logger.info("=" * 60)
        
        stats = {"single": 0, "multi": 0, "failed": 0, "skipped": 0}
        
        for idx, video_path in enumerate(videos, 1):
            self.logger.info(f"[{idx}/{len(videos)}] Processing: {video_path.name}")
            
            if self._is_processed(video_path):
                self.logger.info("  -> Skipped (already processed)")
                stats["skipped"] += 1
                continue
            
            try:
                has_cuts, cut_points = self.detect_shot_cuts(video_path)
                
                original_stem = video_path.stem
                
                if has_cuts:
                    output_name = f"{original_stem}_multi.mp4"
                    output_path = multi_dir / output_name
                    
                    if self.create_multi_shot_clip(video_path, cut_points, output_path):
                        stats["multi"] += 1
                        self.logger.info(f"  -> Multi-shot clip generated: {output_name}")
                        self._save_progress(video_path)
                    else:
                        output_name = f"{original_stem}_single.mp4"
                        output_path = single_dir / output_name
                        
                        if self.create_single_shot_clip(video_path, output_path):
                            stats["single"] += 1
                            self.logger.info(f"  -> Multi-shot failed; single-shot fallback succeeded: {output_name}")
                            self._save_progress(video_path)
                        else:
                            stats["failed"] += 1
                            self.logger.error("  -> Clip generation failed completely")
                else:
                    output_name = f"{original_stem}_single.mp4"
                    output_path = single_dir / output_name
                    
                    if self.create_single_shot_clip(video_path, output_path):
                        stats["single"] += 1
                        self.logger.info(f"  -> Single-shot clip generated: {output_name}")
                        self._save_progress(video_path)
                    else:
                        stats["failed"] += 1
                        self.logger.error("  -> Single-shot clip generation failed")
            
            except Exception as e:
                stats["failed"] += 1
                self.logger.error(f"  -> Processing error: {str(e)}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"Theme '{theme}' processing complete:")
        self.logger.info(f"  single-shot: {stats['single']}")
        self.logger.info(f"  multi-shot: {stats['multi']}")
        self.logger.info(f"  failed: {stats['failed']}")
        self.logger.info(f"  skipped: {stats['skipped']}")
        self.logger.info(f"  total videos: {len(videos)}")
        self.logger.info("=" * 60)
        
        return stats
    
    def process_all(self, themes_input: str = None) -> Dict[str, Dict[str, int]]:
        """Process all routine."""
        if not self.check_ffmpeg():
            self.logger.error("ffmpeg is unavailable; make sure ffmpeg is installed and on PATH")
            return {}
        
        available_themes = self.get_available_themes()
        if not available_themes:
            self.logger.error(f"No theme folders found in {self.input_dir}")
            self.logger.info(f"Expected theme folders: {', '.join(self.themes)}")
            return {}
        
        if themes_input:
            if ',' in themes_input:
                input_themes = [t.strip() for t in themes_input.split(',')]
            else:
                input_themes = themes_input.split()
            
            themes_to_process = []
            for theme in input_themes:
                if theme not in self.themes:
                    self.logger.error(f"Invalid theme: {theme}")
                    self.logger.info(f"Available themes: {', '.join(self.themes)}")
                    return {}
                
                if theme not in available_themes:
                    self.logger.error(f"Theme folder does not exist: {self.input_dir / theme}")
                    return {}
                
                themes_to_process.append(theme)
            
            self.logger.info(f"Processing selected themes: {', '.join(themes_to_process)}")
        else:
            themes_to_process = available_themes
            self.logger.info(f"Processing all available themes: {', '.join(themes_to_process)}")
        
        all_stats = {}
        for theme in themes_to_process:
            try:
                stats = self.process_theme(theme)
                all_stats[theme] = stats
            except KeyboardInterrupt:
                self.logger.warning("\nInterrupt detected; saving progress...")
                self.logger.info(f"Processed {len(self.processed_videos)} videos")
                raise
            except Exception as e:
                self.logger.error(f"Error while processing theme {theme}: {e}")
                all_stats[theme] = {"single": 0, "multi": 0, "failed": 0, "skipped": 0}
        
        results_file = self.output_dir / "shot_detection_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved: {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
        
        total_single = sum(s.get("single", 0) for s in all_stats.values())
        total_multi = sum(s.get("multi", 0) for s in all_stats.values())
        total_failed = sum(s.get("failed", 0) for s in all_stats.values())
        total_skipped = sum(s.get("skipped", 0) for s in all_stats.values())
        total_processed = total_single + total_multi + total_failed
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Shot detection processing summary:")
        self.logger.info("=" * 60)
        self.logger.info(f"Single-shot clips: {total_single}")
        self.logger.info(f"Multi-shot clips: {total_multi}")
        self.logger.info(f"Failed clips: {total_failed}")
        self.logger.info(f"Skipped clips: {total_skipped}")
        self.logger.info(f"Processed in this run: {total_processed} videos")
        
        if total_processed > 0:
            success_rate = (total_single + total_multi) / total_processed * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        self.logger.info("=" * 60)
        
        return all_stats


def main():
    """Main routine."""
    parser = argparse.ArgumentParser(
        description='Video shot-change detector with per-theme processing and resume support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_shot_detector.py
  
  python video_shot_detector.py --theme architecture
  
  python video_shot_detector.py --input ref4d_build/ref_collect/downloaded_videos --output ref4d_build/ref_collect/downloaded_clip
  
  python video_shot_detector.py --list-themes
        """
    )
    
    parser.add_argument(
        '--theme',
        type=str,
        default=None,
        help='Theme(s) to process. Omit to process all themes.\n' +
             'Multiple themes are supported: --theme "architecture,food,landscape" or --theme "architecture food landscape"'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(REFVIDEO_DIR),
        help='Input video directory (default: ref4d_build/ref_collect/downloaded_videos)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(CPU_CLIP_DIR),
        help='Output clip directory (default: ref4d_build/ref_collect/downloaded_clip)'
    )
    parser.add_argument(
        '--ffmpeg',
        type=str,
        default='ffmpeg',
        help='Path to the ffmpeg executable (default: ffmpeg)'
    )
    parser.add_argument(
        '--progress-file',
        type=str,
        default=str(STATE_DIR / 'shot_detection_progress.json'),
        help='Resume progress file (default: shot_detection_progress.json)'
    )
    parser.add_argument(
        '--list-themes',
        action='store_true',
        help='List all available themes'
    )
    
    args = parser.parse_args()
    
    detector = VideoShotDetector(
        input_dir=args.input,
        output_dir=args.output,
        ffmpeg_path=args.ffmpeg,
        progress_file=args.progress_file
    )
    
    if args.list_themes:
        print("=" * 60)
        print("Available themes:")
        print("=" * 60)
        available = detector.get_available_themes()
        for i, theme in enumerate(detector.themes, 1):
            status = "✓" if theme in available else "✗"
            video_count = len(detector.get_theme_videos(theme)) if theme in available else 0
            print(f"{i:2d}. [{status}] {theme:25s} ({video_count} videos)")
        print("=" * 60)
        print(f"Input directory: {args.input}")
        print(f"Found {len(available)} available themes")
        return
    
    print("=" * 60)
    print("Video shot-change detector")
    print("=" * 60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Progress file: {args.progress_file}")
    if args.theme:
        print(f"Themes to process: {args.theme}")
    else:
        print("Themes to process: all available themes")
    print("=" * 60)
    print("Features:")
    print("  - Detect video shot-change points")
    print("  - Generate 10-20 second multi-shot clips that include the cut point")
    print("  - Generate 5-10 second single-shot clips")
    print("  - Support resumable processing")
    print("=" * 60)
    print()
    
    try:
        results = detector.process_all(themes_input=args.theme)
        
        if results:
            print("\nProcessing complete!")
            print(f"Output directory: {args.output}")
            print(f"Result file: {args.output}/shot_detection_results.json")
        else:
            print("\nProcessing failed")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("Progress has been saved; the next run will resume from the checkpoint")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
