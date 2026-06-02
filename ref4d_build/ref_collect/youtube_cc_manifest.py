#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YouTube CC video metadata collector for Ref4D reference data.

This script uses the YouTube Data API v3, supports resume mode, and writes
artifacts into ref4d_build/ref_collect and ref4d_build/ref_collect/state."""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import time
import logging
import socket
import random
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import httplib2
except ImportError:
    print("Install google-api-python-client first: pip install google-api-python-client")
    exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
METADATA_DIR = SCRIPT_DIR
STATE_DIR = SCRIPT_DIR / 'state'
LOG_DIR = SCRIPT_DIR / 'logs'

for _dir in (METADATA_DIR, STATE_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / 'youtube_manifest.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REQUIRED_VIDEO_LICENSE = 'creativeCommon'
MAX_DURATION_SECONDS = 600


CATEGORY_KEYWORDS = {
    'animals_and_ecology': [
        'wildlife', 'animals', 'nature documentary', 'safari', 'marine life',
        'birds', 'mammals', 'ecosystem', 'biodiversity', 'animal behavior',
        'wild animals', 'ecology', 'conservation', 'endangered species',
        'ocean life', 'jungle', 'forest animals', 'pets', 'zoo', 'aquarium',
        'big cats', 'lions', 'tigers', 'leopards', 'cheetah', 'elephants',
        'whales', 'dolphins', 'sharks', 'sea turtles', 'penguins', 'eagles',
        'owls', 'parrots', 'hummingbirds', 'bears', 'wolves', 'foxes',
        'deer', 'monkeys', 'gorillas', 'pandas', 'koalas', 'kangaroos',
        'reptiles', 'snakes', 'lizards', 'crocodiles', 'frogs', 'insects',
        'butterflies', 'bees', 'ants', 'spiders', 'coral reef', 'fish',
        'rainforest', 'savanna', 'wetlands', 'mangrove', 'arctic wildlife',
        'desert animals', 'mountain animals', 'tropical animals', 'nocturnal animals',
        'migration', 'hibernation', 'predator', 'prey', 'food chain',
        'wildlife photography', 'animal documentary', 'nature reserve',
        'wildlife sanctuary', 'animal rescue', 'species protection',
        'habitat restoration', 'wildlife monitoring', 'animal tracking',
        'animal ecology', 'marine organisms', 'avian wildlife', 'mammalian wildlife',
        'biodiversity protection', 'rare animals', 'protected species',
        'wildlife behavior', 'nature reserve animals', 'wildlife camera'
    ],
    'architecture': [
        'architecture', 'building', 'construction', 'modern architecture',
        'contemporary architecture', 'minimalist architecture', 'brutalist',
        'art deco', 'bauhaus', 'gothic architecture', 'baroque architecture',
        'renaissance architecture', 'classical architecture', 'neoclassical',
        'postmodern architecture', 'deconstructivism', 'futuristic architecture',
        'skyscraper', 'high rise', 'residential building', 'commercial building',
        'office building', 'tower', 'landmark', 'monument', 'cathedral',
        'temple', 'mosque', 'museum architecture', 'library architecture',
        'stadium architecture', 'bridge design', 'pavilion', 'villa',
        'interior design', 'exterior design', 'architectural design', 'facade',
        'structure', 'urban design', 'landscape architecture', 'sustainable design',
        'green building', 'eco architecture', 'glass architecture', 'wooden architecture',
        'concrete architecture', 'steel structure', 'architectural details',
        'historical buildings', 'heritage building', 'ancient architecture',
        'castle', 'palace', 'fortress', 'ruins', 'restoration',
        'smart building', 'parametric design', '3D architecture', 'innovative design',
        'architectural visualization', 'building technology', 'prefab architecture',
        'modular construction', 'adaptive reuse', 'mixed use development',
        'building design', 'interior architecture', 'skyscrapers', 'landmark buildings',
        'modern buildings', 'ancient buildings', 'architectural art',
        'architectural aesthetics', 'city design', 'building facade', 'building details'
    ],
    'commercial_marketing': [
        'advertising', 'marketing', 'commercial', 'advertisement', 'promo',
        'tv commercial', 'video ad', 'digital marketing', 'social media marketing',
        'content marketing', 'influencer marketing', 'email marketing',
        'viral marketing', 'guerrilla marketing', 'experiential marketing',
        'brand', 'branding', 'brand identity', 'brand strategy', 'brand awareness',
        'brand building', 'brand positioning', 'rebranding', 'brand launch',
        'brand storytelling', 'brand experience', 'brand design',
        'promotion', 'product launch', 'campaign', 'marketing campaign',
        'promotional video', 'product demo', 'product showcase', 'product review',
        'unboxing', 'testimonial', 'case study', 'success story',
        'business', 'corporate', 'corporate video', 'company profile',
        'startup', 'entrepreneurship', 'business model', 'pitch deck',
        'investor pitch', 'business presentation', 'company culture',
        'marketing strategy', 'sales', 'customer engagement', 'lead generation',
        'conversion optimization', 'call to action', 'marketing funnel',
        'customer journey', 'user acquisition', 'retention strategy',
        'growth hacking', 'market research', 'competitive analysis',
        'video marketing', 'storytelling ad', 'emotional advertising',
        'creative advertising', 'funny commercial', 'inspiring ad',
        'animated commercial', 'motion graphics ad', 'explainer video',
        'advertising campaign', 'marketing promotion', 'business branding',
        'product release', 'digital promotion', 'social media advertising',
        'content marketing campaign', 'brand development', 'creative ad',
        'corporate promotion', 'brand story'
    ],
    'food': [
        'cooking', 'baking', 'grilling', 'roasting', 'frying', 'steaming',
        'boiling', 'sautéing', 'slow cooking', 'pressure cooking', 'air frying',
        'smoking', 'fermenting', 'pickling', 'braising', 'poaching',
        'recipe', 'food', 'cuisine', 'dishes', 'meal', 'dessert',
        'pastry', 'cake', 'bread', 'pizza', 'pasta', 'noodles', 'soup',
        'salad', 'sandwich', 'burger', 'sushi', 'barbecue', 'seafood',
        'steak', 'chicken', 'vegetarian', 'vegan', 'healthy food',
        'restaurant', 'cafe', 'bakery', 'kitchen', 'street food',
        'food truck', 'fine dining', 'casual dining', 'fast food',
        'food court', 'food market', 'food festival', 'buffet',
        'culinary', 'gastronomy', 'gourmet', 'foodie', 'chef',
        'master chef', 'home cooking', 'traditional recipe', 'authentic cuisine',
        'fusion cuisine', 'molecular gastronomy', 'food art', 'plating',
        'meal prep', 'cooking tutorial', 'cooking tips', 'cooking techniques',
        'knife skills', 'ingredients', 'spices', 'seasoning', 'garnish',
        'italian food', 'french cuisine', 'chinese food', 'japanese food',
        'korean food', 'thai food', 'indian cuisine', 'mexican food',
        'mediterranean food', 'asian cuisine', 'american food',
        'food review', 'taste test', 'food challenge', 'mukbang',
        'cooking show', 'food vlog', 'food tour', 'food photography',
        'food culture', 'home cooking recipe', 'restaurant food', 'bakery dessert',
        'chef cooking', 'homestyle dishes', 'western dishes', 'japanese dishes',
        'snacks', 'street snacks', 'restaurant review', 'food tutorial',
        'food preparation', 'food sharing'
    ],
    'industrial_activity': [
        'factory', 'manufacturing', 'industrial', 'production', 'assembly line',
        'mass production', 'industrial process', 'fabrication', 'processing plant',
        'production line', 'manufacturing plant', 'industrial facility',
        'production process', 'quality control', 'lean manufacturing',
        'machinery', 'equipment', 'industrial machinery', 'heavy machinery',
        'construction equipment', 'excavator', 'crane', 'forklift',
        'conveyor belt', 'robotic arm', 'cnc machine', 'lathe', 'mill',
        'engineering', 'mechanical engineering', 'industrial engineering',
        'process engineering', 'automation', 'robotics', 'industrial automation',
        'smart factory', 'industry 4.0', 'iot manufacturing', 'digital factory',
        'heavy industry', 'steel industry', 'metal industry', 'automotive industry',
        'aerospace industry', 'shipbuilding', 'chemical industry', 'petrochemical',
        'oil refinery', 'power plant', 'nuclear plant', 'mining', 'quarry',
        'oil and gas', 'petroleum', 'drilling', 'extraction', 'mining operation',
        'coal mining', 'mineral extraction', 'ore processing', 'smelting',
        'metal casting', 'welding', 'machining', 'forging',
        'construction', 'construction site', 'building construction',
        'civil engineering', 'infrastructure', 'road construction',
        'bridge construction', 'demolition', 'excavation',
        'warehouse', 'logistics', 'distribution center', 'supply chain',
        'inventory management', 'packaging', 'shipping', 'loading',
        'technology', 'innovation', 'industrial design', 'prototyping',
        'testing', 'research and development', 'laboratory', 'industrial safety',
        'industrial manufacturing', 'factory production', 'industrial machinery',
        'automation equipment', 'heavy industry', 'manufacturing line',
        'industrial equipment', 'engineering work', 'mining industry',
        'oil industry', 'chemical plant', 'steel production', 'automotive manufacturing',
        'aerospace manufacturing', 'smart manufacturing', 'industrial robots',
        'warehouse logistics'
    ],
    'landscape': [
        'landscape', 'nature', 'scenery', 'scenic view', 'natural beauty',
        'panorama', 'vista', 'horizon', 'wilderness', 'pristine nature',
        'mountains', 'mountain range', 'peak', 'summit', 'alpine', 'hills',
        'valley', 'canyon', 'gorge', 'cliff', 'plateau', 'highland',
        'mountain pass', 'mountain vista', 'rocky mountains', 'snow peak',
        'beach', 'coastline', 'seashore', 'ocean view', 'sea', 'lake',
        'river', 'stream', 'waterfall', 'cascade', 'rapids', 'hot spring',
        'lagoon', 'bay', 'inlet', 'fjord', 'island', 'archipelago',
        'forest', 'woods', 'jungle', 'rainforest', 'tropical forest',
        'pine forest', 'bamboo forest', 'woodland', 'grove', 'trees',
        'meadow', 'grassland', 'prairie', 'savanna', 'tundra',
        'desert', 'sand dunes', 'oasis', 'badlands', 'cave', 'karst',
        'volcano', 'geothermal', 'geyser', 'glacier', 'iceberg', 'ice field',
        'sunset', 'sunrise', 'twilight', 'golden hour', 'blue hour',
        'clouds', 'cloudscape', 'storm', 'lightning', 'rainbow',
        'starry night', 'milky way', 'northern lights', 'aurora',
        'spring landscape', 'summer landscape', 'autumn foliage', 'fall colors',
        'winter landscape', 'snow landscape', 'cherry blossom', 'flower field',
        'aerial view', 'drone footage', 'bird eye view', 'time lapse',
        'nature photography', 'landscape photography', 'scenic drive',
        'hiking', 'trekking', 'backpacking', 'camping',
        'travel', 'national park', 'nature reserve', 'countryside',
        'scenic spot', 'tourist attraction', 'natural wonder', 'world heritage',
        'natural landscape', 'mountain scenery', 'beach scenery', 'sunset view',
        'sunrise view', 'forest scenery', 'lake view', 'waterfall view',
        'canyon scenery', 'grassland landscape', 'desert scenery',
        'countryside scenery', 'historic scenic site', 'tourist scenic spot',
        'aerial landscape', 'seasonal scenery', 'sea of clouds', 'starry sky'
    ],
    'people_daily': [
        'daily life', 'everyday life', 'lifestyle', 'life style', 'daily routine',
        'morning routine', 'night routine', 'day in the life', 'real life',
        'daily activities', 'daily habits', 'life vlog', 'daily vlog',
        'vlog', 'vlogger', 'personal vlog', 'family vlog', 'travel vlog',
        'student vlog', 'work vlog', 'weekend vlog', 'vacation vlog',
        'family', 'family life', 'home life', 'household', 'parenting',
        'kids', 'children', 'baby', 'marriage', 'couple', 'relationship',
        'siblings', 'grandparents', 'family time', 'family activities',
        'people', 'social life', 'friends', 'friendship', 'gathering',
        'party', 'celebration', 'birthday', 'anniversary', 'reunion',
        'hangout', 'meet up', 'social gathering', 'community',
        'personal', 'personal development', 'self improvement', 'motivation',
        'inspiration', 'life lessons', 'life experience', 'story time',
        'life story', 'challenges', 'overcoming obstacles', 'success',
        'minimalism', 'simple living', 'sustainable living', 'slow living',
        'urban living', 'city life', 'suburban life', 'rural life',
        'apartment living', 'house tour', 'room tour', 'organization',
        'cleaning', 'decluttering', 'home organization', 'productivity',
        'culture', 'cultural', 'traditions', 'customs', 'festival',
        'ceremony', 'ritual', 'heritage', 'local culture', 'lifestyle culture',
        'hobbies', 'leisure', 'entertainment', 'recreation', 'pastime',
        'reading', 'music', 'art', 'crafts', 'diy', 'gardening',
        'pet care', 'cooking at home', 'shopping', 'errands',
        'work life balance', 'student life', 'school life', 'college life',
        'office life', 'remote work', 'work from home', 'study routine',
        'everyday lifestyle', 'daily vlog', 'family daily life', 'life diary',
        'daily sharing', 'life record', 'household life', 'daily chores',
        'personal life', 'social gathering', 'friends gathering', 'simple living',
        'city lifestyle', 'rural lifestyle', 'life tips', 'home living'
    ],
    'sports_competition': [
        'sports', 'sport', 'competition', 'game', 'match', 'tournament',
        'championship', 'league', 'playoff', 'final', 'semifinal',
        'olympic', 'olympics', 'world cup', 'athletic', 'athletics',
        'athlete', 'player', 'team', 'professional sports', 'amateur sports',
        'sports performance', 'sports skills', 'sports technique',
        'football', 'soccer', 'basketball', 'baseball', 'volleyball',
        'tennis', 'badminton', 'table tennis', 'ping pong', 'golf',
        'rugby', 'cricket', 'hockey', 'handball', 'squash',
        'running', 'sprint', 'marathon', 'track and field', 'long jump',
        'high jump', 'pole vault', 'hurdles', 'relay race', 'javelin',
        'shot put', 'discus', 'hammer throw', 'decathlon',
        'swimming', 'diving', 'water polo', 'synchronized swimming',
        'surfing', 'sailing', 'rowing', 'kayaking', 'canoeing',
        'water sports', 'aquatic sports',
        'gymnastics', 'artistic gymnastics', 'rhythmic gymnastics',
        'martial arts', 'karate', 'judo', 'taekwondo', 'boxing',
        'wrestling', 'mma', 'mixed martial arts', 'fencing', 'kung fu',
        'skiing', 'snowboarding', 'ice skating', 'figure skating',
        'speed skating', 'ice hockey', 'curling', 'winter sports',
        'extreme sports', 'skateboarding', 'bmx', 'parkour', 'rock climbing',
        'bouldering', 'mountain biking', 'motocross', 'base jumping',
        'skydiving', 'bungee jumping', 'adventure sports',
        'racing', 'car racing', 'formula 1', 'f1', 'rally', 'drag racing',
        'motorcycle racing', 'motogp', 'nascar', 'motorsport',
        'training', 'workout', 'exercise', 'fitness', 'gym', 'weight lifting',
        'bodybuilding', 'crossfit', 'hiit', 'cardio', 'strength training',
        'sports training', 'conditioning', 'warm up', 'cool down',
        'cycling', 'hiking', 'trail running', 'triathlon', 'endurance sports',
        'adventure race', 'orienteering', 'outdoor sports',
        'competitive sports', 'fitness training', 'soccer match', 'basketball game',
        'tennis match', 'badminton match', 'table tennis match', 'swimming race',
        'track and field', 'marathon race', 'gymnastics routine',
        'martial arts training', 'boxing match', 'skiing competition',
        'extreme sports event', 'racing event', 'athlete training',
        'championship event', 'gym workout'
    ],
    'transportation': [
        'transportation', 'transport', 'transit', 'mobility', 'commute',
        'transportation system', 'public transport', 'mass transit',
        'urban transportation', 'transport infrastructure',
        'cars', 'car', 'automobile', 'vehicle', 'auto', 'sedan',
        'suv', 'truck', 'pickup truck', 'van', 'minivan', 'sports car',
        'luxury car', 'electric car', 'ev', 'hybrid car', 'self driving car',
        'autonomous vehicle', 'classic car', 'vintage car',
        'driving', 'drive', 'road trip', 'highway', 'freeway', 'expressway',
        'traffic', 'road', 'street', 'intersection', 'parking',
        'driving test', 'driving school', 'driving experience',
        'train', 'railway', 'railroad', 'locomotive', 'passenger train',
        'freight train', 'high speed train', 'bullet train', 'metro',
        'subway', 'underground', 'light rail', 'tram', 'streetcar',
        'monorail', 'maglev', 'train station', 'railway station',
        'airplane', 'plane', 'aircraft', 'aviation', 'flight',
        'flying', 'jet', 'airliner', 'commercial aviation', 'private jet',
        'helicopter', 'drone', 'airport', 'terminal', 'runway',
        'air travel', 'airline', 'aviation industry', 'cockpit',
        'bus', 'public bus', 'school bus', 'coach', 'double decker',
        'bus stop', 'bus station', 'bus rapid transit', 'shuttle',
        'trolleybus', 'cable car', 'gondola', 'funicular',
        'motorcycle', 'motorbike', 'bike', 'bicycle', 'cycling',
        'scooter', 'electric scooter', 'moped', 'vespa',
        'boat', 'ship', 'vessel', 'ferry', 'cruise ship', 'yacht',
        'sailboat', 'speedboat', 'catamaran', 'cargo ship', 'container ship',
        'tanker', 'maritime', 'shipping', 'port', 'harbor', 'marina',
        'hyperloop', 'flying car', 'electric vehicle', 'sustainable transport',
        'smart city', 'intelligent transportation', 'mobility as a service',
        'delivery', 'logistics', 'shipping', 'freight', 'cargo',
        'trucking', 'courier', 'package delivery', 'last mile delivery',
        'bridge', 'tunnel', 'viaduct', 'interchange', 'toll road',
        'parking lot', 'gas station', 'charging station', 'rest area',
        'transport vehicles', 'car traffic', 'airplane travel', 'train travel',
        'subway transit', 'public bus', 'ship transport', 'high speed rail',
        'railway system', 'road driving', 'urban transit', 'public transportation',
        'logistics transport', 'electric vehicles', 'bicycle commute',
        'motorcycle ride', 'airport operations', 'train station'
    ]
}


class YouTubeManifestCollector:
    """YouTubeManifestCollector implementation."""
    
    def __init__(
        self, 
        api_key: str, 
        max_results_per_category: int = 50,
        progress_file: str = str(STATE_DIR / 'youtube_collection_progress.json'),
        proxy: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        required_license: str = REQUIRED_VIDEO_LICENSE,
    ):
        """  init   routine."""
        self.api_key = api_key
        self.max_results_per_category = max_results_per_category
        self.progress_file = progress_file
        self.timeout = timeout
        self.max_retries = max_retries
        self.required_license = required_license
        
        http = httplib2.Http(timeout=timeout)
        if proxy:
            import socks
            proxy_info = httplib2.ProxyInfo(
                proxy_type=socks.PROXY_TYPE_HTTP,
                proxy_host=proxy.split('://')[1].split(':')[0],
                proxy_port=int(proxy.split(':')[-1])
            )
            http = httplib2.Http(proxy_info=proxy_info, timeout=timeout)
            logger.info(f"Using proxy: {proxy}")
        
        socket.setdefaulttimeout(timeout)
        
        self.youtube = build('youtube', 'v3', developerKey=api_key, http=http)
        self.manifest = {category: [] for category in CATEGORY_KEYWORDS.keys()}
        self.completed_categories = set()
        
        self.search_history = {}  # {category: {(keyword, order, duration): True}}
        
        self._load_progress()
    
    def _load_progress(self):
        """ load progress routine."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    loaded_manifest = progress_data.get('manifest', {})
                    for category in CATEGORY_KEYWORDS.keys():
                        if category in loaded_manifest:
                            self.manifest[category] = loaded_manifest[category]
                    self.completed_categories = set(progress_data.get('completed_categories', []))
                    
                    search_history_data = progress_data.get('search_history', {})
                    self.search_history = {}
                    for category, history_list in search_history_data.items():
                        self.search_history[category] = {tuple(item): True for item in history_list}
                    
                    logger.info(f"Resumed from progress file: {len(self.completed_categories)} completed categories")
                    for category in self.completed_categories:
                        logger.info(f"  - {category}: {len(self.manifest.get(category, []))} videos")
                    
                    for category, history in self.search_history.items():
                        if history:
                            logger.info(f"  - {category}: {len(history)} keyword combinations already searched")
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}; starting collection from scratch")
        else:
            logger.info("No progress file found; starting collection from scratch")
    
    def _save_progress(self):
        """ save progress routine."""
        try:
            deduped_manifest = {}
            for category, videos in self.manifest.items():
                unique_by_id = {}
                for video in videos:
                    video_id = video.get('video_id', '')
                    if not video_id:
                        continue
                    old = unique_by_id.get(video_id)
                    if old is None or (
                        not self._is_license_verified(old) and self._is_license_verified(video)
                    ):
                        unique_by_id[video_id] = video
                unique_videos = list(unique_by_id.values())
                deduped_manifest[category] = unique_videos
                
                if len(videos) > len(unique_videos):
                    logger.info(f"{category}: {len(videos)} before deduplication, {len(unique_videos)} after deduplication")
            
            search_history_data = {}
            for category, history in self.search_history.items():
                search_history_data[category] = [list(key) for key in history.keys()]
            
            progress_data = {
                'last_update': datetime.now().isoformat(),
                'completed_categories': list(self.completed_categories),
                'manifest': deduped_manifest,
                'search_history': search_history_data
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Progress saved to: {self.progress_file}")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _retry_request(self, request_func, *args, **kwargs):
        """ retry request routine."""
        for attempt in range(self.max_retries):
            try:
                result = request_func(*args, **kwargs)
                return result
            except (socket.timeout, TimeoutError, ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Connection timed out; retrying in {wait_time}s ({attempt + 1}/{self.max_retries})...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Still failed after {self.max_retries} retries: {e}")
                    raise
            except HttpError as e:
                if e.resp.status == 403 and 'quota' in str(e).lower():
                    logger.error("=" * 60)
                    logger.error("API quota has been exhausted.")
                    logger.error("Suggested actions:")
                    logger.error("1. Wait for the quota reset, usually at midnight Pacific Time.")
                    logger.error("2. Visit https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas")
                    logger.error("   Request a quota increase.")
                    logger.error("3. Collected data has been saved; rerun the same command after quota resets.")
                    logger.error("=" * 60)
                    raise
                    if attempt < self.max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"Server error ({e.resp.status}); retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        
    def search_videos_by_keywords(
        self, 
        keywords: List[str], 
        max_results: int = 50,
        category: str = None
    ) -> List[Dict[str, Any]]:
        """Search videos by keywords routine."""
        video_ids = set()
        video_basic_info: List[Dict[str, Any]] = []
        total_ai_filtered = 0
        
        if category and category not in self.search_history:
            self.search_history[category] = {}
        
        order_options = ['relevance', 'date', 'rating', 'viewCount', 'title']
        
        duration_options = ['short', 'medium']
        
        search_combinations = []
        for keyword in keywords:
            for order in order_options:
                for duration in duration_options:
                    combination = (keyword, order, duration)
                    
                    if category and combination in self.search_history.get(category, {}):
                        continue

                    search_combinations.append(combination)
        
        random.shuffle(search_combinations)
        
        total_possible = len(keywords) * len(order_options) * len(duration_options)
        already_searched = len(self.search_history.get(category, {})) if category else 0
        
        logger.info("Search strategy statistics:")
        logger.info(f"  - Keywords: {len(keywords)}")
        logger.info(f"  - Sort orders: {len(order_options)}")
        logger.info(f"  - Duration options: {len(duration_options)}")
        logger.info(f"  - Theoretical combinations: {total_possible}")
        logger.info(f"  - Already searched: {already_searched}")
        logger.info(f"  - Available combinations: {len(search_combinations)}")
        logger.info(f"  - Required license: {self.required_license}")
        
        if len(search_combinations) == 0:
            logger.warning("All search combinations have already been tried; add more keywords or reset search history")
            return []
        
        search_count = 0
        target_searches = len(search_combinations)
        
        for keyword, order, duration in search_combinations:
            if len(video_ids) >= max_results:
                    break
            
            if search_count >= target_searches:
                        break
            
            search_count += 1
            logger.info(f"[{search_count}/{target_searches}] Search: '{keyword}' | {order} | {duration}")
            
            try:
                def search_request():
                    return self.youtube.search().list(
                        part='snippet',
                        q=keyword,
                        type='video',
                        order=order,
                        videoDuration=duration,
                        videoLicense=self.required_license,
                        maxResults=min(10, max_results - len(video_ids)),
                    ).execute()
                
                search_response = self._retry_request(search_request)
                
                if category:
                    self.search_history[category][(keyword, order, duration)] = True
                
                found_in_this_search = 0
                ai_filtered_in_search = 0
                
                for item in search_response.get('items', []):
                    if 'videoId' in item['id']:
                        video_id = item['id']['videoId']
                            
                        if self._is_ai_generated_content_from_search(item):
                            ai_filtered_in_search += 1
                            continue
                        
                        video_ids.add(video_id)
                        
                        snippet = item.get('snippet', {})
                        basic_info = {
                            'category': category,
                            'video_id': video_id,
                            'title': snippet.get('title', f'[search result] {video_id}'),
                            'description': snippet.get('description', ''),
                            'published_at': snippet.get('publishedAt', ''),
                            'channel_title': snippet.get('channelTitle', ''),
                            'thumbnail_url': snippet.get('thumbnails', {}).get('default', {}).get('url', ''),
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                        }
                        video_basic_info.append(basic_info)
                        found_in_this_search += 1
                
                total_ai_filtered += ai_filtered_in_search
                logger.info(f"  -> Added {found_in_this_search} video IDs (total: {len(video_ids)})")
                if ai_filtered_in_search > 0:
                    logger.info(f"  -> AI content filtered during search: {ai_filtered_in_search}")
                
                time.sleep(0.5)
                
            except (HttpError, socket.timeout, TimeoutError, ConnectionError) as e:
                logger.error(f"Error while searching '{keyword}' | {order} | {duration}: {e}")
                if category:
                    self.search_history[category][(keyword, order, duration)] = True
                continue

        logger.info("=" * 60)
        logger.info("Search completion statistics:")
        logger.info(f"  - Tried combinations: {search_count}")
        logger.info(f"  - Videos collected: {len(video_ids)}")
        logger.info(f"  - AI content filtered: {total_ai_filtered}")
        logger.info(f"  - Average efficiency: {len(video_ids)/search_count:.1f} videos/combination" if search_count > 0 else "  - Average efficiency: 0 videos/combination")
        if total_ai_filtered > 0:
            logger.info("AI filtering runs during search and still works even if quota is later exhausted")
        logger.info("=" * 60)
        return video_basic_info
        
    
    def reset_search_history(self, category: str = None):
        """Reset search history routine."""
        if category:
            if category in self.search_history:
                old_count = len(self.search_history[category])
                self.search_history[category] = {}
                logger.info(f"Reset search history for {category}; cleared {old_count} searched combinations")
            else:
                logger.info(f"{category} has no search history")
        else:
            total_cleared = sum(len(history) for history in self.search_history.values())
            self.search_history = {}
            logger.info(f"Reset search history for all categories; cleared {total_cleared} searched combinations")
        
        self._save_progress()
    
    def show_search_history_stats(self):
        """Show search history stats routine."""
        logger.info("=" * 60)
        logger.info("Search history statistics:")
        logger.info("=" * 60)
        
        total_possible = sum(len(keywords) * 5 * 2 for keywords in CATEGORY_KEYWORDS.values())
        for category in CATEGORY_KEYWORDS.keys():
            history_count = len(self.search_history.get(category, {}))
            category_total = len(CATEGORY_KEYWORDS[category]) * 5 * 2
            
            if history_count > 0:
                percentage = (history_count / category_total) * 100 if category_total > 0 else 0
                logger.info(f"{category}: {history_count}/{category_total} ({percentage:.1f}%)")
            else:
                logger.info(f"{category}: search has not started")
        
        total_searched = sum(len(history) for history in self.search_history.values())
        overall_percentage = (total_searched / total_possible) * 100 if total_possible > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"Total: {total_searched}/{total_possible} ({overall_percentage:.1f}%)")
        logger.info("=" * 60)

    def _is_license_verified(self, video: Dict[str, Any]) -> bool:
        return video.get('license', '') == self.required_license

    def _verified_manifest(self) -> Dict[str, List[Dict[str, Any]]]:
        verified = {}
        dropped = 0
        for category, videos in self.manifest.items():
            category_verified = [video for video in videos if self._is_license_verified(video)]
            dropped += len(videos) - len(category_verified)
            verified[category] = category_verified
        if dropped:
            logger.warning(
                f"Dropped {dropped} unverified/non-{self.required_license} videos from manifest export"
            )
        return verified
    
    def update_missing_details(self, category: str = None):
        """Update missing details routine."""
        logger.info("=" * 60)
        logger.info("Updating missing video details...")
        logger.info("=" * 60)
        
        categories_to_update = [category] if category else list(CATEGORY_KEYWORDS.keys())
        
        total_updated = 0
        total_failed = 0
        
        for cat in categories_to_update:
            if cat not in self.manifest:
                        continue
                
            videos_need_update = []
            for video in self.manifest[cat]:
                if video.get('details_fetched', True) == False:
                    videos_need_update.append(video['video_id'])
                elif (video.get('duration_seconds', -1) == 0 and 
                      video.get('definition', '') == 'unknown' and 
                      video.get('license', '') == 'unknown'):
                    videos_need_update.append(video['video_id'])
                elif video.get('title', '').startswith('[details pending]'):
                    videos_need_update.append(video['video_id'])
            
            if not videos_need_update:
                logger.info(f"{cat}: all videos already have detailed metadata")
                continue
            
            logger.info(f"{cat}: found {len(videos_need_update)} videos that need detailed metadata")
            
            try:
                detailed_videos = self.get_video_details(videos_need_update, cat)
                
                if detailed_videos:
                    detailed_map = {v['video_id']: v for v in detailed_videos}
                    
                    updated_videos = []
                    for video in self.manifest[cat]:
                        video_id = video.get('video_id', '')
                        if video_id in detailed_map:
                            updated_videos.append(detailed_map[video_id])
                            total_updated += 1
                        else:
                            updated_videos.append(video)
                    
                    self.manifest[cat] = updated_videos
                    logger.info(f"{cat}: successfully updated detailed metadata for {len(detailed_videos)} videos")
                else:
                    logger.warning(f"{cat}: failed to fetch any detailed metadata")
                    total_failed += len(videos_need_update)
                
            except Exception as e:
                logger.error(f"{cat}: error while updating detailed metadata: {e}")
                total_failed += len(videos_need_update)
                continue

        self._save_progress()
        
        logger.info("=" * 60)
        logger.info("Detailed metadata update complete!")
        logger.info("=" * 60)
        logger.info(f"Updated successfully: {total_updated} videos")
        logger.info(f"Update failed: {total_failed} videos")
        logger.info("=" * 60)
    
    def show_details_status(self):
        """Show details status routine."""
        logger.info("=" * 60)
        logger.info("Video detail metadata status:")
        logger.info("=" * 60)
        
        total_videos = 0
        total_with_details = 0
        total_basic_only = 0
        
        for category in CATEGORY_KEYWORDS.keys():
            if category not in self.manifest:
                logger.info(f"{category}: no videos")
                continue

            videos = self.manifest[category]
            with_details = 0
            basic_only = 0
            
            for video in videos:
                has_complete_details = self._is_license_verified(video)
                
                if has_complete_details:
                    with_details += 1
                else:
                    basic_only += 1
            
            total_videos += len(videos)
            total_with_details += with_details
            total_basic_only += basic_only
            
            if basic_only > 0:
                logger.info(f"{category}: {len(videos)} videos (detailed: {with_details}, basic: {basic_only})")
            else:
                logger.info(f"{category}: {len(videos)} videos (all have detailed metadata)")
        
        logger.info("=" * 60)
        logger.info(f"Total: {total_videos} videos")
        logger.info(f"  - With detailed metadata: {total_with_details}")
        logger.info(f"  - Basic metadata only: {total_basic_only}")
        if total_basic_only > 0:
            logger.info("Use --update-details to fetch missing detailed metadata")
        logger.info("=" * 60)
    
    def _is_ai_generated_content(self, item: Dict[str, Any]) -> bool:
        """ is ai generated content routine."""
        video_id = item.get('id', '')
        title = item.get('snippet', {}).get('title', '')
        
        
        content_details = item.get('contentDetails', {})
        
        ai_disclosure_fields = [
        ]
        
        for field in ai_disclosure_fields:
            if field in content_details:
                field_value = content_details[field]
                logger.debug(f"Found field '{field}': {field_value}")
                if field_value is True or str(field_value).lower() == 'true':
                    logger.info(f"YouTube official AI-content flag detected: [{video_id}] {title}")
                    return True
        
        status = item.get('status', {})
        for field in ai_disclosure_fields:
            if field in status:
                field_value = status[field]
                logger.debug(f"Found field '{field}': {field_value}")
                if field_value is True or str(field_value).lower() == 'true':
                    logger.info(f"YouTube official AI-content flag detected: [{video_id}] {title}")
                    return True
        
        
        ai_keywords_en = [
            'ai generated', 'ai-generated', 'ai created', 'ai made',
            'artificial intelligence', 'generated by ai', 'created by ai',
            'made with ai', 'made by ai', 'ai video', 'ai animation',
            'text to video', 'text-to-video', 'ai tool', 'ai-generated content',
            'synthetically generated', 'machine generated', 'algorithmically generated',
            'sora', 'runway', 'pika', 'stable diffusion', 'midjourney',
            'dall-e', 'dalle', 'synthesia', 'pictory', 'fliki',
            'lumen5', 'invideo ai', 'descript', 'runway ml',
            'gen-2', 'gen-3', 'kling', 'veo', 'dream screen'
        ]
        
        ai_keywords_extra = [
            'ai generated video', 'ai created video', 'ai produced video',
            'ai synthesized video', 'synthetic video', 'machine-made video',
            'algorithm-generated video', 'digitally generated video',
            'text generated video', 'text prompt video', 'ai cartoon',
        ]
        
        all_keywords = ai_keywords_en + ai_keywords_extra
        
        title_lower = title.lower()
        for keyword in all_keywords:
            if keyword in title_lower:
                logger.info(f"AI-content keyword detected in title: [{video_id}] {title} [keyword: '{keyword}']")
                return True
        
        description = item.get('snippet', {}).get('description', '').lower()
        for keyword in all_keywords:
            if keyword in description:
                logger.info(f"AI-content keyword detected in description: [{video_id}] {title} [keyword: '{keyword}']")
                return True
        
        tags = item.get('snippet', {}).get('tags', [])
        for tag in tags:
            tag_lower = tag.lower()
            for keyword in all_keywords:
                if keyword in tag_lower:
                    logger.info(f"AI-content keyword detected in tag: [{video_id}] {title} [tag: '{tag}']")
                    return True
        
        return False
    
    def _is_ai_generated_content_from_search(self, search_item: Dict[str, Any]) -> bool:
        """ is ai generated content from search routine."""
        video_id = search_item.get('id', {}).get('videoId', '')
        snippet = search_item.get('snippet', {})
        title = snippet.get('title', '')
        
        
        ai_keywords_en = [
            'ai generated', 'ai-generated', 'ai created', 'ai made',
            'artificial intelligence', 'generated by ai', 'created by ai',
            'made with ai', 'made by ai', 'ai video', 'ai animation',
            'text to video', 'text-to-video', 'ai tool', 'ai-generated content',
            'synthetically generated', 'machine generated', 'algorithmically generated',
            'altered content', 'synthetic content',
            'sora', 'runway', 'pika', 'stable diffusion', 'midjourney',
            'dall-e', 'dalle', 'synthesia', 'pictory', 'fliki',
            'lumen5', 'invideo ai', 'descript', 'runway ml',
            'gen-2', 'gen-3', 'kling', 'veo', 'dream screen'
        ]
        
        ai_keywords_extra = [
            'ai generated video', 'ai created video', 'ai produced video',
            'ai synthesized video', 'synthetic video', 'machine-made video',
            'algorithm-generated video', 'digitally generated video',
            'text generated video', 'text prompt video', 'ai cartoon',
            'altered video', 'composited video'
        ]
        
        all_keywords = ai_keywords_en + ai_keywords_extra
        
        title_lower = title.lower()
        for keyword in all_keywords:
            if keyword in title_lower:
                logger.info(f"AI filtering during search matched title: [{video_id}] {title} [keyword: '{keyword}']")
                return True
        
        description = snippet.get('description', '').lower()
        for keyword in all_keywords:
            if keyword in description:
                logger.info(f"AI filtering during search matched description: [{video_id}] {title} [keyword: '{keyword}']")
                return True
        
        tags = snippet.get('tags', [])
        for tag in tags:
            tag_lower = tag.lower()
            for keyword in all_keywords:
                if keyword in tag_lower:
                    logger.info(f"AI filtering during search matched tag: [{video_id}] {title} [tag: '{tag}']")
                    return True
        
        return False
    
    def get_video_details(self, video_ids: List[str], category: str) -> List[Dict[str, Any]]:
        """Get video details routine."""
        video_details = []
        ai_filtered_count = 0
        duration_filtered_count = 0
        license_filtered_count = 0
        
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            try:
                def video_request():
                    return self.youtube.videos().list(
                        part='snippet,contentDetails,status',
                        id=','.join(batch_ids)
                    ).execute()
                
                videos_response = self._retry_request(video_request)
                
                for item in videos_response.get('items', []):
                    video_id = item.get('id', '')
                    snippet = item.get('snippet', {})
                    content_details = item.get('contentDetails', {})
                    status = item.get('status', {})
                    title = snippet.get('title', '')
                    
                    if not hasattr(self, '_debug_fields_printed'):
                        self._debug_fields_printed = True
                        logger.info("=" * 60)
                        logger.info("Debug information: field structure returned by the YouTube API")
                        logger.info("-" * 60)
                        logger.info(f"Available top-level fields: {list(item.keys())}")
                        if 'contentDetails' in item:
                            logger.info(f"contentDetails fields: {list(item['contentDetails'].keys())}")
                        if 'status' in item:
                            logger.info(f"status fields: {list(item['status'].keys())}")
                        logger.info("=" * 60)

                    license_value = status.get('license', '')
                    if license_value != self.required_license:
                        license_filtered_count += 1
                        logger.info(
                            f"Skipping non-{self.required_license} video: [{video_id}] {title} "
                            f"(license={license_value or 'missing'})"
                        )
                        continue

                    duration = content_details.get('duration', 'PT0S')
                    duration_seconds = self._parse_duration(duration)

                    if duration_seconds > MAX_DURATION_SECONDS or duration_seconds == 0:
                        duration_filtered_count += 1
                        continue

                    if self._is_ai_generated_content(item):
                        ai_filtered_count += 1
                        continue
                    
                    video_info = {
                        'category': category,
                        'video_id': video_id,
                        'title': title,
                        'description': snippet.get('description', ''),
                        'published_at': snippet.get('publishedAt', ''),
                        'channel_title': snippet.get('channelTitle', ''),
                        'thumbnail_url': snippet.get('thumbnails', {}).get('default', {}).get('url', ''),
                        'duration': duration,
                        'duration_seconds': duration_seconds,
                        'definition': content_details.get('definition', ''),
                        'license': license_value,
                        'privacy_status': status.get('privacyStatus', ''),
                        'license_checked_at': datetime.now().isoformat(),
                        'details_fetched': True,
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    }
                    
                    video_details.append(video_info)
                
                time.sleep(0.5)
                
            except (HttpError, socket.timeout, TimeoutError, ConnectionError) as e:
                logger.error(f"Error while fetching video details: {e}")
                continue

        if ai_filtered_count > 0:
            logger.info(f"AI-content filtering summary: filtered {ai_filtered_count} AI-generated videos")
        if duration_filtered_count > 0:
            logger.info(f"Duration filtering summary: filtered {duration_filtered_count} videos")
        if license_filtered_count > 0:
            logger.info(f"License filtering summary: filtered {license_filtered_count} non-{self.required_license} videos")

        return video_details
    
    def _parse_duration(self, duration: str) -> int:
        """ parse duration routine."""
        import re
        
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def collect_category(self, category: str):
        """Collect category routine."""
        if category not in CATEGORY_KEYWORDS:
            logger.error(f"Invalid category name: {category}")
            logger.info(f"Available categories: {', '.join(CATEGORY_KEYWORDS.keys())}")
            return
        
        keywords = CATEGORY_KEYWORDS[category]
        
        logger.info("=" * 60)
        logger.info(f"Starting collection for category: {category}")
        logger.info(f"Currently stored videos: {len(self.manifest.get(category, []))}")
        logger.info("=" * 60)
        
        try:
            basic_videos = self.search_videos_by_keywords(
                keywords, 
                self.max_results_per_category,
                category=category
            )
            logger.info(f"{category}: found {len(basic_videos)} videos with basic metadata from the search stage")
            
            if basic_videos:
                logger.info(f"Verifying license metadata for {len(basic_videos)} videos...")
                
                video_ids = [video['video_id'] for video in basic_videos]
                
                try:
                    detailed_videos = self.get_video_details(video_ids, category)
                    logger.info(
                        f"{category}: verified {len(detailed_videos)} videos with "
                        f"license={self.required_license}"
                    )
                    
                    if detailed_videos:
                        existing_videos = self.manifest.get(category, [])
                        merged = {video.get('video_id', ''): video for video in existing_videos if video.get('video_id', '')}
                        for video in detailed_videos:
                            merged[video['video_id']] = video
                        self.manifest[category] = list(merged.values())
                        logger.info(f"Saved {len(detailed_videos)} verified videos")
                    
                except Exception as detail_error:
                    logger.error(f"Error while fetching detailed metadata: {detail_error}")
                    logger.warning(
                        "No unverified basic-only candidates were saved. "
                        "Rerun later so license checks can complete before manifest export."
                    )
                
                self._save_progress()
                
                final_count = len(self.manifest[category])
                videos_with_details = 0
                videos_basic_only = 0
                
                for video in self.manifest[category]:
                    has_complete_details = self._is_license_verified(video)
                    
                    if has_complete_details:
                        videos_with_details += 1
                    else:
                        videos_basic_only += 1
                
                logger.info(f"{category}: {final_count} videos after deduplication")
                if videos_basic_only > 0:
                    logger.info(f"  - Complete detailed metadata: {videos_with_details}")
                    logger.info(f"  - Basic metadata only (including title): {videos_basic_only}")
                    logger.info("Basic metadata includes real titles, descriptions, publish times, channel names, and related fields")
                else:
                    logger.info(f"  - All videos have complete detailed metadata: {videos_with_details}")
                
            else:
                logger.warning(f"{category}: no matching videos found")
            
            logger.info("=" * 60)
            logger.info(f"Category {category} collection completed and saved")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error while collecting category {category}: {e}")
            self._save_progress()
            logger.info("Progress saved, including video IDs found so far")
            raise
    
    def collect_all_categories(self):
        """Collect all categories routine."""
        logger.info("Starting YouTube video metadata collection...")
        logger.info(f"Total categories to collect: {len(CATEGORY_KEYWORDS)}")
        
        if self.completed_categories:
            logger.info(f"Skipping {len(self.completed_categories)} completed categories and continuing unfinished work")
        
        try:
            for category in CATEGORY_KEYWORDS.keys():
                if category in self.completed_categories:
                    logger.info(f"Skipping completed category: {category} ({len(self.manifest[category])} videos already stored)")
                    continue

                logger.info(f"Collecting category [{len(self.completed_categories)+1}/{len(CATEGORY_KEYWORDS)}]: {category}")
                
                try:
                    self.collect_category(category)
                    self.completed_categories.add(category)
                    
                except Exception as e:
                    logger.error(f"Error while collecting category {category}: {e}")
                    logger.info("Progress saved; you can continue later")
                    raise
                
                time.sleep(1)

        except KeyboardInterrupt:
            logger.warning("\nInterrupt detected; saving progress...")
            self._save_progress()
            logger.info("Progress saved; the next run will resume from the checkpoint")
            raise
        
        except Exception as e:
            logger.error(f"Error during collection: {e}")
            self._save_progress()
            logger.info("Progress saved")
            raise
        
        logger.info("=" * 60)
        logger.info("All categories have been collected.")
        logger.info(f"Collected {len(self.completed_categories)}/{len(CATEGORY_KEYWORDS)} categories")
    
    def save_manifest(self, output_file: str = str(METADATA_DIR / 'youtube_manifest.json')):
        """Save manifest routine."""
        manifest = self._verified_manifest()
        total_videos = sum(len(videos) for videos in manifest.values())
        
        output_data = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'total_videos': total_videos,
                'categories': list(CATEGORY_KEYWORDS.keys()),
                'filters': {
                    'max_duration_seconds': MAX_DURATION_SECONDS,
                    'video_license': self.required_license,
                    'license_enforced': True,
                    'license_verification': 'search.list videoLicense filter plus videos.list status.license check'
                }
            },
            'categories': manifest
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved to: {output_file}")
        logger.info(f"Total collected videos: {total_videos}")
        
        print("\n=== Collection Statistics ===")
        for category, videos in self.manifest.items():
            print(f"{category}: {len(videos)} videos")
    
    def save_manifest_csv(self, output_file: str = str(METADATA_DIR / 'youtube_manifest.csv')):
        """Save manifest csv routine."""
        import csv
        manifest = self._verified_manifest()
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                'category', 'video_id', 'title', 'duration_seconds',
                'duration', 'definition', 'license', 'privacy_status',
                'published_at', 'channel_title', 'thumbnail_url',
                'license_checked_at', 'url'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for category, videos in manifest.items():
                for video in videos:
                    row = {
                        'category': video.get('category', ''),
                        'video_id': video.get('video_id', ''),
                        'title': video.get('title', ''),
                        'duration_seconds': video.get('duration_seconds', 0),
                        'duration': video.get('duration', ''),
                        'definition': video.get('definition', ''),
                        'license': video.get('license', ''),
                        'privacy_status': video.get('privacy_status', ''),
                        'published_at': video.get('published_at', ''),
                        'channel_title': video.get('channel_title', ''),
                        'thumbnail_url': video.get('thumbnail_url', ''),
                        'license_checked_at': video.get('license_checked_at', ''),
                        'url': video.get('url', '')
                    }
                    writer.writerow(row)
        
        logger.info(f"CSV metadata saved to: {output_file}")


def main():
    """Main routine."""
    parser = argparse.ArgumentParser(
        description='YouTube Creative Commons video metadata collector with resume support'
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='YouTube Data API v3 key'
    )
    parser.add_argument(
        '--max-per-category',
        type=int,
        default=50,
        help='Maximum number of videos to fetch per category (default: 50)'
    )
    parser.add_argument(
        '--output-json',
        default=str(METADATA_DIR / 'youtube_manifest.json'),
        help='JSON output file path (default: youtube_manifest.json)'
    )
    parser.add_argument(
        '--output-csv',
        default=str(METADATA_DIR / 'youtube_manifest.csv'),
        help='CSV output file path (default: youtube_manifest.csv)'
    )
    parser.add_argument(
        '--progress-file',
        default=str(STATE_DIR / 'youtube_collection_progress.json'),
        help='Progress file path (default: youtube_collection_progress.json)'
    )
    parser.add_argument(
        '--proxy',
        default=None,
        help='Proxy server URL, for example: http://127.0.0.1:7890'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retry count (default: 3)'
    )
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Do not generate a CSV file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Continue collection from the last interruption by loading the progress file'
    )
    parser.add_argument(
        '--category',
        default=None,
        help='Collect only one specified category, for example: animals_and_ecology'
    )
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List all available category names'
    )
    parser.add_argument(
        '--show-search-stats',
        action='store_true',
        help='Show search history statistics'
    )
    parser.add_argument(
        '--reset-search-history',
        default=None,
        help='Reset search history for a specified category, or use "all" to reset everything'
    )
    parser.add_argument(
        '--update-details',
        default=None,
        help='Fetch missing detailed metadata for a specified category, or use "all" to update everything'
    )
    parser.add_argument(
        '--show-details-status',
        action='store_true',
        help='Show detailed metadata status for each category'
    )
    
    args = parser.parse_args()
    
    if args.list_categories:
        print("\nAvailable categories:")
        print("=" * 60)
        for i, category in enumerate(CATEGORY_KEYWORDS.keys(), 1):
            keyword_count = len(CATEGORY_KEYWORDS[category])
            print(f"{i:2d}. {category} ({keyword_count} keywords)")
        print("=" * 60)
        print(f"\nTotal categories: {len(CATEGORY_KEYWORDS)}")
        print("\nUsage:")
        print(f"  python {os.path.basename(__file__)} --api-key YOUR_KEY --category animals_and_ecology")
        return

    try:
        collector = YouTubeManifestCollector(
            api_key=args.api_key,
            max_results_per_category=args.max_per_category,
            progress_file=args.progress_file,
            proxy=args.proxy,
            timeout=args.timeout,
            max_retries=args.max_retries
        )
        
        if args.show_search_stats:
            collector.show_search_history_stats()
            return
        
        if args.reset_search_history:
            if args.reset_search_history.lower() == 'all':
                collector.reset_search_history()
            else:
                collector.reset_search_history(args.reset_search_history)
            return
        
        if args.show_details_status:
            collector.show_details_status()
            return
        
        if args.update_details:
            if args.update_details.lower() == 'all':
                collector.update_missing_details()
            else:
                collector.update_missing_details(args.update_details)
            return
        
        
        if args.category:
            logger.info(f"Collecting a single specified category: {args.category}")
            collector.collect_category(args.category)
        else:
            collector.collect_all_categories()
        
        collector.save_manifest(args.output_json)
        
        if not args.no_csv:
            collector.save_manifest_csv(args.output_csv)
        
        print("\n" + "=" * 60)
        print("Collection complete!")
        print(f"JSON file: {args.output_json}")
        if not args.no_csv:
            print(f"CSV file: {args.output_csv}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nCollection interrupted by user; progress has been saved")
        print("Run the same command to continue collection")
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        print("\nAn error occurred, but progress has been saved. Run the same command to continue collection")
        raise


if __name__ == '__main__':
    main()
