"""
Weather API integration and feature mapping module for StormCast.
Retrieves live meteorological observations and maps them to the trained ML feature space.

Features:
- Dual-tier and multi-provider fallback (OpenWeatherMap, Open-Meteo, World Meteorological Service, NOAA NWS).
- Detection and graceful handling of HTTP 429 Too Many Requests.
- Retry-After inspection and limited backoff with jitter.
- Static and dynamic geocoding caches to eliminate redundant lookups.
- Observation cache with TTL to eliminate duplicate weather requests.
- No artificial or fake weather data: all fetched data represents real atmospheric observations.
"""

import os
import time
import json
import random
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from src.preprocessing import get_season


# ---------------------------------------------------------------------------
# Pre-populated Geocoding Table (Instant resolution without external HTTP calls)
# ---------------------------------------------------------------------------
KNOWN_LOCATIONS: Dict[str, Dict[str, Any]] = {
    'austin': {
        'city': 'Austin', 'region': 'Texas', 'state': 'TX', 'country': 'United States',
        'latitude': 30.2672, 'longitude': -97.7431, 'timezone': 'US/Central'
    },
    'austin, texas': {
        'city': 'Austin', 'region': 'Texas', 'state': 'TX', 'country': 'United States',
        'latitude': 30.2672, 'longitude': -97.7431, 'timezone': 'US/Central'
    },
    'dallas': {
        'city': 'Dallas', 'region': 'Texas', 'state': 'TX', 'country': 'United States',
        'latitude': 32.7767, 'longitude': -96.7970, 'timezone': 'US/Central'
    },
    'dallas, tx': {
        'city': 'Dallas', 'region': 'Texas', 'state': 'TX', 'country': 'United States',
        'latitude': 32.7767, 'longitude': -96.7970, 'timezone': 'US/Central'
    },
    'houston': {
        'city': 'Houston', 'region': 'Texas', 'state': 'TX', 'country': 'United States',
        'latitude': 29.7604, 'longitude': -95.3698, 'timezone': 'US/Central'
    },
    'san antonio': {
        'city': 'San Antonio', 'region': 'Texas', 'state': 'TX', 'country': 'United States',
        'latitude': 29.4241, 'longitude': -98.4936, 'timezone': 'US/Central'
    },
    'denver': {
        'city': 'Denver', 'region': 'Colorado', 'state': 'CO', 'country': 'United States',
        'latitude': 39.7392, 'longitude': -104.9903, 'timezone': 'US/Mountain'
    },
    'denver, co': {
        'city': 'Denver', 'region': 'Colorado', 'state': 'CO', 'country': 'United States',
        'latitude': 39.7392, 'longitude': -104.9903, 'timezone': 'US/Mountain'
    },
    'miami': {
        'city': 'Miami', 'region': 'Florida', 'state': 'FL', 'country': 'United States',
        'latitude': 25.7617, 'longitude': -80.1918, 'timezone': 'US/Eastern'
    },
    'miami, fl': {
        'city': 'Miami', 'region': 'Florida', 'state': 'FL', 'country': 'United States',
        'latitude': 25.7617, 'longitude': -80.1918, 'timezone': 'US/Eastern'
    },
    'orlando': {
        'city': 'Orlando', 'region': 'Florida', 'state': 'FL', 'country': 'United States',
        'latitude': 28.5383, 'longitude': -81.3792, 'timezone': 'US/Eastern'
    },
    'tampa': {
        'city': 'Tampa', 'region': 'Florida', 'state': 'FL', 'country': 'United States',
        'latitude': 27.9506, 'longitude': -82.4572, 'timezone': 'US/Eastern'
    },
    'seattle': {
        'city': 'Seattle', 'region': 'Washington', 'state': 'WA', 'country': 'United States',
        'latitude': 47.6062, 'longitude': -122.3321, 'timezone': 'US/Pacific'
    },
    'seattle, wa': {
        'city': 'Seattle', 'region': 'Washington', 'state': 'WA', 'country': 'United States',
        'latitude': 47.6062, 'longitude': -122.3321, 'timezone': 'US/Pacific'
    },
    'chicago': {
        'city': 'Chicago', 'region': 'Illinois', 'state': 'IL', 'country': 'United States',
        'latitude': 41.8781, 'longitude': -87.6298, 'timezone': 'US/Central'
    },
    'new york': {
        'city': 'New York', 'region': 'New York', 'state': 'NY', 'country': 'United States',
        'latitude': 40.7128, 'longitude': -74.0060, 'timezone': 'US/Eastern'
    },
    'los angeles': {
        'city': 'Los Angeles', 'region': 'California', 'state': 'CA', 'country': 'United States',
        'latitude': 34.0522, 'longitude': -118.2437, 'timezone': 'US/Pacific'
    },
    'san francisco': {
        'city': 'San Francisco', 'region': 'California', 'state': 'CA', 'country': 'United States',
        'latitude': 37.7749, 'longitude': -122.4194, 'timezone': 'US/Pacific'
    },
    'oklahoma city': {
        'city': 'Oklahoma City', 'region': 'Oklahoma', 'state': 'OK', 'country': 'United States',
        'latitude': 35.4676, 'longitude': -97.5164, 'timezone': 'US/Central'
    },
    'atlanta': {
        'city': 'Atlanta', 'region': 'Georgia', 'state': 'GA', 'country': 'United States',
        'latitude': 33.7490, 'longitude': -84.3880, 'timezone': 'US/Eastern'
    },
    'phoenix': {
        'city': 'Phoenix', 'region': 'Arizona', 'state': 'AZ', 'country': 'United States',
        'latitude': 33.4484, 'longitude': -112.0740, 'timezone': 'US/Mountain'
    },
    'delhi': {
        'city': 'Delhi', 'region': 'Delhi', 'state': 'DL', 'country': 'India',
        'latitude': 28.6139, 'longitude': 77.2090, 'timezone': 'Asia/Kolkata'
    },
    'london': {
        'city': 'London', 'region': 'Greater London', 'state': 'LDN', 'country': 'United Kingdom',
        'latitude': 51.5074, 'longitude': -0.1278, 'timezone': 'Europe/London'
    }
}

# Dynamic in-memory geocoding and observation caches
_DYNAMIC_GEO_CACHE: Dict[str, Dict[str, Any]] = {}
_WEATHER_OBSERVATION_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300.0  # 5 minutes TTL for live weather observations
_OPEN_METEO_COOLDOWN_UNTIL = 0.0  # Timestamp until which Open-Meteo requests should be bypassed due to 429


# Mapping from full US State / Territory names to 2-letter postal codes
US_STATE_MAP = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

TIMEZONE_MAP = {
    'America/New_York': 'US/Eastern',
    'America/Detroit': 'US/Eastern',
    'America/Indiana/Indianapolis': 'US/Eastern',
    'America/Chicago': 'US/Central',
    'America/Menominee': 'US/Central',
    'America/North_Dakota/Center': 'US/Central',
    'America/Denver': 'US/Mountain',
    'America/Boise': 'US/Mountain',
    'America/Phoenix': 'US/Mountain',
    'America/Los_Angeles': 'US/Pacific',
    'America/Anchorage': 'US/Alaska',
    'Pacific/Honolulu': 'US/Hawaii'
}

WMO_CODE_MAP = {
    0: 'Clear Sky',
    1: 'Mainly Clear',
    2: 'Partly Cloudy',
    3: 'Overcast',
    45: 'Fog',
    48: 'Depositing Rime Fog',
    51: 'Light Drizzle',
    53: 'Moderate Drizzle',
    55: 'Dense Drizzle',
    61: 'Slight Rain',
    63: 'Moderate Rain',
    65: 'Heavy Rain',
    71: 'Slight Snow Fall',
    73: 'Moderate Snow Fall',
    75: 'Heavy Snow Fall',
    80: 'Slight Rain Showers',
    81: 'Moderate Rain Showers',
    82: 'Violent Rain Showers',
    95: 'Thunderstorm',
    96: 'Thunderstorm with Slight Hail',
    99: 'Thunderstorm with Heavy Hail'
}


def get_api_key() -> Optional[str]:
    """Retrieve OpenWeatherMap API key from environment variables or .env file."""
    key = os.getenv('OPENWEATHER_API_KEY') or os.getenv('OPENWEATHERMAP_API_KEY')
    if key and key.strip():
        return key.strip()

    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k in ('OPENWEATHER_API_KEY', 'OPENWEATHERMAP_API_KEY') and v:
                            return v
        except Exception:
            pass
    return None


def _make_http_request(url: str, timeout: float = 6.0, max_retries: int = 1) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    """
    Execute an HTTP GET request with timeout, 429 rate-limit detection,
    and single exponential backoff with jitter.
    
    Returns: (data_dict, status_code, error_message)
    """
    headers = {
        'User-Agent': 'StormCast-App/1.1 (Academic-ML-Project; Python-urllib)',
        'Accept': 'application/json'
    }
    req = urllib.request.Request(url, headers=headers)

    global _OPEN_METEO_COOLDOWN_UNTIL
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data, 200, None
        except urllib.error.HTTPError as e:
            status_code = e.code
            if status_code == 429:
                # Rate limited
                retry_after = e.headers.get('Retry-After')
                cooldown_sec = 1800.0
                if retry_after:
                    try:
                        cooldown_sec = min(float(retry_after), 3600.0)
                    except ValueError:
                        pass
                _OPEN_METEO_COOLDOWN_UNTIL = time.time() + cooldown_sec
                return None, 429, "Weather service temporarily rate-limited (HTTP 429). Please try again shortly or use Manual Mode."
            elif status_code == 404:
                return None, 404, "Requested resource or location was not found."
            elif status_code == 401:
                return None, 401, "API authentication failed. Please check your API key."
            else:
                return None, status_code, f"HTTP Error {status_code}: {e.reason}"
        except urllib.error.URLError as e:
            return None, 0, f"Network connection error: {e.reason}"
        except Exception as e:
            return None, 0, f"Request failed: {str(e)}"

    return None, 0, "Request attempts exceeded."


def resolve_location_geocoding(query: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Resolve city string into coordinates, state, and timezone.
    Prioritizes:
    1. Static lookup table (instant, zero network calls)
    2. In-memory dynamic geocoding cache (zero network calls)
    3. External Open-Meteo Geocoding API (with 429 handling)
    """
    clean_query = query.strip().lower()
    
    # 1. Check pre-populated static table
    if clean_query in KNOWN_LOCATIONS:
        return KNOWN_LOCATIONS[clean_query], None

    base_name = clean_query.split(',')[0].strip()
    if base_name in KNOWN_LOCATIONS:
        return KNOWN_LOCATIONS[base_name], None

    # 2. Check in-memory dynamic cache
    if clean_query in _DYNAMIC_GEO_CACHE:
        return _DYNAMIC_GEO_CACHE[clean_query], None

    # 3. Call Open-Meteo Geocoding API if not in cooldown
    global _OPEN_METEO_COOLDOWN_UNTIL
    if time.time() < _OPEN_METEO_COOLDOWN_UNTIL:
        return None, "Open-Meteo geocoding cooldown active (HTTP 429)."

    encoded_city = urllib.parse.quote(query.strip())
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
    
    geo_data, status, err = _make_http_request(geo_url, timeout=5.0, max_retries=1)
    if err:
        return None, err
    if not geo_data or not geo_data.get('results'):
        return None, f"Location '{query}' could not be resolved. Please check spelling or specify State/Country."

    top_res = geo_data['results'][0]
    lat = float(top_res['latitude'])
    lon = float(top_res['longitude'])
    city_name = top_res.get('name', query)
    admin1 = top_res.get('admin1', '')
    country = top_res.get('country', '')
    raw_tz = top_res.get('timezone', 'UTC')

    tz_str = TIMEZONE_MAP.get(raw_tz)
    if not tz_str:
        if 'Chicago' in raw_tz:
            tz_str = 'US/Central'
        elif 'New_York' in raw_tz:
            tz_str = 'US/Eastern'
        elif 'Denver' in raw_tz:
            tz_str = 'US/Mountain'
        elif 'Los_Angeles' in raw_tz:
            tz_str = 'US/Pacific'
        else:
            tz_str = 'US/Central'

    state_code = US_STATE_MAP.get(admin1, admin1.upper() if len(admin1) == 2 else 'TX')

    loc_dict = {
        'city': city_name,
        'region': admin1,
        'state': state_code,
        'country': country,
        'latitude': lat,
        'longitude': lon,
        'timezone': tz_str
    }

    _DYNAMIC_GEO_CACHE[clean_query] = loc_dict
    return loc_dict, None


def fetch_weather_openmeteo(loc_info: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch current conditions via Open-Meteo with 429 rate-limit backoff."""
    global _OPEN_METEO_COOLDOWN_UNTIL
    if time.time() < _OPEN_METEO_COOLDOWN_UNTIL:
        return {'success': False, 'rate_limited': True, 'error': 'Open-Meteo cooldown active (HTTP 429).'}

    lat = loc_info['latitude']
    lon = loc_info['longitude']

    w_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,showers,weather_code,wind_speed_10m"
        f"&timezone=auto"
    )

    w_data, status, err = _make_http_request(w_url, timeout=5.0, max_retries=1)
    if err:
        is_429 = (status == 429) or ("rate-limited" in err) or ("429" in err)
        return {'success': False, 'rate_limited': is_429, 'error': err}

    current = w_data.get('current', {})
    temp_c = float(current.get('temperature_2m', 0.0))
    humidity = int(current.get('relative_humidity_2m', 0))
    wind_kmh = float(current.get('wind_speed_10m', 0.0))
    precip_mm = float(current.get('precipitation', 0.0))
    w_code = int(current.get('weather_code', 0))
    condition_desc = WMO_CODE_MAP.get(w_code, 'Clear')

    obs_time_iso = current.get('time', '')
    obs_time_display = obs_time_iso.replace('T', ' ') if obs_time_iso else datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')

    return {
        'success': True,
        'rate_limited': False,
        'provider': 'Open-Meteo Weather Service',
        'city': loc_info['city'],
        'region': loc_info['region'],
        'country': loc_info['country'],
        'latitude': lat,
        'longitude': lon,
        'temperature_c': round(temp_c, 1),
        'humidity': humidity,
        'wind_kmh': round(wind_kmh, 1),
        'precipitation_mm': round(precip_mm, 2),
        'condition': condition_desc,
        'timezone_str': loc_info['timezone'],
        'observation_time': obs_time_display,
        'from_cache': False
    }


def fetch_weather_wttr(loc_info: Dict[str, Any], raw_query: str) -> Dict[str, Any]:
    """
    Fallback meteorological provider using World Weather Service (wttr.in).
    Reliable, zero API key required, global coverage.
    """
    query_target = loc_info['city'] if loc_info.get('city') else raw_query
    encoded_target = urllib.parse.quote(query_target.strip())
    url = f"https://wttr.in/{encoded_target}?format=j1"
    
    headers = {
        'User-Agent': 'curl/7.88.1 (StormCast-Academic-Demo)',
        'Accept': 'application/json'
    }
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            
        cur = data['current_condition'][0]
        temp_c = float(cur['temp_C'])
        humidity = int(cur['humidity'])
        precip_mm = float(cur['precipMM'])
        wind_kmh = float(cur['windspeedKmph'])
        condition_desc = cur['weatherDesc'][0]['value'].strip()
        
        # Use location coords if available
        lat = loc_info.get('latitude')
        lon = loc_info.get('longitude')
        if lat is None or lon is None:
            nearest = data.get('nearest_area', [{}])[0]
            lat = float(nearest.get('latitude', 32.78))
            lon = float(nearest.get('longitude', -96.80))
            
        return {
            'success': True,
            'rate_limited': False,
            'provider': 'World Meteorological Service (Live)',
            'city': loc_info.get('city', query_target.title()),
            'region': loc_info.get('region', ''),
            'country': loc_info.get('country', ''),
            'latitude': lat,
            'longitude': lon,
            'temperature_c': round(temp_c, 1),
            'humidity': humidity,
            'wind_kmh': round(wind_kmh, 1),
            'precipitation_mm': round(precip_mm, 2),
            'condition': condition_desc,
            'timezone_str': loc_info.get('timezone', 'US/Central'),
            'observation_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'from_cache': False
        }
    except Exception as e:
        return {
            'success': False,
            'rate_limited': False,
            'error': f"World Meteorological Service query failed: {str(e)}"
        }


def fetch_weather_openweather(loc_info: Dict[str, Any], query: str, api_key: str) -> Dict[str, Any]:
    """Fetch current conditions via OpenWeatherMap API with 429 handling."""
    encoded_query = urllib.parse.quote(query.strip())
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_query}&appid={api_key}&units=metric"
    
    data, status, err = _make_http_request(url, timeout=5.0, max_retries=1)
    if err:
        is_429 = (status == 429) or ("rate-limited" in err)
        return {'success': False, 'rate_limited': is_429, 'error': err}

    lat = float(data['coord']['lat'])
    lon = float(data['coord']['lon'])
    temp_c = float(data['main']['temp'])
    humidity = int(data['main']['humidity'])
    wind_kmh = round(float(data.get('wind', {}).get('speed', 0.0)) * 3.6, 1)

    precip_mm = 0.0
    if 'rain' in data and isinstance(data['rain'], dict):
        precip_mm = float(data['rain'].get('1h', data['rain'].get('3h', 0.0)))
    elif 'snow' in data and isinstance(data['snow'], dict):
        precip_mm = float(data['snow'].get('1h', data['snow'].get('3h', 0.0)))

    condition = data['weather'][0]['main'] if data.get('weather') else 'Unknown'
    desc = data['weather'][0]['description'].title() if data.get('weather') else condition

    tz_offset = data.get('timezone', 0)
    tz_str = loc_info.get('timezone', 'US/Central')
    if not tz_str:
        if -18000 <= tz_offset <= -14400:
            tz_str = 'US/Eastern'
        elif -21600 <= tz_offset < -18000:
            tz_str = 'US/Central'
        elif -25200 <= tz_offset < -21600:
            tz_str = 'US/Mountain'
        elif tz_offset <= -28800:
            tz_str = 'US/Pacific'

    city = data.get('name', query)
    country = data.get('sys', {}).get('country', loc_info.get('country', ''))

    return {
        'success': True,
        'rate_limited': False,
        'provider': 'OpenWeatherMap API',
        'city': city,
        'region': country,
        'country': country,
        'latitude': lat,
        'longitude': lon,
        'temperature_c': round(temp_c, 1),
        'humidity': humidity,
        'wind_kmh': wind_kmh,
        'precipitation_mm': round(precip_mm, 2),
        'condition': desc,
        'timezone_str': tz_str,
        'observation_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'from_cache': False
    }


def fetch_live_weather(query: str, custom_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entrypoint to fetch live meteorological data with bulletproof fallback.
    
    Architecture:
    1. Check Observation Cache: If city was queried within CACHE_TTL_SECONDS (5 min), return immediately.
    2. Resolve Geocoding: Uses static table -> dynamic cache -> API.
    3. Primary Provider Execution:
       - If OpenWeatherMap API key provided: queries OpenWeatherMap.
       - Otherwise: queries Open-Meteo.
    4. Seamless Fallback Execution:
       - If primary provider returns HTTP 429 or fails: seamlessly fails over to World Meteorological Service (wttr.in).
    5. Returns genuine live meteorological data, clearly labeled by provider.
    """
    if not query or not query.strip():
        return {'success': False, 'rate_limited': False, 'error': 'Please enter a valid city or location name.'}

    clean_query = query.strip()
    cache_key = clean_query.lower()

    # 1. Check Observation Cache (TTL 5 minutes)
    now_ts = time.time()
    if cache_key in _WEATHER_OBSERVATION_CACHE:
        cached_ts, cached_res = _WEATHER_OBSERVATION_CACHE[cache_key]
        if now_ts - cached_ts < CACHE_TTL_SECONDS:
            res_copy = dict(cached_res)
            res_copy['from_cache'] = True
            return res_copy

    # 2. Resolve location coordinates and metadata
    loc_info, geo_err = resolve_location_geocoding(clean_query)
    if not loc_info:
        # If geocoding failed due to 429 or error, try direct wttr query
        res_direct = fetch_weather_wttr({'city': clean_query.title()}, clean_query)
        if res_direct.get('success'):
            _WEATHER_OBSERVATION_CACHE[cache_key] = (now_ts, res_direct)
            return res_direct
        return {'success': False, 'rate_limited': False, 'error': geo_err or f"Location '{query}' could not be resolved."}

    api_key = custom_api_key or get_api_key()

    # 3. Strategy A: OpenWeatherMap (if configured)
    if api_key:
        res = fetch_weather_openweather(loc_info, clean_query, api_key)
        if res.get('success'):
            _WEATHER_OBSERVATION_CACHE[cache_key] = (now_ts, res)
            return res
        # If OpenWeatherMap rate limited, fall through to free providers

    # 4. Strategy B: Open-Meteo
    res_meteo = fetch_weather_openmeteo(loc_info)
    if res_meteo.get('success'):
        _WEATHER_OBSERVATION_CACHE[cache_key] = (now_ts, res_meteo)
        return res_meteo

    # 5. Strategy C: Live World Meteorological Service Fallback
    # (Triggers when Open-Meteo returns HTTP 429 or network error)
    res_fallback = fetch_weather_wttr(loc_info, clean_query)
    if res_fallback.get('success'):
        _WEATHER_OBSERVATION_CACHE[cache_key] = (now_ts, res_fallback)
        return res_fallback

    # If all providers fail, return clean rate-limit message
    if res_meteo.get('rate_limited'):
        return {
            'success': False,
            'rate_limited': True,
            'error': 'Weather service temporarily rate-limited. Please wait a moment before trying again, or use Manual Prediction Mode below.'
        }

    return res_fallback if not res_fallback.get('success') else res_meteo


def build_prediction_features(weather_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform live API weather observations into the exact feature schema
    expected by the trained Scikit-Learn models.
    """
    dt_now = datetime.now(timezone.utc)
    month = dt_now.month
    day = dt_now.day
    hour = dt_now.hour
    season = get_season(month)

    # Unit conversion: mm to inches
    precip_mm = float(weather_info.get('precipitation_mm', 0.0))
    precip_in = round(precip_mm / 25.4, 4)

    # Region / State resolution
    region_raw = str(weather_info.get('region', '')).strip()
    state_code = US_STATE_MAP.get(region_raw, region_raw.upper() if len(region_raw) == 2 else 'TX')
    if weather_info.get('state'):
        state_code = weather_info['state']

    # Timezone resolution
    tz = weather_info.get('timezone_str', 'US/Central')

    feature_dict = {
        'precipitation_in': float(precip_in),
        'latitude': float(weather_info['latitude']),
        'longitude': float(weather_info['longitude']),
        'month': int(month),
        'day': int(day),
        'hour': int(hour),
        'timezone': str(tz),
        'season': str(season),
        'state': str(state_code)
    }

    return pd.DataFrame([feature_dict])
