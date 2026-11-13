"""
Weather API integration and feature mapping module for StormCast.
Retrieves live meteorological observations and maps them to the trained ML feature space.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from src.preprocessing import get_season


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

# Mapping from IANA timezone strings to standard US TimeZones used in ASOS records
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

# WMO Weather Interpretation Codes (WW)
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
    """
    Retrieve OpenWeatherMap API key from environment variables or .env file.
    Does not crash if none is found.
    """
    key = os.getenv('OPENWEATHER_API_KEY') or os.getenv('OPENWEATHERMAP_API_KEY')
    if key and key.strip():
        return key.strip()

    # Try local .env file
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


def fetch_weather_openweather(query: str, api_key: str) -> Dict[str, Any]:
    """Fetch current conditions via OpenWeatherMap API."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_query}&appid={api_key}&units=metric"
    req = urllib.request.Request(url, headers={'User-Agent': 'StormCast/1.0'})
    
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        
    lat = float(data['coord']['lat'])
    lon = float(data['coord']['lon'])
    temp_c = float(data['main']['temp'])
    humidity = int(data['main']['humidity'])
    wind_kmh = round(float(data.get('wind', {}).get('speed', 0.0)) * 3.6, 1)
    
    # Precipitation in last 1 hour (mm)
    precip_mm = 0.0
    if 'rain' in data and isinstance(data['rain'], dict):
        precip_mm = float(data['rain'].get('1h', data['rain'].get('3h', 0.0)))
    elif 'snow' in data and isinstance(data['snow'], dict):
        precip_mm = float(data['snow'].get('1h', data['snow'].get('3h', 0.0)))
        
    condition = data['weather'][0]['main'] if data.get('weather') else 'Unknown'
    desc = data['weather'][0]['description'].title() if data.get('weather') else condition

    # Timezone offset in seconds
    tz_offset = data.get('timezone', 0)
    # Estimate standard US timezone if inside contiguous US
    tz_str = 'US/Central'
    if -18000 <= tz_offset <= -14400:
        tz_str = 'US/Eastern'
    elif -21600 <= tz_offset < -18000:
        tz_str = 'US/Central'
    elif -25200 <= tz_offset < -21600:
        tz_str = 'US/Mountain'
    elif tz_offset <= -28800:
        tz_str = 'US/Pacific'

    city = data.get('name', query)
    country = data.get('sys', {}).get('country', '')

    return {
        'success': True,
        'provider': 'OpenWeatherMap',
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
        'observation_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    }


def fetch_weather_openmeteo(query: str) -> Dict[str, Any]:
    """
    Fetch current conditions via Open-Meteo Geocoding & Forecast APIs.
    Requires no API keys; highly reliable for global locations.
    """
    # 1. Geocoding search
    encoded_city = urllib.parse.quote(query.strip())
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
    geo_req = urllib.request.Request(geo_url, headers={'User-Agent': 'StormCast/1.0'})
    
    with urllib.request.urlopen(geo_req, timeout=8) as resp:
        geo_data = json.loads(resp.read().decode('utf-8'))
        
    results = geo_data.get('results')
    if not results or len(results) == 0:
        return {
            'success': False,
            'error': f"Location '{query}' could not be resolved. Please check the spelling or specify State/Country."
        }
        
    top_res = results[0]
    lat = float(top_res['latitude'])
    lon = float(top_res['longitude'])
    city_name = top_res.get('name', query)
    admin1 = top_res.get('admin1', '')
    country = top_res.get('country', '')
    raw_tz = top_res.get('timezone', 'UTC')
    
    # Map timezone to ASOS conventions
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

    # 2. Weather observation request
    w_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,showers,weather_code,wind_speed_10m"
        f"&timezone=auto"
    )
    w_req = urllib.request.Request(w_url, headers={'User-Agent': 'StormCast/1.0'})
    with urllib.request.urlopen(w_req, timeout=8) as resp:
        w_data = json.loads(resp.read().decode('utf-8'))

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
        'provider': 'Open-Meteo Weather Service',
        'city': city_name,
        'region': admin1,
        'country': country,
        'latitude': lat,
        'longitude': lon,
        'temperature_c': round(temp_c, 1),
        'humidity': humidity,
        'wind_kmh': round(wind_kmh, 1),
        'precipitation_mm': round(precip_mm, 2),
        'condition': condition_desc,
        'timezone_str': tz_str,
        'observation_time': obs_time_display
    }


def fetch_live_weather(query: str, custom_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entrypoint to fetch live meteorological data.
    Uses OpenWeatherMap if an API key is provided, or seamlessly utilizes Open-Meteo.
    Handles network errors, invalid queries, and timeouts gracefully.
    """
    if not query or not query.strip():
        return {'success': False, 'error': 'Please enter a valid city or location name.'}

    api_key = custom_api_key or get_api_key()

    # If API key is available, attempt OpenWeatherMap first
    if api_key:
        try:
            res = fetch_weather_openweather(query.strip(), api_key)
            if res.get('success'):
                return res
        except urllib.error.HTTPError as e:
            if e.code == 401:
                pass  # Key invalid or unactivated; fallback to Open-Meteo
            elif e.code == 404:
                return {'success': False, 'error': f"City '{query}' not found by OpenWeatherMap."}
        except Exception:
            pass  # Fall through to Open-Meteo

    # Open-Meteo endpoint (no API key required)
    try:
        return fetch_weather_openmeteo(query.strip())
    except urllib.error.URLError as e:
        return {
            'success': False,
            'error': f"Network connection error while reaching weather service ({e.reason})."
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unable to retrieve live weather data: {str(e)}"
        }


def build_prediction_features(weather_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform live API weather observations into the exact feature schema
    expected by the trained Scikit-Learn models.
    
    Conversions performed:
    - Precipitation: mm -> inches (1 in = 25.4 mm)
    - Coordinates: float latitude and longitude
    - Temporal features: month, day, hour (UTC), season
    - Categorical: timezone, state
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
