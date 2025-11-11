# weatherstack_integration.py
import requests
import os

class WeatherstackService:
    def __init__(self):
        self.api_key = os.getenv('WEATHERSTACK_API_KEY', 'your_api_key_here')
        self.base_url = "http://api.weatherstack.com/"
    
    def get_real_time_weather(self, lat, lon, location_name):
        """Get true real-time weather data"""
        try:
            # Use location name for better accuracy
            params = {
                'access_key': self.api_key,
                'query': location_name,
                'units': 'm'
            }
            
            response = requests.get(f"{self.base_url}current", params=params)
            data = response.json()
            
            return self._parse_weatherstack_data(data)
            
        except Exception as e:
            print(f"Weatherstack API error: {e}")
            return self._get_fallback_data()
    
    def _parse_weatherstack_data(self, data):
        """Parse Weatherstack specific format"""
        current = data.get('current', {})
        
        return {
            'temperature': current.get('temperature'),
            'humidity': current.get('humidity'),
            'pressure': current.get('pressure'),
            'rainfall_current': current.get('precip', 0),  # CURRENT rainfall mm
            'rainfall_1h': current.get('precip', 0),  # Last hour rainfall
            'wind_speed': current.get('wind_speed'),
            'wind_dir': current.get('wind_dir'),
            'visibility': current.get('visibility'),
            'cloud_cover': current.get('cloudcover'),
            'feels_like': current.get('feelslike'),
            'uv_index': current.get('uv_index'),
            'weather_descriptions': current.get('weather_descriptions', []),
            'data_source': 'Weatherstack (Real-time)',
            'last_updated': data.get('location', {}).get('localtime')
        }