# weather_service.py - COMPLETE VERSION
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class RealTimeWeatherService:
    def __init__(self):
        # Setup the Open-Meteo API client with cache and retry
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # 1 hour cache
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Tamil Nadu coordinates for fallback
        self.tamil_nadu_coordinates = {
            'Chennai': (13.0827, 80.2707),
            'Coimbatore': (11.0168, 76.9558),
            'Madurai': (9.9252, 78.1198),
            'Trichy': (10.7905, 78.7047),
            'Cuddalore': (11.7447, 79.7680),
            'Nagapattinam': (10.7667, 79.8417),
            'Thanjavur': (10.7870, 79.1378),
            'Salem': (11.6643, 78.1460)
        }
    
    def get_real_time_weather(self, latitude, longitude, district_name=""):
        """Get real-time weather data from Open-Meteo API"""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "surface_pressure"],
                "hourly": ["precipitation", "rain", "soil_moisture_0_to_1cm"],
                "daily": ["precipitation_sum", "rain_sum", "precipitation_hours"],
                "timezone": "auto",
                "forecast_days": 3
            }
            
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Current weather data
            current = response.Current()
            hourly = response.Hourly()
            daily = response.Daily()
            
            # Extract rainfall data
            hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
            hourly_rain = hourly.Variables(1).ValuesAsNumpy()
            
            # Calculate rainfall metrics
            rainfall_1h = hourly_precipitation[0] if len(hourly_precipitation) > 0 else 0
            rainfall_6h = np.sum(hourly_precipitation[:6]) if len(hourly_precipitation) >= 6 else 0
            rainfall_24h = np.sum(hourly_precipitation[:24]) if len(hourly_precipitation) >= 24 else 0
            
            # Get forecast data
            daily_precipitation = daily.Variables(0).ValuesAsNumpy()
            forecast_rainfall = daily_precipitation[0] if len(daily_precipitation) > 0 else 0
            
            weather_data = {
                'temperature': float(current.Variables(0).Value()),
                'humidity': float(current.Variables(1).Value()),
                'pressure': float(current.Variables(4).Value()),
                'current_precipitation': float(current.Variables(2).Value()),
                'current_rain': float(current.Variables(3).Value()),
                'rainfall_1h': float(rainfall_1h),
                'rainfall_6h': float(rainfall_6h),
                'rainfall_24h': float(rainfall_24h),
                'forecast_rainfall': float(forecast_rainfall),
                'soil_moisture': float(hourly.Variables(2).ValuesAsNumpy()[0] if len(hourly.Variables(2).ValuesAsNumpy()) > 0 else 0.5),
                'data_source': 'Open-Meteo API',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'coordinates': f"{latitude:.4f}, {longitude:.4f}"
            }
            
            print(f"✅ Successfully fetched real-time weather for {district_name}")
            return weather_data
            
        except Exception as e:
            print(f"❌ Weather API error for {district_name}: {e}")
            print("🔄 Using fallback weather data...")
            return self.get_fallback_weather_data(district_name)
    
    def get_fallback_weather_data(self, district_name):
        """Provide realistic fallback data when API fails"""
        # Simulate realistic Tamil Nadu weather patterns
        base_rainfall = {
            'Chennai': np.random.uniform(5, 25),  # Coastal - moderate rain
            'Cuddalore': np.random.uniform(10, 40),  # Coastal - higher rain
            'Nagapattinam': np.random.uniform(8, 35),  # Coastal
            'Thanjavur': np.random.uniform(3, 20),  # Delta region
            'Trichy': np.random.uniform(2, 15),  # Inland
            'Madurai': np.random.uniform(1, 10),  # Dry region
            'Coimbatore': np.random.uniform(2, 12),  # Western ghats
            'Salem': np.random.uniform(1, 8)  # Dry region
        }
        
        rainfall = float(base_rainfall.get(district_name, np.random.uniform(1, 20)))
        
        return {
            'temperature': float(np.random.uniform(25, 35)),
            'humidity': float(np.random.uniform(60, 90)),
            'pressure': float(np.random.uniform(1000, 1015)),
            'current_precipitation': float(rainfall),
            'current_rain': float(rainfall),
            'rainfall_1h': float(rainfall * 0.1),
            'rainfall_6h': float(rainfall * 0.4),
            'rainfall_24h': float(rainfall),
            'forecast_rainfall': float(rainfall * 1.2),
            'soil_moisture': float(np.random.uniform(0.3, 0.8)),
            'data_source': 'Fallback Data',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'coordinates': 'Unknown'
        }
    
    def get_weather_alerts(self, weather_data):
        """Generate weather alerts based on current conditions"""
        alerts = []
        
        if weather_data['rainfall_24h'] > 50:
            alerts.append({
                'level': 'HIGH',
                'message': 'Heavy rainfall warning - Flood risk elevated',
                'type': 'rainfall'
            })
        
        if weather_data['rainfall_24h'] > 100:
            alerts.append({
                'level': 'SEVERE',
                'message': 'Extreme rainfall alert - Immediate action required',
                'type': 'extreme_rain'
            })
            
        if weather_data['soil_moisture'] > 0.7:
            alerts.append({
                'level': 'MEDIUM',
                'message': 'High soil saturation - Reduced drainage capacity',
                'type': 'soil_moisture'
            })
            
        return alerts
    
    def batch_get_weather_data(self, districts_data):
        """Get weather data for multiple districts efficiently"""
        weather_results = {}
        
        for district, data in districts_data.items():
            lat = data['lat']
            lon = data['lon']
            weather_results[district] = self.get_real_time_weather(lat, lon, district)
            
        return weather_results