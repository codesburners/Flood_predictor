# flood_predictor.py - UPDATED WITH REAL WEATHER
import pandas as pd
import numpy as np
import joblib
from weather_service import RealTimeWeatherService

class FloodPredictor:
    def __init__(self):
        self.weather_service = RealTimeWeatherService()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained ML model (if available)"""
        try:
            self.model = joblib.load('flood_prediction_model.pkl')
            print("✅ Loaded pre-trained flood prediction model")
        except:
            print("ℹ️ No pre-trained model found. Using rule-based prediction.")
            self.model = None
    
    def predict_flood_risk(self, location_data, use_real_weather=True):
        """Predict flood risk using real-time weather data"""
        try:
            if use_real_weather:
                # Get real-time weather data
                weather_data = self.weather_service.get_real_time_weather(
                    location_data['lat'], 
                    location_data['lon'],
                    location_data.get('name', 'Unknown')
                )
            else:
                # Use fallback data
                weather_data = self.weather_service.get_fallback_weather_data(
                    location_data.get('name', 'Unknown')
                )
            
            # Calculate risk using enhanced factors
            risk_score = self._calculate_comprehensive_risk(location_data, weather_data)
            
            # Get weather alerts
            weather_alerts = self.weather_service.get_weather_alerts(weather_data)
            
            return {
                'predicted_risk': risk_score,
                'weather_data': weather_data,
                'weather_alerts': weather_alerts,
                'risk_factors': self._get_risk_factors(location_data, weather_data),
                'prediction_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': weather_data['data_source']
            }
            
        except Exception as e:
            print(f"❌ Flood prediction error: {e}")
            return self._get_fallback_prediction(location_data)
    
    def _calculate_comprehensive_risk(self, location_data, weather_data):
        """Calculate comprehensive flood risk score (0-10)"""
        # Extract factors
        rainfall_24h = weather_data['rainfall_24h']
        rainfall_forecast = weather_data['forecast_rainfall']
        soil_moisture = weather_data['soil_moisture']
        elevation = location_data['elevation']
        population_density = location_data['population_density']
        drainage_capacity = location_data['drainage_capacity']
        distance_to_river = location_data['distance_to_river']
        
        # Normalize factors (0-1 scale)
        rainfall_factor = min(1.0, rainfall_24h / 100)  # 100mm = max risk
        forecast_factor = min(1.0, rainfall_forecast / 120)  # 120mm forecast = max risk
        elevation_factor = max(0, 1 - (elevation / 200))  # Lower elevation = higher risk
        population_factor = min(1.0, population_density / 30000)
        drainage_factor = 1 - drainage_capacity  # Poor drainage = higher risk
        river_proximity_factor = min(1.0, 1 / (distance_to_river + 0.1))
        soil_moisture_factor = max(0, (soil_moisture - 0.3) / 0.5)  # Higher moisture = higher risk
        
        # Weighted risk calculation (based on historical flood data patterns)
        weights = {
            'rainfall_current': 0.25,
            'rainfall_forecast': 0.15,
            'elevation': 0.15,
            'drainage': 0.12,
            'river_proximity': 0.12,
            'soil_moisture': 0.10,
            'population': 0.08,
            'humidity': 0.03
        }
        
        humidity_factor = weather_data['humidity'] / 100
        
        # Calculate weighted risk score
        risk_score = (
            weights['rainfall_current'] * rainfall_factor +
            weights['rainfall_forecast'] * forecast_factor +
            weights['elevation'] * elevation_factor +
            weights['drainage'] * drainage_factor +
            weights['river_proximity'] * river_proximity_factor +
            weights['soil_moisture'] * soil_moisture_factor +
            weights['population'] * population_factor +
            weights['humidity'] * humidity_factor
        ) * 10  # Scale to 0-10
        
        # Apply ML model if available
        if self.model:
            try:
                ml_features = self._prepare_ml_features(location_data, weather_data)
                ml_risk = self.model.predict([ml_features])[0]
                # Blend rule-based and ML prediction
                risk_score = 0.7 * risk_score + 0.3 * ml_risk
            except:
                pass  # Fallback to rule-based
        
        return min(10, max(0, risk_score))
    
    def _get_risk_factors(self, location_data, weather_data):
        """Get detailed risk factor breakdown"""
        rainfall_24h = weather_data['rainfall_24h']
        elevation = location_data['elevation']
        
        return {
            'rainfall_impact': min(1.0, rainfall_24h / 100),
            'elevation_impact': max(0, 1 - (elevation / 200)),
            'drainage_impact': 1 - location_data['drainage_capacity'],
            'river_proximity_impact': min(1.0, 1 / (location_data['distance_to_river'] + 0.1)),
            'population_impact': min(1.0, location_data['population_density'] / 30000),
            'soil_moisture_impact': max(0, (weather_data['soil_moisture'] - 0.3) / 0.5)
        }
    
    def _prepare_ml_features(self, location_data, weather_data):
        """Prepare features for ML model prediction"""
        return [
            weather_data['rainfall_24h'],
            weather_data['rainfall_6h'],
            weather_data['soil_moisture'],
            weather_data['humidity'],
            location_data['elevation'],
            location_data['population_density'],
            location_data['distance_to_river'],
            location_data['drainage_capacity']
        ]
    
    # In flood_predictor.py - make sure this method exists:
def _get_fallback_prediction(self, location_data):
    """Fallback prediction when main logic fails"""
    return {
        'predicted_risk': float(np.random.uniform(3, 8)),
        'weather_data': self.weather_service.get_fallback_weather_data('Unknown'),
        'weather_alerts': [],
        'risk_factors': {
            'rainfall_impact': 0.5,
            'elevation_impact': 0.3,
            'drainage_impact': 0.4,
            'river_proximity_impact': 0.2,
            'population_impact': 0.3,
            'soil_moisture_impact': 0.3
        },
        'prediction_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': 'Fallback System'
    }