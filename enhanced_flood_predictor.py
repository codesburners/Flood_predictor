# enhanced_flood_predictor.py
class EnhancedFloodPredictor:
    def __init__(self):
        self.weatherstack = WeatherstackService()
        self.open_meteo = ExistingWeatherService()  # Keep as fallback
    
    def predict_flood_risk_enhanced(self, location_data, use_real_time=True):
        """Enhanced prediction with real-time weather data"""
        if use_real_time:
            # Get TRUE real-time data from Weatherstack
            weather_data = self.weatherstack.get_real_time_weather(
                location_data['lat'], 
                location_data['lon'],
                location_data.get('name', 'Unknown')
            )
        else:
            # Fallback to Open-Meteo
            weather_data = self.open_meteo.get_weather_data(
                location_data['lat'], 
                location_data['lon']
            )
        
        # Enhanced risk calculation with real-time factors
        risk_score = self._calculate_enhanced_risk(location_data, weather_data)
        
        return {
            'predicted_risk': risk_score,
            'weather_data': weather_data,
            'data_source': weather_data.get('data_source', 'Unknown'),
            'real_time_indicators': self._get_real_time_indicators(weather_data),
            'flash_flood_warning': self._check_flash_flood_risk(weather_data)
        }
    
    def _calculate_enhanced_risk(self, location_data, weather_data):
        """Calculate risk with real-time factors"""
        base_risk = self._calculate_base_risk(location_data)
        
        # REAL-TIME ADJUSTMENTS
        real_time_factors = {
            'current_rain_intensity': self._get_rain_intensity_factor(weather_data),
            'recent_rain_trend': self._get_rain_trend_factor(weather_data),
            'soil_saturation': self._calculate_current_saturation(weather_data),
            'immediate_storm_risk': self._get_storm_risk(weather_data)
        }
        
        # Apply real-time adjustments
        adjusted_risk = base_risk
        for factor, weight in real_time_factors.items():
            adjusted_risk *= weight
        
        return min(10, adjusted_risk)
    
    def _get_rain_intensity_factor(self, weather_data):
        """Factor based on current rainfall intensity"""
        current_rain = weather_data.get('rainfall_current', 0)
        
        if current_rain > 50:  # mm/hour - extreme rainfall
            return 1.8
        elif current_rain > 25:  # mm/hour - heavy rainfall
            return 1.5
        elif current_rain > 10:  # mm/hour - moderate rainfall
            return 1.2
        else:
            return 1.0
    
    def _check_flash_flood_risk(self, weather_data):
        """Check for immediate flash flood risk"""
        current_rain = weather_data.get('rainfall_current', 0)
        recent_rain = weather_data.get('rainfall_1h', 0)
        
        if current_rain > 30 or recent_rain > 50:
            return {
                'level': 'HIGH',
                'message': f'Flash flood warning! Current rainfall: {current_rain}mm/h',
                'recommended_actions': ['Immediate evacuation in low-lying areas', 'Close flood gates', 'Alert rescue teams']
            }
        elif current_rain > 15:
            return {
                'level': 'MEDIUM', 
                'message': f'Heavy rainfall ongoing: {current_rain}mm/h',
                'recommended_actions': ['Monitor water levels', 'Prepare evacuation plans', 'Alert communities']
            }
        else:
            return {
                'level': 'LOW',
                'message': 'No immediate flash flood risk',
                'recommended_actions': ['Continue monitoring', 'Review drainage systems']
            }