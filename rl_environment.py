# rl_environment.py
import numpy as np
import gym
from gym import spaces
from datetime import datetime, timedelta
import pandas as pd

class FloodMitigationEnv(gym.Env):
    def __init__(self, districts_data, weather_service, max_teams=10, max_budget=100000, horizon=7):
        super(FloodMitigationEnv, self).__init__()
        
        self.districts_data = districts_data
        self.weather_service = weather_service
        self.max_teams = max_teams
        self.max_budget = max_budget
        self.horizon = horizon
        self.district_names = list(districts_data.keys())
        
        # Action space: For each district, choose from 6 mitigation actions + do nothing
        self.action_space = spaces.MultiDiscrete([7] * len(districts_data))
        
        # Observation space: [risk_scores, rainfall, teams_used, budget_used, day, weather_conditions...]
        self.observation_space = spaces.Box(
            low=0, high=100, 
            shape=(len(districts_data) * 5 + 4,), dtype=np.float32
        )
        
        self.mitigation_actions = {
            1: {'name': 'Emergency_Evacuation', 'cost': 5000, 'teams': 3, 'effectiveness': 0.8, 'duration': 1},
            2: {'name': 'Temporary_Barriers', 'cost': 3000, 'teams': 2, 'effectiveness': 0.6, 'duration': 3},
            3: {'name': 'Pump_Installation', 'cost': 4000, 'teams': 2, 'effectiveness': 0.7, 'duration': 4},
            4: {'name': 'Drainage_Clearing', 'cost': 800, 'teams': 1, 'effectiveness': 0.4, 'duration': 2},
            5: {'name': 'Sandbag_Deployment', 'cost': 500, 'teams': 1, 'effectiveness': 0.3, 'duration': 2},
            6: {'name': 'Warning_System', 'cost': 300, 'teams': 1, 'effectiveness': 0.2, 'duration': 1}
        }
        
        self.reset()
    
    def reset(self):
        self.current_day = 0
        self.teams_used = 0
        self.budget_used = 0
        self.schedule = []
        
        # Initialize with real weather data
        self._update_weather_data()
        
        # Initialize risk scores based on current conditions
        self.risk_scores = {}
        for district in self.district_names:
            self.risk_scores[district] = self._calculate_initial_risk(district)
        
        return self._get_observation()
    
    def _update_weather_data(self):
        """Update real-time weather data for all districts"""
        self.weather_data = {}
        for district in self.district_names:
            loc_data = self.districts_data[district]
            weather = self.weather_service.get_real_time_weather(loc_data['lat'], loc_data['lon'], district)
            self.weather_data[district] = weather
    
    def _calculate_initial_risk(self, district):
        """Calculate initial risk based on real weather and location factors"""
        weather = self.weather_data[district]
        loc_data = self.districts_data[district]
        
        # Use similar risk calculation as main predictor
        rainfall_factor = min(1.0, weather['rainfall_24h'] / 100)
        elevation_factor = max(0, 1 - (loc_data['elevation'] / 200))
        drainage_factor = 1 - loc_data['drainage_capacity']
        river_proximity_factor = min(1.0, 1 / (loc_data['distance_to_river'] + 0.1))
        soil_moisture_factor = max(0, (weather['soil_moisture'] - 0.3) / 0.5)
        
        risk_score = (
            0.25 * rainfall_factor +
            0.15 * elevation_factor +
            0.12 * drainage_factor +
            0.12 * river_proximity_factor +
            0.10 * soil_moisture_factor +
            0.08 * min(1.0, loc_data['population_density'] / 30000) +
            0.03 * (weather['humidity'] / 100)
        ) * 10
        
        return min(10, max(0, risk_score))
    
    def _get_observation(self):
        """Get current state observation with real weather data"""
        obs = []
        
        # District states with real weather
        for district in self.district_names:
            weather = self.weather_data[district]
            loc_data = self.districts_data[district]
            
            obs.extend([
                self.risk_scores[district],
                weather['rainfall_24h'],
                weather['forecast_rainfall'],
                weather['soil_moisture'],
                loc_data['elevation'] / 200  # Normalized
            ])
        
        # Global state
        obs.extend([
            self.teams_used / self.max_teams,
            self.budget_used / self.max_budget,
            self.current_day / self.horizon,
            len(self.schedule) / (len(self.district_names) * 3)  # Normalized schedule length
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        """Execute actions and advance environment"""
        reward = 0
        actions_taken = []
        
        # Execute actions for each district
        for i, (district, action_idx) in enumerate(zip(self.district_names, actions)):
            if action_idx == 0:  # Do nothing
                continue
                
            action = self.mitigation_actions[action_idx]
            cost = action['cost']
            teams = action['teams']
            
            # Check constraints
            if (self.teams_used + teams <= self.max_teams and 
                self.budget_used + cost <= self.max_budget):
                
                # Apply action effect
                risk_reduction = self.risk_scores[district] * action['effectiveness']
                self.risk_scores[district] = max(0, self.risk_scores[district] - risk_reduction)
                self.teams_used += teams
                self.budget_used += cost
                
                # Record action
                action_record = {
                    'district': district,
                    'action': action['name'],
                    'start_day': self.current_day + 1,
                    'duration': action['duration'],
                    'teams_required': teams,
                    'cost': cost,
                    'risk_reduction': risk_reduction
                }
                self.schedule.append(action_record)
                actions_taken.append(action_record)
                
                # Reward components
                reward += risk_reduction * 10  # Reward for risk reduction
                reward -= cost / 1000  # Small penalty for cost
        
        # Advance time
        self.current_day += 1
        
        # Update environment with new weather and natural risk progression
        self._update_environment()
        
        # Calculate final reward
        total_risk = sum(self.risk_scores.values())
        reward -= total_risk  # Penalty for remaining risk
        
        # Check terminal state
        done = (self.current_day >= self.horizon) or (total_risk < 2) or (self.budget_used >= self.max_budget)
        
        info = {
            'actions_taken': actions_taken,
            'total_risk': total_risk,
            'teams_used': self.teams_used,
            'budget_used': self.budget_used,
            'schedule': self.schedule.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _update_environment(self):
        """Update environment state based on weather changes"""
        # Update weather data
        old_weather = self.weather_data.copy()
        self._update_weather_data()
        
        # Update risks based on weather changes
        for district in self.district_names:
            # Calculate weather impact
            old_rainfall = old_weather[district]['rainfall_24h']
            new_rainfall = self.weather_data[district]['rainfall_24h']
            rainfall_change = new_rainfall - old_rainfall
            
            # Rainfall increases risk
            rainfall_impact = rainfall_change / 50  # Normalize impact
            self.risk_scores[district] = min(10, self.risk_scores[district] + rainfall_impact)
            
            # Soil moisture impact
            soil_moisture = self.weather_data[district]['soil_moisture']
            if soil_moisture > 0.7:
                self.risk_scores[district] = min(10, self.risk_scores[district] + 0.5)
    
    def render(self):
        """Render current environment state"""
        print(f"Day {self.current_day}")
        print(f"Total Risk: {sum(self.risk_scores.values()):.2f}")
        print(f"Teams Used: {self.teams_used}/{self.max_teams}")
        print(f"Budget Used: ₹{self.budget_used:,}/₹{self.max_budget:,}")
        for district in self.district_names:
            print(f"  {district}: Risk {self.risk_scores[district]:.2f}, "
                  f"Rainfall: {self.weather_data[district]['rainfall_24h']:.1f}mm")