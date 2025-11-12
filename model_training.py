import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from tamilnadu_data import TamilNaduData
from weather_service import RealTimeWeatherService
from flood_predictor import FloodPredictor


def generate_training_samples(
    tn_data: TamilNaduData,
    weather_service: RealTimeWeatherService,
    predictor: FloodPredictor,
    num_augments_per_district: int = 30,
) -> pd.DataFrame:
    """Bootstrap a dataset by pairing district features with varied weather
    and using the existing rule-based scorer (without ML blend) as noisy labels."""

    rows: List[Dict[str, Any]] = []
    districts = tn_data.districts

    # Helper to compute the rule-based score only
    def rule_based_score(location_data: Dict[str, Any], weather_data: Dict[str, Any]) -> float:
        # Copy core of _calculate_comprehensive_risk without ML blending
        rainfall_24h = weather_data['rainfall_24h']
        rainfall_forecast = weather_data['forecast_rainfall']
        soil_moisture = weather_data['soil_moisture']
        elevation = location_data['elevation']
        population_density = location_data['population_density']
        drainage_capacity = location_data['drainage_capacity']
        distance_to_river = location_data['distance_to_river']
        rainfall_factor = min(1.0, rainfall_24h / 100)
        forecast_factor = min(1.0, rainfall_forecast / 120)
        elevation_factor = max(0, 1 - (elevation / 200))
        population_factor = min(1.0, population_density / 30000)
        drainage_factor = 1 - drainage_capacity
        river_proximity_factor = min(1.0, 1 / (distance_to_river + 0.1))
        soil_moisture_factor = max(0, (soil_moisture - 0.3) / 0.5)
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
        score = (
            weights['rainfall_current'] * rainfall_factor +
            weights['rainfall_forecast'] * forecast_factor +
            weights['elevation'] * elevation_factor +
            weights['drainage'] * drainage_factor +
            weights['river_proximity'] * river_proximity_factor +
            weights['soil_moisture'] * soil_moisture_factor +
            weights['population'] * population_factor +
            weights['humidity'] * humidity_factor
        ) * 10.0
        return float(max(0.0, min(10.0, score)))

    rng = np.random.default_rng(42)
    for name, loc in districts.items():
        # Get a baseline real/fallback weather snapshot
        base_weather = weather_service.get_fallback_weather_data(name)
        for _ in range(num_augments_per_district):
            # Randomly perturb weather around baseline to create variety
            weather = dict(base_weather)
            weather['rainfall_24h'] = float(max(0.0, weather['rainfall_24h'] * rng.uniform(0.5, 2.0)))
            weather['rainfall_6h'] = float(max(0.0, weather['rainfall_6h'] * rng.uniform(0.5, 2.0)))
            weather['soil_moisture'] = float(min(0.95, max(0.1, weather['soil_moisture'] + rng.normal(0, 0.05))))
            weather['humidity'] = float(min(100.0, max(15.0, weather['humidity'] + rng.normal(0, 5)))))

            # Label via rule-based scorer
            y = rule_based_score(loc, weather)

            rows.append({
                # Features (align with FloodPredictor._prepare_ml_features)
                'rainfall_24h': weather['rainfall_24h'],
                'rainfall_6h': weather['rainfall_6h'],
                'soil_moisture': weather['soil_moisture'],
                'humidity': weather['humidity'],
                'elevation': loc['elevation'],
                'population_density': loc['population_density'],
                'distance_to_river': loc['distance_to_river'],
                'drainage_capacity': loc['drainage_capacity'],
                # Target
                'risk_score': y,
                # Meta
                'district': name
            })

    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame) -> Dict[str, Any]:
    X = df[['rainfall_24h', 'rainfall_6h', 'soil_moisture', 'humidity',
            'elevation', 'population_density', 'distance_to_river', 'drainage_capacity']].values
    y = df['risk_score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=1337,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        'r2': float(r2_score(y_test, preds)),
        'mae': float(mean_absolute_error(y_test, preds)),
        'timestamp': datetime.utcnow().isoformat()
    }
    return {'model': model, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description="Train flood risk ML model and save to flood_prediction_model.pkl")
    parser.add_argument('--augments', type=int, default=30, help='Augmented samples per district')
    parser.add_argument('--out', type=str, default='flood_prediction_model.pkl', help='Output model path')
    parser.add_argument('--metrics', type=str, default='model_training_metrics.json', help='Where to write training metrics')
    args = parser.parse_args()

    tn_data = TamilNaduData()
    weather_service = RealTimeWeatherService()
    predictor = FloodPredictor()

    df = generate_training_samples(tn_data, weather_service, predictor, num_augments_per_district=args.augments)
    result = train_model(df)
    joblib.dump(result['model'], args.out)

    with open(args.metrics, 'w', encoding='utf-8') as f:
        json.dump(result['metrics'], f, indent=2)

    print(f"✅ Saved model to {args.out}")
    print(f"📊 Metrics: R2={result['metrics']['r2']:.3f}, MAE={result['metrics']['mae']:.3f}")


if __name__ == '__main__':
    main()


