## Tamil Nadu Flood Risk Prediction & Mitigation System

### Overview
This app provides real-time flood risk assessment for Tamil Nadu districts and generates optimized mitigation schedules. It blends a rule-based risk model with a scikit-learn model and includes an AI-enhanced (RL-inspired) scheduler.

### Components
- `app.py`: Streamlit UI.
- `flood_predictor.py`: Risk scoring (rule-based + ML blend).
- `mitigation_scheduler.py`: Dynamic mitigation plan generator.
- `rl_agent.py`: AI-enhanced scheduler with persistent learning.
- `tamilnadu_data.py`: District features.
- `weather_service.py`: Open-Meteo live weather with fallback.

### ML Model
- Artifact: `flood_prediction_model.pkl` (scikit-learn).
- Features (8): rainfall_24h, rainfall_6h, soil_moisture, humidity, elevation, population_density, distance_to_river, drainage_capacity.
- Label: bootstrapped from the rule-based scorer.

### Train/Re-train
Prereqs: Python 3.10+, see `requirements.txt`.

Run:

```bash
python model_training.py --augments 30 --out flood_prediction_model.pkl --metrics model_training_metrics.json
```

This generates synthetic-but-realistic samples per district, trains a RandomForest, evaluates, and saves:
- model: `flood_prediction_model.pkl`
- metrics: `model_training_metrics.json`

### AI (RL) Scheduler
- Loads and saves learning state at `rl_models/rl_agent_state.json`.
- Learning persists across sessions automatically when generating schedules in the AI mode.

### Run the App
```bash
streamlit run app.py
```

### Notes
- `enhanced_flood_predictor.py` references Weatherstack and is not wired into the app.
- Add your own data and replace the bootstrapped label approach when historical flood outcomes are available.


