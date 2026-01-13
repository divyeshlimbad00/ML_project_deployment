# CardioSense

Educational demo: Heart disease risk predictor with a Bootstrap UI, Flask backend, and a scikit-learn model trained on `project_week/cardio_train.csv`.

## Features
- Responsive Bootstrap landing page and predictor form
- Backend API endpoint `/api/predict` (JSON) and `/predict` HTML form
- Results page with Chart.js visualizations
- Training script: `model/train_model.py` saves `models/cardio_model.pkl`

## Quick start (local)
1. Create environment (recommended):
   - Python 3.10+ or conda
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train model (creates `models/cardio_model.pkl`):
   ```bash
   python model/train_model.py
   ```
4. Run the app:
   ```bash
   python app.py
   # or
   gunicorn app:app
   ```
5. Visit http://localhost:5000

## Endpoints
- GET `/` : Landing page
- GET/POST `/predict` : Predictor form and results page
- POST `/api/predict` : JSON API, expects fields: `height, weight, ap_hi, ap_lo, cholesterol, smoke, alco, active`
- GET `/api/stats` : Returns basic dataset stats used for charts

## UI/UX notes
- Built with Bootstrap 5 for responsive layout and accessibility.
- Minimal custom CSS for clean visuals; Chart.js used for donut/pie charts.
- Focus on clear CTA, readable typography, large form inputs and single-column flow on mobile.

## Deployment (Heroku)
1. Create Heroku app and add remote:
   ```bash
   heroku create my-cardiosense-app
   git push heroku main
   ```
2. Ensure `Procfile` and `requirements.txt` are present (already included).

## Tests
Run the basic unittest suite:
```bash
python -m unittest
```

## Notes
- This project is for educational purposes only and is not medical advice.
