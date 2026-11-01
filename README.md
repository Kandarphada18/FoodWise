# FoodWise — Restaurant Food Waste Prediction

FoodWise is a full-stack restaurant food-waste workbench. It uses React + Vite, FastAPI, SQLite, Pandas, NumPy and scikit-learn. No hosted database is required.

## Features
- Account registration and login with hashed passwords and bearer tokens.
- SQLite-backed restaurant records and prediction history.
- CSV import/export for historical food-waste records.
- Train a Random Forest regression model from labelled records.
- Generate predictions only after a model has been trained on sufficient real records.
- Dashboard summaries, category analytics and actionable data-based recommendations.

## Run locally

### Backend
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
API docs: http://127.0.0.1:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Set `VITE_API_URL=http://127.0.0.1:8000` in `frontend/.env.local` if needed.

## Data and model
Import a CSV containing at least `quantity_prepared`, `quantity_sold`, `customer_count`, and `actual_waste`. Optional columns: `record_date`, `food_category`, `average_daily_demand`, `previous_waste`, `season`, `special_event`, `day_of_week`. Add at least 30 valid labelled records before training. The model uses a chronological holdout split where dates are available and reports MAE, RMSE and R². Metrics are not shown until training succeeds.

## Deploy
- Backend: deploy this repository on Render as a Python web service. Build: `pip install -r requirements.txt`; start: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Frontend: deploy `frontend/` on Vercel with root directory `frontend` and set `VITE_API_URL` to the Render service URL.

## Storage warning
SQLite is appropriate for a simple workbench and local development. Render's default filesystem is ephemeral, so records can be lost on redeploy/restart. For durable hosted records, attach a persistent disk to the Render service and set `SQLITE_PATH` to a file on that disk, or later move to a managed database.
