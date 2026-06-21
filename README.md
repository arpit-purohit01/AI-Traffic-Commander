# Gridlock AI 🚦
**Traffic Incident & Resource Optimization Engine**

Developed by **Arpit Purohit**  
GitHub: [github.com/arpit-purohit01](https://github.com/arpit-purohit01)

---

## 📌 Project Overview
Gridlock AI is an enterprise-grade Traffic Management system designed to predict traffic incident clearance delays and autonomously deploy mitigation resources. The system leverages a **Weighted Ensemble Machine Learning Pipeline** for extremely accurate delay forecasting and a **Multi-Objective Linear Programming (PuLP)** engine to balance risk, operational cost, and response times.

### 🎯 Key Outcomes & Capabilities
- **Unprecedented Prediction Accuracy:** By combining CatBoost (40%), XGBoost (30%), ExtraTrees (20%), and Linear Regression (10%), the ensemble pipeline radically reduces Mean Absolute Error (MAE) compared to baseline models.
- **Cost-Aware Autonomous Deployment:** The PuLP solver mathematically decides the exact number of Ambulances, Tow Trucks, Officers, Cranes, and ERT units to deploy. It guarantees constraint satisfaction (e.g., minimum 1 Ambulance and 1 ERT for accidents) while minimizing a weighted objective of `1.0 * Risk + 0.001 * Cost`.
- **RAG & Explainable AI (XAI):** The backend incorporates **Conformal Prediction** for strict 95% confidence bounds, **SHAP values** for feature contribution transparency, and a **K-Nearest Neighbors (KNN)** retrieval engine to synthesize historical RAG advisories.
- **Enterprise MLOps & Security:** The FastAPI backend is guarded by `slowapi` rate limiting (anti-DDoS), features Population Stability Index (PSI) drift monitoring to detect data degradation, and safely handles SQLite/PostgreSQL fallbacks.
- **Live Command Center UI:** A stunning dark-mode React dashboard displaying a real-time `react-leaflet` interactive map, counterfactual insights ("What if we don't close the road?"), and a dynamic 24-hour simulation sweep.

---

## ⚙️ Technology Stack

### Backend (Python/FastAPI)
* **Framework:** FastAPI
* **Machine Learning:** CatBoost, XGBoost, Scikit-Learn (ExtraTrees, KNN, Linear Regression)
* **Optimization Engine:** PuLP (Coin-OR Branch and Cut solver)
* **Explainability & Security:** SHAP, Conformal Prediction Math, SlowAPI (Rate Limiting)
* **Database:** SQLAlchemy (PostgreSQL with automatic fallback to SQLite `events.db`)

### Frontend (React/Vite)
* **Framework:** React + Vite
* **Data Visualization:** Recharts (Dynamic LineCharts, PieChart Gauges, BarCharts)
* **Interactive Mapping:** React-Leaflet, Leaflet.js
* **Styling:** Custom CSS with Glassmorphism and CSS Grid

---

## 📊 Dataset Requirements
The model is trained on historical, anonymized traffic incident logs.
* **Link (If Publicly Hosted):** Provide your dataset link here or point to `Astram event data_anonymized.csv`
* **Core Features:** `event_cause`, `priority`, `requires_road_closure`, `zone`, `hour_of_day`, `day_of_week`.
* **Preprocessing:** The `train_model.py` script automatically imputes missing values, strips extreme outliers (>480 mins), engineers temporal features (weekends, quarters), and scales the data utilizing a serialized `ColumnTransformer`.

---

## 🚀 How to Run the Project Locally

To launch the entire Gridlock AI system on your machine, you need to start both the Python backend and the React frontend simultaneously.

### 1. Start the Backend (FastAPI)
The backend serves the ML models, runs the PuLP optimizer, and handles the SQLite database logging.

```bash
cd backend
# Create and activate your virtual environment (if not already done)
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows

# Install Dependencies
pip install requirements.txt

# Train the Models (Generates artifacts in /backend/models)
python train_model.py

# Launch the FastAPI Server (Runs on port 8000)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*API Docs available at: `http://localhost:8000/docs`*

### 2. Start the Frontend (React + Vite)
The frontend is the visual command dashboard.

```bash
cd frontend

# Install Dependencies
npm install
npm install recharts react-leaflet leaflet

# Launch the Vite Development Server (Runs on port 5173)
npm run dev
```

### 3. View the Dashboard
Open your web browser and navigate to:
**`http://localhost:5173`**

You can now use the control panel on the left to simulate a traffic incident, view the RAG advisories, check the counterfactual statistics, and watch the PuLP optimizer deploy resources in real-time!

---

*This project is built and maintained by Arpit Purohit.*
