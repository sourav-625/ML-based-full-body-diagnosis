# full_body_diagnosis_ml

This project implements a **two-stage diagnostic ML pipeline** using synthetic data.

### Stage 1: System Status Prediction
Predicts 9 major body system statuses (CNS, Cardio, etc.) using RandomForestClassifier (multi-output).

### Stage 2: Disease Prediction
For each system with abnormal status (=1), a specific submodel predicts diseases for that system.

---

## Run Instructions

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data:
   ```bash
   python data/generate_diagnosis_data.py
   ```

3. Train models:
   ```bash
   python src/train_pipeline.py
   ```

4. Run full pipeline (train + predict demo):
   ```bash
   python main.py
   ```

---

Models are saved in `models/`. The project uses 24 input bio-signals and 9 system labels.
