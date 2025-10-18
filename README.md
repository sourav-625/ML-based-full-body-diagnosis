# ML-based Full body Diagnosis

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
   python main.py
   ```

4. Run full pipeline (train + predict demo):
   ```bash
   python main.py
   ```

---

The project uses 24 input bio-signals and 9 body-part based labels along with more than 30 disease based labels. All models are being trained on the same training data while focusing on different features.

Addition of a new model is simple - add the required feature columns to the training data and add the model to the second level of ML-pipeline. The model architecture of all models is the same in the secod stage of the ML-pipeline

Models are being tested and fine-tuned.
