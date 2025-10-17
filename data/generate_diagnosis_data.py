import numpy as np, pandas as pd, os
from datetime import datetime, timedelta

np.random.seed(42)
N = 1000
start = datetime.now() - timedelta(days=365)
timestamps = [(start + timedelta(days=np.random.rand()*365)).isoformat() for _ in range(N)]

# Generate 24 features (simplified for brevity)
df = pd.DataFrame({
    'Timestamp': timestamps,
    'EEG_Alpha_Power_uV2': np.random.normal(30, 8, N),
    'EEG_Beta_Power_uV2': np.random.normal(18, 6, N),
    'fNIRS_HbO_uM': np.random.normal(40, 10, N),
    'fNIRS_HbR_uM': np.random.normal(20, 8, N),
    'ECG_HR_bpm': np.random.normal(72, 12, N),
    'ECG_QRS_ms': np.random.normal(100, 10, N),
    'PPG_Amp_norm': np.random.rand(N),
    'PPG_TT_ms': np.random.normal(250, 40, N),
    'EIS_500Hz_Ohm': np.random.normal(500, 120, N),
    'EIS_Phase_deg': np.random.normal(-30, 10, N),
    'NIR_Abs_850nm': np.random.normal(0.6, 0.15, N),
    'Fluid_pH': np.random.normal(7.2, 0.5, N),
    'Fluid_Conductivity_uS_cm': np.random.normal(600, 200, N),
    'VOC_Sensor1': np.random.rand(N),
    'VOC_Sensor2': np.random.rand(N),
    'Thermal_MeanTemp_C': np.random.normal(36.5, 0.6, N),
    'Thermal_DeltaTemp_C': np.random.normal(0.3, 0.4, N),
    'MS_Hb_Index': np.random.normal(14, 3, N),
    'MS_Melanin_Index': np.random.normal(40, 15, N),
    'Fundus_VesselTortuosity': np.random.normal(1.0, 0.2, N),
    'Age': np.random.randint(18, 80, N),
    'Sex': np.random.choice([0, 1], N),
    'BMI': np.random.normal(24, 4, N)
})

# Create 9 binary system statuses
for sys in ['CNS','Cardio','Renal','GI','Skeletal','Skin','Eyes','Nasal','Repro']:
    df[f'{sys}_Status'] = np.random.choice([0,1], N)

# Simple disease columns per system
for sys in ['CNS','Cardio','Renal','GI','Skeletal','Skin','Eyes','Nasal','Repro']:
    for i in range(3):
        df[f'{sys}_Disease{i+1}'] = (df[f'{sys}_Status'] * np.random.choice([0,1], N, p=[0.8,0.2]))

os.makedirs('data', exist_ok=True)
df.to_csv('data/diagnosis_data.csv', index=False)
print('diagnosis_data.csv generated with shape:', df.shape)
