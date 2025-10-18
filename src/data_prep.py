import pandas as pd
def load_data(path='../data/diagnosis_data.csv'): return pd.read_csv(path)
def split_data(df):
    X = df.iloc[:,1:25]
    Y_status = df[[c for c in df.columns if c.endswith('_Status')]]
    Y_diseases = {sys: df[[c for c in df.columns if c.startswith(sys+'_Disease')]]
                  for sys in ['CNS','Cardio','Renal','GI','Skeletal','Skin','Eyes','Nasal','Repro']}
    return X, Y_status, Y_diseases
