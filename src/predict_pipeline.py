from ..src.utils import load_model
from ..src.data_prep import load_data, split_data

def predict_pipeline(X):
    main_model = load_model('main_status_model.pkl')
    status_preds = main_model.predict(X)
    import pandas as pd
    systems = ['CNS','Cardio','Renal','GI','Skeletal','Skin','Eyes','Nasal','Repro']
    status_df = pd.DataFrame(status_preds, columns=[f'{s}_Status' for s in systems])

    disease_results = {}
    for sys in systems:
        if status_df[f'{sys}_Status'].iloc[0] == 1:
            sub_model = load_model(f'{sys.lower()}_disease_model.pkl')
            disease_preds = sub_model.predict(X)
            disease_results[sys] = disease_preds.tolist()
        else:
            disease_results[sys] = 'System normal'
    return status_df, disease_results
