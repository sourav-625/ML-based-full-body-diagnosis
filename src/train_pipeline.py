from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from ..src.data_prep import load_data, split_data
from ..src.utils import save_model
import os

def train_all():
    df = load_data()
    X, Y_status, Y_diseases = split_data(df)
    main_model = MultiOutputClassifier(RandomForestClassifier())
    main_model.fit(X, Y_status)
    save_model(main_model, 'main_status_model.pkl')

    for sys, y in Y_diseases.items():
        if y.empty: continue
        model = MultiOutputClassifier(RandomForestClassifier())
        model.fit(X, y)
        save_model(model, f'{sys.lower()}_disease_model.pkl')

if __name__ == '__main__':
    train_all()
    print('All models trained and saved in models/.')
