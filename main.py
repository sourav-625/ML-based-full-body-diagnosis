from src.train_pipeline import train_all
from src.predict_pipeline import predict_pipeline
from src.data_prep import load_data, split_data

if __name__ == '__main__':
    train_all()
    df = load_data()
    X, _, _ = split_data(df)
    sample = X.sample(1, random_state=42)
    status, diseases = predict_pipeline(sample)
    print('Sample Input Prediction:')
    print(status)
    print('Disease Details:', diseases)
