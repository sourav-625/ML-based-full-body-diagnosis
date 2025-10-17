import joblib, os
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
def save_model(model, name): joblib.dump(model, os.path.join(MODELS_DIR, name))
def load_model(name): return joblib.load(os.path.join(MODELS_DIR, name))
