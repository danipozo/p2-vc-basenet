from tensorflow import keras
from pathlib import Path

def save_models(models, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(models):
        file_name = 'model-'+str(i)+'.json'
        with open(path / file_name, 'w') as f:
            f.write(model.to_json())

def load_models(path):
    models_json = Path(path).glob('*.json')
    models_json = [ open(str(m), 'r').read() for m in models_json ]

    return [ keras.models.model_from_json(m) for m in models_json ]
