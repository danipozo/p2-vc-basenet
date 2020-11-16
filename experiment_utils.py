from tensorflow import keras
from pathlib import Path

import subprocess
import os
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if os.getenv('PAPERSPACE'):
    install('pandas')

import pandas as pd


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

def save_results(results, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    for i, (m, h) in enumerate(results):
        f_name = 'results-'+str(i)+'.json'
        with open(path / f_name, 'w') as f:
            f.write('[')
            f.write(m.to_json())
            f.write(',')
            hist_df = pd.DataFrame(h.history)
            f.write(hist_df.to_json())
            f.write(']')
