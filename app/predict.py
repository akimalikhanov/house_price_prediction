import numpy as np
import pandas as pd
import joblib
from pipeline import trasnform_pipe

def price_predict(data, model_path='models/xgb_model.sav', pipe_path='models/test_prep_pipe.bin'):
    data=trasnform_pipe(data, pipe_path)
    model=joblib.load(model_path)
    y_pred=model.predict(data)
    return y_pred[0]
