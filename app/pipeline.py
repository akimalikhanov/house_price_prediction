import numpy as np
import pandas as pd
import joblib

def trasnform_pipe(data, pipe_path='models/test_prep_pipe.bin'):
    pipeline=joblib.load(pipe_path)
    data['In Capital']='yes' if any(x in data['Location'] for x in ['q.', 'm.', 'r.']) else 'no'
    data['Near Subway']='yes' if any(x in data['Location'] for x in ['m.']) else 'no'
    data_tr=pipeline.transform(pd.DataFrame.from_records(data, index=[0]))
    return data_tr
