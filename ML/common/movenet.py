import os
import pathlib
import tensorflow_hub as hub

_dir = pathlib.Path(__file__).parent.resolve()
model_path = os.path.join(_dir, 'movenet', 'singlepose-thunder')

## LOAD MOVENET MODEL
# _model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4")
_model = hub.load(model_path)
movenet = _model.signatures['serving_default']