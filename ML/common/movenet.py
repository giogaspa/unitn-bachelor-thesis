import tensorflow_hub as hub

## LOAD MOVENET MODEL
_model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4")
movenet = _model.signatures['serving_default']