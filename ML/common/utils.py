import numpy as np
import tensorflow as tf

from .params import *
from .movenet import *

def get_cordinates_from(normalized_coord):
    return [int(normalized_coord[1] * image_width), int(normalized_coord[0] * image_height) - movenet_image_landscape_height_offset, normalized_coord[2]]
    
def get_keypoint_axes_from_prediction(prediction, kp='nose'):
    coords = prediction[KEY_POINTS.get(kp)]
    return get_cordinates_from(coords[:3])

def compute_key_points(image):
    # Ridimensiona il frame e aggiungi un padding se necessario. Il ridimensionamento trasforma i dati in float32
    tImage = tf.image.resize_with_pad(image, movenet_image_height, movenet_image_width)

    # Converti image data type da float32 -> int32 
    tImage = tf.cast(tImage, dtype=tf.int32)

    # Aggiungi una dimensione (in testa) in modo da essere compatibile con la shape dei dati richiesta dal modello
    tImage = tf.expand_dims(tImage, axis=0)

    # Movenet si aspetta in input: A frame of video or an image, 
    # represented as an int32 tensor of shape: 192x192x3. 
    # Channels order: RGB with values in [0, 255].
    kp = movenet(tImage)
    kp_coords = np.squeeze(kp['output_0'].numpy())

    norm = {}

    for idx in range(len(kp_coords)):
        coord = get_cordinates_from(kp_coords[idx])
        norm[KEY_POINTS_NAMES[idx]] = [(coord[:2]), coord[2]]

    raw = {}
    for idx in range(len(kp_coords)):
        coord = kp_coords[idx]
        raw[KEY_POINTS_NAMES[idx]] = [(coord[:2]), coord[2]]

    return (norm, raw)