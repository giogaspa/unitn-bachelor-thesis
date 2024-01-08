import numpy as np
import tensorflow as tf
import keras

CLASS_TRANSFORMER = {
    "center": 0,
    "wrong-center-forward": 1,
    "wrong-center-backward": 2,
    "right": 3,
    "wrong-right-forward": 4,
    "wrong-right-backward": 5,
    "left": 6,
    "wrong-left-forward": 7,
    "wrong-left-backward": 8,
}

def train(data):
    print("Train model")

    positions_number = 9 # calcolare in base alle classi del tensore

    training_data = data[:,:14]
    training_data = training_data.astype(np.float32)
    training_data = tf.constant(training_data)

    training_labels = data[:,14]
    new_labels = []
    for l in training_labels:
        new_labels.append(CLASS_TRANSFORMER[l])

    training_labels = np.array(new_labels).reshape([len(training_labels), 1])
    training_labels = tf.constant(training_labels)


    model = keras.Sequential([
        keras.layers.Dense(72, input_dim=14, activation='relu'),
        keras.layers.Dense(36, activation='relu'),
        keras.layers.Dense(18, activation='relu'),
        keras.layers.Dense(positions_number, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(training_data, training_labels, epochs=1000, verbose=1)

    return model