import os
import numpy as np
import tensorflow as tf
import keras

from common.params import DATASET_PATH, exercises
from common.utils import split_dataset

exerciseId = 'ex1'

def train():
    print("Train model")

    ## Load data
    dataset_filepath = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}_processed.csv')
    raw_data = np.genfromtxt(dataset_filepath, delimiter=",", dtype=None, encoding='UTF8')
    raw_data = raw_data[1:,:]

    class_transformer = exercises[exerciseId]["transformer"]
    class_number = len(class_transformer)
    input_dimension = len(raw_data[0]) - 1 # Remove label column

    train_data, other_data = split_dataset(raw_data, test_size=0.4)
    cv_data, test_data = split_dataset(other_data, test_size=0.5)

    training_data = train_data[:,:-1]
    training_data = training_data.astype(np.float32)
    training_data = tf.constant(training_data)

    training_labels = train_data[:,-1]
    new_labels = []
    for l in training_labels:
        new_labels.append(class_transformer[l])

    training_labels = np.array(new_labels).reshape([len(training_labels), 1])
    training_labels = tf.constant(training_labels)

    #############################
    #           MODEL           #
    #############################
    model = keras.Sequential([
        keras.layers.Dense(72, input_dim=input_dimension, activation='relu'),
        keras.layers.Dense(36, activation='relu'),
        keras.layers.Dense(class_number, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(training_data, training_labels, epochs=300, verbose=1)

    ######################################
    #           MODEL EVALUATE           #
    ######################################
    cv_values = cv_data[:,:-1].astype(np.float32)
    cv_values = tf.constant(cv_values)

    cv_labels = []
    for l in cv_data[:,-1]:
        cv_labels.append(class_transformer[l])

    cv_labels = np.array(cv_labels).reshape([-1, 1])
    cv_labels = tf.constant(cv_labels)

    print("[red]Evaluate model on CV dataset[/red]")
    model.evaluate(cv_values, cv_labels)

    test_values = test_data[:,:-1].astype(np.float32)
    test_values = tf.constant(test_values)

    test_labels = []
    for l in test_data[:,-1]:
        test_labels.append(class_transformer[l])

    test_labels = np.array(test_labels).reshape([-1, 1])
    test_labels = tf.constant(test_labels)

    print("[red]Evaluate model on test dataset[/red]")
    model.evaluate(test_values, test_labels)

    """
    piegamenti-laterali-del-capo_1705671044_w_angle:    128  36 -   500E
    piegamenti-laterali-del-capo_1705671044_w_angle_2:  72   36 -   300E

    print("[red]Predict 1[/red]")
    print(test_data[0][-1])
    prepared_data = tf.constant(test_data[0,:-1].astype(np.float32).reshape([1, 16]))
    predicted = model.predict(prepared_data)
    predicted = tf.nn.softmax(predicted).numpy()
    print(np.argmax(predicted))
    """

    return model