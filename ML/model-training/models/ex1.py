import os
from typing import List
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
import keras
import warnings
import csv
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import InputLayer, Dense, Dropout

from sklearn.datasets import make_classification 
from sklearn.linear_model import LogisticRegression

from common.params import DATASET_PATH, exercises
from common.utils import split_dataset

exerciseId = 'ex1'
exercise_path =  os.path.join(DATASET_PATH, exerciseId, 'evaluations')

class_transformer = exercises[exerciseId]["transformer"]
class_number = len(class_transformer)

def train(configuration):
    ## Load data
    dataset_filepath = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}_processed.csv')
    raw_data = np.genfromtxt(dataset_filepath, delimiter=",", dtype=None, encoding='UTF8')
    raw_data = raw_data[1:,:]

    input_dimension = len(raw_data[0]) - 1 # Remove label column

    train_data, other_data = split_dataset(raw_data, test_size=0.4)
    cv_data, test_data = split_dataset(other_data, test_size=0.5)

    #print_dataset_stats('original', raw_data)
    #print_dataset_stats('train', train_data)
    #print_dataset_stats('cross-validation', cv_data)
    #print_dataset_stats('test-data', test_data)

    training_data = train_data[:,:-1]
    training_data = training_data.astype(np.float32)
    training_data = tf.constant(training_data)

    training_labels = train_data[:,-1]
    new_labels = []
    for l in training_labels:
        new_labels.append(class_transformer[l])

    training_labels = np.array(new_labels).reshape([len(training_labels), 1])
    training_labels = tf.constant(training_labels)

    cv_values = tf.constant(cv_data[:,:-1].astype(np.float32))
    cv_labels = []
    for l in cv_data[:,-1]:
        cv_labels.append(class_transformer[l])

    cv_labels = np.array(cv_labels).reshape([-1, 1])
    cv_labels = tf.constant(cv_labels)

    test_values = tf.constant(test_data[:,:-1].astype(np.float32))
    test_labels = []
    for l in test_data[:,-1]:
        test_labels.append(class_transformer[l])

    test_labels = np.array(test_labels).reshape([-1, 1])
    test_labels = tf.constant(test_labels) 

    """     
        ####################################
        #           LINEAR MODEL           #
        ####################################
        log_score = multinomial_logistic_regression_score(train_data, cv_data, test_data)
        if(log_score == 0):
            print('NON CONVERGE! Le classi non sono separabili linearmente -> crea NN!')
        else:
            print(f"Il modello con regression lineare converge con accuratezza {log_score:0.1%}")
    """

    #############################
    #           MODEL           #
    #############################
    model = Sequential(name=configuration["name"])
    model.add(InputLayer(input_shape=(input_dimension,), name='Input'))

    for layer, units in enumerate(configuration["configuration"]):
        model.add( Dense(units=units, activation='relu', name=f"Hidden-{layer+1}-{units}"))
        model.add( Dropout(0.5))

    model.add(Dense(class_number, activation='softmax', name='Output')) # OUT LAYER

    model.compile(
        # OPTIMIZER
        optimizer='adam',
        # LOSS FUNCTION TO MINIMIZE
        loss='categorical_crossentropy',
        # METRICS TO MONITOR
        metrics=['accuracy']
    )

    oh_trainint_labels = keras.utils.to_categorical(training_labels)
    oh_cv_labels = keras.utils.to_categorical(cv_labels)
    oh_test_labels = keras.utils.to_categorical(test_labels)

    history = model.fit(training_data, oh_trainint_labels, epochs=300, verbose=0, validation_data=(cv_values, oh_cv_labels))
    
    evaluate_model(model,(cv_values, oh_cv_labels), (test_values, oh_test_labels), history.history, configuration)

    return model

def multinomial_logistic_regression_score(train_data, cv_data, test_data):

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, verbose=1, penalty='l2', C=1.0)
    
    train_data_list = train_data[:,:-1].astype(np.float32)
    train_labels = transform_labels_to_numbers(train_data[:,-1])

    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        try:
            model.fit(train_data_list, train_labels)

            train_score = model.score(train_data_list, train_labels)

            cv_data_list = cv_data[:,:-1].astype(np.float32)
            cv_labels = transform_labels_to_numbers(cv_data[:,-1])
            cv_score = model.score(cv_data_list, cv_labels)

            test_data_list = test_data[:,:-1].astype(np.float32)
            test_labels = transform_labels_to_numbers(test_data[:,-1])
            test_score = model.score(test_data_list, test_labels)

            print(f"N. iterations {model.n_iter_[0]}")
            print(f"Train score {train_score:.1%}")
            print(f"CV score {cv_score:.1%}%")
            print(f"Test score {test_score:.1%}%")


            return cv_score
        except ConvergenceWarning:
            # do something in response
            print('NON CONVERGE!')
            return 0

def transform_labels_to_numbers(labels):
    return np.array(list(map(label_to_number, labels))).astype(str)

def transform_numbers_to_labels(numbers):
    return np.array(list(map(number_to_label, numbers))).astype(int)

def label_to_number(l):
    return class_transformer[l]

def number_to_label(n):
    mapper = list(class_transformer.keys())
    return mapper[n]

def test_logistic_regression():
    # generate a random binary classification dataset 
    X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, n_redundant=0, n_classes=10, random_state=1) 

    print(X[:,:-1])
    
    # create a logistic regression classifier 
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', verbose=1, penalty='l2', C=1.0) 
    
    # fit the classifier to the data 
    clf.fit(X, y) 
    
    # calculate the accuracy of the classifier 
    accuracy = clf.score(X, y) 
    
    # print the accuracy 
    print("Accuracy of logistic regression classifier:", accuracy) 

def make_layers_combinations(layers: list[int], units:list[int]) -> dict:
    combinations = {}
    for l_num in layers:
        input = np.repeat([units], l_num, axis=0)
        layer_combinations = np.array(np.meshgrid(*input)).T.reshape(-1, l_num)
        combinations[l_num] = layer_combinations

    return combinations

def make_nn_models(input_dim) -> list[keras.Model]:
    layers_config = [2]
    units_config = [128, 64]

    combinations = make_layers_combinations(layers_config, units_config)

    models = []

    for layer in combinations:
        for c in combinations[layer]:
            model = make_nn_model(c, input_dim)
            models.append({"model": model, "layers": len(c), "units": c})

    return models

def save_evaluations(evaluations):
    keys = evaluations[0].keys()

    filepath = os.path.join(DATASET_PATH, exerciseId, 'evaluations', 'evaluations.csv')

    with open(filepath, 'a+', encoding='UTF8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        #dict_writer.writeheader()
        dict_writer.writerows(evaluations)

def make_nn_model(units: list[int], input_dim) -> keras.Model: 
    layers_number = len(units)
    units_text = "_".join(map(str, units))
    model_name = f"L{layers_number}_U{units_text}"
    model = keras.Sequential(name=model_name)
    model.add(keras.layers.InputLayer(input_shape=(input_dim,), name='Input'))

    for l, u in enumerate(units):
        model.add( keras.layers.Dense(units=u, activation='relu', name=f"Hidden-{l+1}-{u}"))
        model.add( keras.layers.Dropout(0.5))
                        
    model.add(keras.layers.Dense(class_number, activation='linear', name='Output')) # OUT LAYER

    return model

def evaluate_model(model: keras.Model, cv, test, history, configuration):
        fig, (acc_plot, loss_plot) = plt.subplots(2, sharex=True)
        fig.supxlabel('epoch')
        
        evaluate_cv = model.evaluate(cv[0], cv[1])

        #evaluate_test = model.evaluate(test[0], test[1])
        #print(evaluate_test)

        linestyle = '-'
        acc_plot.plot(history['accuracy'], label=configuration['name'], linestyle=linestyle)
        acc_plot.plot(history['val_accuracy'], label=f"CV_{configuration['name']}", linestyle=linestyle)
        
        loss_plot.plot(history['loss'], label=configuration['name'], linestyle=linestyle)
        loss_plot.plot(history['val_loss'], label=f"CV_{configuration['name']}", linestyle=linestyle)

        fig.suptitle(configuration['name'])

        acc_plot.set(ylabel='Accuracy')
        acc_plot.legend(loc='lower right')
        acc_plot.text(200,0.7,f"CV: {evaluate_cv[1]:.1%}")
        #acc_plot.text(200,0.6,f"Test: {evaluate_test[1]:.1%}")

        loss_plot.set(ylabel='Loss')
        loss_plot.legend(loc='upper right')
        loss_plot.text(200,1,f"CV: {evaluate_cv[0]:.2}")
        #loss_plot.text(200,0.8,f"Test: {evaluate_test[0]:.2}")

        filepath = os.path.join(DATASET_PATH, exerciseId, 'evaluations', f"{configuration['name']}.png")
        fig.savefig(filepath)
        #acc_plot.clear()
        #loss_plot.clear()

        m = {
            "name": configuration["name"],
            "layers": configuration["layers"],
            "units": configuration["units"],
            "cv_accuracy": evaluate_cv[1],
            #"cv_loss": evaluate_cv[0], 
            #"test_loss": evaluate_test[0], 
            #"test_accuracy": evaluate_test[1],
        }

        #print(f"Test: {evaluate_test[1]:.1%}")

        save_evaluations([m])

def print_dataset_stats(name, dataset):
    cli_separator = f"{'-'*37}"
    stats = np.unique(dataset[:,-1], return_counts=True)

    print(cli_separator)
    print(f"Total {name} data: {dataset.shape[0]}")

    for idx, label in enumerate(stats[0]):
        print(f"{label}: {stats[1][idx]}")
    print(cli_separator)