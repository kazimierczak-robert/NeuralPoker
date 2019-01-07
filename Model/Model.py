import os
from tensorflow import keras
from Sets.SetImporter import SetImporter
from Sets.SetGenerator import SetGenerator
import tensorflow as tf
import numpy as np


class Model:
    all_cards: int
    dataset: SetImporter
    model: keras.models.Sequential
    dir_path = "./Model/GeneratedModels"

    def __init__(self, dataset=None):
        self.dataset = dataset
        self.model = None
        self.all_cards = self.dataset.all_cards if dataset is not None else 0

    def convert_input_to_one_hot(self, input, all_cards):
        columns = None

        for ith_column in range(0, all_cards*2):
            column = input[:, ith_column]
            if ith_column % 2 == 0:  # suit
                one_hot_column = tf.one_hot(column, 4)
                if ith_column == 0:
                    columns = one_hot_column
                else:
                    columns = tf.concat([columns, one_hot_column], 1)
            else:  # rank
                one_hot_column = tf.one_hot(column, 13)
                columns = tf.concat([columns, one_hot_column], 1)

        return columns

    def create(self, neurons_in_hidden_layer):
        if self.dataset is not None:
            self.model = keras.models.Sequential()
            # 17 = 13 ranks + 4 suits
            input_layer = keras.layers.Input((17*self.dataset.all_cards,), name="input_layer")
            hidden_layer = keras.layers.Dense(neurons_in_hidden_layer, activation=tf.nn.relu,
                                              name="hidden_layer")(input_layer)
            # 1 output in NN
            if self.dataset.all_cards == 2:
                output_layer = keras.layers.Dense(1, activation=tf.nn.sigmoid, name="output_layer")(hidden_layer)
                self.model = keras.models.Model(input_layer, output_layer)
            # 2 outputs in NN
            else:
                output_1 = keras.layers.Dense(10, activation=tf.nn.softmax, name="output_layer_1")(hidden_layer)
                output_2 = keras.layers.Dense(1, activation=tf.nn.sigmoid, name="output_layer_2")(hidden_layer)
                self.model = keras.models.Model(input_layer, [output_1, output_2])
            self.compile()

    def train(self):
        if all([self.dataset is not None, self.model is not None]):
            # verbose = 0 - don't print
            if self.dataset.all_cards == 2:
                self.model.fit(self.convert_input_to_one_hot(self.dataset.input, self.dataset.all_cards),
                               self.dataset.output, epochs=1, steps_per_epoch=self.dataset.number_of_samples, verbose=0)
            else:
                self.model.fit(self.convert_input_to_one_hot(self.dataset.input, self.dataset.all_cards),
                               [self.dataset.output[:, 0], self.dataset.output[:, 1]], epochs=1,
                               steps_per_epoch=self.dataset.number_of_samples, verbose=0)

    def save(self, output_file_name):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        with open(output_file_name + ".json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(output_file_name + ".h5")

    def load(self, input_file_name, all_cards_no):
        with open(input_file_name + ".json", "r") as json_file:
            self.model = tf.keras.models.model_from_json(json_file.read())
        self.model.load_weights(input_file_name + ".h5")
        self.all_cards = all_cards_no
        self.compile()

    # https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
    # Keras Regression metrics:
    # - Mean Squared Error
    # - Mean Absolute Error
    # - Mean Absolute Percentage Error
    # - Cosine Proximity
    # Keras Classification Metrics
    # - Binary Accuracy
    # - Categorical Accuracy
    # - Sparse Categorical Accuracy
    # - Top k Categorical Accuracy
    # - Sparse Top k Categorical Accuracy
    def compile(self):
        if self.all_cards == 2:
            # metrics for regression
            self.model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse',
                               metrics=['mse'])
        else:
            # metrics for classification and regression
            self.model.compile(optimizer=tf.train.AdamOptimizer(),
                               loss={'output_layer_1': 'sparse_categorical_crossentropy', 'output_layer_2': 'mse'},
                               metrics={'output_layer_1': 'acc', 'output_layer_2': 'mse'})

    def test(self, test_set: SetImporter):
        if test_set.all_cards == 2:
            eval_result = self.model.evaluate(self.convert_input_to_one_hot(test_set.input, test_set.all_cards),
                                              test_set.output, steps=test_set.number_of_samples, verbose=0)
        else:
            eval_result = self.model.evaluate(self.convert_input_to_one_hot(test_set.input, test_set.all_cards),
                                              [test_set.output[:, 0], test_set.output[:, 1]],
                                              steps=test_set.number_of_samples, verbose=1)
        for idx in range(0, len(self.model.metrics_names)):
            print(self.model.metrics_names[idx] + ":", eval_result[idx])

    def predict(self, input: list):
        b = SetGenerator.convert_char_input_to_int_list(input)
        a = self.convert_input_to_one_hot(np.asarray([b]), len(input))
        results = self.model.predict(a, steps=1)
        if len(input) == 2:
            return results[0][0]
        elif all([len(input) >= 5, len(input) <= 7]):
            return np.argmax(results[0][0]), results[1][0][0]
        return None
