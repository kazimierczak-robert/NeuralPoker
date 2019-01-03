from tensorflow import keras
from SetImporter import SetImporter
import tensorflow as tf


class Model:
    dataset: SetImporter
    model: keras.models.Sequential

    def __init__(self, dataset=None):
        self.dataset = dataset

    def convert_input_to_one_hot(self, input):
        columns = None

        for ith_column in range(0, self.dataset.all_cards*2):
            column = input[:, ith_column]
            if ith_column % 2 == 0: # suit
                one_hot_column = tf.one_hot(column, 4)
                if ith_column == 0:
                    columns = one_hot_column
                else:
                    columns = tf.concat([columns, one_hot_column], 1)
            else: # rank
                one_hot_column = tf.one_hot(column, 13)
                columns = tf.concat([columns, one_hot_column], 1)

        return columns

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
    def create(self):
        self.model = keras.models.Sequential()
        # 17 = 13 ranks + 4 suits
        input_layer = keras.layers.Input((17*self.dataset.all_cards,))
        second_layer = keras.layers.Dense(7, activation=tf.nn.relu)(input_layer)
        # 1 output in NN
        if self.dataset.all_cards == 2:
            output_layer = keras.layers.Dense(1, activation=tf.nn.sigmoid)(second_layer)
            self.model = keras.models.Model(input_layer, output_layer)
            # metrics for regression
            self.model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse',
                               metrics='mse')
        # 2 outputs in NN
        else:
            output_1 = keras.layers.Dense(10, activation=tf.nn.softmax)(second_layer)
            output_2 = keras.layers.Dense(1, activation=tf.nn.sigmoid)(second_layer)
            self.model = keras.models.Model(input_layer, [output_1, output_2])
            # metrics for classification and regression
            self.model.compile(optimizer=tf.train.AdamOptimizer(), loss={'dense_1': 'sparse_categorical_crossentropy',
                                                                         'dense_2': 'mse'},
                               metrics={'dense_1': 'acc', 'dense_2': 'mse'})

    def train(self):
        if self.dataset.all_cards == 2:
            self.model.fit(self.convert_input_to_one_hot(self.dataset.input), self.dataset.output, epochs=150,
                           steps_per_epoch=self.dataset.number_of_samples)
        else:
            self.model.fit(self.convert_input_to_one_hot(self.dataset.input), [self.dataset.output[:, 0],
                                                                               self.dataset.output[:, 1]], epochs=1,
                           steps_per_epoch=self.dataset.number_of_samples)

    def save(self, output_file_name):
        with open(output_file_name + ".json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(output_file_name + "model.h5")

    def load(self, input_file_name):
        with open(input_file_name + ".json", "r") as json_file:
            self.model = tf.keras.models.model_from_json(json_file.read())
        self.model.load_weights(input_file_name + "model.h5")

    def test(self, test_set: SetImporter):
        if test_set.all_cards == 2:
            eval_result = self.model.evaluate(self.convert_input_to_one_hot(test_set.input), test_set.output,
                                              steps=test_set.number_of_samples)
        else:
            eval_result = self.model.evaluate(self.convert_input_to_one_hot(test_set.input), [test_set.output[:, 0],
                                                                                              test_set.output[:, 1]],
                                              steps=test_set.number_of_samples)
        for idx in range(0, len(self.model.metrics_names)):
            print(self.model.metrics_names[idx] + ":", eval_result[idx])


train_test = SetImporter("Sets\poker-hand-training-5.data", 5)
test_test = SetImporter("Sets\poker-hand-test-5.data", 5)
model = Model(train_test)
model.create()
model.train()
model.test(test_test)