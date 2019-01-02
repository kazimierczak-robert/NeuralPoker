from tensorflow import keras
from SetImporter import SetImporter
import tensorflow as tf

class Model:
    dataset: SetImporter
    model: keras.models.Sequential

    def __init__(self, dataset):
        self.dataset = dataset

    def create(self):
        self.model = keras.models.Sequential()
        # 17 = 13 ranks + 4 suits
        self.model.add(keras.layers.Dense(7, input_dim=17*self.dataset.on_board, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(7, activation=tf.nn.relu))
        output_1 = keras.layers.Dense(10, activation=tf.nn.softmax)
        output_2 = keras.layers.Dense(1, activation=tf.nn.sigmoid)
        output = tf.concat(1, [output_1, output_2])
        self.model.add(output)
        self.model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        self.model.fit(self.dataset.input, self.dataset.output, epochs=150)
