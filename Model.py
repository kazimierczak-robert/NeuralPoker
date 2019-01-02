from tensorflow import keras
from SetImporter import SetImporter
import tensorflow as tf

class Model:
    dataset: SetImporter
    model: keras.models.Sequential

    def __init__(self, dataset):
        self.dataset = dataset

    def convert_input_to_one_hot(self):
        columns = None

        with tf.Session() as sess:
            for ith_column in range(0, self.dataset.all_cards*2):
                column = self.dataset.input[:, ith_column]
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


    def create(self):
        self.model = keras.models.Sequential()
        # 17 = 13 ranks + 4 suits
        input_layer = keras.layers.Input((17*self.dataset.all_cards,))
        second_layer = keras.layers.Dense(7, activation=tf.nn.relu)(input_layer)
        if self.dataset.all_cards == 2:
            output_layer = keras.layers.Dense(1, activation=tf.nn.sigmoid)(second_layer)
            self.model = keras.models.Model(input_layer, output_layer)
        else:
            output_1 = keras.layers.Dense(10, activation=tf.nn.softmax)(second_layer)
            output_2 = keras.layers.Dense(1, activation=tf.nn.sigmoid)(second_layer)
            self.model = keras.models.Model(input_layer, [output_1, output_2])

        self.model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        if self.dataset.all_cards == 2:
            self.model.fit(self.convert_input_to_one_hot(), self.dataset.output, epochs=150, steps_per_epoch=self.dataset.number_of_samples)
        else:
            self.model.fit(self.convert_input_to_one_hot(), [self.dataset.output[:,0], self.dataset.output[:,1]], epochs=1, steps_per_epoch=self.dataset.number_of_samples)


importer = SetImporter("Sets\poker-hand-training-5.data", 5)
model = Model(importer)
model.create()
model.train()