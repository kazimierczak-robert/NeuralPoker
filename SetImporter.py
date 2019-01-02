import numpy as np


class SetImporter:
    input: np.ndarray
    number_of_samples: int
    on_board: int
    output: np.ndarray

    def __init__(self, file_name, on_board):
        self.on_board = on_board
        self.divide_dataset(np.loadtxt(file_name, delimiter=','))
        self.number_of_samples = self.input.shape[0]

    def divide_dataset(self, dataset):
        self.input = dataset[:, 0:self.on_board*2]
        self.output = dataset[:, self.on_board*2:]


# importer = SetImporter("Sets\poker-hand-training-5.data", 5)
