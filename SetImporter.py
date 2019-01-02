import numpy as np


class SetImporter:
    input: np.ndarray
    output: np.ndarray

    def __init__(self, file_name, on_board):
        self.divide_dataset(np.loadtxt(file_name, delimiter=','), on_board)

    def divide_dataset(self, dataset, on_board):
        self.input = dataset[:, 0:on_board*2]
        self.output = dataset[:, on_board*2:]


# importer = SetImporter("Sets\poker-hand-training-5.data", 5)
