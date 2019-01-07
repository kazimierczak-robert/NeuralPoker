import numpy as np


class SetImporter:
    input: np.ndarray
    number_of_samples: int
    all_cards: int
    output: np.ndarray

    def __init__(self, file_name, all_cards):
        self.all_cards = all_cards
        self.divide_dataset(np.loadtxt(file_name, delimiter=','))
        self.number_of_samples = self.input.shape[0]

    def divide_dataset(self, dataset):
        self.input = dataset[:, 0:self.all_cards*2]
        self.output = dataset[:, self.all_cards*2:]
