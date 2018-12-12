import tensorflow as tf


class OneHot:
    one_hot_hands: list
    one_hot_suits: list
    one_hot_ranks: list

    def __init__(self):
        with tf.Session() as sess:
            self.one_hot_suits = self.get_one_hot_vector(4).eval()
            self.one_hot_ranks = self.get_one_hot_vector(13).eval()
            self.one_hot_hands = self.get_one_hot_vector(10).eval()

    @staticmethod
    def get_one_hot_vector(set_stop):
        return tf.one_hot(list(range(0, set_stop)), set_stop)

    def get_one_hot_probes(self, file_path):
        with open(file_path) as file:
            one_hot_probes = []
            for probe in file:
                splitted_probe = probe.split(',')
                if len(splitted_probe) == 5:
                    one_hot_probes.append(self.prepare_one_hot_probe_5(splitted_probe))
                elif any([len(splitted_probe) == 12, len(splitted_probe) == 14, len(splitted_probe) == 16]):
                    one_hot_probes.append(self.prepare_one_hot_probe_12_16(splitted_probe))
        return one_hot_probes

    def prepare_one_hot_probe_5(self, splitted_probe):
        one_hot_probe = []
        suit_rank_hot = ""
        for idx in range(0, len(splitted_probe) - 1):
            if idx % 2 == 0:
                suit_rank_hot += self.one_hot_suits[splitted_probe[idx] - 1].tolist()
            else:
                suit_rank_hot += self.one_hot_ranks[splitted_probe[idx] - 1].tolist()
        one_hot_probe.append(suit_rank_hot)
        one_hot_probe.append(splitted_probe[len(splitted_probe) - 1])
        return one_hot_probe

    def prepare_one_hot_probe_12_16(self, splitted_probe):
        one_hot_probe = []
        suit_rank_hot = []
        for idx in range(0, len(splitted_probe) - 2):
            if idx % 2 == 0:
                suit_rank_hot += self.one_hot_suits[int(splitted_probe[idx]) - 1].tolist()
            else:
                suit_rank_hot += self.one_hot_ranks[int(splitted_probe[idx]) - 1].tolist()
        one_hot_probe.append(suit_rank_hot)
        one_hot_probe.append(splitted_probe[len(splitted_probe) - 2])
        one_hot_probe.append(splitted_probe[len(splitted_probe) - 1])
        return one_hot_probe

one_hot = OneHot()
one_hot.get_one_hot_probes("./Sets/poker-hand-training-7.data")