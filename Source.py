from itertools import combinations
from Libraries.deuces.deuces import Card, Deck, Evaluator
import math
from Libraries.MonteCarlo.Monte_Carlo_Poker_Simulation import MonteCarlo
import random
import tensorflow as tf


class OneHot:
    one_hot_colours: list
    one_hot_figures: list
    one_hot_hand: list

    def __init__(self):
        self.one_hot_colours = self.get_one_hot_vector(4)
        self.one_hot_figures = self.get_one_hot_vector(13)
        self.one_hot_hand = self.get_one_hot_vector(10)

    @staticmethod
    def get_one_hot_vector(set_stop):
        return tf.one_hot(list(range(1, set_stop + 1)), set_stop)


class TrainingSetPreparer:
    evaluator = Evaluator()
    generated_probes = []

    # Prepare files for 4 cases with card_parameter: 2, 5, 6, 7
    # 2 card in hand 0 visible on table (2)
    # 2 card in hand 3 visible on table (5)
    # 2 card in hand 4 visible on table (6)
    # 2 card in hand 5 visible on table (7)
    def generate_training_set(self, on_board_parameter):
        if all([on_board_parameter >= 3, on_board_parameter <= 5]):
            # training set
            with open("poker-hand-training-"+str(on_board_parameter+2)+".data", "w") as training_set:
                for i in range(0, 4):  # 32768
                    training_set.write(self.generate_probe(on_board_parameter)+"\n")
                    training_set.flush()
            # test set
            with open("poker-hand-test-"+str(on_board_parameter+2)+".data", "w") as test_set:
                for i in range(0, 2):  # 4096
                    test_set.write(self.generate_probe(on_board_parameter)+"\n")
                    test_set.flush()

        elif on_board_parameter == 0:
            deck = Deck.GetFullDeck()
            for i in range(0, len(deck)):
                deck[i] = Card.int_to_str(deck[i])  # fiqure+colour, i.e. Ad - ace diamond

            pair_combinations = list(combinations(deck, 2))  # list of tuples
            test_probes = random.sample(pair_combinations, len(pair_combinations) // 10)  # 10% for testing
            training_probes = (list(set(pair_combinations) - set(test_probes)))
            # print(training_probes[0][0], training_probes[0][1])
            mc = MonteCarlo(times=10000)  # 'times' says how many times it should play
            with open("poker-hand-test-" + str(on_board_parameter + 2) + ".data", "w") as test_set:
                print("Test set")
                for idx in range(len(test_probes)):
                    # total_villains - how many oponents
                    print(idx)
                    equity = mc.preflop(card1=test_probes[idx][0], card2=test_probes[idx][1], total_villains=4)
                    string = ','.join(map(str, self.get_ints_from_card_symbols(test_probes[idx][0]) + self.get_ints_from_card_symbols(test_probes[idx][1]) + [equity]))
                    test_set.write(string + "\n")
                    test_set.flush()

            with open("poker-hand-training-" + str(on_board_parameter + 2) + ".data", "w") as training_set:
                print("Test set")
                for idx in range(len(training_probes)):
                    # total_villains - how many oponents
                    print(idx)
                    equity = mc.preflop(card1=training_probes[idx][0], card2=training_probes[idx][1], total_villains=4)
                    training_set.write(','.join(map(str, self.get_ints_from_card_symbols(training_probes[idx][0]) + self.get_ints_from_card_symbols(training_probes[idx][1]) + [equity])) + "\n")
                    training_set.flush()

    def generate_probe(self, on_board_parameter):
        ready_probe = True
        while ready_probe:
            deck = Deck()
            in_hand = deck.draw(2)
            on_board = deck.draw(on_board_parameter)

            set_probe = set(in_hand + on_board)
            ready_probe = False
            for generate_probe in self.generated_probes:
                if generate_probe & set_probe == set_probe:
                    ready_probe = True
                    break

        self.generated_probes.append(set_probe)
        hand_strength = self.evaluator.evaluate(on_board, in_hand)

        probe = []
        for card in in_hand + on_board:
            probe.append(int(math.log2(Card.get_suit_int(card))) + 1)  # colour (library: 1, 2, 4, 8)
            probe.append(Card.get_rank_int(card) + 1)  # figure (library: 0, 1, 2,...,12)

        # 1: "Royal Straight Flush"
        # 2: "Straight Flush"
        # 3: "Four of a Kind"
        # 4: "Full House"
        # 5: "Flush"
        # 6: "Straight"
        # 7: "Three of a Kind"
        # 8: "Two Pair
        # 9: "Pair"
        # 10: "High Card"

        # library: 1 - Straight Flush, so must +1
        hand = 1 if hand_strength == 1 else self.evaluator.get_rank_class(hand_strength) + 1
        probe.append(hand)
        probe.append(1-self.evaluator.get_five_card_rank_percentage(hand_strength))
        return ','.join(map(str, probe))  # change list of number to string, where number is separated by ','

    @staticmethod
    def get_ints_from_card_symbols(card):
        rank_char = card[0]
        suit_char = card[1]
        rank_int = Card.CHAR_RANK_TO_INT_RANK[rank_char] + 1
        suit_int = int(math.log2(Card.CHAR_SUIT_TO_INT_SUIT[suit_char])) + 1
        return [suit_int, rank_int]

training_set = TrainingSetPreparer()
TrainingSetPreparer.generate_training_set(training_set, 0)
#one_hot = OneHot()
