from itertools import combinations
from Libraries.deuces.deuces import Card, Deck, Evaluator
import math
from Libraries.MonteCarlo.Monte_Carlo_Poker_Simulation import MonteCarlo
import os
import random


class TrainingSetPreparer:
    # Hands:
    # 0: Royal Straight Flush
    # 1: Straight Flush
    # 2: Four of a Kind
    # 3: Full House
    # 4: Flush
    # 5: Straight
    # 6: Three of a Kind
    # 7: Two Pair
    # 8: Pair
    # 9: High Card

    # Suits:
    # 1: Spades (s)
    # 2: Hearts (h)
    # 3: Diamonds (d)
    # 4: Clubs (c)

    # Ranks:
    # 1 : 2 (analogically for ranks 3, 4, ..., 9)
    # 9: 10 (T)
    # 10: Jack (J)
    # 11: Queen (Q)
    # 12: King (K)
    # 13: Ace (A)
    dir_path = "./Sets"
    evaluator = Evaluator()
    generated_probes = []

    def generate_probe_3_5(self, on_board_parameter):
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
            probe.append(int(math.log2(Card.get_suit_int(card))) + 1)  # suit (library: 1, 2, 4, 8)
            probe.append(Card.get_rank_int(card) + 1)  # rank (library: 0, 1, 2,...,12)

        # library: 0 - Straight Flush
        hand = 0 if hand_strength == 1 else self.evaluator.get_rank_class(hand_strength)
        probe.append(hand)
        probe.append(1-self.evaluator.get_five_card_rank_percentage(hand_strength))
        return ','.join(map(str, probe))  # change list of number to string, where number is separated by ','

    # Prepare files for 4 cases with on_board_parameter: 0, 3, 4, 5
    # 2 card in hand 0 visible on table (0)
    # 2 card in hand 3 visible on table (3)
    # 2 card in hand 4 visible on table (4)
    # 2 card in hand 5 visible on table (5)
    def generate_sets(self, on_board_parameter):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        if all([on_board_parameter >= 3, on_board_parameter <= 5]):
            # training set
            self.save_to_file_3_5("training", on_board_parameter, 32768)
            # test set
            self.save_to_file_3_5("test", on_board_parameter, 4096)

        elif on_board_parameter == 0:
            deck = Deck.GetFullDeck()
            for i in range(0, len(deck)):
                deck[i] = Card.int_to_str(deck[i])  # rank+suit, i.e. Ad - ace diamond

            pair_combinations = list(combinations(deck, 2))  # list of tuples
            test_probes = random.sample(pair_combinations, len(pair_combinations) // 10)  # 10% for testing
            training_probes = (list(set(pair_combinations) - set(test_probes)))

            self.save_to_file_0("training", training_probes)
            self.save_to_file_0("test", test_probes)

    def save_to_file_0(self, filename_infix, probes):
        with open(self.dir_path + "/poker-hand-" + filename_infix + "-2.data", "w") as set:
            mc = MonteCarlo(times=10000)  # 'times' says how many times it should play
            for idx in range(len(probes)):
                # total_villains - how many opponents
                equity = mc.preflop(card1=probes[idx][0], card2=probes[idx][1], total_villains=4)
                string = ','.join(map(str, self.get_ints_from_card_symbols(
                    probes[idx][0]) + self.get_ints_from_card_symbols(probes[idx][1]) + [equity]))
                set.write(string + "\n")
                set.flush()

    def save_to_file_3_5(self, filename_infix, on_board_parameter, set_size):
        with open(self.dir_path + "/poker-hand-" + filename_infix + "-" + str(on_board_parameter + 2) + ".data", "w") as set:
            self.generated_probes = []
            for idx in range(0, set_size):
                set.write(self.generate_probe_3_5(on_board_parameter) + "\n")
                set.flush()

    @staticmethod
    def get_ints_from_card_symbols(card):
        rank_char = card[0]
        suit_char = card[1]
        rank_int = Card.CHAR_RANK_TO_INT_RANK[rank_char] + 1
        suit_int = int(math.log2(Card.CHAR_SUIT_TO_INT_SUIT[suit_char])) + 1
        return [suit_int, rank_int]


training_set = TrainingSetPreparer()
TrainingSetPreparer.generate_sets(training_set, 3) # on board
