from itertools import combinations
from Libraries.deuces.deuces import Card, Deck, Evaluator
import math
from Libraries.MonteCarlo.Monte_Carlo_Poker_Simulation import MonteCarlo
import os
import random
import time

class SetGenerator:
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

    # Prepare files for 4 cases with on_board_parameter: 0, 3, 4, 5
    # 2 card in hand 0 visible on table (0)
    # 2 card in hand 3 visible on table (3)
    # 2 card in hand 4 visible on table (4)
    # 2 card in hand 5 visible on table (5)
    def generate_sets(self, on_board_parameter):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        if all([on_board_parameter >= 3, on_board_parameter <= 5]):
            deck = Deck.GetFullDeck()
            for i in range(0, len(deck)):
                deck[i] = Card.int_to_str(deck[i])  # rank+suit, i.e. Ad - ace diamond

            pair_combinations = list(combinations(deck, 2))  # list of tuples
            all_probes = list()

            for pair_combination in pair_combinations:
                pair = list(pair_combination)
                deck_without_pair = deck.copy()
                deck_without_pair.remove(pair[0])
                deck_without_pair.remove(pair[1])

                for i in range(0, 30):
                    pair_and_on_board = pair.copy()
                    deck_without_pair_and_on_board = deck_without_pair.copy()
                    for j in range(0, on_board_parameter):
                        pair_and_on_board.append(deck_without_pair_and_on_board.pop(deck_without_pair_and_on_board.index(
                            random.choice(deck_without_pair_and_on_board))))
                    all_probes.append(tuple(pair_and_on_board))

            test_probes = random.sample(all_probes, len(all_probes) // 10)  # 10% for testing
            training_probes_set = set(all_probes) - set(test_probes)
            training_probes = (list(training_probes_set))

            if on_board_parameter == 3:
                self.save_to_file_3("training", training_probes)
                self.save_to_file_3("test", test_probes)
            elif on_board_parameter == 4:
                self.save_to_file_4("training", training_probes)
                self.save_to_file_4("test", test_probes)
            elif on_board_parameter == 5:
                self.save_to_file_5("training", training_probes)
                self.save_to_file_5("test", test_probes)

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

    def save_to_file_3(self, filename_infix, probes):
        with open(self.dir_path + "/poker-hand-" + filename_infix + "-5.data", "w") as set:
            mc = MonteCarlo(times=10000)  # 'times' says how many times it should play
            for idx in range(len(probes)):
                print(idx)
                # start_time = time.time()
                probe = list(probes[idx])
                cards = list()
                board = list()
                for c in probe[0:2]:
                    cards.append(Card.new(c))
                for c in probe[2:]:
                    board.append(Card.new(c))

                hand_strength = self.evaluator.evaluate(cards, board)
                # library: 0 - Straight Flush
                hand = 0 if hand_strength == 1 else self.evaluator.get_rank_class(hand_strength)

                # total_villains - how many opponents
                equity = mc.flop(card1=probe[0], card2=probe[1], fcard1=probe[2], fcard2=probe[3], fcard3=probe[4], total_villains=4)
                string = ','.join(map(str,
                                      self.get_ints_from_card_symbols(probe[0]) +
                                      self.get_ints_from_card_symbols(probe[1]) +
                                      self.get_ints_from_card_symbols(probe[2]) +
                                      self.get_ints_from_card_symbols(probe[3]) +
                                      self.get_ints_from_card_symbols(probe[4]) + [hand] + [equity]))
                set.write(string + "\n")
                set.flush()
                # print("--- %s seconds ---" % (time.time() - start_time))

    def save_to_file_4(self, filename_infix, probes):
        with open(self.dir_path + "/poker-hand-" + filename_infix + "-6.data", "w") as set:
            mc = MonteCarlo(times=10000)  # 'times' says how many times it should play
            for idx in range(len(probes)):
                probe = list(probes[idx])
                cards = list()
                board = list()
                for c in probe[0:2]:
                    cards.append(Card.new(c))
                for c in probe[2:]:
                    board.append(Card.new(c))

                hand_strength = self.evaluator.evaluate(cards, board)
                # library: 0 - Straight Flush
                hand = 0 if hand_strength == 1 else self.evaluator.get_rank_class(hand_strength)

                # total_villains - how many opponents
                equity = mc.turn(card1=probe[0], card2=probe[1], fcard1=probe[2],
                                 fcard2=probe[3], fcard3=probe[4], tcard=probe[5], total_villains=4)
                string = ','.join(map(str,
                                      self.get_ints_from_card_symbols(probe[0]) +
                                      self.get_ints_from_card_symbols(probe[1]) +
                                      self.get_ints_from_card_symbols(probe[2]) +
                                      self.get_ints_from_card_symbols(probe[3]) +
                                      self.get_ints_from_card_symbols(probe[4]) +
                                      self.get_ints_from_card_symbols(probe[5]) + [hand] + [equity]))
                set.write(string + "\n")
                set.flush()

    def save_to_file_5(self, filename_infix, probes):
        with open(self.dir_path + "/poker-hand-" + filename_infix + "-7.data", "w") as set:
            mc = MonteCarlo(times=10000)  # 'times' says how many times it should play
            for idx in range(len(probes)):
                probe = list(probes[idx])
                cards = list()
                board = list()
                for c in probe[0:2]:
                    cards.append(Card.new(c))
                for c in probe[2:]:
                    board.append(Card.new(c))

                hand_strength = self.evaluator.evaluate(cards, board)
                # library: 0 - Straight Flush
                hand = 0 if hand_strength == 1 else self.evaluator.get_rank_class(hand_strength)

                # total_villains - how many opponents
                equity = mc.river(card1=probe[0], card2=probe[1], fcard1=probe[2],
                                 fcard2=probe[3], fcard3=probe[4], tcard=probe[5], rcard=probe[6], total_villains=4)
                string = ','.join(map(str,
                                      self.get_ints_from_card_symbols(probe[0]) +
                                      self.get_ints_from_card_symbols(probe[1]) +
                                      self.get_ints_from_card_symbols(probe[2]) +
                                      self.get_ints_from_card_symbols(probe[3]) +
                                      self.get_ints_from_card_symbols(probe[4]) +
                                      self.get_ints_from_card_symbols(probe[5]) +
                                      self.get_ints_from_card_symbols(probe[6]) + [hand] + [equity]))
                set.write(string + "\n")
                set.flush()

    @staticmethod
    def get_ints_from_card_symbols(card):
        rank_char = card[0]
        suit_char = card[1]
        rank_int = Card.CHAR_RANK_TO_INT_RANK[rank_char] + 1
        suit_int = int(math.log2(Card.CHAR_SUIT_TO_INT_SUIT[suit_char])) + 1
        return [suit_int, rank_int]