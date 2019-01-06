from itertools import combinations
from Libraries.deuces.deuces import Card, Deck, Evaluator
import math
from Libraries.MonteCarlo.Monte_Carlo_Poker_Simulation import MonteCarlo
import os
import random
import itertools


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

    dir_path = "./Sets/GeneratedSets"
    evaluator = Evaluator()

    # Prepare files for 4 cases with all_cards_no: 2, 5, 6, 7
    # 2 card in hand 0 visible on table (2)
    # 2 card in hand 3 visible on table (5)
    # 2 card in hand 4 visible on table (6)
    # 2 card in hand 5 visible on table (7)
    def generate_sets(self, all_cards_no):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        deck = Deck.GetFullDeck()
        for i in range(0, len(deck)):
            # convert int to string: rank+suit, i.e. Ad - ace diamond
            deck[i] = Card.int_to_str(deck[i])
        # list of tuples
        pair_combinations = list(combinations(deck, 2))

        training_probes = []
        test_probes = []
        if all([all_cards_no >= 5, all_cards_no <= 7]):

            all_probes = list()
            for pair_combination in pair_combinations:
                pair = list(pair_combination)
                deck_without_pair = deck.copy()
                deck_without_pair.remove(pair[0])
                deck_without_pair.remove(pair[1])

                for i in range(0, 30):
                    pair_and_on_board = pair.copy()
                    deck_without_pair_and_on_board = deck_without_pair.copy()
                    for j in range(0, all_cards_no - 2):
                        pair_and_on_board.append(deck_without_pair_and_on_board.pop(
                            deck_without_pair_and_on_board.index(
                                random.choice(deck_without_pair_and_on_board))))
                    all_probes.append(tuple(pair_and_on_board))

            test_probes = random.sample(all_probes, len(all_probes) // 10)  # 10% for testing
            training_probes_set = set(all_probes) - set(test_probes)
            training_probes = (list(training_probes_set))

        elif all_cards_no == 2:
            test_probes = random.sample(pair_combinations, len(pair_combinations) // 10)  # 10% for testing
            training_probes = (list(set(pair_combinations) - set(test_probes)))

        self.save_to_file(all_cards_no, 2500, training_probes, "training")
        self.save_to_file(all_cards_no, 2500, test_probes, "test")

    def save_to_file(self, all_cards_no, monte_carlo_rounds, probes, training_or_test):
        with open("{}/poker-hand-{}-{}.data".format(self.dir_path, training_or_test, all_cards_no), "w") as set:
            mc = MonteCarlo(times=monte_carlo_rounds)  # 'times' says how many times it should play
            for idx in range(len(probes)):
                print(idx)
                probe = list(probes[idx])
                # total_villains - how many opponents
                if all_cards_no == 2:
                    equity = mc.preflop(card1=probe[0], card2=probe[1], total_villains=4)
                    set.write("{}\n".format(','.join(map(str, list(itertools.chain.from_iterable(
                        map(self.get_ints_from_card_symbols, probe))) + [equity]))))

                elif all([all_cards_no >= 5, all_cards_no <= 7]):
                    hand_cards, board_cards = self.convert_and_divide_string_probe_to_ints(probe)
                    hand_strength = self.evaluator.evaluate(hand_cards, board_cards)
                    # 0 - Straight Flush
                    hand = 0 if hand_strength == 1 else self.evaluator.get_rank_class(hand_strength)

                    if all_cards_no == 5:
                        equity = mc.flop(card1=probe[0], card2=probe[1], fcard1=probe[2], fcard2=probe[3],
                                         fcard3=probe[4], total_villains=4)
                    elif all_cards_no == 6:
                        equity = mc.turn(card1=probe[0], card2=probe[1], fcard1=probe[2],
                                         fcard2=probe[3], fcard3=probe[4], tcard=probe[5], total_villains=4)
                    elif all_cards_no == 7:
                        equity = mc.river(card1=probe[0], card2=probe[1], fcard1=probe[2],
                                          fcard2=probe[3], fcard3=probe[4], tcard=probe[5], rcard=probe[6],
                                          total_villains=4)
                    set.write("{}\n".format(','.join(map(str, list(itertools.chain.from_iterable(
                        map(self.get_ints_from_card_symbols, probe))) + [hand] + [equity]))))
                set.flush()

    @staticmethod
    def convert_and_divide_string_probe_to_ints(probe):
        hand_cards = list()
        for card in probe[0:2]:
            # card in string format to integer format
            hand_cards.append(Card.new(card))

        board_cards = list()
        for card in probe[2:]:
            # card in string format to integer format
            board_cards.append(Card.new(card))
        return hand_cards, board_cards
    
    @staticmethod
    def get_ints_from_card_symbols(card):
        rank_char = card[0]
        suit_char = card[1]
        rank_int = Card.CHAR_RANK_TO_INT_RANK[rank_char] + 1
        suit_int = int(math.log2(Card.CHAR_SUIT_TO_INT_SUIT[suit_char])) + 1
        return [suit_int, rank_int]
