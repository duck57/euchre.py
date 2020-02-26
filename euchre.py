#!/usr/bin/env python3
# coding=UTF-8
# -*- coding: UTF-8 -*-
# vim: set fileencoding=UTF-8 :

"""
Double-deck bid euchre

Implementation is similar to the rules given by Craig Powers 
https://www.pagat.com/euchre/bideuch.html

Notable differences (to match how I learned in high school calculus) include:
    * Minimum bid of 6 (which can be stuck to the dealer)
    * Shooters and loners are separate bids (guessing as ±18 for shooter, similar to a loner)
    * Shooters are a mandatory 2 card exchange with your partner
    * Trump isn't announced until after bidding has conculded
    * Winner of bid leads the first hand
    * Winning your bid gives you (tricks eaned + 2) points

Mothjab is a funny word with no current meaning.
"""

from enum import Enum
from itertools import cycle, islice
import random
from typing import List, Optional, Tuple, Iterable, Iterator, Set


class Suit(Enum):
    JOKER = (0, "🃏")
    CLUB = (1, "♣")
    DIAMOND = (2, "♢")
    SPADE = (3, "♠")
    HEART = (4, "♡")
    TRUMP = (5, "T")

    def plural_name(self) -> str:
        if self != self.TRUMP:
            return self.name + "S"
        return self.name

    def __str__(self):
        return self.plural_name()

    def __repr__(self):
        return self._value_[1]

    def value(self):
        return self._value_[0]


class Rank(Enum):
    JOKER = (0, "🃟")
    ACE_LO = (1, "a")
    TWO = (2, "2")
    THREE = (3, "3")
    FOUR = (4, "4")
    FIVE = (5, "5")
    SIX = (6, "6")
    SEVEN = (7, "7")
    EIGHT = (8, "8")
    NINE = (9, "9")
    TEN = (10, "🔟")
    JACK = (11, "J")
    QUEEN = (12, "Q")
    KING = (13, "K")
    ACE_HI = (14, "A")
    LEFT_BOWER = (15, "L")
    RIGHT_BOWER = (16, "R")

    def long_display_name(self) -> str:
        if self == self.ACE_HI:
            return "ACE"
        if self == self.ACE_LO:
            return "ACE"
        return self.name

    def value(self):
        return self._value_[0]

    def __repr__(self):
        return self._value_[1]

    def __str__(self):
        return self.long_display_name()


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.d_rank = rank
        self.d_suit = suit

    def card_name(self) -> str:
        return self.rank.long_display_name() + " of " + self.suit.plural_name()

    def __str__(self) -> str:
        return self.card_name()

    def __repr__(self):
        return f"{repr(self.d_rank)}{repr(self.d_suit)}"


# make a euchre deck
def make_euchre_deck() -> List[Card]:
    deck = []
    for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE_HI]:
        for suit in [Suit.HEART, Suit.CLUB, Suit.SPADE, Suit.DIAMOND]:
            deck.append(Card(rank, suit))
    return deck


# a.k.a. a double euchre deck
def make_pinochle_deck() -> List[Card]:
    return make_euchre_deck() + make_euchre_deck()


class Player:
    hand: List[Card]  # sorted
    name: str
    tricks: int = 0

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"{self.name}"


class Team:
    players: Set[Player]
    score: int = 0
    bid: int

    def __init__(self, players: Set[Player]):
        self.players = players

    def __str__(self):
        return str(self.players)


# main method
def main(handedness: int = 4):
    double_deck(handedness)


# normal euchre would be an entirely different function
def double_deck(handedness: int = 4) -> None:
    deck: List[Card] = make_pinochle_deck()
    plist: List[Player]
    tea_lst: Set[Team]
    victory_threshold: int = 50
    mercy_rule: int = -50
    bad_ai_end: int = -25

    # set up teams and players
    if handedness == 3:
        plist = [Player("Juan"), Player("Sarah"), Player("Turia")]
        tea_lst = {Team({plist[0]}), Team({plist[1]}), Team({plist[2]})}
    else:
        plist = [
            Player("Nelson"),
            Player("Eustace"),
            Player("Samantha"),
            Player("Wyclef"),
        ]
        tea_lst = {Team({plist[0], plist[2]}), Team({plist[1], plist[3]})}
    deal_order: Iterator[Player] = cycle(plist)
    next(deal_order)
    next(deal_order)  # initial dealer for 4-hands should be South

    def victory_check() -> Tuple[int, Optional[Team]]:
        scores_all_bad: bool = True
        for team in tea_lst:
            if team.score >= victory_threshold:
                return 1, team
            if team.score < mercy_rule:
                return -1, team
            if team.score > bad_ai_end:
                scores_all_bad = False
        if scores_all_bad:
            return -1, None
        return 0, None

    # play the game
    v: Tuple[int, Optional[Team]] = victory_check()
    while v[0] == 0:
        dealer: Player = next(deal_order)
        bid_order: List[Player] = list(islice(deal_order, handedness))
        play_hand(deck, handedness, bid_order, tea_lst)
        v = victory_check()

    # display final scores
    for t in tea_lst:
        print(t.score)


def play_hand(
    deck: List[Card], handedness: int, bid_order: List[Player], teams: Set[Team]
) -> None:
    random.shuffle(deck)
    hand_size: int = len(deck) // handedness  # could be hardcoded
    ppt: int = len(bid_order) // len(teams)
    bid_cycle: Iterator[Player] = cycle(bid_order)

    # deal the cards
    idx: int = 0
    print(f"Dealer: {bid_order[-1]}")
    for player in bid_order:
        player.hand = deck[idx : idx + hand_size]
        idx += hand_size
        print(f"{player}: {player.hand}")

    # bidding

    # play the tricks
    while hand_size > 0:
        # play_trick(trump)  # TODO
        hand_size -= 1

    # calculate scores
    for t in teams:
        t.score += random.randrange(-12, 14)


if __name__ == "__main__":
    main()
