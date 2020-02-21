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
import random
from typing import List


class Suit(Enum):
    JOKER = 0
    CLUB = 1
    DIAMOND = 2
    SPADE = 3
    HEART = 4
    TRUMP = 5

    def plural_name(self) -> str:
        if self != self.TRUMP:
            return self.name + "S"
        return self.name


class Rank(Enum):
    JOKER = 0
    ACE_LO = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE_HI = 14
    LEFT_BOWER = 15
    RIGHT_BOWER = 16

    def disp_name(self) -> str:
        if self == self.ACE_HI:
            return "ACE"
        if self == self.ACE_LO:
            return "ACE"
        return self.name


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.d_rank = rank
        self.d_suit = suit

    def cardname(self) -> str:
        return self.rank.disp_name() + " of " + self.suit.plural_name()

    def __str__(self) -> str:
        return self.cardname()


# make a euchre deck
def make_euchre_deck() -> List[Card]:
    deck = []
    for rank in range(9, 15):
        for suit in range(1, 5):
            deck.append(Card(Rank(rank), Suit(suit)))
    return deck


# a.k.a. a double euchre deck
def make_pinochle_deck() -> List[Card]:
    return make_euchre_deck() + make_euchre_deck()


class Player:
    hand: List[Card]
    name: str

    def __init__(self, name: str):
        self.name = name


class Team:
    players: List[Player]
    score: int

    def __init__(self, players: List[Player]):
        self.players = players


# main method
def main(handedness: int = 4):
    double_deck(handedness)


# normal euchre would be an entirely different function
def double_deck(handedness: int = 4):
    deck: List[Card] = make_pinochle_deck()
    random.shuffle(deck)
    l: int = len(deck) // handedness  # could be hardcoded

    plist: List[Player]
    tea_lst: List[Team]

    # set up teams and players
    if handedness == 3:
        plist = [Player("One"), Player("Two"), Player("Third")]
        tea_lst = [Team([plist[0]]), Team([plist[1]]), Team([plist[2]])]
    else:
        plist = [Player("North"), Player("East"), Player("South"), Player("West")]
        tea_lst = [Team([plist[0], plist[2]]), Team([plist[1], plist[3]])]

    idx = 0
    for player in plist:
        player.hand = deck[idx:idx+l]
        print(f"\n{player.name}")
        for card in player.hand:
            print(card)


if __name__ == "__main__":
    main()
