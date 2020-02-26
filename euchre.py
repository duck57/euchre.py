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
    JOKER = (0, "🃏", 0, "white")
    CLUB = (1, "♣", 3, "black")
    DIAMOND = (2, "♢", 4, "red")
    SPADE = (3, "♠", 1, "black")
    HEART = (4, "♡", 2, "red")
    TRUMP = (5, "T", 5, "yes")

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
    TEN = (10, "⒑")
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
    team: "Team"
    player_type: int = 0

    def __init__(self, name: str, bot: int = 1):
        self.name = name
        self.is_bot = bot

    def __str__(self):
        return f"{self.name}"

    def choose_trump(self) -> Tuple[Optional[Suit], bool]:
        if self.is_bot:
            s = random.choice(
                [Suit.HEART, Suit.SPADE, Suit.CLUB, Suit.DIAMOND, None, False]
            )
            r = False if s else random.choice([True, False])
            return (None if not s else s), r

    def make_bid(self, valid_bids: Set[int]) -> int:
        if self.is_bot:
            bid: int = random.randint(-4, 25)
            if bid not in valid_bids:
                bid = random.randint(-5, 10)
            return bid


class Team:
    players: Set[Player]
    score: int = 0
    bid: int = 0

    def __init__(self, players: Set[Player]):
        for player in players:
            player.team = self
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
        print(f"Dealer: {next(deal_order)}")
        bid_order: List[Player] = list(islice(deal_order, handedness))
        play_hand(deck, handedness, bid_order, tea_lst)
        v = victory_check()

    # display final scores
    for t in tea_lst:
        print(t.score)


def bidding(
    bid_order: List[Player], *, min_bid: int = 6
) -> List[Player]:
    first_round: bool = True
    count: int = 1
    hands: int = len(bid_order)
    wp: Player
    valid_bids: Set[int] = {6, 7, 8, 9, 10, 11, 12}
    if len(bid_order) == 4:
        valid_bids |= {18, 24}
    else:
        valid_bids |= {13, 14, 15, 16}
    wb: int = 0
    bid_order = cycle(bid_order)

    for p in bid_order:
        bid: int = p.make_bid(valid_bids)

        # everyone has passed
        if count == hands:
            if first_round:  # stuck the dealer
                wb = min_bid
                print(f"Dealer {p} got stuck with {min_bid}")
                wp = p
            else:  # someone won the bid
                wb = min_bid - 1
            break

        # player passes
        if bid < min_bid:
            print(f"{p} passes")
            count += 1
            continue

        # bid successful
        min_bid = bid + 1
        wp = p
        count = 1
        first_round = False
        print(f"{p} bids {bid}")

    wp.team.bid = wb
    return list(islice(bid_order, 3, 3 + hands))


def play_trick(
    play_order: List[Player], trump: Optional[Suit], is_low: bool = False
) -> List[Player]:
    random.choice(play_order).tricks += 1
    return play_order


def play_hand(
    deck: List[Card], handedness: int, bid_order: List[Player], teams: Set[Team]
) -> None:
    random.shuffle(deck)
    hand_size: int = len(deck) // handedness  # could be hardcoded
    ppt: int = len(bid_order) // len(teams)
    bid_cycle: Iterator[Player] = cycle(bid_order)

    # deal the cards
    idx: int = 0
    for player in bid_order:
        player.hand = deck[idx : idx + hand_size]
        player.tricks = 0
        idx += hand_size
        print(f"{player}: {player.hand}")

    # bidding
    play_order: List[Player] = bidding(bid_order)
    lead: Player = play_order[0]

    # declare Trump
    trump, is_low = lead.choose_trump()
    t_d: str
    if trump:
        t_d = str(trump)
    elif is_low:
        t_d = "LoNo"
    else:
        t_d = "HiNo"
    print(f"{lead} bid {lead.team.bid} {t_d}")

    # play the tricks
    while hand_size > 0:
        play_order = play_trick(play_order, trump, is_low)
        hand_size -= 1

    # calculate scores
    for t in teams:
        tr_t: int = 0
        ls: int = 0
        for p in t.players:
            tr_t += p.tricks
        if t.bid:
            if t.bid > 16:  # only in 4 hand
                ls = t.bid
                t.bid = 12

            if tr_t < t.bid:
                print(f"{t} got Euchred and fell {t.bid-tr_t} short")
                t.score -= t.bid
            elif ls:
                t.score += ls
            else:
                t.score += tr_t + 2
        else:
            t.score += tr_t
        t.bid = 0  # reset for next time


if __name__ == "__main__":
    main(4)
