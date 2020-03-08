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
    * Trump isn't announced until after bidding has concluded
    * Winner of bid leads the first hand
    * Winning your bid gives you (tricks earned + 2) points

Mothjab is a funny word with no current meaning.
"""

from enum import Enum
from itertools import cycle, islice
import random
from typing import List, Optional, Tuple, Iterable, Iterator, Set, Callable, Dict
from copy import deepcopy


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

    def color(self):
        return self._value_[3]

    def opposite(self) -> "Suit":
        return op_s(self)

    def __lt__(self, other):
        return self.value() < other.value()

    def __eq__(self, other):
        return self.value() == other.value()

    def __hash__(self):
        return self.value()


def op_s(s: Suit) -> Suit:
    if s == Suit.CLUB:
        return Suit.SPADE
    if s == Suit.SPADE:
        return Suit.CLUB
    if s == Suit.HEART:
        return Suit.DIAMOND
    if s == Suit.DIAMOND:
        return Suit.HEART
    return s


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

    def __lt__(self, other):
        return self.value() < other.value()

    def __eq__(self, other):
        return self.value() == other.value()

    def __hash__(self):
        return self.value()


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
        # return f"{repr(self.rank)}{repr(self.suit)}"

    def __eq__(self, other):
        return True if (self.rank == other.rank and self.suit == other.suit) else False

    def __lt__(self, other):
        if self.suit < other.suit:
            return True
        if self.suit > other.suit:
            return False
        if self.rank < other.rank:
            return True
        return False

    def beats(self, other: "Card", is_low: bool) -> bool:
        """Assumes the first played card of equal value wins"""
        if other.suit != self.suit and other.suit != Suit.TRUMP:
            return False  # must follow suit
        return (self < other) if is_low else (self > other)


def display_key(c: Card) -> int:
    if c.rank == Rank.LEFT_BOWER:
        return c.d_suit.opposite().value() * 100 + 15
    return c.d_suit.value() * 100 + c.rank.value()


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
    next_player: "Player"
    is_bot: int
    desired_trump: Tuple[Optional[Suit], bool]

    def __init__(self, name: str, bot: int = 1):
        self.name = name
        self.is_bot = bot

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"Player {self.name}"

    def choose_trump(self) -> Tuple[Optional[Suit], bool]:
        if self.is_bot:
            return self.desired_trump

    def make_bid(self, valid_bids: Set[int], d: List[Card], handedness: int) -> int:
        bid: int = 0
        if self.is_bot:
            # setup
            test_bids: Dict[Tuple[Optional[Suit], bool], int] = {
                (Suit.SPADE, False): 0,
                (Suit.HEART, False): 0,
                (Suit.DIAMOND, False): 0,
                (Suit.CLUB, False): 0,
                (None, False): 0,  # HiNo
                (None, True): 0,  # LoNo
            }
            for card in self.hand:
                d.remove(card)

            # test the bids by simulating games
            for t in test_bids.keys():
                # internal setup
                h_p: List[Card] = deepcopy(self.hand)
                d_p: List[Card] = deepcopy(d)
                if t[0]:
                    make_cards_trump(h_p, t[0])
                    make_cards_trump(d_p, t[0])
                # largest cards first this time
                h_p.sort(key=power_trump_key, reverse=not t[1])
                d_p.sort(key=power_trump_key, reverse=not t[1])
                my_trump: List[Card] = follow_suit(Suit.TRUMP, h_p)
                my_other: List[Card] = [c for c in h_p if (c.suit != Suit.TRUMP)]
                mystery_trump: List[Card] = follow_suit(Suit.TRUMP, d_p)
                mystery_other: List[Card] = [c for c in d_p if (c.suit != Suit.TRUMP)]

                # simulate a game
                for card in h_p:
                    upset_check: List[Card] = follow_suit(
                        card.suit, d_p, False
                    )[:handedness-1]
                    if not upset_check:
                        continue
                    if card.beats(upset_check[-1], t[1]):
                        test_bids[t] += 1
                    for c in upset_check:
                        d_p.remove(c)
                print(f"{t}: {test_bids[t]}")

            # pick the biggest
            bid = max(test_bids.values())
            self.desired_trump = random.choice(
                [k for k in test_bids.keys() if (test_bids[k] == bid)]
            )
            bid += 2 if handedness == 4 else 0  # count on two tricks from your partner
            if bid > len(self.hand):
                bid = max(valid_bids)  # call a loner
        return bid if bid in valid_bids else 0

    def trumpify_hand(self, trump_suit: Optional[Suit], is_lo: bool = False):
        """Marks the trump suit and sort the hands"""
        if trump_suit:
            make_cards_trump(self.hand, trump_suit)
        self.hand.sort(reverse=is_lo, key=self.choose_sort_key(trump_suit))

    def choose_sort_key(self, trump: Optional[Suit]) -> Callable:
        if not self.is_bot:
            return display_key
        return power_trump_key

    def play_card(
        self,
        trick_in_progress: List[Card],
        handedness: int,
        is_low: bool,
        discard_pile: Optional[List[Card]],
        unplayed: Optional[List[Card]],
    ) -> Card:
        valid_cards: List[Card] = []
        c: Card
        if trick_in_progress:
            valid_cards = follow_suit(trick_in_progress[0].suit, self.hand, strict=True)
        if not valid_cards:
            valid_cards = self.hand
        if not self.is_bot:
            pass  # replace with pick_card_human
        if self.is_bot == 1:
            c = valid_cards[-1]  # simple AI algorithm
        if self.is_bot == 2:
            pass
        self.hand.remove(c)
        return c  # use discard_pile and unplayed


def rank_first_key(c: Card) -> int:
    return c.rank.value() * 10 + c.suit.value()


def suit_first_key(c: Card) -> int:
    return c.suit.value() * 100 + c.rank.value()


def power_trump_key(c: Card) -> int:
    return rank_first_key(c) + (1000 if c.suit == Suit.TRUMP else 0)


def make_cards_trump(h: List[Card], trump_suit: Suit):
    for card in h:
        if card.suit == trump_suit:
            card.suit = Suit.TRUMP
            if card.rank == Rank.JACK:
                card.rank = Rank.RIGHT_BOWER
        if card.rank == Rank.JACK and card.suit == trump_suit.opposite():
            card.suit = Suit.TRUMP
            card.rank = Rank.LEFT_BOWER


class Team:
    players: Set[Player]
    score: int = 0
    bid: int = 0

    def __init__(self, players: Set[Player]):
        for player in players:
            player.team = self
        self.players = players

    def __repr__(self):
        return str(self.players)


class Game:
    teams: Set[Team] = set()
    trump: Optional[Suit] = None
    low_win: bool = False
    discard_pile: List[Card] = []
    unplayed_cards: List[Card] = []
    deck_generator: Callable
    handedness: int
    hand_size: int
    current_dealer: Player

    def __init__(self, players: List[Player], deck_gen: Callable):
        for p in players:
            self.teams.add(p.team)
        self.deck_generator = deck_gen
        self.current_dealer = players[2]  # initial dealer for 4-hands should be South
        self.handedness = len(players)
        self.hand_size = 48 // self.handedness  # adjust this for single-deck
        for i in range(self.handedness - 1):
            players[i].next_player = players[i + 1]
        players[self.handedness - 1].next_player = players[0]

    def play_hand(self) -> None:
        deck: List[Card] = self.deck_generator()
        random.shuffle(deck)
        print(f"Dealer: {self.current_dealer}")
        po: List[Player] = get_play_order(self.current_dealer, self.handedness)
        po.append(self.current_dealer)
        po.pop(0)  # because the dealer doesn't lead bidding

        # deal the cards
        idx: int = 0
        for player in po:
            player.hand = deck[idx : idx + self.hand_size]
            player.tricks = 0
            idx += self.hand_size

        # bidding
        lead: Player = bidding(po)

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

        # modify hands if trump called
        for player in po:
            player.trumpify_hand(trump, is_low)
            print(f"{player}: {player.hand}")

        # play the tricks
        hand_size: int = self.hand_size
        while hand_size > 0:
            lead = self.play_trick(lead, is_low)
            hand_size -= 1

        # calculate scores
        for t in self.teams:
            tr_t: int = 0
            ls: int = 0
            for p in t.players:
                tr_t += p.tricks
            if t.bid:
                if t.bid > 16:  # only in 4 hand
                    ls = t.bid
                    t.bid = 12

                if tr_t < t.bid:
                    print(f"{t} got Euchred and fell {t.bid - tr_t} short of {t.bid}")
                    t.score -= t.bid
                elif ls:
                    print(f"{t} won the loner, the absolute madman!")
                    t.score += ls
                else:
                    print(f"{t} beat their bid of {t.bid} with {tr_t} tricks")
                    t.score += tr_t + 2
            else:
                t.score += tr_t
            print(t.score)
            t.bid = 0  # reset for next time
        self.current_dealer = self.current_dealer.next_player

    def play_trick(self, lead: Player, is_low: bool = False) -> Player:
        p: Player = lead
        po: List[Player] = []
        cards: List[Card] = []

        # play the cards
        while p not in po:
            po.append(p)
            c: Card = p.play_card(
                cards, self.handedness, is_low, self.discard_pile, self.unplayed_cards
            )
            cards.append(c)
            # print(f"{p.name} played {repr(c)}")
            p = p.next_player

        # find the winner
        i: int = 0
        w: int = 0
        wc: Card = cards[0]
        for c in cards:
            if c.beats(wc, is_low):
                wc = c
                w = i
            i += 1
        po[w].tricks += 1
        # print(f"{po[w].name} won the trick")
        return po[w]


def get_play_order(lead: Player, handedness: int) -> List[Player]:
    po: List[Player] = [lead]
    for i in range(handedness - 1):
        po.append(po[-1].next_player)
    return po


# main method
def main(handedness: int = 4):
    double_deck(handedness)


# normal euchre would be an entirely different function
def double_deck(handedness: int = 4) -> None:
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
    g: Game = Game(plist, make_pinochle_deck)

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
        g.play_hand()
        v = victory_check()

    # display final scores
    for t in tea_lst:
        print(t.score)


def bidding(bid_order: List[Player], *, min_bid: int = 6) -> Player:
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
        bid: int = p.make_bid(valid_bids, make_pinochle_deck(), hands)

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
    return wp


def follow_suit(s: Optional[Suit], cs: List[Card], strict: bool = True) -> List[Card]:
    if not s:
        return cs
    return [c for c in cs if (c.suit == s or c.suit == Suit.TRUMP and not strict)]


if __name__ == "__main__":
    main(4)
