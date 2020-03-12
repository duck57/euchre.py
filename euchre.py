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
from datetime import datetime
import configparser
import click


class Color:
    WHITE = (0, "FFF")
    BLACK = (1, "000")
    RED = (2, "F00")
    YES = (99, "")

    def __init__(self, value: int, hexa: str):
        self.v = value
        self.hex = hexa

    def __hash__(self) -> int:
        return self.v


class Suit(Enum):
    JOKER = (0, "🃏", "JOKER", Color.WHITE)
    CLUB = (1, "♣", "SPADE", Color.BLACK)
    DIAMOND = (2, "♢", "HEART", Color.RED)
    SPADE = (3, "♠", "CLUB", Color.BLACK)
    HEART = (4, "♡", "DIAMOND", Color.RED)
    TRUMP = (5, "T", "TRUMP", Color.YES)

    opposite: "Suit"
    v: int

    def __init__(self, value: int, symbol: str, opposite: str, color: Color):
        self.v = value
        self.symbol = symbol
        self.color = color
        try:
            self.opposite = self._member_map_[opposite]
            self._member_map_[opposite].opposite = self
        except KeyError:
            pass

    @property
    def plural_name(self) -> str:
        if self != self.TRUMP:
            return self.name + "S"
        return self.name

    def __str__(self):
        return self.plural_name

    def __repr__(self):
        return self.symbol

    def __lt__(self, other):
        return self.v < other.v

    def __eq__(self, other):
        return self.v == other.v

    def __hash__(self) -> int:
        return self.v


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

    def __init__(self, value: int, chr: str):
        self.v = value
        self.char = chr

    @property
    def long_display_name(self) -> str:
        if self == self.ACE_HI:
            return "ACE"
        if self == self.ACE_LO:
            return "ACE"
        return self.name

    def __repr__(self):
        return self.char

    def __str__(self):
        return self.long_display_name

    def __lt__(self, other):
        return self.v < other.v

    def __eq__(self, other):
        return self.v == other.v

    def __hash__(self):
        return self.v


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.d_rank = rank
        self.d_suit = suit

    @property
    def card_name(self) -> str:
        return self.rank.long_display_name + " of " + self.suit.plural_name

    def __str__(self) -> str:
        return self.card_name

    def __repr__(self):
        return f"{repr(self.d_rank)}{repr(self.d_suit)}"
        # return f"{repr(self.rank)}{repr(self.suit)}"  # debugging

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
    return (
        c.d_suit.opposite.v * 100 + 15
        if c.rank == Rank.LEFT_BOWER
        else c.d_suit.v * 100 + c.rank.v
    )


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
    team: "Team"
    next_player: "Player"
    desired_trump: Tuple[Optional[Suit], bool]

    def __init__(self, name: str, bot: int = 1):
        self.name: str = name
        self.is_bot: int = bot
        self.tricks: int = 0
        self.hand: List[Card] = []

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
                    upset_check: List[Card] = follow_suit(card.suit, d_p, False)[
                        : handedness - 1
                    ]
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
        trick_in_progress: "List[Tuple[Card, Player]]",
        handedness: int,
        is_low: bool,
        discard_pile: Optional[List[Card]],
        unplayed: Optional[List[Card]],
    ) -> Card:
        valid_cards: List[Card] = []
        if trick_in_progress:
            valid_cards = follow_suit(
                trick_in_progress[0][0].suit, self.hand, strict=True
            )
        if not valid_cards:
            valid_cards = self.hand
        c: Card = {0: human_pick_card, 1: simple_select_card, 2: advanced_card_ai}[
            self.is_bot
        ](valid_cards, self.hand, trick_in_progress, is_low, discard_pile, unplayed)
        self.hand.remove(c)
        return c  # use discard_pile and unplayed

    @property
    def teammates(self) -> "Set[Player]":
        return self.team.players - {self}


def human_pick_card(
    valid_cards: List[Card],
    full_hand: List[Card],
    trick_in_progress: List[Tuple[Card, Player]],
    is_low: bool,
    discard_pile: List[Card],
    unplayed: List[Card],
    *args,
) -> Card:
    print(trick_in_progress)
    proper_picks: List[int] = [
        i for i in range(len(full_hand)) if full_hand[i] in valid_cards
    ]
    selection_indices: List[str] = []
    for j in range(len(full_hand)):
        selection_indices.append(f"{j:2}" if j in proper_picks else "  ")
    print("  ".join([repr(c) for c in full_hand]))
    print("  ".join(selection_indices))
    return valid_cards[-1]


def simple_select_card(
    valid_cards: List[Card],
    full_hand: List[Card],
    trick_in_progress: List[Tuple[Card, Player]],
    is_low: bool,
    discard_pile: List[Card],
    unplayed: List[Card],
    *args,
) -> Card:
    return valid_cards[-1]


def advanced_card_ai(
    valid_cards: List[Card],
    full_hand: List[Card],
    trick_in_progress: List[Tuple[Card, Player]],
    is_low: bool,
    discard_pile: List[Card],
    unplayed: List[Card],
    *args,
    **kwargs,
) -> Card:
    return valid_cards[-1]


def rank_first_key(c: Card) -> int:
    return c.rank.v * 10 + c.suit.v


def suit_first_key(c: Card) -> int:
    return c.suit.v * 100 + c.rank.v


def power_trump_key(c: Card) -> int:
    return rank_first_key(c) + (1000 if c.suit == Suit.TRUMP else 0)


def make_cards_trump(h: List[Card], trump_suit: Suit):
    for card in h:
        if card.suit == trump_suit:
            card.suit = Suit.TRUMP
            if card.rank == Rank.JACK:
                card.rank = Rank.RIGHT_BOWER
        if card.rank == Rank.JACK and card.suit == trump_suit.opposite:
            card.suit = Suit.TRUMP
            card.rank = Rank.LEFT_BOWER


class Team:
    bid: int = 0

    def __init__(self, players: List[Player]):
        for player in players:
            player.team = self
        self.players: Set[Player] = set(players)
        self.bid_history: List[str] = []
        self.tricks_taken: List[int] = []
        self.score_changes: List[int] = []

    def __repr__(self):
        return str(self.players)

    @property
    def score(self):
        return sum(self.score_changes)

    @score.setter
    def score(self, value: int):
        self.score_changes.append(value)


class Game:
    def __init__(self, players: List[Player], deck_gen: Callable):
        self.trump: Optional[Suit] = None
        self.low_win: bool = False
        self.discard_pile: List[Card] = []
        self.unplayed_cards: List[Card] = []
        self.teams: Set[Team] = set([p.team for p in players])
        self.deck_generator: Callable = deck_gen
        self.current_dealer = players[2]  # initial dealer for 4-hands should be South
        self.handedness: int = len(players)
        self.hand_size: int = 48 // self.handedness  # adjust this for single-deck
        for i in range(self.handedness - 1):
            players[i].next_player = players[i + 1]
        players[self.handedness - 1].next_player = players[0]

        c = configparser.ConfigParser()
        c.read("constants.cfg")
        self.valid_bids: Set[int] = set(
            [int(i) for i in c["ValidBids"][str(self.handedness)].split(",")]
        )
        self.victory_threshold: int = c["Scores"].getint("victory")
        self.mercy_rule: int = c["Scores"].getint("mercy")
        self.bad_ai_end: int = c["Scores"].getint("broken_ai")

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
        lead: Player = bidding(po, self.valid_bids)

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
            bid: int = t.bid
            for p in t.players:
                tr_t += p.tricks
            if bid:
                # loners and shooters
                if bid > self.hand_size:
                    ls = bid
                    bid = self.hand_size

                if tr_t < bid:
                    print(f"{t} got Euchred and fell {bid - tr_t} short of {bid}")
                    t.score = -bid
                elif ls:
                    print(f"Someone on {t} won the loner, the absolute madman!")
                    t.score = ls
                else:
                    print(f"{t} beat their bid of {bid} with {tr_t} tricks")
                    t.score = tr_t + 2
            else:  # tricks the non-bidding team earned
                t.score = tr_t

            # bookkeeping
            t.bid_history.append(f"{ls if ls else bid} {t_d}" if bid else str(None))
            t.tricks_taken.append(tr_t)
            print(f"{t}: {t.score}")
            t.bid = 0  # reset for next time
        self.current_dealer = self.current_dealer.next_player

    def play_trick(self, lead: Player, is_low: bool = False) -> Player:
        p: Player = lead
        po: List[Player] = []
        trick_in_progress: List[Tuple[Card, Player]] = []

        # play the cards
        while p not in po:
            po.append(p)
            c: Card = p.play_card(
                trick_in_progress,
                self.handedness,
                is_low,
                self.discard_pile,
                self.unplayed_cards,
            )
            trick_in_progress.append((c, p))
            # print(f"{p.name} played {repr(c)}")
            p = p.next_player

        # find the winner
        w: Tuple[Card, Player] = trick_in_progress[0]
        for cpt in trick_in_progress:
            if cpt[0].beats(w[0], is_low):
                w = cpt
        w[1].tricks += 1
        # print(f"{w[1].name} won the trick")
        return w[1]

    def write_log(self, splitter: str = "\t|\t"):
        f = open(f"{str(datetime.now()).split('.')[0]}.gamelog", "w")
        t_l: List[Team] = list(self.teams)  # give a consistent ordering

        # headers
        f.write(splitter)
        f.write(splitter.join([f"{t}\t\t" for t in t_l]))
        f.write(f"\n{splitter}")
        f.write(splitter.join(["Bid\tTricks Taken\tScore Change" for _ in t_l]))
        f.write(f"\nHand{splitter}")
        f.write(splitter.join(["===\t===\t===" for _ in t_l]))

        # body
        for hand in range(len(t_l[0].bid_history)):
            f.write(f"\n{hand+1}{splitter}")
            f.write(
                splitter.join(
                    [
                        f"{t.bid_history[hand]}\t{t.tricks_taken[hand]}\t{t.score_changes[hand]}"
                        for t in t_l
                    ]
                )
            )

        # totals
        f.write(f"\n{splitter}")
        f.write(splitter.join(["===\t===\t===" for _ in t_l]))
        f.write(f"\nTotals{splitter}")
        f.write(
            splitter.join(
                [
                    f"{sum([1 for b in t.bid_history if b != str(None)])}\t{sum(t.tricks_taken)}\t{sum(t.score_changes)}"
                    for t in t_l
                ]
            )
        )
        f.close()

    def victory_check(self) -> Tuple[int, Optional[Team]]:
        scores_all_bad: bool = True
        for team in self.teams:
            if team.score >= self.victory_threshold:
                return 1, team
            if team.score < self.mercy_rule:
                return -1, team
            if team.score > self.bad_ai_end:
                scores_all_bad = False
        if scores_all_bad:
            return -1, None
        return 0, None

    def play(self) -> None:
        v: Tuple[int, Optional[Team]] = self.victory_check()
        while v[0] == 0:
            self.play_hand()
            v = self.victory_check()
        print(f"Final Scores")
        for t in self.teams:
            print(f"{t}: {t.score}")


def get_play_order(lead: Player, handedness: int) -> List[Player]:
    po: List[Player] = [lead]
    for i in range(handedness - 1):
        po.append(po[-1].next_player)
    return po


# normal euchre would be an entirely different function because of the kitty
@click.command()
@click.option(
    "--handedness",
    "-h",
    type=click.Choice(["33", "42", "63", "62", "82", "84"]),
    default="42",
    help="10s place: Number of players, 1s place: number of teams.",
)
@click.option(
    "--humans",
    "-p",
    multiple=True,
    default=[],
    type=click.INT,
    help="index of a human player, repeatable",
)
def main(handedness: int, humans: List[int]) -> Game:
    handedness: int = int(handedness)
    hands: int = handedness // 10
    teams: int = handedness % 10
    ppt: int = hands // teams
    player_handles: List[str] = {
        3: ["P1", "P2", "P3"],
        4: ["North", "East", "South", "West"],
        6: ["P1", "East", "P2", "North", "P3", "West"],
        8: ["P1", "North", "P2", "East", "Spare", "South", "P3", "West"],
    }[hands]
    if len(humans) == 1 and humans[0] < hands:
        player_handles[humans[0]] = "Human"
    plist: List[Player] = make_players(player_handles)
    for h in humans:
        if h < hands:
            plist[h].is_bot = 0
    for i in range(teams):
        print([i + j * teams for j in range(ppt)])
        Team([plist[i + j * teams] for j in range(ppt)])
    g: Game = Game(plist, make_pinochle_deck)
    g.play()
    g.write_log()
    return g


def make_players(handles: List[str]) -> List[Player]:
    c = configparser.ConfigParser()
    c.read("players.cfg")
    return [Player(c[h]["name"], int(c[h]["is_bot"])) for h in handles]


def bidding(bid_order: List[Player], valid_bids: Set[int]) -> Player:
    first_round: bool = True
    count: int = 1
    hands: int = len(bid_order)
    wp: Player
    wb: int = 0
    bid_order = cycle(bid_order)
    min_bid: int = min(valid_bids)
    max_bid: int = max(valid_bids)

    for p in bid_order:
        # everyone has passed
        if count == hands:
            if first_round:  # stuck the dealer
                wb = min_bid
                print(f"Dealer {p} got stuck with {min_bid}")
                wp = p
            else:  # someone won the bid
                wb = min_bid - 1
            break

        # end bidding early for a loner
        if min_bid > max_bid:
            break

        # get the bid
        bid: int = p.make_bid(valid_bids, make_pinochle_deck(), hands)

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
    main()
