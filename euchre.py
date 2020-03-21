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

from enum import Enum, unique
from itertools import cycle
import random
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
    Iterable,
    Set,
    Callable,
    Dict,
    NamedTuple,
    TextIO,
    Union,
)
from copy import deepcopy
from datetime import datetime
import configparser
import click
import os

o: Optional[TextIO] = None
debug: Optional[bool] = False
log_dir: str = "logs/"


def p(msg):
    global o
    click.echo(msg, o)


def px(msg) -> None:
    global debug
    if debug:
        p(msg)


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


@unique
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


@unique
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

    def __init__(self, value: int, char: str):
        self.v = value
        self.char = char

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

    def __hash__(self) -> int:
        return key_trump_power(self)

    def beats(self, other: "Card", is_low: bool) -> bool:
        """Assumes the first played card of equal value wins"""
        if self.suit not in {other.suit, Suit.TRUMP}:
            return False  # must follow suit
        return (self < other) if is_low else (self > other)

    def follows_suit(self, s: Suit, strict: bool = True) -> bool:
        return self.suit == s or self.suit == Suit.TRUMP and not strict


def key_display4human(c: Card) -> int:
    return (
        c.d_suit.opposite.v * 100 + 15
        if c.rank == Rank.LEFT_BOWER
        else c.d_suit.v * 100 + c.rank.v
    )


suits: List[Suit] = [Suit.HEART, Suit.SPADE, Suit.DIAMOND, Suit.CLUB]
euchre_ranks: List[Rank] = [
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE_HI,
]
poker_ranks: List[Rank] = [
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
] + euchre_ranks


class Hand(List[Card]):
    def __str__(self) -> str:
        return "  ".join([repr(x) for x in self])

    def __add__(self, other) -> "Hand":
        return Hand(list(self) + list(other))

    def trumpify(self, trump_suit: Optional[Suit]) -> "Hand":
        if not trump_suit:
            return self
        for card in self:
            if card.suit == trump_suit:
                card.suit = Suit.TRUMP
                if card.rank == Rank.JACK:
                    card.rank = Rank.RIGHT_BOWER
            if card.rank == Rank.JACK and card.suit == trump_suit.opposite:
                card.suit = Suit.TRUMP
                card.rank = Rank.LEFT_BOWER
        return self


def make_deck(r: List[Rank], s: List[Suit]) -> Hand:
    return Hand(Card(rank, suit) for rank in r for suit in s)


def make_euchre_deck() -> Hand:
    """Single euchre deck"""
    return make_deck(euchre_ranks, suits)


def make_pinochle_deck() -> Hand:
    """a.k.a. a double euchre deck"""
    return make_euchre_deck() + make_euchre_deck()


def make_standard_deck() -> Hand:
    """
    Standard 52 card deck
    Perfect for 52 pick-up
    """
    return make_deck(poker_ranks, suits)


@unique
class Bid(Enum):
    LOW_NO = (None, True)
    CLUBS = (Suit.CLUB, False)
    DIAMONDS = (Suit.DIAMOND, False)
    SPADES = (Suit.SPADE, False)
    HEARTS = (Suit.HEART, False)
    HI_NO = (None, False)

    def __init__(self, s: Optional[Suit], lo: bool):
        self.trump_suit: Optional[Suit] = s
        self.is_low: bool = lo


class Player:
    team: "Team"
    next_player: "Player"
    desired_trump: Bid
    shoot_strength: int

    def __init__(self, name: str, bot: int = 1):
        self.name: str = name
        self.is_bot: int = bot
        self.tricks: int = 0
        self.hand: Hand = Hand()
        self.bid_estimates: Dict[Bid, int] = {}
        self.reset_bids()
        self.card_count: Dict[Card, int] = {}

    def reset_bids(self) -> None:
        for t in Bid:
            self.bid_estimates[t] = 0

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"Player {self.name}"

    def choose_trump(self) -> Bid:
        if not self.is_bot:
            p(self.hand)  # give a closer look at your hand before bidding
            self.desired_trump = Bid[
                click.prompt(
                    "Declare Trump",
                    type=click.Choice([c for c in Bid.__members__], False),
                ).upper()
            ]
        return self.desired_trump

    def make_bid(
        self,
        valid_bids: List[int],
        d: Hand,
        handedness: int,
        min_bid: int = 0,
        leading_player: "Optional[Player]" = None,
    ) -> int:
        if self.is_bot and max(self.bid_estimates.values()) == 0:
            # simulate a hand for the bots
            for card in self.hand:
                d.remove(card)
            self.bid_estimates = {
                t: simulate_hand(
                    h_p=deepcopy(self.hand), d_p=deepcopy(d), handedness=handedness, t=t
                )
                for t in Bid
            }
        elif not self.is_bot:
            # show hand for the humans
            self.hand.sort(key=key_display4human)
            p(self.hand)
        if self.is_bot:
            # pick the biggest
            # any decisions based on the current winning bid should happen here
            bid: int = max(self.bid_estimates.values())
            self.desired_trump = random.choice(
                [k for k in self.bid_estimates.keys() if (self.bid_estimates[k] == bid)]
            )
            # don't outbid your partner (within reason)
            if leading_player in self.teammates and bid - min_bid < 2:
                return 0
            # can you do it by yourself?
            if bid == len(self.hand) - 1:
                return valid_bids[-2]  # call a shooter
            elif bid == len(self.hand):
                return valid_bids[-1]  # call a loner
            # don't bid outrageously if you don't have to
            if bid in range(min_bid, min_bid + self.shoot_strength + 2):
                return bid
            # count on two tricks from your partner
            return bid + self.shoot_strength * len(self.teammates)
        else:
            return int(
                click.prompt(
                    "How much to bid",
                    type=click.Choice(
                        ["0"] + [str(x) for x in valid_bids if (x >= min_bid)], False,
                    ),
                )
            )

    def trumpify_hand(self, trump_suit: Optional[Suit], is_lo: bool = False) -> None:
        """Marks the trump suit and sort the hands"""
        if trump_suit:
            self.hand.trumpify(trump_suit)
        self.hand.sort(
            reverse=is_lo
            if self.is_bot
            else False,  # bots have [-1] as the "best" card
            key=self.choose_sort_key(trump_suit),
        )
        self.card_count = reset_unplayed(self.hand, ts=trump_suit)

    def choose_sort_key(self, trump: Optional[Suit] = None) -> Callable:
        if not self.is_bot:
            return key_display4human
        return key_trump_power

    def play_card(self, trick_in_progress: "Trick", /, **kwargs,) -> Card:
        for x in trick_in_progress:
            self.card_count[x.card] -= 1
        c: Card = {0: pick_card_human, 1: pick_card_simple, 2: pick_card_advance}[
            self.is_bot
        ](
            follow_suit(  # valid cards
                trick_in_progress[0].card.suit if trick_in_progress else None,
                self.hand,
                strict=None,
            ),
            full_hand=self.hand,  # your hand
            trick_in_progress=trick_in_progress,  # current trick
            pl=self,
            unplayed=unwrap_cc(self.card_count),
            **kwargs,  # any other useful info
        )
        self.hand.remove(c)  # why can't .remove act like .pop?
        return c  # use discard_pile and unplayed

    @property
    def teammates(self) -> "Set[Player]":
        return self.team.players - {self}

    # the following two methods could be re-used for hearts

    def send_shooter(self, cards: int) -> List[Card]:
        if not cards:
            return []
        if self.is_bot:
            return self.hand[-cards:]
        # human
        return [
            pick_card_human(self.hand, self.hand, None, p_word="send")
            for _ in range(cards)
        ]

    def receive_shooter(self, handedness: int, bid: Bid) -> None:
        for pl in self.teammates:
            self.hand += pl.send_shooter(self.shoot_strength)
        self.hand.sort(
            key=self.choose_sort_key(), reverse=bid.is_low if self.is_bot else False
        )
        if isinstance(self.hand[-1], Card):
            return  # keep old rules without deleting code
        # discard extra cards
        for _ in range(len(self.teammates) * self.shoot_strength):
            if self.is_bot:
                self.hand.remove(
                    pick_card_advance(self.hand, is_low=not bid.is_low, pl=self)
                )
            else:
                self.hand.remove(
                    pick_card_human(self.hand, self.hand, None, p_word="discard")
                )


def simulate_hand(*, h_p: Hand, d_p: Hand, t: Bid, **kwargs) -> int:
    def slice_by_suit(h: Hand, s: Suit) -> Hand:
        return follow_suit(
            s,
            sorted(
                h.trumpify(t.trump_suit), key=key_trump_power, reverse=not t.is_low,
            ),
        )

    return sum(
        [
            len(
                estimate_tricks_by_suit(
                    my_suit=slice_by_suit(h_p, s),
                    mystery_suit=slice_by_suit(d_p, s),
                    is_low=t.is_low,
                    is_trump=(s == Suit.TRUMP),
                )
            )
            for s in suits + [Suit.TRUMP]
        ]
    )


def estimate_tricks_by_suit(
    my_suit: Iterable[Card],
    mystery_suit: Iterable[Card],
    is_low: bool,
    is_trump: Optional[bool] = False,
    strict: bool = False,
) -> Hand:
    """
    Slices up your hand and unplayed cards to estimate which suit has the most potential
    :param my_suit: list of your cards presumed of the same suit
    :param mystery_suit: unplayed cards of the suit
    :param is_low: lo no?
    :param is_trump: unused
    :param strict: True to pick a trick, False to estimate total tricks in a hand
    :return:
    """
    est = Hand()
    for rank in (
        euchre_ranks
        if is_low
        else [Rank.RIGHT_BOWER, Rank.LEFT_BOWER] + list(reversed(euchre_ranks))
    ):
        me: List[Card] = match_by_rank(my_suit, rank)
        oth: List[Card] = match_by_rank(mystery_suit, rank)
        # p((me, rank, oth))  # debugging
        est.extend(me)
        if oth and (strict or not me and not strict):
            break  # there are mystery cards that beat your cards
    return est


def pick_card_human(
    valid_cards: Hand, full_hand: Hand, trick_in_progress: "Optional[Trick]", **kwargs
) -> Card:
    if trick_in_progress is not None:
        p(trick_in_progress if trick_in_progress else "Choose the lead.")
    proper_picks: List[int] = [
        i for i in range(len(full_hand)) if full_hand[i] in valid_cards
    ]
    p("  ".join([repr(c) for c in full_hand]))
    p(
        "  ".join(
            [f"{j:2}" if j in proper_picks else "  " for j in range(len(full_hand))]
        )
    )
    return full_hand[
        int(
            click.prompt(
                f"Index of card to {kwargs.get('p_word', 'play')}",
                type=click.Choice([str(pp) for pp in proper_picks], False),
                show_choices=False,
                default=proper_picks[0] if len(proper_picks) == 1 else None,
            )
        )
    ]


def pick_card_simple(valid_cards: Hand, **kwargs,) -> Card:
    return pick_card_advance(valid_cards, **kwargs)


def pick_card_advance(valid_cards: Hand, **kwargs,) -> Card:
    pl: Player = kwargs.get("pl")
    my_hand: Hand = pl.hand
    tp: Trick = kwargs.get("trick_in_progress")
    is_low: bool = kwargs.get("is_low")
    handedness: int = kwargs.get("handedness")
    discard: Hand = kwargs.get("discard")
    unplayed: Hand = kwargs.get("unplayed", Hand())
    broken: Dict[Suit, Union[Team, None, bool]] = kwargs.get("broken_suits")
    trump: Optional[Suit] = kwargs.get("trump")

    def winning_leads(ss: List[Suit], st: bool = True) -> List[Card]:
        wl: List[Card] = []
        for s in ss:
            wl.extend(
                estimate_tricks_by_suit(
                    follow_suit(s, valid_cards, True),
                    follow_suit(s, unplayed, True),
                    is_low,
                    strict=st,
                )
            )
        return wl

    if not tp:  # you have the lead
        safer_suits: List[Suit] = [
            s for s in broken.keys() if broken[s] is False or broken[s] == pl.team
        ] if broken else suits
        w: List[Card] = []
        if safer_suits:  # unbroken suits to lead aces
            px("Checking suits")
            w += winning_leads(safer_suits)
        else:  # lead with good trump
            px("Leading with a good trump")
            w += winning_leads([Suit.TRUMP])
        if not w:  # try a risky ace
            px("Risky bet")
            w += winning_leads(suits, st=bool(pl.teammates))
        if not w and pl.teammates:  # seamless passing of the lead
            is_low = not is_low
            w += winning_leads(suits + [Suit.TRUMP], st=False)
            px("Lead pass")
        if not w:  # YOLO time
            px("YOLO")
            return random.choice(valid_cards)
        px(w)
        return random.choice(w)
    # you don't have the lead
    # win if you can (and the current lead isn't on your team)
    # play garbage otherwise
    junk_ranks: Set[Rank] = (
        {Rank.ACE_HI, Rank.KING} if is_low else {Rank.NINE, Rank.TEN, Rank.JACK}
    ) | {Rank.QUEEN}
    wc, wp = tp.winner(is_low)
    w = Hand(c for c in valid_cards if c.beats(wc, is_low))
    junk_cards = Hand(h for h in valid_cards if h not in w)
    if w:  # you have something that can win
        if wp in pl.teammates and junk_cards:  # your partner is winning
            if wc.rank in junk_ranks:  # but their card is rubbish
                return random.choice(w)
            return random.choice(junk_cards)
        return random.choice(w)
    return random.choice(junk_cards)


def match_by_rank(c: Iterable[Card], r: Rank) -> List[Card]:
    return [x for x in c if (x.rank == r)]


def key_rank_first(c: Card) -> int:
    return c.rank.v * 10 + c.suit.v


def key_suit_first(c: Card) -> int:
    return c.suit.v * 100 + c.rank.v


def key_trump_power(c: Card) -> int:
    return key_rank_first(c) + (1000 if c.suit == Suit.TRUMP else 0)


class TrickPlay(NamedTuple):
    card: Card
    played_by: Player

    def beats(self, other: "TrickPlay", is_low: bool = False):
        return self.card.beats(other.card, is_low)


class Trick(List[TrickPlay]):
    def winner(self, is_low: bool) -> Optional[TrickPlay]:
        if not self:
            return None
        w: TrickPlay = self[0]
        for cpt in self:
            if cpt.beats(w, is_low):
                w = cpt
        return w


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
        return "/".join([pl.name for pl in self.players])

    @property
    def score(self):
        return sum(self.score_changes)

    @score.setter
    def score(self, value: int):
        self.score_changes.append(value)

    def hand_tab(self, hand: Optional[int], tab: str = "\t") -> str:
        return tab.join(
            [
                str(self.bid_history[hand]),
                str(self.tricks_taken[hand]),
                str(self.score_changes[hand]),
            ]
            if hand is not None
            else [
                str(sum([1 for b in self.bid_history if b != str(None)])),
                str(sum(self.tricks_taken)),
                str(self.score),
            ]
        )


def unwrap_cc(cc: Dict[Card, int]) -> Hand:
    h = Hand()
    for card, count in cc.items():
        h.extend([card for _ in range(count)])
    return h


def reset_unplayed(
    your_hand: Optional[Hand] = None,
    deck_type: Callable[[], Hand] = make_pinochle_deck,
    duplicity: int = 2,
    ts: Optional[Suit] = None,
) -> Dict[Card, int]:
    up: Dict[Card, int] = {c: duplicity for c in deck_type().trumpify(ts)}
    if not your_hand:
        return up
    for card in your_hand:
        up[card] -= 1
    return up


class Game:
    def __init__(
        self,
        players: List[Player],
        deck_gen: Callable,
        *,
        threshold: Optional[int],
        start: str = str(datetime.now()).split(".")[0],
    ):
        self.trump: Optional[Suit] = None
        self.low_win: bool = False
        self.discard_pile: Hand = Hand()
        self.teams: Set[Team] = set([pl.team for pl in players])
        self.deck_generator: Callable = deck_gen
        self.current_dealer = players[2]  # initial dealer for 4-hands should be South
        self.handedness: int = len(players)
        self.hand_size: int = 48 // self.handedness  # adjust this for single-deck
        for i in range(self.handedness - 1):
            players[i].next_player = players[i + 1]
        players[self.handedness - 1].next_player = players[0]
        for pl in players:
            pl.shoot_strength = {3: 3, 4: 2, 6: 1, 8: 1}[self.handedness]
        self.start_time: str = start

        c = configparser.ConfigParser()
        c.read("constants.cfg")
        self.valid_bids: List[int] = [
            int(i) for i in c["ValidBids"][str(self.handedness)].split(",")
        ]
        if threshold is not None and threshold > 0:  # negative thresholds get dunked on
            self.victory_threshold: int = threshold
            self.mercy_rule: int = -threshold
            self.bad_ai_end: int = -threshold // 2
        else:
            self.victory_threshold: int = c["Scores"].getint("victory")
            self.mercy_rule: int = c["Scores"].getint("mercy")
            self.bad_ai_end: int = c["Scores"].getint("broken_ai")
        self.suit_safety: Dict[Suit, Union[None, bool, Team]] = {}
        self.reset_suit_safety()

    def reset_suit_safety(self) -> None:
        self.suit_safety = {s: False for s in suits}

    def play_hand(self) -> None:
        deck: Hand = self.deck_generator()
        random.shuffle(deck)
        hn: int = len(self.current_dealer.team.score_changes) + 1
        p(f"\nHand {hn}")
        p(f"Dealer: {self.current_dealer}")
        po: List[Player] = get_play_order(self.current_dealer, self.handedness)
        po.append(po.pop(0))  # because the dealer doesn't lead bidding
        self.reset_suit_safety()  # reset the unsafe suits
        self.discard_pile = Hand()

        # deal the cards
        idx: int = 0
        for player in po:
            player.hand = Hand(deck[idx : idx + self.hand_size])
            player.tricks = 0
            player.card_count = reset_unplayed(
                player.hand, deck_type=self.deck_generator
            )
            idx += self.hand_size
            player.reset_bids()

        # bidding
        lead: Player = bidding(po, self.valid_bids)

        # declare Trump
        trump: Bid = lead.choose_trump()
        p(f"{lead} bid {lead.team.bid} {trump.name}\n")

        # modify hands if trump called
        [player.trumpify_hand(trump.trump_suit, trump.is_low) for player in po]
        self.suit_safety[trump.trump_suit] = None

        # check for shooters and loners
        lone: Optional[Player] = None
        if lead.team.bid > self.hand_size:
            if lead.team.bid < 2 * self.hand_size:
                lead.receive_shooter(self.handedness, trump)
            lone = lead

        # play the tricks
        for _ in range(self.hand_size):
            lead = self.play_trick(lead, trump.is_low, lone)

        # calculate scores
        p(f"Hand {hn} scores:")
        for t in self.teams:
            tr_t: int = 0
            ls: int = 0
            bid: int = t.bid
            for pl in t.players:
                tr_t += pl.tricks
            if bid:
                # loners and shooters
                if lone:
                    ls = bid
                    bid = self.hand_size

                if tr_t < bid:
                    p(f"{t} got Euchred and fell {bid - tr_t} short of {bid}")
                    t.score = -bid if not ls else -bid * 3 // 2
                elif ls:
                    p(f"{lone} won all alone, the absolute madman!")
                    t.score = ls
                else:
                    p(f"{t} beat their bid of {bid} with {tr_t} tricks")
                    t.score = tr_t + 2
            else:  # tricks the non-bidding team earned
                p(f"{t} earned {tr_t} tricks")
                t.score = tr_t

            # bookkeeping
            t.bid_history.append(
                f"{ls if ls else bid} {trump.name}" if bid else str(None)
            )
            t.tricks_taken.append(tr_t)
            p(f"{t}: {t.score}")
            t.bid = 0  # reset for next time
        self.current_dealer = self.current_dealer.next_player

    def play_trick(
        self, lead: Player, is_low: bool = False, lone: Optional[Player] = None
    ) -> Player:
        pl: Player = lead
        po: List[Player] = []
        trick_in_progress: Trick = Trick()

        # play the cards
        while pl not in po:
            if lone and pl in lone.teammates:
                pl = pl.next_player
                continue
            trick_in_progress.append(
                TrickPlay(
                    pl.play_card(
                        trick_in_progress,
                        handedness=self.handedness,
                        is_low=is_low,
                        discarded=self.discard_pile,
                        broken_suits=self.suit_safety,
                        trump=self.trump,
                    ),
                    pl,
                )
            )
            for p2 in po:
                p2.card_count[trick_in_progress[-1].card] -= 1
            p(f"{pl.name} played {repr(trick_in_progress[-1].card)}")
            po.append(pl)
            pl = pl.next_player

        # find the winner
        w: TrickPlay = trick_in_progress.winner(is_low)
        l: Card = trick_in_progress[0].card
        w.played_by.tricks += 1
        p(f"{w.played_by.name} won the trick\n")
        if w.card.suit != l.suit:
            self.suit_safety[l.suit] = (
                True if self.suit_safety[l.suit] else w.played_by.team
            )
        return w.played_by

    def write_log(self, splitter: str = "\t|\t"):
        stop_time: str = str(datetime.now()).split(".")[0]
        f: TextIO = open(os.path.join(log_dir, f"{self.start_time}.gamelog"), "w")
        t_l: List[Team] = list(self.teams)  # give a consistent ordering

        def w(msg):
            click.echo(msg, f)

        # headers
        w(splitter.join([self.start_time] + [f"{t}\t\t" for t in t_l]))
        w(splitter.join([""] + ["Bid\tTricks Taken\tScore Change" for _ in t_l]))
        w(splitter.join(["Hand"] + ["===\t===\t===" for _ in t_l]))
        w(  # body
            "\n".join(
                [
                    splitter.join([f"{hand + 1}"] + [t.hand_tab(hand) for t in t_l])
                    for hand in range(len(t_l[0].bid_history))
                ]
            )
        )
        # totals
        w(splitter.join([stop_time] + ["===\t===\t===" for _ in t_l]))
        w(splitter.join(["Totals"] + [t.hand_tab(None) for t in t_l]))
        f.close()

    def victory_check(self) -> Tuple[int, Optional[Team]]:
        scorecard: List[Team] = sorted(self.teams, key=score_key)
        best_score: int = scorecard[-1].score
        if best_score < self.bad_ai_end:
            return -2, None  # everyone went too far negative
        if best_score == scorecard[-2].score:
            return 0, None  # keep playing for a tie
        if best_score > self.victory_threshold:  # a team won
            return 1, scorecard[-1]
        if scorecard[0].score < self.mercy_rule:  # a team lost
            return -1, scorecard[0]  # should never tie for last
        return 0, None

    def play(self) -> None:
        v: Tuple[int, Optional[Team]] = self.victory_check()
        global o
        while v[0] == 0:
            self.play_hand()
            v = self.victory_check()

        def final_score(pf: Callable = print):
            pf(f"\nFinal Scores")
            for t in self.teams:
                pf(f"{t}: {t.score}")
            pf(f"({len(self.current_dealer.team.bid_history)} hands)")

        final_score(p)
        if o:  # final scores to terminal
            final_score()


def score_key(t: Team) -> int:
    return t.score


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
    type=click.Choice(
        [
            "33",
            "42",
            "63",
            "62",
            # Turn off 8-handed for now
            # "82",
            # "84",
        ],
        False,
    ),
    default="42",
    help="""
    Two digits: number of players followed by number of teams.
    
    63 = six players divided into three teams of 2 players each
    """,
)
@click.option(
    "--humans",
    "-p",
    multiple=True,
    default=[],
    type=click.IntRange(0, 8),
    help="List index of a human player, repeatable",
)
@click.option(
    "--all-bots",
    type=click.BOOL,
    is_flag=True,
    help="All-bot action for testing and demos",
)
@click.option(
    "--required-points",
    "-v",
    "points",
    type=click.IntRange(4, None),
    help="""
    Victory threshold (v): positive integer
    
    team.score > v: victory
    
    team.score < -v: mercy rule loss
    
    all team scores < -v/2: everyone loses 
    """,
)
def main(
    handedness: int, humans: List[int], all_bots: bool, points: Optional[int]
) -> Game:
    handedness: int = int(handedness)
    hands: int = handedness // 10
    teams: int = handedness % 10
    ppt: int = hands // teams
    start_time: str = str(datetime.now()).split(".")[0]
    global o
    global debug
    if not humans:  # assume one human player as default
        humans = [random.randrange(hands)]
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if all_bots:
        humans = []
        o = open(os.path.join(log_dir, f"{start_time}.gameplay"), "w")
        debug = True
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
    # set up teams
    [Team([plist[i + j * teams] for j in range(ppt)]) for i in range(teams)]
    g: Game = Game(plist, make_pinochle_deck, threshold=points, start=start_time)
    g.play()
    g.write_log()
    return g


def make_players(handles: List[str]) -> List[Player]:
    c = configparser.ConfigParser()
    c.read("players.cfg")
    return [Player(c[h]["name"], int(c[h]["is_bot"])) for h in handles]


def bidding(bid_order: List[Player], valid_bids: List[int]) -> Player:
    first_round: bool = True
    count: int = 1
    hands: int = len(bid_order)
    wp: Optional[Player] = None
    wb: int = 0
    bid_order = cycle(bid_order)
    min_bid: int = min(valid_bids)
    max_bid: int = max(valid_bids)

    for pl in bid_order:
        # everyone has passed
        if count == hands:
            if first_round:  # stuck the dealer
                wb = min_bid
                p(f"Dealer {pl} got stuck with {min_bid}")
                if pl.is_bot:  # dealer picks suit
                    pl.make_bid(valid_bids, make_pinochle_deck(), hands)
                wp = pl
            else:  # someone won the bid
                wb = min_bid - 1
            break

        # end bidding early for a loner
        if min_bid > max_bid:
            wb = max_bid
            break

        # get the bid
        bid: int = pl.make_bid(valid_bids, make_pinochle_deck(), hands, min_bid, wp)

        # player passes
        if bid < min_bid:
            p(f"{pl} passes")
            count += 1
            continue

        # bid successful
        min_bid = bid + 1
        wp = pl
        count = 1
        first_round = False
        p(f"{pl} bids {bid}")

    wp.team.bid = wb
    return wp


def follow_suit(
    s: Optional[Suit], cs: Iterable[Card], strict: Optional[bool] = True
) -> Hand:
    # strict filtering
    if not s:
        return cs if isinstance(cs, Hand) else Hand(cs)
    if strict is not None:
        return Hand(c for c in cs if (c.follows_suit(s, strict)))

    # strict is None gives cards that follow suit
    if valid_cards := follow_suit(s, cs, True):
        return valid_cards
    # if valid_cards := follow_suit(s, cs, False):
    #     return valid_cards
    return follow_suit(None, cs, None)


if __name__ == "__main__":
    main()
