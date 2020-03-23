# coding=UTF-8
# -*- coding: UTF-8 -*-
# vim: set fileencoding=UTF-8 :


"""
Shared objects and functions for many card games, especially those
where you take tricks
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
    Generic,
    TypeVar,
    Type,
)
from copy import deepcopy
from datetime import datetime
import configparser
import os
import click
import abc

base_log_dir: str = "logs/"


def game_out_dir(gamename: Union[str, bytes]) -> str:
    return os.path.join(base_log_dir, gamename)


@abc.abstractmethod
def get_game_name() -> bytes:
    return os.path.basename(__file__).split(".py")[0]


def make_players(
    handles: List[str],
    player_type: "Union[Iterable[Type[PlayerType]], Type[PlayerType]]",
) -> "List[PlayerType]":
    c = configparser.ConfigParser()
    c.read("players.cfg")
    if not isinstance(player_type, Iterable):
        player_type = {player_type}
    player_type = cycle(player_type)
    return [
        pt(c[h]["name"], int(c[h]["is_bot"])) for pt, h in zip(player_type, handles)
    ]


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


class Cardstock:
    """
    Thought the name would be punny.  Think of this as BaseCard.
    """

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

    def beats(self, other: "Cardstock", is_low: bool) -> bool:
        """Assumes the first played card of equal value wins"""
        if self.suit not in {other.suit, Suit.TRUMP}:
            return False  # must follow suit
        return (self < other) if is_low else (self > other)

    def follows_suit(self, s: Suit, strict: bool = True) -> bool:
        return self.suit == s or self.suit == Suit.TRUMP and not strict


class Card(Cardstock):
    pass


CardType = TypeVar("CardType", bound=Cardstock)


def key_display4human(c: Cardstock) -> int:
    return (
        c.d_suit.opposite.v * 100 + 15
        if c.rank == Rank.LEFT_BOWER
        else c.d_suit.v * 100 + c.rank.v
    )


def match_by_rank(c: Iterable[CardType], r: Rank) -> List[CardType]:
    return [x for x in c if (x.rank == r)]


def key_rank_first(c: CardType) -> int:
    return c.rank.v * 10 + c.suit.v


def key_suit_first(c: CardType) -> int:
    return c.suit.v * 100 + c.rank.v


def key_trump_power(c: CardType) -> int:
    return key_rank_first(c) + (1000 if c.suit == Suit.TRUMP else 0)


class TrickPlay(NamedTuple):
    card: CardType
    played_by: "PlayerType"

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


class Hand(List[CardType]):
    def __str__(self) -> str:
        return "  ".join([repr(x) for x in self])

    def __add__(self, other: Union[Iterable[CardType], CardType]) -> "Hand":
        if isinstance(other, Iterable):
            self.extend(other)
        if isinstance(other, Cardstock):
            self.append(other)
        return self

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

    def __mul__(self, other) -> "Hand":
        assert isinstance(other, int)
        for _ in range(other - 1):
            self.extend(self)
        return self


def make_deck(r: List[Rank], s: List[Suit]) -> Hand:
    return Hand(Card(rank, suit) for rank in r for suit in s)


def make_euchre_deck() -> Hand:
    """Single euchre deck"""
    return make_deck(euchre_ranks, suits)


def make_pinochle_deck() -> Hand:
    """a.k.a. a double euchre deck"""
    return make_euchre_deck() * 2


def make_standard_deck() -> Hand:
    """
    Standard 52 card deck
    Perfect for 52 pick-up
    """
    return make_deck(poker_ranks, suits)


def follow_suit(
    s: Optional[Suit], cs: Iterable[CardType], strict: Optional[bool] = True
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


class WithScore:
    def __init__(self):
        self.score_changes: List[int] = []

    @property
    def score(self):
        return sum(self.score_changes)

    @score.setter
    def score(self, value: int):
        self.score_changes.append(value)


class MakesBid:
    bid: int = 0


class BasePlayer(abc.ABC):
    next_player: "PlayerType"

    def __init__(self, name: str, bot: int = 1):
        self.name: str = name
        self.is_bot: int = bot
        self.hand: Hand = Hand()

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"Player {self.name}"

    @abc.abstractmethod
    def play_card(self, *args, **kwargs) -> CardType:
        return self.hand.pop(-1)


PlayerType = TypeVar("PlayerType", bound=BasePlayer)


class BaseTeam:
    pass


TeamType = TypeVar("TeamType", bound=BaseTeam)


class TeamPlayer(BasePlayer, abc.ABC):
    team: TeamType


class BaseGame(abc.ABC):
    @abc.abstractmethod
    def play(self):
        pass


GameType = TypeVar("GameType", bound=BaseGame)
