# coding=UTF-8
# -*- coding: UTF-8 -*-
# vim: set fileencoding=UTF-8 :


"""
Shared objects and functions for many card games, especially those
where you take tricks
"""
import itertools
from enum import Enum, unique
from itertools import cycle
import random
from math import ceil
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
    Iterator,
    Any,
)
from copy import deepcopy
from datetime import datetime
import configparser
import os
import click
import abc
import inspect

base_log_dir: str = "logs/"


def game_out_dir(gamename: Union[str, bytes]) -> str:
    return os.path.join(base_log_dir, gamename)


@abc.abstractmethod
def get_game_name() -> str:
    return os.path.basename(__file__).split(".py")[0]


def p0(*msg):
    """
    dummy function for use when you don't want any output
    """
    pass


def now() -> str:
    return str(datetime.now()).split(".")[0]


def make_players(
    handles: List[str],
    player_type: "Union[Iterable[Type[PlayerType]], Type[PlayerType]]",
    calling_game: "GameType",
) -> "List[PlayerType]":
    if not isinstance(player_type, Iterable):
        player_type = {player_type}
    player_type = cycle(player_type)
    return [pt(calling_game, h) for pt, h in zip(player_type, handles)]


class Color:
    WHITE = (0, "FFF")
    BLACK = (1, "000")
    RED = (2, "F00")
    YES = (99, "00F")
    BLUE = (-5, "00F")
    INTERNATIONAL_KLEIN_BLUE = (153, "002FA7")
    GREEN = (3, "0F0")
    YELLOW = (5, "FF0")

    def __init__(self, value: int, hexa: str):
        self.v = value
        self.hex = hexa

    def __hash__(self) -> int:
        return self.v

    @property
    def hex_code(self) -> str:
        return f"#{self.hex}"


@unique
class Suit(Enum):
    JOKER_WHITE = (0, "🃏", "JOKER_WHITE", Color.WHITE)
    JOKER = (0, "🃏", "JOKER", Color.YES)
    CLUB = (1, "♣", "SPADE", Color.BLACK)
    DIAMOND = (2, "♢", "HEART", Color.RED)
    SPADE = (3, "♠", "CLUB", Color.BLACK)
    HEART = (4, "♡", "DIAMOND", Color.RED)
    TRUMP = (5, "T", "TRUMP", Color.YES)
    JOKER_RED = (0, "🃏", "JOKER_BLACK", Color.RED)
    JOKER_BLACK = (0, "🃏", "JOKER_RED", Color.BLACK)

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
    SUPER_JOKER = (17, "*")

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

    def follows_suit(self, s: Optional[Suit], strict: bool = True) -> bool:
        return s is None or (self.suit == s or self.suit == Suit.TRUMP and not strict)

    @property
    def value(self) -> int:
        return 1


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
    def winner(self, is_low: bool = False) -> Optional[TrickPlay]:
        if not self:
            return None
        w: TrickPlay = self[0]
        for cpt in self:
            if cpt.beats(w, is_low):
                w = cpt
        return w

    def __add__(self, other):
        if isinstance(other, Iterable):
            self.extend(other)
        elif isinstance(other, TrickPlay):
            self.append(other)
        return self

    @property
    def points(self) -> int:
        return 1

    @property
    def cards(self) -> "Hand":
        return Hand(c.card for c in self)

    def follow_suit(
        self, strict: bool = False, ot: "Optional[Type[Trick]]" = None
    ) -> "TrickType":
        if not ot:
            ot: Type[Trick] = Trick
        return ot(x for x in self if x.card.follows_suit(self.lead_suit, strict))

    @property
    def lead(self) -> Optional[CardType]:
        return self[0].card if self else None

    @property
    def lead_suit(self) -> Optional[Suit]:
        return self.lead.suit if self.lead else None


TrickType = TypeVar("TrickType", bound=Trick)


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
        return Hand([c for c in self for _ in range(other)])

    @property
    def points(self) -> int:
        return sum([c.value for c in self])

    @property
    def pointable(self) -> "Hand":
        """
        :return: Hand of cards with positive point values, intended for Hearts
        """
        return Hand(c for c in self if c.value > 0)

    @property
    def point_free(self) -> "Hand":
        return Hand(c for c in self if c.value < 1)

    @property
    def pcv(self) -> int:
        return self.pointable.points

    def add_jokers(self, j: int = 0, c: Type[Cardstock] = Card) -> "Hand":
        if j < 1:
            return self
        if j % 2:
            self.append(c(Rank.JOKER, random.choice([Suit.JOKER, Suit.JOKER_WHITE])))
        for _ in range(j // 2):
            self.append(c(Rank.JOKER, Suit.JOKER_RED))
            self.append(c(Rank.JOKER, Suit.JOKER_BLACK))
        return self


def make_deck(
    r: List[Rank], s: List[Suit], c: Type[Cardstock] = Card, jokers: int = 0
) -> Tuple[Hand, int]:
    return (
        Hand(c(rank, suit) for rank in r for suit in s).add_jokers(jokers, c),
        len(r) * len(s) + jokers,
    )


def make_euchre_deck(c: Type[Cardstock] = Card, jokers: int = 0) -> Tuple[Hand, int]:
    """Single euchre deck"""
    return make_deck(euchre_ranks, suits, c, jokers)


def make_pinochle_deck(c: Type[Cardstock] = Card, jokers: int = 0) -> Tuple[Hand, int]:
    """a.k.a. a double euchre deck"""
    return (make_euchre_deck(c) * 2)[0].add_jokers(jokers, c), 48 + jokers


def make_standard_deck(c: Type[Cardstock] = Card, jokers: int = 0) -> Tuple[Hand, int]:
    """
    Standard 52 card deck
    Perfect for 52 pick-up
    """
    return make_deck(poker_ranks, suits, c, jokers)


def make_minimum_sized_deck(
    m: Callable[[Type[Cardstock], int], Tuple[Hand, int]],
    c: Type[Cardstock] = Card,
    jokers: int = 0,
    minimum_size: int = 0,
    minimum_copies: int = 1,
) -> Hand:
    """
    multiples the deck's size until it is greater or equal to the minimum number of cards
    Jokers are added at the very end unless the value is negative
    """
    deck, size = m(c, 0 if jokers > 0 else -jokers)
    if minimum_size < 1:
        return deck
    multiplicity: int = max(ceil((minimum_size - jokers) / size), minimum_copies)
    return (deck * multiplicity).add_jokers(jokers)


def follow_suit(
    s: Optional[Suit],
    cs: Iterable[CardType],
    strict: Optional[bool] = True,
    allow_points: bool = True,
    ok_empty: bool = False,
    **kwargs,
) -> Hand:
    """
    :param s: suit to follow
    :param cs: cards to filter
    :param strict: count trump cards as following suit?
    :param allow_points: (for Hearts), used on first trick
                         or when leading before heartbreak
    :param ok_empty: return an empty Hand if no cards match if true
                     else return all input cards
    :return: cards that follow suit
    """

    # print(s, cs, strict, allow_points, ok_empty)  # debugging
    if not cs:
        # stop here on an empty input
        return Hand(cs)
    if strict is None:
        # activate a preset
        strict = True
        ok_empty = False
    if not allow_points:
        # filter out the pointable cards and try again
        return follow_suit(s, [c for c in cs if (c.value < 1)], strict, True, ok_empty)
    if not ok_empty:
        # take 1
        valid_cards = follow_suit(s, cs, strict, allow_points, True)
        if valid_cards:
            return valid_cards
        # try again with no suit requirements
        valid_cards = follow_suit(None, cs, strict, allow_points, True)
        if valid_cards:
            return valid_cards
        # final try where you allow pointable cards
        # this should return the whole input set
        return follow_suit(None, cs, strict, True, True)
    # return cards that follow suit
    return Hand(c for c in cs if (c.follows_suit(s, strict)))


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
        self.short_name: str = self.name[:-1].split("_")[0].lower()


class WithScore:
    def __init__(self):
        self.score_changes: List[int] = []

    @property
    def score(self) -> int:
        return sum(self.score_changes)

    @score.setter
    def score(self, value: int):
        self.score_changes.append(value)

    @abc.abstractmethod
    def hand_tab(self, hand: Optional[int], tab: str = "\t") -> str:
        """
        :param hand: hand number
        :param tab: separator
        :return: an expanded list of score changes for the hand
        """
        if hand is None:
            return str(self.score)
        return str(self.score_changes[hand])


class MakesBid:
    bid: int = 0


class BasePlayer(abc.ABC):
    next_player: "PlayerType"
    team: "TeamType"
    previous_player: "PlayerType"
    opposite_player: "Optional[PlayerType]" = None

    def __init__(self, g: "GameType", /, name: str, bot: int = 1):
        self.name: str = name
        self.is_bot: int = bot
        self.hand = Hand()
        self.in_game: GameType = g  # allow access to the game in which you're playing
        self.tricks_taken: List[Hand] = []
        self.sort_key: Callable[[CardType], int] = key_display4human
        self.passed_cards = Hand()

    @property
    def deck(self) -> Hand:
        return self.in_game.deck

    @property
    def card_count(self) -> Hand:
        """
        :return: list of remaining cards to be played
        """
        o = deepcopy(self.in_game.unplayed_cards)
        for c in self.hand:
            o.remove(c)
        return o

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"Player {self.name}"

    @abc.abstractmethod
    def pick_card(
        self, valid_cards: Hand, **kwargs,
    ):
        pass

    def play_card(self, trick_in_progress: "TrickType", /, **kwargs,) -> CardType:
        vr: Optional[Rank] = kwargs.get("valid_rank")
        c: CardType = self.pick_card(
            follow_suit(  # valid cards
                trick_in_progress.lead_suit
                if trick_in_progress
                else kwargs.get("force_suit"),
                Hand(c for c in self.hand if c.rank == vr) if vr else self.hand,
                strict=None,
                allow_points=kwargs.get("points_ok", True),
                ok_empty=False,
            ),
            trick_in_progress=trick_in_progress,  # current trick
            **kwargs,  # any other useful info
        )
        # below line is for debugging trump calls
        # print(c, self.in_game.played_cards, self.in_game.unplayed_cards)
        self.in_game.played_cards.append(c)
        self.in_game.unplayed_cards.remove(c)
        self.hand.remove(c)
        return c

    @property
    def teammates(self) -> "Set[PlayerType]":
        return self.team.players - {self}

    def send_shooter(
        self, cards: int, p_word: str = "send", prompt="Send a card to your friend."
    ) -> List[Card]:
        """
        Passes cards
        Name is from its original use in euchre
        :param cards: number of cards to send
        :param p_word: changes the word in the inline prompt for humans
        :param prompt: replaces "choose the lead" at the top of the prompt for humans
        :return: a list of cards to be sent
        """
        if not cards or not self.hand:
            return []
        out: List[CardType] = (
            self.hand[-cards:]
            if self.is_bot
            else [
                self.pick_card(self.hand, p_word=p_word, prompt=prompt)
                for _ in range(cards)
            ]
        )
        for c in out:
            self.hand.remove(c)
            self.passed_cards.append(c)
        return out

    def sort_hand(self, is_low: bool = False) -> None:
        """
        Sorts your current hand
        bots have [-1] as the "best" card
        """
        self.hand.sort(key=self.sort_key, reverse=is_low if self.is_bot else False)

    def receive_cards(
        self, cards_in: Iterable[CardType], /, *, sort_low: bool = False, **kwargs
    ) -> None:
        self.hand += cards_in
        self.sort_hand(sort_low)


PlayerType = TypeVar("PlayerType", bound=BasePlayer)


def get_play_order(lead: PlayerType) -> List[PlayerType]:
    p_c = lead
    out = []
    while p_c not in out:
        out.append(p_c)
        p_c = p_c.next_player
    return out


class BaseComputer(BasePlayer, abc.ABC):
    def __init__(self, g: "GameType", /, name: str, **kwargs):
        BasePlayer.__init__(self, g, name, 1)


class BaseHuman(BasePlayer, abc.ABC):
    def __init__(self, g: "GameType", /, name: str, **kwargs):
        BasePlayer.__init__(self, g, name, 0)

    def pick_card(self, valid_cards: Hand, **kwargs,) -> CardType:
        trick_in_progress = kwargs.get("trick_in_progress")
        if not trick_in_progress:
            print(kwargs.get("prompt", "Choose the lead."))
        proper_picks: List[int] = [
            i for i in range(len(self.hand)) if self.hand[i] in valid_cards
        ]
        print("  ".join([repr(c) for c in self.hand]))
        print(
            "  ".join(
                [f"{j:2}" if j in proper_picks else "  " for j in range(len(self.hand))]
            )
        )
        return self.hand[
            int(
                click.prompt(
                    f"Index of card to {kwargs.get('p_word', 'play')}",
                    type=click.Choice([str(pp) for pp in proper_picks], False),
                    show_choices=False,
                    default=proper_picks[0] if len(proper_picks) == 1 else None,
                )
            )
        ]

    def receive_cards(
        self,
        cards_in: Iterable[CardType],
        /,
        *,
        sort_low: bool = False,
        from_player: Optional[PlayerType] = None,
        **kwargs,
    ) -> None:
        super(BaseHuman, self).receive_cards(cards_in, sort_low=sort_low, **kwargs)
        summary: str = f"Received {Hand(cards_in)}"
        if from_player:
            summary += f" from {from_player}"
        print(summary)


class BaseTeam:
    def __init__(self, players: "Iterable[PlayerType]"):
        for player in players:
            player.team = self
        self.players: Set[PlayerType] = set(players)

    def __repr__(self):
        return "/".join([pl.name for pl in self.players])


TeamType = TypeVar("TeamType", bound=BaseTeam)


def deal(
    players: List[PlayerType],
    deck: Hand,
    minimum_kitty_size: int = 0,
    shuffled: bool = True,
    replace: bool = True,
    fixed_hand_size: Optional[int] = None,
) -> Hand:
    """
    Enforces an equal-size deal
    :param fixed_hand_size: deal this many cards instead of dealing until you run out
    :param players: list of players to whom to deal
    :param deck: the deck from which the cards are dealt
    :param shuffled: is the deck shuffled first?
    :param minimum_kitty_size: the minimum size of the kitty
    :param replace: replaces the player's hand if true; appends if false
    :return: the kitty
    """
    deck_size: int = len(deck)
    handedness = len(players)
    # for card in self.deck:  # for debugging
    #     assert card.suit != Suit.TRUMP, f"{card} {repr(card)}"
    if shuffled:
        random.shuffle(deck)
    k_size: int = minimum_kitty_size + (deck_size - minimum_kitty_size) % handedness
    k_dex: int = deck_size if k_size == 0 else -k_size
    kitty: Hand = Hand(deck[k_dex:])
    for i in range(handedness):
        p: PlayerType = players[i]
        h: Hand = deepcopy(Hand(deck[i:k_dex:handedness][:fixed_hand_size]))
        if replace:
            p.hand = h
        else:
            p.hand += h
        p.sort_hand()
    return kitty


_global_options = [
    click.option(
        "--humans",
        "-p",
        multiple=True,
        default=[],
        type=click.IntRange(0, 8),
        help="List index of a human player, repeatable",
    ),
    click.option(
        "--all-bots",
        type=click.BOOL,
        is_flag=True,
        help="All-bot action for testing and demos",
    ),
    click.option(
        "--required-points",
        "-v",
        "points",
        type=click.IntRange(1, None),
        help="""
        Victory threshold (v): positive integer

        \b
        team.score > v: victory
        team.score < -v: mercy rule loss
        all team scores < -v/2: everyone loses 
        """,
    ),
    click.option(
        "--handedness",
        "-h",
        type=click.IntRange(3, 10),
        default=4,
        help="Number of players in the game",
    ),
    click.option(
        "--team-size",
        "-t",
        type=click.IntRange(1, 5),
        help="""
        Number of players per team. 
        
        \b
        1 = normal hearts
        2 = standard 4-player euchre
        """,
    ),
    click.option(
        "--pass-size",
        "-z",
        type=click.IntRange(1, 5),
        help="(maximum) Number of cards to pass",
    ),
    click.option(
        "--minimum-hand-size",
        type=click.IntRange(1, None),
        help="Minimum size of the hand",
    ),
    click.option(
        "--add-jokers",
        type=click.IntRange(0, None),
        help="number of jokers to add to the deck",
    ),
    click.option(
        "--minimum-kitty-size",
        type=click.IntRange(0, None),
        help="minimum number of cards in the kitty",
    ),
    click.option(
        "--deck-replication",
        type=click.IntRange(1, None),
        help="Minimum number of times the deck gets replicated",
    ),
]


def common_options(func):
    return add_options(func, *_global_options)


def add_options(func, *options):
    for o in options:
        func = o(func)
    return func


def make_and_play_game(
    game: "Type[BaseGame]",
    logging_directory: Optional[str],
    start_time: Optional[str] = None,
    **kwargs,
):
    start_time = start_time if start_time else now()
    g = game(start=start_time, **kwargs)
    g.play()
    g.write_log(logging_directory)


class BaseGame(abc.ABC):
    def __init__(
        self,
        *,
        deck_generator: Callable[
            [Type[Cardstock], int], Tuple[Hand, int]
        ] = make_standard_deck,
        human_player_type: Type[BaseHuman],
        computer_player_type: Type[BaseComputer],
        card_type: Type[Cardstock] = Card,
        team_type: Type[BaseTeam],
        game_name: str,
        force_multiple_teams: bool = True,
        handedness: int = 4,
        team_size: int = 1,
        points: int,
        fixed_hand_size: Optional[int] = None,
        start: str = now(),
        pass_size: int = None,
        add_jokers: int = None,
        minimum_kitty_size: int = 0,
        minimum_hand_size: int = 10,
        deck_replication: int = 1,
        single_human_name: str = "You",
        all_bots: bool = False,
        humans: List[int] = None,
        shuffle_deck: bool = True,
        **kwargs,
    ):
        """
        The basic setup for a trick-taking card game.
        Most of these params should either be from the Click kwargs or
        left at their default values rather than parsing them in the
        constructor of child classes.

        :param deck_generator: function that generates the deck of cards
        :param human_player_type: Class of human players
        :param computer_player_type: Class for computer players
        :param card_type: Class of card to use
        :param team_type: Class of team
        :param game_name: string of the game's name, should be the same as the filename
        :param force_multiple_teams: make sure all the players aren't on the same team
        :param handedness: number of players in the game
        :param team_size: number of players on each team
        :param points: victory threshold
        :param single_human_name: the name override for games with one human player
        :param start: string of game start time, used for logging
        :param pass_size: number of cards to pass in passing games
        :param add_jokers: number of jokers to add to the deck
        :param fixed_hand_size: fix hand size instead of dealing all available cards
        :param minimum_kitty_size: minimum number of leftover cards on each deal
        :param minimum_hand_size: minimum hand size, determines deck replication
        :param deck_replication: at least this many copies of the deck are created
        :param all_bots: set True for all-bot testing action
        :param humans: index locations of human players
        :param shuffle_deck: shuffle before dealing?
        :param kwargs: additional arguments to alter the game's setup and behavior
        """

        """
        Basic setup and checks
        """
        self.handedness: int = 4 if handedness is None else handedness
        # sanity checking
        if not team_size:
            team_size = 1
        assert (
            self.handedness % team_size == 0
        ), f"{self.handedness} players can't divide into teams of {team_size}"
        if force_multiple_teams:
            assert (  # also handles negative team sizes
                self.handedness // team_size > 1
            ), f"There's no game if everyone is on the same team"

        # constants
        self.victory_threshold: int = points
        self.start_time: str = start if start else now()
        c = configparser.ConfigParser()
        c.read("constants.cfg")

        def g_h() -> str:
            return f"{game_name.capitalize()}-{self.handedness}"

        if pass_size is None:
            try:
                pass_size = c.getint("Shoot Strength", g_h())
            except configparser.NoOptionError:
                try:
                    pass_size = c.getint("Shoot Strength", game_name)
                except configparser.NoOptionError:
                    pass_size = c.getint("Shoot Strength", "Default")
        self.pass_size: int = pass_size if pass_size else 0

        """
        Make the deck
        """
        # the initial constant configuration
        self.kitty: Hand = Hand()
        j = 0 if add_jokers is None else 0
        self.minimum_kitty_size = (
            0 if minimum_kitty_size is None else minimum_kitty_size
        )

        # actually make the deck
        self.deck = make_minimum_sized_deck(
            deck_generator,
            card_type,
            j,
            self.handedness * (minimum_hand_size if minimum_hand_size else 1),
            deck_replication if deck_replication else 1,
        )

        # calculate hand sizes and card counts
        self.hand_size: int = fixed_hand_size if fixed_hand_size else (
            len(self.deck) - self.minimum_kitty_size
        ) // self.handedness
        self.fhs: Optional[int] = fixed_hand_size
        self.suit_safety: Dict[Suit, Union[None, bool, TeamType, PlayerType]] = {}
        self.reset_suit_safety()

        """
        Setup players and teams
        """
        # Get names
        try:  # preset name schema
            player_names = c["Names"][g_h()].strip().split("\n")
        except KeyError:  # names randomly from a list
            pph = c["Names"]["Grand Name Gallery"].strip().split("\n")
            est: str = f"{self.handedness} is way too many players!\n"
            est += "Edit the grand name gallery in constants.cfg to have "
            est += f"at least {self.handedness-len(pph)} more names"
            assert self.handedness <= len(pph), est
            player_names = pph[
                (j := random.randrange(len(pph) - self.handedness)) : j
                + self.handedness
            ]

        # calculate player types
        if not humans:  # assume one human player as default
            humans = [random.randrange(self.handedness)]
        if all_bots:
            humans = []
        if len(humans) == 1 and humans[0] < self.handedness:
            player_names[humans[0]] = single_human_name if single_human_name else "You"
        self.players: List[PlayerType] = make_players(
            player_names,
            [
                (human_player_type if i in humans else computer_player_type)
                for i in range(self.handedness)
            ],
            self,
        )

        # make teams
        self.teams: Set[TeamType] = {
            team_type(self.players[i :: self.handedness // team_size])
            for i in range(self.handedness // team_size)
        }
        # set up player rotation
        for i in range(self.handedness):
            nxt: PlayerType = self.players[(i + 1) % self.handedness]
            self.players[i].next_player = nxt
            nxt.previous_player = self.players[i]
            self.players[i].opposite_player = (
                self.players[(i + self.handedness // 2) % self.handedness]
                if self.handedness % 2 == 0
                else None
            )
        # initial dealer for 4-hands should be South
        self.current_dealer: PlayerType = self.players[2 % self.handedness]
        self.played_cards = Hand()
        self.unplayed_cards = deepcopy(self.deck)
        self.shuffle_deck: bool = shuffle_deck if shuffle_deck is not None else True

    def reset_suit_safety(self) -> None:
        self.suit_safety = {s: False for s in suits}

    def deal(
        self, *, minimum_kitty_size: int = None, shuffled: bool = None, **kwargs,
    ) -> Hand[CardType]:

        # kwargs nonsense
        if minimum_kitty_size is None:
            minimum_kitty_size = self.minimum_kitty_size
        if shuffled is None:
            shuffled = self.shuffle_deck

        # actually deal
        self.kitty = deal(
            self.players,
            self.deck,
            minimum_kitty_size,
            shuffled,
            True,
            self.hand_size if self.fhs else None,
        )

        # stuff for card counting
        self.reset_suit_safety()
        self.played_cards = Hand()
        self.unplayed_cards = deepcopy(self.deck)

        # we're all done!
        return self.kitty

    @abc.abstractmethod
    def play(self):
        pass

    @abc.abstractmethod
    def play_hand(self, dealer: PlayerType) -> PlayerType:
        pass

    @abc.abstractmethod
    def play_trick(self, lead: PlayerType) -> PlayerType:
        pass

    @abc.abstractmethod
    def victory_check(self) -> Tuple[int, Optional[WithScore]]:
        pass

    @abc.abstractmethod
    def write_log(self, ld: str, splitter: str = "\t|\t") -> None:
        stop_time: str = now()
        f: TextIO = open(os.path.join(ld, f"{self.start_time}.gamelog"), "w")
        t_l: List[BaseTeam] = list(self.teams)  # give a consistent ordering

        def w(msg):
            click.echo(msg, f)

        # Headers
        # Body
        # Conclusion

        f.close()


def make_null_deck(c: Type[Cardstock]) -> Hand:
    return Hand()


class TestGame(BaseGame, abc.ABC):
    """
    A test harness to give players a predefined deal
    Use me to check for consistency in rare behavior
    """

    p_type: Type[BaseComputer]

    def __init__(
        self,
        team_size: int,
        card_type: Type[Cardstock],
        team_type: Type[BaseTeam],
        preset_hands: List[Hand],
        **kwargs,
    ):
        self.handedness = len(preset_hands)
        super().__init__(
            [str(x) for x in range(self.handedness)],
            [self.p_type for _ in range(self.handedness)],
            team_size,
            card_type,
            deck_generator=make_null_deck,
            team_type=team_type,
            victory_threshold=0,
        )
        for p in preset_hands:
            self.deck.extend(p)
        self.p_hands = preset_hands
        self.kitty = kwargs.get("kitty", Hand())

    def victory_check(self) -> Tuple[int, Optional[WithScore]]:
        return 1, None

    def write_log(self, ld: str, splitter: str = "\t|\t") -> None:
        pass

    def deal(
        self, minimum_kitty_size: int = 0, shuffled: bool = False, **kwargs
    ) -> Hand[CardType]:
        for i in range(self.handedness):
            self.players[i].hand = self.p_hands[i]
        return self.kitty()


GameType = TypeVar("GameType", bound=BaseGame)


def is_prime(n: int):
    """
    Primality test
    from https://stackoverflow.com/questions/15285534/
    """
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n ** 0.5)
    f = 5
    while f <= r:
        if n % f == 0:
            return False
        if n % (f + 2) == 0:
            return False
        f += 6
    return True


class PassPacket:
    """
    This is a one-to-one player to card ratio
    """

    def __init__(
        self,
        from_player: BasePlayer,
        direction: Callable,
        *,
        c: Optional[CardType] = None,
        to_player: Optional[PlayerType] = None,
        **kwargs,
    ):
        self.from_player = from_player
        if direction == pass_shoot:
            assert to_player is not None
        self.direction = direction
        self.card: CardType = c if c else type(from_player.hand[0])(
            Rank.JOKER, Suit.JOKER
        )
        self.specified_player = to_player
        self.low_sort = kwargs.get("sort_low", False)

    def __repr__(self):
        return f"PassPacket {self.card} from {self.from_player} to {self.to_player}"

    @property
    def to_player(self) -> PlayerType:
        return (
            self.specified_player
            if self.specified_player
            else {
                pass_left: self.from_player.next_player,
                pass_right: self.from_player.previous_player,
                pass_across: self.from_player.opposite_player
                if self.from_player.opposite_player
                else self.from_player,
                pass_hold: self.from_player,
            }[self.direction]
        )

    def collect_card(self) -> CardType:
        self.card = self.from_player.send_shooter(
            1,
            p_word=self.direction.__name__.replace("_", " "),
            prompt=f"Send a card to {self.to_player}",
        )[0]
        return self.card

    def send_card(self) -> None:
        self.to_player.receive_cards(
            [self.card], sort_low=self.low_sort, from_player=self.from_player
        )


class PassList(List[PassPacket]):
    def __init__(
        self,
        playlist: List[PlayerType],
        directions: List[Callable],
        specific_destination: Optional[Iterable[PlayerType]] = None,
        **kwargs,
    ):
        if specific_destination and not isinstance(specific_destination, Iterator):
            specific_destination = cycle(specific_destination)
        super().__init__(
            [
                PassPacket(
                    p,
                    d,
                    to_player=next(specific_destination)
                    if specific_destination
                    else None,
                    **kwargs,
                )
                for p, d in itertools.product(playlist, directions)
                if d != pass_hold
            ]
        )

    def collect_cards(self) -> None:
        [p.collect_card() for p in self]

    def distribute_cards(self) -> None:
        [p.send_card() for p in self]


def pass_cards(
    playlist: List[PlayerType], directions: List[Callable], kitty: Hand,
) -> Hand:
    if pass_kitty in directions:
        return pass_kitty(playlist, kitty, len(directions))
    where_to = PassList(playlist, directions)
    where_to.collect_cards()
    # this is a separate step so that players may not re-pass a card
    where_to.distribute_cards()
    # should be a clean pass-through, added for simpler function signature
    return kitty


def pass_kitty(
    playlist: List[PlayerType],
    kitty: Hand,
    strength: int = 3,
    minimum_kitty_size: int = 0,
) -> Hand:
    for p in playlist:
        kitty += p.send_shooter(strength, prompt="Send a card to the kitty.")
    kitty = deal(playlist, kitty, replace=False, minimum_kitty_size=minimum_kitty_size)
    return kitty


def pass_left():
    return "left"


def pass_right():
    return "right"


def pass_across():
    return "across"


def pass_hold():
    return "hold"


def pass_shoot():
    return "shoot"


pass_order_all = [
    pass_left,
    pass_right,
    pass_across,
    pass_hold,
    pass_kitty,
]
pass_order_ms_hearts_exe = pass_order_all[:-1]
pass_order_no_hold_even = pass_order_all[:3]
pass_order_no_hold_odd = pass_order_all[:2]


def pass_n(p: Callable, /) -> str:
    return p.__name__.split("_")[-1]


def pass2s(p: Callable, /) -> str:
    if p == pass_hold:
        return "to yourself"
    if p == pass_shoot:
        return "to your partner"
    return ("" if p == pass_across else "to the ") + pass_n(p)


def pass_choice_display_list(pc: List[Callable], /) -> Dict[Union[str, chr], Callable]:
    long: Dict[str, Callable] = {pass_n(p): p for p in pc}
    s: Dict[chr, Callable] = {k[0]: long[k] for k in long}
    return {**long, **s}
