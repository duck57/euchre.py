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


def make_players(
    handles: List[str],
    player_type: "Union[Iterable[Type[PlayerType]], Type[PlayerType]]",
    deck: "Hand",
) -> "List[PlayerType]":
    c = configparser.ConfigParser()
    c.read("players.cfg")
    if not isinstance(player_type, Iterable):
        player_type = {player_type}
    player_type = cycle(player_type)
    return [pt(h, deck) for pt, h in zip(player_type, handles)]


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

    def follows_suit(self, s: Suit, strict: bool = True) -> bool:
        return self.suit == s or self.suit == Suit.TRUMP and not strict

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
        return ot(x for x in self if x.card.follows_suit(self[0].card.suit, strict))


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
        for _ in range(other - 1):
            self.extend(self)
        return self


def make_deck(r: List[Rank], s: List[Suit], c: Type[Cardstock] = Card) -> Hand:
    return Hand(c(rank, suit) for rank in r for suit in s)


def make_euchre_deck(c: Type[Cardstock] = Card) -> Hand:
    """Single euchre deck"""
    return make_deck(euchre_ranks, suits, c)


def make_pinochle_deck(c: Type[Cardstock] = Card) -> Hand:
    """a.k.a. a double euchre deck"""
    return make_euchre_deck(c) * 2


def make_standard_deck(c: Type[Cardstock] = Card) -> Hand:
    """
    Standard 52 card deck
    Perfect for 52 pick-up
    """
    return make_deck(poker_ranks, suits, c)


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
        self.short_name: str = self.name[:-1].split("_")[0].lower()


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
    team: "TeamType"
    previous_player: "PlayerType"
    opposite_player: "Optional[PlayerType]" = None

    def __init__(self, name: str, bot: int = 1, deck: Optional[Hand] = None):
        if not deck:
            deck = Hand()
        self.name: str = name
        self.is_bot: int = bot
        self.hand: Hand = Hand()
        self.deck: Hand = deepcopy(deck)
        self.card_count: Hand = deepcopy(deck)
        self.tricks_taken: List[TrickType] = []
        self.sort_key: Callable[[CardType], int] = key_display4human

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
        try:
            for x in trick_in_progress:
                self.card_count.remove(x.card)
        except ValueError:
            # This works fine with euchre.py
            # Print is commented out for Hearts
            # boosted nines create problems with this when playing Hearts
            # print(f"{self}: {trick_in_progress.cards} not in {self.card_count}")
            pass
        return self.pick_card(
            follow_suit(  # valid cards
                trick_in_progress[0].card.suit if trick_in_progress else None,
                self.hand,
                strict=None,
            ),
            full_hand=self.hand,  # your hand
            trick_in_progress=trick_in_progress,  # current trick
            pl=self,
            unplayed=self.card_count,
            **kwargs,  # any other useful info
        )

    @property
    def teammates(self) -> "Set[PlayerType]":
        return self.team.players - {self}

    def send_shooter(
        self, cards: int, p_word: str = "send", prompt="Send a card to your friend."
    ) -> List[Card]:
        if not cards:
            return []
        out = (
            self.hand[-cards:]
            if self.is_bot
            else [
                self.pick_card(self.hand, p_word=p_word, prompt=prompt)
                for _ in range(cards)
            ]
        )
        self.card_count += out
        for c in out:
            try:
                self.hand.remove(c)
            except ValueError:
                print(f"{self}: {c} not found in {self.hand}")
        return out

    def sort_hand(self, is_low: bool = False) -> None:
        """
        Sorts your current hand
        bots have [-1] as the "best" card
        """
        self.hand.sort(key=self.sort_key, reverse=is_low if self.is_bot else False)

    def receive_cards(self, cards_in: Iterable[CardType]) -> None:
        self.hand += cards_in
        self.sort_hand()
        # print(self, cards_in, "\n", self.card_count)
        for c in cards_in:
            self.card_count.remove(c)

    def reset_unplayed(self, ts: Optional[Suit] = None) -> Hand:
        dk: Hand = deepcopy(self.deck)
        dk.trumpify(ts)
        dk.sort(key=key_display4human)
        for card in self.hand:
            dk.remove(card)
        self.card_count = dk
        return dk

    def pass_left(self, c: Union[CardType, List[CardType]]) -> None:
        self.next_player.receive_cards(list(c))

    def pass_right(self, c: Union[CardType, List[CardType]]) -> None:
        self.previous_player.receive_cards(list(c))

    def pass_across(self, c: Union[CardType, List[CardType]]) -> None:
        if self.opposite_player:
            self.opposite_player.receive_cards(list(c))
        else:
            self.pass_hold(c)

    def pass_hold(self, c: Union[CardType, List[CardType]]) -> None:
        self.receive_cards(list(c))


PlayerType = TypeVar("PlayerType", bound=BasePlayer)


def get_play_order(lead: PlayerType) -> List[PlayerType]:
    p_c = lead
    out = []
    while p_c not in out:
        out.append(p_c)
        p_c = p_c.next_player
    return out


class BaseComputer(BasePlayer, abc.ABC):
    pass


class BaseHuman(BasePlayer, abc.ABC):
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
        return self.hand.pop(
            int(
                click.prompt(
                    f"Index of card to {kwargs.get('p_word', 'play')}",
                    type=click.Choice([str(pp) for pp in proper_picks], False),
                    show_choices=False,
                    default=proper_picks[0] if len(proper_picks) == 1 else None,
                )
            )
        )


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
) -> Hand:
    """
    Enforces an equal-size deal
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
    k_dex: Optional[int] = deck_size if k_size == 0 else -k_size
    kitty: Hand = Hand(deck[k_dex:])
    for i in range(handedness):
        h: Hand = deepcopy(Hand(deck[i:k_dex:handedness]))
        if replace:
            players[i].hand = h
        else:
            players[i].hand += h
        players[i].reset_unplayed()
    return kitty


class BaseGame(abc.ABC):
    def __init__(
        self,
        player_names: List[str],
        player_types: List[Type[BasePlayer]],
        team_size: int,
        card_type: Type[Cardstock],
        deck_generator: Callable[[Type[Cardstock]], Hand],
        team_type: Type[BaseTeam],
        victory_threshold: int,
        start: str = str(datetime.now()).split(".")[0],
    ):
        # constants
        self.victory_threshold = victory_threshold
        self.start_time: str = start
        # make the deck
        self.kitty: Hand[CardType] = Hand()
        self.deck: Hand[CardType] = deck_generator(card_type)
        self.suit_safety: Dict[Suit, Union[None, bool, TeamType]] = {}
        self.reset_suit_safety()
        # create players and teams
        self.players: List[PlayerType] = make_players(
            player_names, player_types, self.deck,
        )
        self.handedness: int = len(self.players)
        assert (
            self.handedness % team_size == 0 and self.handedness // team_size > 1
        ), f"{self.handedness} players can't divide into teams of {team_size}"
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
            self.players[i].deck = deepcopy(self.deck)
        # initial dealer for 4-hands should be South
        self.current_dealer: PlayerType = self.players[2]

    def reset_suit_safety(self) -> None:
        self.suit_safety = {s: False for s in suits}

    def deal(
        self, minimum_kitty_size: int = 0, shuffled: bool = True
    ) -> Hand[CardType]:
        self.kitty = deal(self.players, self.deck, minimum_kitty_size, shuffled, True)
        # print([f"{p} {p.card_count}" for p in self.players])
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
        stop_time: str = str(datetime.now()).split(".")[0]
        f: TextIO = open(os.path.join(ld, f"{self.start_time}.gamelog"), "w")
        t_l: List[BaseTeam] = list(self.teams)  # give a consistent ordering

        def w(msg):
            click.echo(msg, f)

        # Headers
        # Body
        # Conclusion

        f.close()


GameType = TypeVar("GameType", bound=BaseGame)


def is_prime(n: int):
    """
    Primality test
    from https://stackoverflow.com/questions/15285534/isprime-function-for-python-language
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


def pass_cards(
    playlist: List[PlayerType], directions: List[Callable], kitty: Hand,
) -> None:
    if pass_kitty in directions:
        return pass_kitty(playlist, kitty, len(directions))
    where_to: List[Tuple[PlayerType, Callable]] = [
        x for x in itertools.product(playlist, directions) if x[1] != pass_hold
    ]
    # collect cards
    cards: List[CardType] = [
        p.send_shooter(1, a.__name__.replace("_", " ")) for p, a in where_to
    ]
    # pass the cards
    for i in range(len(cards)):
        getattr(where_to[i][0], where_to[i][1].__name__)(cards[i])


def pass_kitty(
    playlist: List[PlayerType],
    kitty: Hand,
    strength: int = 3,
    minimum_kitty_size: int = 0,
) -> None:
    for p in playlist:
        kitty += p.send_shooter(strength)
    kitty = deal(playlist, kitty, replace=False, minimum_kitty_size=minimum_kitty_size)


pass_left = BasePlayer.pass_left
pass_right = BasePlayer.pass_right
pass_across = BasePlayer.pass_across
pass_hold = BasePlayer.pass_hold
