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
from itertools import cycle, islice
import random
from typing import (
    List,
    Optional,
    Tuple,
    Iterable,
    Iterator,
    Set,
    Callable,
    Dict,
    NamedTuple,
)
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


class Hand(List[Card]):
    def __str__(self) -> str:
        return "  ".join([repr(x) for x in self])

    def __add__(self, other) -> "Hand":
        return Hand(list(self) + list(other))

    def trumpify(self, trump_suit: Suit):
        for card in self:
            if card.suit == trump_suit:
                card.suit = Suit.TRUMP
                if card.rank == Rank.JACK:
                    card.rank = Rank.RIGHT_BOWER
            if card.rank == Rank.JACK and card.suit == trump_suit.opposite:
                card.suit = Suit.TRUMP
                card.rank = Rank.LEFT_BOWER


# make a euchre deck
def make_euchre_deck() -> Hand:
    return Hand(Card(rank, suit) for rank in euchre_ranks for suit in suits)


# a.k.a. a double euchre deck
def make_pinochle_deck() -> Hand:
    return make_euchre_deck() + make_euchre_deck()


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

    def reset_bids(self) -> None:
        for t in Bid:
            self.bid_estimates[t] = 0

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"Player {self.name}"

    def choose_trump(self) -> Bid:
        if not self.is_bot:
            print(self.hand)  # give a closer look at your hand before bidding
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
            print(self.hand)
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

    def choose_sort_key(self, trump: Optional[Suit]) -> Callable:
        if not self.is_bot:
            return key_display4human
        return key_trump_power

    def play_card(self, trick_in_progress: "List[TrickPlay]", /, **kwargs,) -> Card:
        c: Card = {0: pick_card_human, 1: pick_card_simple, 2: pick_card_advance}[
            self.is_bot
        ](
            follow_suit(  # valid cards
                trick_in_progress[0].card.suit if trick_in_progress else None,
                self.hand,
                strict=None,
            ),
            self.hand,  # your hand
            trick_in_progress,  # current trick
            **kwargs,  # any other useful info
        )
        self.hand.remove(c)  # why can't .remove act like .pop?
        return c  # use discard_pile and unplayed

    @property
    def teammates(self) -> "Set[Player]":
        return self.team.players - {self}

    def send_shooter(self, cards: int) -> List[Card]:
        if not cards:
            return []
        if self.is_bot:
            return self.hand[-cards:]
        # human
        # display hand
        # prompt for card indices until you have enough
        return self.hand[-cards:]

    def receive_shooter(self, handedness: int, trump: Suit) -> None:
        for p in self.teammates:
            self.hand += p.send_shooter(self.shoot_strength)
        # discard extra cards
        for _ in range(len(self.teammates) * self.shoot_strength):
            pass
        pass


def simulate_hand(*, h_p: Hand, d_p: Hand, t: Bid, handedness: int) -> int:
    # you: int = 0
    if t.trump_suit:
        h_p.trumpify(t.trump_suit)
        d_p.trumpify(t.trump_suit)
    # largest cards first this time
    h_p.sort(key=key_trump_power, reverse=not t.is_low)
    d_p.sort(key=key_trump_power, reverse=not t.is_low)
    # my_trump: Hand = follow_suit(Suit.TRUMP, h_p)
    # my_other: Hand = [c for c in h_p if (c.suit != Suit.TRUMP)]
    # mystery_trump: Hand = follow_suit(Suit.TRUMP, d_p)
    # mystery_other: Hand = [c for c in d_p if (c.suit != Suit.TRUMP)]

    return sum(
        [
            estimate_tricks_by_suit(
                my_suit=follow_suit(s, h_p),
                others=follow_suit(s, d_p),
                is_low=t.is_low,
                is_trump=(s == Suit.TRUMP),
            )
            for s in suits + [Suit.TRUMP]
        ]
    )

    # # simulate a hand
    # while h_p:
    #     # pick the smallest card that beats everything else, if any
    #     gain_trick: bool = False
    #     lead_suit: Suit = Suit.JOKER
    #     for card in h_p:
    #         best_opp: Card = follow_suit(card.suit, d_p, None)[-1]
    #         if not best_opp.beats(card, t.is_low):
    #             gain_trick = True
    #         if gain_trick:
    #             h_p.remove(card)
    #             lead_suit = card.suit
    #             you += 1
    #             break
    #     if not gain_trick:
    #         lead_suit = random.choice([Suit.HEART, Suit.SPADE, Suit.DIAMOND, Suit.CLUB])
    #         others += 1
    #         h_p.remove(follow_suit(lead_suit, h_p, None)[0])
    #     fs_cards: Hand = follow_suit(lead_suit, d_p, gain_trick)
    #     fixed_count: int = 0
    #     if fs_cards:
    #         d_p.remove(fs_cards.pop(-1))  # the best card was played
    #         fixed_count += 1
    #     if fs_cards:
    #         d_p.remove(fs_cards.pop(random.randrange(len(fs_cards))))
    #         fixed_count += 1
    #     for _ in range(handedness - 1 - fixed_count):  # best card and player are out
    #         d_p.pop(random.randrange(len(d_p)))
    # return you


def estimate_tricks_by_suit(
    my_suit: Hand, others: Hand, is_low: bool, is_trump: bool
) -> int:
    est: int = 0
    for rank in (
        euchre_ranks
        if is_low
        else ([Rank.RIGHT_BOWER, Rank.LEFT_BOWER] if is_trump else [])
        + list(reversed(euchre_ranks))
    ):
        count: int = len([x for x in my_suit if (x.rank == rank)])
        est += count
        if not count:
            break
    return est


def pick_card_human(
    valid_cards: Hand, full_hand: Hand, trick_in_progress: "List[TrickPlay]", **kwargs
) -> Card:
    print(trick_in_progress if trick_in_progress else "Choose the lead.")
    proper_picks: List[int] = [
        i for i in range(len(full_hand)) if full_hand[i] in valid_cards
    ]
    print("  ".join([repr(c) for c in full_hand]))
    print(
        "  ".join(
            [f"{j:2}" if j in proper_picks else "  " for j in range(len(full_hand))]
        )
    )
    return full_hand[
        int(
            click.prompt(
                "Index of card to play",
                type=click.Choice([str(p) for p in proper_picks], False),
                show_choices=False,
                default=proper_picks[0] if len(proper_picks) == 1 else None,
            )
        )
    ]


def pick_card_simple(
    valid_cards: Hand, full_hand: Hand, trick_in_progress: "List[TrickPlay]", **kwargs,
) -> Card:
    return pick_card_advance(valid_cards, full_hand, trick_in_progress, **kwargs)


def pick_card_advance(
    valid_cards: Hand, full_hand: Hand, trick_in_progress: "List[TrickPlay]", **kwargs,
) -> Card:
    return valid_cards[-1]


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
        return "/".join([p.name for p in self.players])

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


class Game:
    def __init__(
        self, players: List[Player], deck_gen: Callable, *, threshold: Optional[int]
    ):
        self.trump: Optional[Suit] = None
        self.low_win: bool = False
        self.discard_pile: Hand = Hand()
        self.unplayed_cards: Hand = Hand()
        self.teams: Set[Team] = set([p.team for p in players])
        self.deck_generator: Callable = deck_gen
        self.current_dealer = players[2]  # initial dealer for 4-hands should be South
        self.handedness: int = len(players)
        self.hand_size: int = 48 // self.handedness  # adjust this for single-deck
        for i in range(self.handedness - 1):
            players[i].next_player = players[i + 1]
        players[self.handedness - 1].next_player = players[0]
        for p in players:
            p.shoot_strength = {3: 3, 4: 2, 6: 1, 8: 1}[self.handedness]

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
        self.unsafe_suits: Set[Suit] = set()

    def play_hand(self) -> None:
        deck: Hand = self.deck_generator()
        random.shuffle(deck)
        hn: int = len(self.current_dealer.team.score_changes) + 1
        print(f"\nHand {hn}")
        print(f"Dealer: {self.current_dealer}")
        po: List[Player] = get_play_order(self.current_dealer, self.handedness)
        po.append(po.pop(0))  # because the dealer doesn't lead bidding
        self.unsafe_suits -= self.unsafe_suits  # reset the unsafe suits

        # deal the cards
        idx: int = 0
        for player in po:
            player.hand = Hand(deck[idx : idx + self.hand_size])
            player.tricks = 0
            idx += self.hand_size
            player.reset_bids()

        # bidding
        lead: Player = bidding(po, self.valid_bids)

        # declare Trump
        trump: Bid = lead.choose_trump()
        print(f"{lead} bid {lead.team.bid} {trump.name}\n")

        # modify hands if trump called
        [player.trumpify_hand(trump.trump_suit, trump.is_low) for player in po]

        # play the tricks
        for _ in range(self.hand_size):
            lead = self.play_trick(lead, trump.is_low)

        # calculate scores
        print(f"Hand {hn} scores:")
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
                print(f"{t} earned {tr_t} tricks")
                t.score = tr_t

            # bookkeeping
            t.bid_history.append(
                f"{ls if ls else bid} {trump.name}" if bid else str(None)
            )
            t.tricks_taken.append(tr_t)
            print(f"{t}: {t.score}")
            t.bid = 0  # reset for next time
        self.current_dealer = self.current_dealer.next_player

    def play_trick(self, lead: Player, is_low: bool = False) -> Player:
        p: Player = lead
        po: List[Player] = []
        trick_in_progress: List[TrickPlay] = []

        # play the cards
        while p not in po:
            po.append(p)
            trick_in_progress.append(
                TrickPlay(
                    p.play_card(
                        trick_in_progress,
                        handedness=self.handedness,
                        is_low=is_low,
                        discarded=self.discard_pile,
                        unplayed=self.unplayed_cards,
                        broken_suits=self.unsafe_suits,
                    ),
                    p,
                )
            )
            print(f"{p.name} played {repr(trick_in_progress[-1].card)}")
            p = p.next_player

        # find the winner
        w: TrickPlay = trick_in_progress[0]
        l: Card = w.card
        for cpt in trick_in_progress:
            if cpt.beats(w, is_low):
                w = cpt
        w.played_by.tricks += 1
        print(f"{w.played_by.name} won the trick\n")
        if w.card.suit != l.suit:
            self.unsafe_suits |= {l.suit}
        return w.played_by

    def write_log(self, splitter: str = "\t|\t"):
        f = open(f"{str(datetime.now()).split('.')[0]}.gamelog", "w")
        t_l: List[Team] = list(self.teams)  # give a consistent ordering

        # headers
        f.write(splitter.join([""] + [f"{t}\t\t" for t in t_l]))
        f.write(
            splitter.join(["\n"] + ["Bid\tTricks Taken\tScore Change" for _ in t_l])
        )
        f.write(splitter.join(["\nHand"] + ["===\t===\t===" for _ in t_l]))

        # body
        f.write(
            "\n"
            + "\n".join(
                [
                    splitter.join([f"{hand+1}"] + [t.hand_tab(hand) for t in t_l])
                    for hand in range(len(t_l[0].bid_history))
                ]
            )
        )

        # totals
        f.write(splitter.join(["\n"] + ["===\t===\t===" for _ in t_l]))
        f.write(splitter.join(["\nTotals"] + [t.hand_tab(None) for t in t_l]))
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
        while v[0] == 0:
            self.play_hand()
            v = self.victory_check()
        print(f"\nFinal Scores")
        for t in self.teams:
            print(f"{t}: {t.score}")


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
    if not humans:  # assume one human player as default
        humans = [random.randrange(hands)]
    if all_bots:
        humans = []
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
    g: Game = Game(plist, make_pinochle_deck, threshold=points)
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

    for p in bid_order:
        # everyone has passed
        if count == hands:
            if first_round:  # stuck the dealer
                wb = min_bid
                print(f"Dealer {p} got stuck with {min_bid}")
                if p.is_bot:  # dealer picks suit
                    p.make_bid(valid_bids, make_pinochle_deck(), hands)
                wp = p
            else:  # someone won the bid
                wb = min_bid - 1
            break

        # end bidding early for a loner
        if min_bid > max_bid:
            wb = max_bid
            break

        # get the bid
        bid: int = p.make_bid(valid_bids, make_pinochle_deck(), hands, min_bid, wp)

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


def follow_suit(s: Optional[Suit], cs: Hand, strict: Optional[bool] = True) -> Hand:
    # strict filtering
    if not s:
        return cs
    if strict is not None:
        return Hand(c for c in cs if (c.follows_suit(s, strict)))

    # strict is None gives cards that follow suit
    if valid_cards := follow_suit(s, cs, True):
        return valid_cards
    # if valid_cards := follow_suit(s, cs, False):
    #     return valid_cards
    return cs


if __name__ == "__main__":
    main()
