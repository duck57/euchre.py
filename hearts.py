#!venv/bin/python
# coding=UTF-8
# -*- coding: UTF-8 -*-
# vim: set fileencoding=UTF-8 :

"""
Hearts

Rules outlined at https://www.pagat.com/reverse.hearts.html

Some optional rules have been applied in excess of what is demanded by MS Hearts.exe
"""

from cardstock import *

debug: Optional[bool] = False
o: Optional[TextIO] = None
log_dir: str = game_out_dir(os.path.basename(__file__).split(".py")[0])


def p(msg):
    global o
    click.echo(msg, o)


def px(msg) -> None:
    global debug
    if debug:
        p(msg)


class HeartCard(Cardstock):
    @property
    def value(self) -> int:
        if self.suit == Suit.HEART:
            return 1
        if self.suit == Suit.SPADE and self.rank == Rank.QUEEN:
            return 13
        if self.suit == Suit.DIAMOND and self.rank == Rank.JACK:
            return -10
        return 0


class PointHeartCard(HeartCard):
    @property
    def value(self) -> int:
        if self.suit != Suit.HEART:
            # balance & enhance Q♠️, J♦️
            return 2 * super().value
        if self.rank < Rank.QUEEN:
            return self.rank.v
        if self.rank == Rank.ACE_HI:
            return 15
        return 11


class DirtHeartCard(PointHeartCard):
    """
    Modified from the DNFH scoring:
    hearts are worth more, jacks & kings are helpful
    """

    @property
    def value(self) -> int:
        if self.rank == Rank.QUEEN:
            # queens are 13 each
            if self.suit == Suit.SPADE:
                return super().value
            return 13
        if self.suit == Suit.HEART:
            if self.rank < Rank.KING:
                return super().value
            return {Rank.KING: 5, Rank.ACE_HI: 1}[self.rank]
        if self.rank == Rank.JACK and self.suit.color != Color.RED:
            return -6
        if self.rank == Rank.KING:
            return -1
        return super().value


def key_points(c: HeartCard) -> int:
    return 100 * c.value + c.rank.v


class HeartPlayer(BasePlayer, WithScore, abc.ABC):
    left: "HeartPlayer"
    right: "HeartPlayer"
    across: "HeartPlayer"

    def __init__(self, g: "GameType", /, name: str, is_bot: int = 1):
        BasePlayer.__init__(self, g, name, is_bot)
        WithScore.__init__(self)
        self.trick_history: List[Hand] = []

    @staticmethod
    def allow_points(first_trick: bool, broken_heart: bool, lead: bool) -> bool:
        if first_trick:
            return False
        if broken_heart:
            return True
        if lead:
            return False
        return True

    def can_moonshot(
        self, *, current_trick: "Optional[HeartTrick]" = None, **kwargs
    ) -> bool:
        k: int = len(self.in_game.kitty)
        deck_total: int = self.deck.pcv
        my_points: Dict[str, int] = {
            "my cards": self.hand.pcv,
            "remaining": Hand(sorted(self.card_count, key=key_points)[:-k]).pcv
            if k
            else self.card_count.pcv,
        }
        # assume the best cards are in the kitty
        # look at the current trick
        if current_trick:
            cont, this = self.can_win_this_trick(current_trick)
            if not cont:
                return False
            my_points["this trick"] = current_trick.cards.pcv
        # can't look at kitty cards until the end of the hand
        # otherwise total the tricks you've already taken
        if sum(my_points.values()) < deck_total:
            return False
        # simulate playing the rest of the game
        cc: Hand = self.card_count
        cc.sort(key=key_rank_first, reverse=True)
        if not (points_remaining := cc.pcv):
            return True
        points_possible: int = 0
        for c in sorted(self.hand, key=key_rank_first, reverse=True):
            t = Trick([TrickPlay(c, self)])
            for _ in range(self.in_game.handedness - 1):
                if not cc:
                    break
                if can_beat_trick(t, cc):
                    t.append(TrickPlay(cc.pop(0), self))
                else:
                    x = follow_suit(t.lead_suit, cc)[-1]
                    t.append(TrickPlay(x, self))
                    cc.remove(x)
            if t.winning_card() == c:
                points_possible += t.cards.pcv
        return points_possible >= points_remaining

    def can_win_this_trick(
        self, t: "HeartTrick", /, *, valid_leads: Optional[Hand] = None
    ) -> Tuple[bool, int]:
        """
        Checks if you can win the trick
        :param valid_leads: list of valid cards to lead, used to check if Hearts is broken
        :param t: the trick in question
        :return: whether you can win it and the expected point change
        """
        m = can_beat_trick(t, self.hand)
        return bool(m), t.points + m[0].value if m else 0

    def play_card(self, trick_in_progress: "TrickType", /, **kwargs,) -> CardType:
        # p(f"{trick_in_progress} {kwargs}")
        first_trick: bool = kwargs.get("first", False)
        lead: bool = not trick_in_progress
        broken_heart: bool = kwargs.get("broken_heart", True)
        return super().play_card(
            trick_in_progress,
            points_ok=self.allow_points(first_trick, broken_heart, lead),
            force_suit=Suit.CLUB if first_trick and lead else None,
            **kwargs,
        )

    def trt(self, hand: Optional[int] = None) -> int:
        """
        :param hand: for which hand?  None = whole game
        :return: number of tricks taken
        """
        if hand is None:
            return (
                len([c for t in self.trick_history for c in t])
                // self.in_game.handedness
            )
        return len(self.trick_history[hand]) // self.in_game.handedness

    def hand_tab(self, hand: Optional[int], tab: str = "\t") -> str:
        """
        For prettier end-of-game bookkeeping
        :param hand: summary of which hand?  None = entire game
        :param tab: separator value
        :return: A summary of the hand
        """
        return tab.join(
            [
                str(self.trt(hand)),
                str(self.score if hand is None else self.score_changes[hand]),
            ]
        )

    @property
    def current_trick(self) -> Hand:
        return self.trick_history[-1]

    def call_pass(
        self,
        pass_size: int,
        /,
        valid_passes: List[Callable] = None,
        *,
        current_stack: List[Callable] = None,
        one_direction: bool = False,
        **kwargs,
    ) -> List[Callable]:
        if pass_size < 1:
            return current_stack
        if not valid_passes:
            valid_passes = pass_order_ms_hearts_exe
        if current_stack:
            if current_stack[-1] in [pass_hold, pass_kitty] or one_direction:
                valid_passes = [current_stack[-1]]  # hold ends card selection
                one_direction = True  # auto-fill the rest of the slots
            else:  # kitty can only be called on the first card
                valid_passes = [v for v in valid_passes if v != pass_kitty]
        else:
            current_stack = []
        return self.call_pass(
            pass_size - 1,
            valid_passes,
            current_stack=current_stack
            + [
                self.pick_pass(
                    valid_passes, remaining=pass_size, one_direction=one_direction
                )
            ],
        )

    @staticmethod
    @abc.abstractmethod
    def pick_pass(vp_list: List[Callable], **kwargs) -> Callable:
        return random.choice(vp_list)


def can_beat_trick(t: Trick, h: Hand) -> Hand:
    if not t:
        # don't check if you have lead, that's for some other function
        return h
    if not (fs := follow_suit(t.lead_suit, h, strict=True, ok_empty=True)):
        # nothing follows suit
        return fs
    return Hand(c for c in fs if c.rank > t.winning_card().rank)


class HeartTeam(BaseTeam, WithScore):
    def hand_tab(self, hand: Optional[int], tab: str = "\t") -> str:
        pass

    @property
    def score(self):
        return sum(pl.score for pl in self.players)


def key_score(pl: WithScore):
    return pl.score


class HumanPlayer(HeartPlayer, BaseHuman):
    def __init__(self, g: "GameType", /, name: str):
        HeartPlayer.__init__(self, g, name, 0)
        BaseHuman.__init__(self, g, name)

    def pick_pass(self, vp_list: List[Callable], **kwargs) -> Callable:
        if len(vp_list) == 1:  # don't bother users if they can't make a choice
            return vp_list[0]
        dct: Dict[str, Callable] = pass_choice_display_list(vp_list)
        count: Optional[int] = kwargs.get("remaining")
        preamble: str = f"(Up to {count} more) " if not kwargs.get(
            "one_direction"
        ) and isinstance(count, int) else ""
        return dct[
            click.prompt(
                type=click.Choice(dct.keys(), False),
                show_choices=False,
                default=pass_n(next(self.in_game.pass_order)),
                text=f"{preamble}Where to pass",
            ).lower()
        ]


class ComputerPlayer(HeartPlayer, BaseComputer):
    sort_key = key_points

    def pick_card(
        self,
        valid_cards: Hand,
        trick_in_progress: "Optional[HeartTrick]" = None,
        **kwargs,
    ):
        """
        Assumes valid_cards is sorted from least to greatest rank
        If it feels that it can shoot the moon, it plays high, otherwise it goes low
        """
        valid_cards.sort(key=key_rank_first)
        # check if you can moonshot
        if self.can_moonshot(current_trick=trick_in_progress):
            return random.choice(
                [c for c in valid_cards if c.rank == valid_cards[-1].rank]
            )
        # picking a card to follow
        if trick_in_progress:
            take_trick: Hand = can_beat_trick(trick_in_progress, valid_cards)
            dump_cards = Hand(c for c in valid_cards if c not in take_trick)
            can_follow_suit: bool = valid_cards[0].follows_suit(
                trick_in_progress.lead_suit
            )
            point_ahead_winners = Hand(
                c for c in take_trick if c.value <= -trick_in_progress.points
            )
            # grab negative (or zero) points when it is safe to do so
            if point_ahead_winners:
                # no remaining point cards
                # or you come out ahead even if you get some more point cards
                if Hand(
                    sorted(self.card_count.pointable, key=key_points)[
                        -(self.in_game.handedness - len(trick_in_progress) - 1) :
                    ]
                ).points <= abs(trick_in_progress.points):
                    return random.choice(point_ahead_winners)
                # you are the last player for the trick
                if len(trick_in_progress) == self.in_game.handedness - 1:
                    return random.choice(point_ahead_winners)
            # see if you can dump point cards on another suit
            if not can_follow_suit and dump_cards.pointable:
                # dump big guns first
                if big_cards := Hand(c for c in dump_cards.pointable if c.value > 9):
                    return random.choice(big_cards)
                return random.choice(dump_cards.pointable)
            # try the biggest card that won't take the trick
            if dump_cards:
                return dump_cards[-1]
            # you may be expected to take the trick
            return sorted(valid_cards, key=key_points)[0]

        # choose a lead (but you can't moonshot anymore)
        if self.card_count.pcv < 1:
            # only point-free cards remain
            # go big
            return valid_cards[-1]
        # take something reasonably small
        return random.choice(
            [c for c in valid_cards if c.rank.v <= valid_cards[0].rank.v]
        )

    @staticmethod
    def pick_pass(vp_list: List[Callable], **kwargs) -> Callable:
        return random.choice(vp_list)


def key_heart_trick_sort(tp: TrickPlay) -> int:
    return tp.card.rank.v


class HeartTrick(Trick):
    @property
    def points(self) -> int:
        return sum([c.value for c in self.cards])

    def winner(
        self, is_low: bool = False, purified: bool = False, display: bool = True
    ) -> Optional[TrickPlay]:
        length: int = len(self)
        if length == 0:
            if display:
                p(f"No one won.")
            return None
        if length == 1:
            if display:
                p(f"One winning card.")
            return self[0]
        if not purified:
            self.sort(key=key_heart_trick_sort, reverse=is_low)
            return self.follow_suit().winner(
                is_low=is_low, purified=True, display=display
            )
        # check for duplicate winners
        if self[-2].card == self[-1].card:
            if display:
                p(f"No clear winner, trying again. {self.cards}")
            return HeartTrick([x for x in self if x.card != self[-1].card]).winner(
                purified=True, display=display
            )
        return self[-1]

    def winning_card(self, is_low: bool = False) -> CardType:
        return self.winner(display=False).card

    def follow_suit(
        self, strict: bool = True, ot: "Optional[Type[Trick]]" = None
    ) -> "TrickType":
        return super().follow_suit(strict, HeartTrick)


class Hearts(BaseGame):
    def __init__(
        self,
        *,
        preset: Optional[str] = "Normal",
        deck_type: Optional[str] = "Normal",
        points: Optional[int] = 100,
        pass_size: Optional[int] = 3,
        unify_passing: Optional[bool] = True,
        custom_calls_enabled: Optional[bool] = False,
        boost9: Optional[bool] = True,
        custom_pass_order: Optional[List[Callable]] = None,
        no_hold: Optional[bool] = False,
        allow_kitty: Optional[bool] = False,
        **kwargs,
    ):
        """
        A game of Hearts

        :param preset: basic game setup
        :param deck_type: scoring rules to use
        :param points: end the game once a player has accumulated this many points
        :param pass_size: (maximum) number of cards to pass at the start of a hand
        :param unify_passing: require all cards to be passed the same direction each hand if True
        :param custom_calls_enabled: the dealer calls pass if True
        :param boost9: leading or following suit with a 9 causes an extra-large trick if true
        :param custom_pass_order: overrides no_hold and allow_kitty if set
        :param no_hold: skips the hold phase of passing if True; requires all cards to be passed with custom calling
        :param allow_kitty: allow players to pass to the kitty if True
        :param kwargs: extra stuff to pass along to BaseGame
        """

        # set up the scoring rules
        preset: chr = preset[0].upper() if preset else "N"
        deck_type: Type[HeartCard] = {
            "N": HeartCard,
            "D": DirtHeartCard,
            "S": PointHeartCard,
        }.get(
            deck_type[0].upper() if deck_type else preset, HeartCard,
        )
        kwargs["points"] = (
            points if points else {"N": 100, "D": 450, "S": 500}.get(preset, 100)
        )
        kwargs["pass_size"] = pass_size if pass_size is not None else 3
        self.unified_passing: bool = (
            unify_passing
            if unify_passing is not None
            else (True if deck_type == DirtHeartCard else False)
        )
        self.player_calling: bool = custom_calls_enabled if (
            custom_calls_enabled is not None
        ) else (True if deck_type == DirtHeartCard else False)
        self.boost9: bool = boost9 if boost9 is not None else True

        super().__init__(
            human_player_type=HumanPlayer,
            computer_player_type=ComputerPlayer,
            **kwargs,
            team_type=HeartTeam,
            game_name="Hearts",
            card_type=deck_type,
        )

        # set up passing order
        self.valid_calls: List[Callable] = custom_pass_order
        # it's your responsibility to make your custom call order behave properly
        if not self.valid_calls:
            """
            The default pass order
            With everything enabled: left, right, across, hold, kitty
            [pass across is only for even numbers of players]
            """
            self.valid_calls = [pass_left, pass_right]
            if not self.handedness % 2:
                self.valid_calls.append(pass_across)
            if not no_hold:
                self.valid_calls.append(pass_hold)
            if allow_kitty or allow_kitty is None and deck_type == DirtHeartCard:
                self.valid_calls.append(pass_kitty)
        self.pass_order = cycle(self.valid_calls)

        # moonshots are instant victory under DNFH rules
        self.moon_points: int = (
            self.deck.pointable.points
            if deck_type != DirtHeartCard
            else self.victory_threshold ** 2
        )
        self.sun_cards: int = len(self.deck)

    def team_scores(self, pf: Callable = print) -> List[TeamType]:
        teams_by_score = sorted(self.teams, key=key_score)
        pf(f"Team scores:")
        for t in teams_by_score:
            pf(f"{t}: {t.score}")
        return teams_by_score

    def victory_check(
        self, pf: Callable = print
    ) -> Tuple[int, Union[TeamType, PlayerType, None]]:
        for pl in self.players:  # replace me with a list comprehension
            if pl.score == 100:  # if your score is exactly 100
                pl.score = -50  # it's a saving throw
        teams_by_score = self.team_scores(
            p0 if len(self.teams) == self.handedness else pf
        )
        if sorted(self.players, key=key_score)[-1].score < self.victory_threshold:
            return 0, None  # Game is not over
        if teams_by_score[1].score == teams_by_score[0].score:
            return -1, None  # Tie
        return 1, teams_by_score[0]

    def play_hand(self, dealer: HeartPlayer) -> HeartPlayer:
        # call pass before dealing the cards
        pass_dir: List[Callable] = dealer.call_pass(
            self.pass_size,
            self.valid_calls if self.player_calling else [next(self.pass_order)],
            one_direction=self.unified_passing,
        )
        if len(pass_dir) == len([x for x in pass_dir if x == pass_dir[0]]):
            # all cards go to the same direction
            if pass_dir[0] != pass_hold:
                p(f"Passing {self.pass_size} cards {pass2s(pass_dir[0])}")
            else:
                p(f"Holding cards")
        else:
            p(f"Pass cards {', '.join([pass_n(pd) for pd in pass_dir])}")

        # deal
        self.deal()
        hn: int = len(dealer.score_changes) + 1
        po: List[HeartPlayer] = get_play_order(dealer)
        for pl in po:
            pl.trick_history.append(Hand())
        p(f"Hand {hn}, dealt by {dealer}")

        # pass cards
        self.kitty = pass_cards(po, pass_dir, self.kitty)

        # 2 of clubs lead
        lead_history: List[HeartPlayer] = []
        start_rank: Union[Rank, bool] = False
        for r in poker_ranks:
            lead_history = [pl for pl in po if HeartCard(r, Suit.CLUB) in pl.hand]
            if lead_history:  # complex to handle duplications and the kitty
                start_rank = r
                # handle ties
                lead_history = lead_history[:1]  # first player starting with dealer
                break
            else:
                p(f"{repr(r)}♣️ must be in the kitty")
        bh, f = False, True

        # play tricks
        while lead_history[0].hand:
            lead, bh, f, start_rank = self.play_trick(
                lead_history[-1], force_low=start_rank, broken_heart=bh, first=f
            )
            lead_history.append(lead)

        # tally score
        p(f"Hand {hn} results:")
        point_grid: Dict[HeartPlayer, int] = {
            pl: pl.current_trick.pointable.points for pl in po
        }
        for pl in po:
            recent: Hand = pl.trick_history[-1]
            if len(recent) == len(self.deck):
                # sun shot
                for x in po:
                    point_grid[x] = 2 * self.moon_points
                    if x == pl:
                        point_grid[x] = 0
            if point_grid[pl] == self.moon_points:
                # moonshot
                for x in po:
                    point_grid[x] = self.moon_points
                    if x == pl:
                        point_grid[x] = 0
            point_grid[pl] += recent.point_free.points  # add cleansing cards
        for pl in po:
            sc: int = point_grid[pl]
            pl.score = sc
            p(f"{pl}: {'+' if sc > 0 else ''}{sc} to total {pl.score}")
        if self.kitty:
            lead_history[1].current_trick.extend(self.kitty)
            p(f"({lead_history[1]} got {self.kitty} from the kitty)")
            assert len(self.kitty) < self.minimum_kitty_size + len(self.players)
        return dealer.next_player

    def play_trick(
        self,
        lead: HeartPlayer,
        first: bool = False,
        broken_heart: bool = True,
        force_low: Optional[Rank] = Rank.TWO,
    ) -> Tuple[HeartPlayer, bool, bool, None]:
        po: List[HeartPlayer] = get_play_order(lead)
        t = HeartTrick()
        p(f"\n{lead} starts")

        # play cards
        def play_round() -> Tuple[bool, bool]:
            boost9: bool = False
            bh: bool = broken_heart
            f = force_low
            for player in po:
                # p(player.hand)
                c: CardType = player.play_card(
                    t, first=first, broken_heart=bh, valid_rank=f
                )
                f = False
                t.append(TrickPlay(c, player))
                p(f"{player} played {repr(c)}")
                if c.suit == Suit.HEART and not bh:
                    p("Hearts has been broken!")
                    bh = True
                if c.suit == t[0].card.suit and c.rank == Rank.NINE and self.boost9:
                    p("Boosted 9!")
                    boost9 = True
                l_suit: Suit = t.lead_suit
                if c.suit != l_suit:
                    self.suit_safety[l_suit] = (
                        True if self.suit_safety[l_suit] else player
                    )
            return bh, boost9

        w: Optional[TrickPlay] = None
        while not w:
            broken_heart, repeat = play_round()
            w = None if repeat else t.winner()
            if not lead.hand:  # out of cards
                break
            force_low = False
        if w:
            lead = w.played_by
        lead.tricks_taken.append(t.cards)
        lead.current_trick.extend(t.cards)
        km: str = " plus the kitty" if self.kitty and first else ""
        p(f"{lead} gets the cards{km}.\n")
        return lead, broken_heart, False, None

    def play(self):
        v: int = 0
        w = None
        global o
        while v < 1:
            self.current_dealer = self.play_hand(self.current_dealer)
            v, w = self.victory_check(p)
            if v == -1:
                p("It's a tie, keep going!")
                continue
            # self.team_scores(p0 if len(self.teams) == self.handedness else p)
            p("\n")
        self.team_scores(print if o else p0)
        p(f"{w} wins!")

    def write_log(self, ld: str, splitter: str = "\t|\t") -> None:
        stop_time: str = str(datetime.now()).split(".")[0]
        f: TextIO = open(os.path.join(ld, f"{self.start_time}.gamelog"), "w")

        def w(msg):
            click.echo(msg, f)

        # headers
        w(splitter.join([self.start_time] + [f"{t}\t" for t in self.players]))
        w(splitter.join([""] + ["Tricks Taken\tScore Change" for _ in self.players]))
        w(splitter.join(["Hand"] + ["===\t===" for _ in self.players]))
        w(  # body
            "\n".join(
                [
                    splitter.join(
                        [f"{hand + 1}"] + [t.hand_tab(hand) for t in self.players]
                    )
                    for hand in range(len(self.players[0].score_changes))
                ]
            )
        )
        # totals
        w(splitter.join([stop_time] + ["===\t===" for _ in self.players]))
        w(splitter.join(["Totals"] + [t.hand_tab(None) for t in self.players]))

        # team stats
        if len(self.teams) < len(self.players):  # more than 1 player per "team"
            w("\n\n")
            w("\t".join(["Team Score History"] + [f"{t}" for t in self.teams]))
            w(  # body
                "\n".join(
                    [
                        "\t".join(
                            [f"{hand + 1}"]
                            + [
                                str(
                                    sum(
                                        [
                                            sum(pj.score_changes[: hand + 1])
                                            for pj in t.players
                                        ]
                                    )
                                )
                                for t in self.teams
                            ]
                        )
                        for hand in range(len(self.players[0].score_changes))
                    ]
                )
            )
            # w("\n\n")
            # self.team_scores(w)
        f.close()


@click.command()
@common_options
@click.option(
    "--no-hold", type=click.BOOL, help="All cards must be passed every round if True",
)
@click.option(
    "--cat-passable",
    "allow_kitty",
    type=click.BOOL,
    help="Allows cards to be passed to the kitty and shuffled if set",
)
@click.option(
    "--score-rules",
    "-s",
    type=click.Choice(
        ["Normal", "Dirty", "DNFH", "Spot", "N", "D", "S"], case_sensitive=False
    ),
    default="Normal",
    help="Rules for scoring cards",
)
@click.option(
    "--call-passes",
    "custom_calls_enabled",
    type=click.BOOL,
    help="Players call their own pass rules for each hand",
)
@click.option(
    "--unify-passing", type=click.BOOL, help="Prohibit split pass calls",
)
@click.option(
    "--game",
    "-g",
    "preset",
    type=click.Choice(
        ["Normal", "Dirty", "DNFH", "Spot", "N", "D", "S"], case_sensitive=False,
    ),
    help="""
    Preset setups for Hearts variants, overridden by other options

    \b
    Normal = as close to vanilla MS Hearts as this gets
    Dirty/DNFH = loads o' special rules
    Spot = Spot hearts (you take one point for each heart on the hearts)
    """,
)
@click.option(
    "--boost9",
    type=click.BOOL,
    default=True,
    help="Leading or following suit with a 9 doubles the trick size if True",
)
def main(**kwargs):
    global o
    global debug
    global log_dir
    if kwargs.get("all_bots"):
        st: str = str(datetime.now()).split(".")[0]
        o = open(os.path.join(log_dir, f"{st}.gameplay"), "w")
        kwargs["start_time"] = st
        debug = True
    make_and_play_game(Hearts, log_dir, **kwargs)


if __name__ == "__main__":
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    main()
