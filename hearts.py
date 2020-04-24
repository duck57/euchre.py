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

    def __init__(self, g: "GameType", /, name: str):
        BasePlayer.__init__(self, g, name)
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

    @property
    def can_moonshot(self, **kwargs) -> bool:
        k: int = kwargs.get("kitty_size", 0)
        h: int = kwargs.get("handedness", 4 - k)
        deck_total: int = self.deck.pcv
        my_points: Dict[str, int] = {
            "my cards": self.hand.pcv,
            "remaining": Hand(sorted(self.card_count, key=key_points)[:-k]).pcv
            if k
            else self.card_count.pcv,
        }
        # assume the best cards are in the kitty
        # look at the current trick
        if ct := kwargs.get("current_trick"):
            cont, this = self.can_win_this_trick(ct)
            if not cont:
                return False
            my_points["this trick"] = ct.cards.pcv
        # can't look at kitty cards until the end of the hand
        # otherwise total the tricks you've already taken
        return sum(my_points.values()) >= deck_total

    def can_win_this_trick(
        self, t: "HeartTrick", /, *, valid_leads: Optional[Hand] = None
    ) -> Tuple[bool, int]:
        """
        Checks if you can win the trick
        :param valid_leads: list of valid cards to lead, used to check if Hearts is broken
        :param t: the trick in question
        :return: whether you can win it and the expected point change
        """
        if not valid_leads:
            valid_leads = self.hand
        if not t:
            return True, 0
        return False, 0

    def play_card(self, trick_in_progress: "TrickType", /, **kwargs,) -> CardType:
        # p(f"{trick_in_progress} {kwargs}")
        first_trick: bool = kwargs.get("first", False)
        lead: bool = not trick_in_progress
        broken_heart: bool = kwargs.get("broken_heart", True)
        # p(
        #     "\t".join(
        #         [
        #             " ".join(["First", "trick?", first_trick]),
        #             " ".join(["Heartbreak?", broken_heart]),
        #             " ".join(["Lead", lead]),
        #             " ".join(
        #                 [
        #                     "Points",
        #                     "ok?",
        #                     self.allow_points(first_trick, broken_heart, lead),
        #                 ]
        #             ),
        #         ]
        #     )
        # )
        return super().play_card(
            trick_in_progress,
            points_ok=self.allow_points(first_trick, broken_heart, lead),
            force_suit=Suit.CLUB if first_trick and lead else None,
            **kwargs,
        )

    def hand_tab(self, hand: Optional[int], tab: str = "\t") -> str:
        return tab.join(
            [str(len(self.trick_history[hand])), str(self.score_changes[hand])]
            if hand is not None
            else [str(sum([len(x) for x in self.trick_history])), str(self.score)]
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
    ) -> List[Callable]:
        if pass_size < 1:
            return current_stack
        if not valid_passes:
            valid_passes = pass_order_all
        if current_stack:
            if current_stack[-1] in [pass_hold, pass_kitty]:
                valid_passes = [current_stack[-1]]  # hold ends card selection
            else:  # kitty can only be called on the first card
                valid_passes = [v for v in valid_passes if v != pass_kitty]
        else:
            current_stack = []
        return self.call_pass(
            pass_size - 1,
            valid_passes,
            current_stack=current_stack + [self.pick_pass(valid_passes)],
        )

    @staticmethod
    @abc.abstractmethod
    def pick_pass(vp_list: List[Callable]) -> Callable:
        return random.choice(vp_list)


class HeartTeam(BaseTeam, WithScore):
    @property
    def score(self):
        return sum(pl.score for pl in self.players)


def key_score(pl: WithScore):
    return pl.score


class HumanPlayer(HeartPlayer, BaseHuman):
    @staticmethod
    def pick_pass(vp_list: List[Callable]) -> Callable:
        if len(vp_list) == 1:  # don't bother users if they can't make a choice
            return vp_list[0]
        return pass_choice_display_list(vp_list)[0]


class ComputerPlayer(HeartPlayer, BaseComputer):
    sort_key = key_points

    def pick_card(self, valid_cards: Hand, **kwargs):
        c: CardType = valid_cards[-1]
        return c

    @staticmethod
    def pick_pass(vp_list: List[Callable]) -> Callable:
        return random.choice(vp_list)


def key_heart_trick_sort(tp: TrickPlay) -> int:
    return tp.card.rank.v


class HeartTrick(Trick):
    @property
    def points(self) -> int:
        return sum([c.value for c in self.cards])

    def winner(
        self, is_low: bool = False, purified: bool = False
    ) -> Optional[TrickPlay]:
        length: int = len(self)
        if length == 0:
            p(f"No one won.")
            return None
        if length == 1:
            p(f"One winning card.")
            return self[0]
        if not purified:
            self.sort(key=key_heart_trick_sort, reverse=is_low)
            return self.follow_suit().winner(is_low=is_low, purified=True)
        # check for duplicate winners
        if self[-2].card == self[-1].card:
            p(f"No clear winner, trying again. {self.cards}")
            return HeartTrick([x for x in self if x.card != self[-1].card]).winner(
                purified=True
            )
        return self[-1]

    def follow_suit(
        self, strict: bool = True, ot: "Optional[Type[Trick]]" = None
    ) -> "TrickType":
        return super().follow_suit(strict, HeartTrick)


class Hearts(BaseGame):
    def __init__(
        self,
        *,
        player_names: List[str],
        player_types: List[Type[BasePlayer]],
        team_size: int,
        card_type: Type[Cardstock],
        victory_threshold: int,
        start_time: str,
        **kwargs,
    ):
        super().__init__(
            player_names,
            player_types,
            team_size,
            card_type,
            make_standard_deck,
            HeartTeam,
            victory_threshold,
            start=start_time,
        )
        self.deck = self.deck * (self.handedness // 6 + 1)

        # set up passing order
        self.valid_calls: List[Callable] = kwargs.get("cpo")
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
            if kwargs.get("allow_holding"):
                self.valid_calls.append(pass_hold)
            if kwargs.get("pass_to_kitty_enabled"):
                self.valid_calls.append(pass_kitty)
        self.pass_order = cycle(self.valid_calls)
        self.player_calling: bool = kwargs.get("custom_calls_enabled", False)

        self.pass_size = kwargs.get("pass_size", 3)
        # moonshots are instant victory under DNFH rules
        self.moon_points: int = self.deck.pointable.points if card_type != DirtHeartCard else victory_threshold ** 2
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
        # deal
        self.deal()
        hn: int = len(dealer.score_changes) + 1
        po: List[HeartPlayer] = get_play_order(dealer)
        for pl in po:
            pl.trick_history.append(Hand())
        p(f"Hand {hn}, dealt by {dealer}")

        # pass cards
        pass_dir: List[Callable] = dealer.call_pass(
            self.pass_size,
            self.valid_calls if self.player_calling else [next(self.pass_order)],
        )
        if len(pass_dir) == len([x for x in pass_dir if x == pass_dir[0]]):
            # all cards go to the same direction
            if pass_dir[0] != pass_hold:
                p(f"Passing {self.pass_size} cards {pass2s(pass_dir[0])}")
            else:
                p(f"Holding cards")
        else:
            p(f"Pass cards {', '.join([pass_n(pd) for pd in pass_dir])}")
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
        p(f"{lead} starts")

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
                if c.suit == t[0].card.suit and c.rank == Rank.NINE:
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
@click.option(
    "--handedness",
    "-h",
    type=click.IntRange(3, 10),
    default=4,
    help="Number of players in the game",
)
@click.option(
    "--humans",
    "-p",
    multiple=True,
    default=[],
    type=click.IntRange(0, 10),
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
    help="Victory threshold (v): positive integer",
    default=4,  # make this over 5 or else it gets overridden
)
@click.option(
    "--team-size",
    "-t",
    type=click.IntRange(1, 5),
    default=1,
    help="Number of players per team. 1 = normal hearts",
)
@click.option(
    "--pass-size",
    "-z",
    type=click.IntRange(1, 5),
    default=3,
    help="(maximum) Number of cards to pass",
)
@click.option(
    "--no-hold",
    type=click.BOOL,
    is_flag=True,
    help="All cards must be passed every round if set",
)
@click.option(
    "--cat-passable",
    "allow_kitty",
    type=click.BOOL,
    is_flag=True,
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
    "custom_call_pass",
    type=click.BOOL,
    is_flag=True,
    help="Players call their own pass rules for each hand",
)
@click.option(
    "--game",
    "-g",
    "preset",
    type=click.Choice(
        ["Normal", "Dirty", "DNFH", "Spot", "N", "D", "S"], case_sensitive=False,
    ),
    default="Normal",
    help="""
    Preset setups for Hearts variants, overridden by other options
    
    Normal = as close to vanilla MS Hearts as this gets
    
    Dirty/DNFH = loads o' special rules
    
    Spot = Spot hearts (you take one point for each heart on the hearts)
    """,
)
def main(
    handedness: int,
    humans: List[int],
    all_bots: bool,
    points: int,
    team_size: int,
    score_rules: str,
    preset: str,
    pass_size: int,
    no_hold: bool,
    allow_kitty: bool,
    custom_call_pass: bool,
) -> None:
    # global constants
    start_time: str = str(datetime.now()).split(".")[0]
    global o
    global debug
    global log_dir

    deck_dict: Dict[str, Type[HeartCard]] = {
        "N": HeartCard,
        "D": DirtHeartCard,
        "S": PointHeartCard,
    }
    if points == 4:  # using a preset without modification
        points = {"n": 100, "d": 450, "s": 500,}[preset[0].lower()]
        deck: Type[HeartCard] = deck_dict[preset[0].upper()]
        if deck == DirtHeartCard:
            allow_kitty = True
            no_hold = False
            custom_call_pass = True
    else:
        deck: Type[HeartCard] = deck_dict[score_rules[0].upper()]

    player_handles: List[str] = {
        3: ["Juan", "Sarah", "Turia"],
        4: ["Nelson", "Eustace", "Samantha", "Wyclef"],
    }.get(
        handedness,
        [
            "Juan",
            "Nelson",
            "Sarah",
            "Eustace",
            "Ahmed",
            "Samantha",
            "Turia",
            "Wyclef",
            "Amanda",
            "Miranda",
        ],
    )[
        :handedness
    ]
    if not humans:  # assume one human player as default
        humans = [random.randrange(handedness)]
    if all_bots:
        humans = []
        o = open(os.path.join(log_dir, f"{start_time}.gameplay"), "w")
        debug = True
    if len(humans) == 1 and humans[0] < handedness:
        player_handles[humans[0]] = "You"
    p_types: List[Type[HeartPlayer]] = [ComputerPlayer for _ in range(handedness)]
    for n in humans:
        if n < len(p_types):
            p_types[n] = HumanPlayer
    h = Hearts(
        player_names=player_handles,
        player_types=p_types,
        team_size=team_size,
        card_type=deck,
        victory_threshold=points,
        start_time=start_time,
        pass_size=pass_size,
        pass_to_kitty_enabled=allow_kitty,
        allow_holding=not no_hold,
        custom_calls_enabled=custom_call_pass,
    )
    h.play()
    h.write_log(log_dir)


if __name__ == "__main__":
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    main()
