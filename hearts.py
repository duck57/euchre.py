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

    def __init__(self, name: str, bot: int = 1, deck: Optional[Hand] = None):
        BasePlayer.__init__(self, name, bot, deck)
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

    def play_card(self, trick_in_progress: "TrickType", /, **kwargs,) -> CardType:
        # p(f"{trick_in_progress} {kwargs}")
        first_trick: bool = kwargs.get("first", False)
        lead: bool = not trick_in_progress
        if lead and first_trick:
            p("Two of clubs to start")
            return self.hand.pop(0)  # two of clubs
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
            **kwargs,
        )

    def reset_unplayed(self, ts: Optional[Suit] = None) -> Hand:
        self.tricks_taken = []
        return super().reset_unplayed(ts)

    def hand_tab(self, hand: Optional[int], tab: str = "\t") -> str:
        return tab.join(
            [str(len(self.trick_history[hand])), str(self.score_changes[hand]),]
            if hand is not None
            else [str(sum([len(x) for x in self.trick_history])), str(self.score),]
        )

    @property
    def current_trick(self) -> Hand:
        return self.trick_history[-1]


class HeartTeam(BaseTeam, WithScore):
    @property
    def score(self):
        return sum(pl.score for pl in self.players)


def key_score(pl: WithScore):
    return pl.score


class HumanPlayer(HeartPlayer, BaseHuman):
    pass


class ComputerPlayer(HeartPlayer, BaseComputer):
    sort_key = key_points

    def pick_card(self, valid_cards: Hand, **kwargs):
        c: CardType = valid_cards[-1]
        self.hand.remove(c)
        return c


def key_heart_trick_sort(tp: TrickPlay) -> int:
    return tp.card.rank.v


class HeartTrick(Trick):
    @property
    def points(self) -> int:
        return sum([c.value for c in self.cards])

    def winner(self, is_low: bool = False) -> Optional[TrickPlay]:
        if not self:
            p(f"No one won.")
            return None
        if len(self.cards) == 1:
            return self[0]
        fs: List[TrickPlay] = sorted(self.follow_suit(), key=key_heart_trick_sort)
        # check for duplicate winners
        try:
            if fs[-2].card == fs[-1].card:
                p(f"No clear winner, trying again.")
                return HeartTrick([x for x in fs if x.card != fs[-1].card]).winner()
        except IndexError:
            p(f"{fs}")  # debugging
        return fs[-1]

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
        pass_order: List[Callable],
        pass_size: int,
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
        self.pass_order = cycle(pass_order)
        self.pass_size = pass_size
        for pl in self.players:
            pl.deck = deepcopy(self.deck)
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
        # pass cards
        p(f"Hand {hn}, dealt by {dealer}")
        pass_cards(po, [next(self.pass_order)] * self.pass_size, self.kitty)
        # 2 of clubs lead
        el: List[HeartPlayer] = []
        for r in poker_ranks:
            el = [pl for pl in po if HeartCard(r, Suit.CLUB) in pl.hand]
            if el:  # complex to handle duplications and the kitty
                break
        lead: HeartPlayer = el[0]
        bh, f = False, True
        # play tricks
        while lead.hand:
            lead, bh, f = self.play_trick(lead, broken_heart=bh, first=f)

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
        return dealer.next_player

    def play_trick(
        self, lead: HeartPlayer, first: bool = False, broken_heart: bool = True
    ) -> Tuple[HeartPlayer, bool, bool]:
        po: List[HeartPlayer] = get_play_order(lead)
        t = HeartTrick()
        p(f"{lead} starts")

        # play cards
        def play_round() -> Tuple[bool, bool]:
            boost9: bool = False
            bh: bool = broken_heart
            for player in po:
                # p(player.hand)
                c: CardType = player.play_card(t, first=first, broken_heart=bh)
                t.append(TrickPlay(c, player))
                p(f"{player} played {repr(c)}")
                if c.suit == Suit.HEART and not bh:
                    p("Hearts has been broken!")
                    bh = True
                if c.suit == t[0].card.suit and c.rank == Rank.NINE:
                    p("Boosted 9!")
                    boost9 = True
            return bh, boost9

        w: Optional[TrickPlay] = None
        while not w:
            broken_heart, repeat = play_round()
            w = None if repeat else t.winner()
            if not lead.hand:  # out of cards
                break
        if w:
            lead = w.played_by
        if self.kitty:  # should only be on the first trick
            p(f"Plus {self.kitty} from the kitty")
            lead.current_trick.extend(self.kitty)
            self.kitty = Hand()  # reset the kitty for the next hand
        lead.tricks_taken.append(t)
        lead.current_trick.extend(t.cards)
        p(f"{lead} gets the cards.\n")
        return lead, broken_heart, False

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
    default=100,
)
@click.option(
    "--team-size",
    "-t",
    type=click.IntRange(1, 5),
    default=1,
    help="Number of players per team. 1 = normal hearts",
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
    "--preset",
    "-p",
    type=click.Choice(
        ["Custom", "Normal", "Dirty", "DNFH", "Spot", "C", "N", "D", "S"],
        case_sensitive=False,
    ),
    default="Custom",
    help="""
    Preset setups for Hearts variants, overrules individual selections if not Custom
    
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
) -> None:
    start_time: str = str(datetime.now()).split(".")[0]
    global o
    global debug
    global log_dir
    if not humans:  # assume one human player as default
        humans = [random.randrange(handedness)]
    if all_bots:
        humans = []
        o = open(os.path.join(log_dir, f"{start_time}.gameplay"), "w")
        debug = True
    game_type: str = preset[0].lower()
    deck_type: str = score_rules[0].upper()
    deck_dict: Dict[str, Type[HeartCard]] = {
        "N": HeartCard,
        "D": DirtHeartCard,
        "S": PointHeartCard,
    }
    deck: Type[HeartCard] = deck_dict[deck_type]
    if game_type != "c":
        handedness = 4
        team_size = 1
        deck = deck_dict[game_type.upper()]
        points = {
            "n": 100,
            "d": 300,
            "s": 500,
        }
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
        pass_order=pass_order_all,
        pass_size=3,
    )
    h.play()
    h.write_log(log_dir)


if __name__ == "__main__":
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    main()
