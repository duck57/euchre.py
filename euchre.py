#!venv/bin/python
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


class Player(BasePlayer, abc.ABC):
    desired_trump: Bid
    shoot_strength: int

    def __init__(self, name: str, bot: int = 1, deck=None):
        super().__init__(name, bot, deck)
        self.tricks: int = 0
        self.bid_estimates: Dict[Bid, int] = {}
        self.reset_bids()

    def reset_bids(self) -> None:
        for t in Bid:
            self.bid_estimates[t] = 0

    @property
    def choose_trump(self) -> Bid:
        return self.desired_trump

    @abc.abstractmethod
    def make_bid(
        self,
        valid_bids: List[int],
        d: Hand,
        handedness: int,
        min_bid: int = 0,
        leading_player: "Optional[Player]" = None,
    ) -> int:
        pass

    def trumpify_hand(self, trump_suit: Optional[Suit], is_lo: bool = False) -> None:
        """Marks the trump suit and sort the hands"""
        if trump_suit:
            self.hand.trumpify(trump_suit)
        self.sort_hand(is_lo)
        self.card_count = self.reset_unplayed(ts=trump_suit)

    def receive_shooter(self, handedness: int, bid: Bid) -> None:
        for pl in self.teammates:
            self.receive_cards(pl.send_shooter(self.shoot_strength))
            if not pl.is_bot:
                print()
        self.sort_hand(bid.is_low)


class HumanPlayer(BaseHuman, Player):
    def __init__(self, name, deck=None):
        super().__init__(name, 0, deck)

    @property
    def choose_trump(self) -> Bid:
        p(self.hand)  # give a closer look at your hand before bidding
        bids: List[str] = [c for c in Bid.__members__]
        bids.extend([Bid[c].short_name for c in Bid.__members__])
        bid: str = click.prompt(
            "Declare Trump", type=click.Choice(bids, False), show_choices=False,
        ).upper()
        return Bid[[b for b in Bid.__members__ if (bid in b)][0]]

    def make_bid(
        self,
        valid_bids: List[int],
        d: Hand,
        handedness: int,
        min_bid: int = 0,
        leading_player: "Optional[Player]" = None,
    ) -> int:
        self.hand.sort(key=key_display4human)
        p(self.hand)
        return int(
            click.prompt(
                "How much to bid",
                type=click.Choice(
                    ["0"] + [str(x) for x in valid_bids if (x >= min_bid)], False,
                ),
            )
        )


class ComputerPlayer(BaseComputer, Player):
    sort_key = key_trump_power

    def __init__(self, name, deck=None):
        super().__init__(name, 1, deck)

    def make_bid(
        self,
        valid_bids: List[int],
        d: Hand,
        handedness: int,
        min_bid: int = 0,
        leading_player: "Optional[Player]" = None,
    ) -> int:
        if max(self.bid_estimates.values()) == 0:
            # simulate a hand
            for card in self.hand:
                d.remove(card)
            self.bid_estimates = {
                t: self.simulate_hand(
                    h_p=deepcopy(self.hand), d_p=deepcopy(d), handedness=handedness, t=t
                )
                for t in Bid
            }
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

    def pick_card(self, valid_cards: Hand, **kwargs,) -> Card:
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
                    self.estimate_tricks_by_suit(
                        follow_suit(s, valid_cards, True),
                        follow_suit(s, unplayed, True),
                        is_low,
                        strict=st,
                    )
                )
            return wl

        if not tp:  # you have the lead
            safer_suits: List[Suit] = [
                s for s in broken.keys() if broken[s] is False or broken[s] == self.team
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
                w += winning_leads(suits, st=bool(self.teammates))
            if not w and self.teammates:  # seamless passing of the lead
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
            if wp in self.teammates and junk_cards:  # your partner is winning
                if wc.rank in junk_ranks:  # but their card is rubbish
                    return random.choice(w)
                return random.choice(junk_cards)
            return random.choice(w)
        return random.choice(junk_cards)

    def play_card(self, trick_in_progress: "Trick", /, **kwargs,) -> Card:
        """
        This is necessary so that cards aren't double-played by CPU players
        Humans pop a raw index, but a card needs to be removed manually for computers
        """
        c: Card = super().play_card(trick_in_progress, **kwargs)
        self.hand.remove(c)
        return c

    def simulate_hand(self, *, h_p: Hand, d_p: Hand, t: Bid, **kwargs) -> int:
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
                    self.estimate_tricks_by_suit(
                        my_suit=slice_by_suit(h_p, s),
                        mystery_suit=slice_by_suit(d_p, s),
                        is_low=t.is_low,
                        is_trump=(s == Suit.TRUMP),
                    )
                )
                for s in suits + [Suit.TRUMP]
            ]
        )

    @staticmethod
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
        :return: winning cards for the suit
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


class Team(BaseTeam, MakesBid, WithScore):
    def __init__(self, players: Iterable[BasePlayer]):
        BaseTeam.__init__(self, players)
        MakesBid.__init__(self)
        WithScore.__init__(self)
        self.bid_history: List[str] = []
        self.tricks_taken: List[int] = []

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


class BidEuchre(BaseGame):
    def __init__(
        self,
        *,
        player_names: List[str],
        player_types: List[Type[BasePlayer]],
        team_size: int,
        victory_threshold: int,
        start_time: str,
    ):
        super().__init__(
            player_names,
            player_types,
            team_size,
            Card,
            make_pinochle_deck,
            Team,
            victory_threshold,
            start=start_time,
        )
        self.trump: Optional[Suit] = None
        self.low_win: bool = False
        self.hand_size: int = len(self.deck) // self.handedness
        for pl in self.players:
            pl.shoot_strength = {3: 3, 4: 2, 6: 1, 8: 1}[self.handedness]

        c = configparser.ConfigParser()
        c.read("constants.cfg")
        self.valid_bids: List[int] = [
            int(i) for i in c["ValidBids"][str(self.handedness)].split(",")
        ]
        if (
            victory_threshold is not None and victory_threshold > 0
        ):  # negative thresholds get dunked on
            self.victory_threshold: int = victory_threshold
            self.mercy_rule: int = -victory_threshold
            self.bad_ai_end: int = -victory_threshold // 2
        else:
            self.victory_threshold: int = c["Scores"].getint("victory")
            self.mercy_rule: int = c["Scores"].getint("mercy")
            self.bad_ai_end: int = c["Scores"].getint("broken_ai")

    def bidding(self, bid_order: List[Player]) -> Player:
        first_round: bool = True
        count: int = 1
        hands: int = len(bid_order)
        wp: Optional[Player] = None
        wb: int = 0
        bid_order = cycle(bid_order)
        min_bid: int = min(self.valid_bids)
        max_bid: int = max(self.valid_bids)

        for pl in bid_order:
            # everyone has passed
            if count == hands:
                if first_round:  # stuck the dealer
                    wb = min_bid
                    p(f"Dealer {pl} got stuck with {min_bid}")
                    if pl.is_bot:  # dealer picks suit
                        pl.make_bid(self.valid_bids, make_pinochle_deck(), hands)
                    wp = pl
                else:  # someone won the bid
                    wb = min_bid - 1
                break

            # end bidding early for a loner
            if min_bid > max_bid:
                wb = max_bid
                break

            # get the bid
            bid: int = pl.make_bid(
                self.valid_bids, deepcopy(self.deck), hands, min_bid, wp
            )

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

    def play_hand(self, dealer: Player) -> Player:
        self.deal()
        hn: int = len(dealer.team.score_changes) + 1
        p(f"\nHand {hn}")
        p(f"Dealer: {dealer}")
        po: List[Player] = get_play_order(dealer)
        po.append(po.pop(0))  # because the dealer doesn't lead bidding
        self.reset_suit_safety()  # reset the unsafe suits

        # deal the cards
        for pl in po:
            pl.tricks = 0
            pl.reset_bids()

        # bidding
        lead: Player = self.bidding(po)

        # declare Trump
        trump: Bid = lead.choose_trump
        p(trump)
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
        return dealer.next_player

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
                        broken_suits=self.suit_safety,
                        trump=self.trump,
                    ),
                    pl,
                )
            )
            p(f"{pl.name} played {repr(trick_in_progress[-1].card)}")
            po.append(pl)
            pl = pl.next_player

        # deal with card counting
        for i in range(len(trick_in_progress)):
            for j in range(i + 1, len(po)):
                try:
                    trick_in_progress[i].played_by.card_count.remove(
                        trick_in_progress[j].card
                    )
                except ValueError:
                    p((trick_in_progress[j].card, trick_in_progress[i].played_by))

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

    def write_log(self, ld: str, splitter: str = "\t|\t") -> None:
        stop_time: str = str(datetime.now()).split(".")[0]
        f: TextIO = open(os.path.join(ld, f"{self.start_time}.gamelog"), "w")
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
            self.current_dealer = self.play_hand(self.current_dealer)
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
) -> None:
    handedness: int = int(handedness)
    hands: int = handedness // 10
    teams: int = handedness % 10
    ppt: int = hands // teams
    start_time: str = str(datetime.now()).split(".")[0]
    global o
    global debug
    global log_dir
    if not humans:  # assume one human player as default
        humans = [random.randrange(hands)]
    if all_bots:
        humans = []
        o = open(os.path.join(log_dir, f"{start_time}.gameplay"), "w")
        debug = True
    player_handles: List[str] = {
        3: ["Alice", "Bob", "Charlie"],
        4: ["North", "East", "South", "West"],
        6: ["Richard", "Abbott", "Gallagher", "Mortimer", "Costello", "Alice"],
        8: [
            "Nelson",
            "Mortimer",
            "Eustace",
            "Costello",
            "Samantha",
            "Richard",
            "Wyclef",
            "Abbott",
        ],
    }[hands]
    if len(humans) == 1 and humans[0] < hands:
        player_handles[humans[0]] = "Human"
    p_types: List[Type[Player]] = [ComputerPlayer for _ in range(handedness)]
    for n in humans:
        if n < len(p_types):
            p_types[n] = HumanPlayer
    g = BidEuchre(
        player_names=player_handles,
        player_types=p_types,
        team_size=ppt,
        victory_threshold=points,
        start_time=start_time,
    )
    g.play()
    g.write_log(log_dir)


if __name__ == "__main__":
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    main()
