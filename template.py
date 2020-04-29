#!venv/bin/python
# coding=UTF-8
# -*- coding: UTF-8 -*-
# vim: set fileencoding=UTF-8 :

"""
Module-level docstring here

This file (template.py) is marked +x for copying to new game files
"""

from cardstock import *


"""
The lines between here and the end of px need to be copied to your
game file or else output logging won't work.
"""

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


class SomeGame(BaseGame):
    """
    Implement all the abstract methods, gameplay, etc…
    """

    pass


@click.command()
@common_options
# other click options go here
def main(**kwargs):
    global o
    global debug
    global log_dir
    if kwargs.get("all_bots"):
        o = open(
            os.path.join(log_dir, f"{str(datetime.now()).split('.')[0]}.gameplay"), "w"
        )
        debug = True
    make_and_play_game(SomeGame, log_dir, **kwargs)


if __name__ == "__main__":
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    main()
