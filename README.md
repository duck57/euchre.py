# Double-deck bid euchre

Implementation is similar to [the rules given by Craig Powers](
https://www.pagat.com/euchre/bideuch.html).

Notable differences (to match how I learned in high school calculus) include:
* Minimum bid of `hand size / 2` (which can be stuck to the dealer)
* Trump isn't announced until after bidding has concluded
* Shooters and loners are separate bids
  * Shooters are worth `± 1.5 * hand size` points
  * Loners are worth double the hand size if you win and subtract the same as a shooter if you lose
  * Shooters are a mandatory donation (think Hearts) 
  * Shooting cards to your teammate happens between trump declaration and the first hand
  * The lone player in a shooter will have spare cards at the hand's end 
* Winner of bid leads the first hand
* Winning your bid gives you `(tricks earned + 2)` points


# Technical Notes

* `click` is the only external dependency
* `black` is in `requirements.txt` for developers: if you submit a PR, please blacken your code first
* `euchre.py --help` displays the other options to customize gameplay
  * `-h` changes the number of players for 3 or 6 hand games
  * `-v` modifies the victory thresholds, useful if you're short on time
  * `--all-bots` lets the computer play itself and puts the output to a file
* Running `euchre.py` without options plays a 4-handed game with one human player in a random seat
* There is a walrus or two, so Python 3.8 is required

# Future Plans

### Hearts?

Hearts may be in the cards for the future.  The major obstacle to implementing
Hearts is thinking of an algorithm for the computer players, especially
deciding when to shoot the moon or not and preventing other players from
successful moonshots.

Implementing the code for shooters made it the obvious next game.  Doing so
will require a refactor and some class inheritance for `Game` and `Player`.

### Bridge

### Single-Deck Euchre
