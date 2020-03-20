# Double-deck bid euchre

Implementation is similar to [the rules given by Craig Powers](
https://www.pagat.com/euchre/bideuch.html).

Notable differences (to match how I learned in high school calculus) include:
* Minimum bid of `hand size / 2` (which can be stuck to the dealer)
* Trump isn't announced until after bidding has concluded
* Shooters and loners are separate bids
  * Shooters are worth `1.5 * hand size` points
  * Loners are worth twice the hand size
  * Losing a loner or shooter subtracts ⅚ the points you would have earned if you had won your bet
  * Shooters are a mandatory exchange with your partner(s)
  * Shooting cards to your teammate happens between trump declaration and the first hand 
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