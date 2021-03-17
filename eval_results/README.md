# Evaluate Results
Stores evaluation of ChePT model. Evaluations were produced using the `evaluate.py` script.

## Structure
Results are stored in JSON format. The keys explain which metric is being stored. Many of the values are averages over the total games played.
Thus, when running `evaluate.py`, use the `--n_games` flag to get an sufficient average.

The files are split by experiment. The name of each evaluation file corresponds to the model version which evaluated.
Each file here was done with experiments using 500 games.

### Move evaluation
Notably, the last 4 lines in the file cover some move evaluation metrics that we created. To find these metrics, while evaluating, we query ``Stockfish`` to 
predict a move independent of ours and evaluate that move in comparison to ours. We split the out PGN into thirds for ``early``, ``middle``, and ``late`` game.

We take the difference between the score for the move ChePT predicted and the move Stockfish predicted. We normalize each section by the sum of the
absolute value of the Stockfish move scores.

### Interpretation
These scores can correspond to the percent difference between the best Stockfish move and the predicted ChePT move. Negative values indicate that
ChePT made worse moves--while positive means ChePT made better moves.

For example, as result such as:
  
    "Average early game move evaluation": -0.3266849714737965,

Means that in the early game, ChePT's moves were roughly 32% worse than Stockfish's best move.
