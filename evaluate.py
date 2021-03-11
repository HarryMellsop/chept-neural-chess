import chess
import numpy as np
import torch
from tqdm import tqdm
import model
import utils
import chess.engine

MASK_CHAR = u"\u2047"
engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


def get_prediction(game_str, gpt_model, stoi, itos, sample=False):
    x = game_str + MASK_CHAR
    x = torch.tensor([stoi[s] for s in x], dtype=torch.long)[None,...].to(device)

    pred = utils.sample(gpt_model, x, 10, sample=sample)[0]
    completion = ''.join([itos[int(i)] for i in pred])
    pred = completion.split(MASK_CHAR)[1].split(' ')[0]

    return pred


def bot_vs_stockfish(game_str, gpt_model, stoi, itos):
    winner = None
    final_illegal = 0
    first_bad_move = -1
    board = chess.Board()
    illegal_moves = []
    bot_move_count = 0

    while True:
        comp_move = engine.play(board, chess.engine.Limit(time=0.05))
        game_str += board.san(comp_move.move) + ' '
        board.push(comp_move.move)

        if board.is_checkmate():
            winner = "STOCKFISH"
            break

        # bot turn
        # handle cases where game str is larger than block size
        if len(game_str) >= 504:
            break
        bot_move = get_prediction(game_str, gpt_model, stoi, itos)
        bot_move_count += 1

        try:
            board.push_san(bot_move)
            illegal_moves.append(0)
        except ValueError:
            if first_bad_move == -1:
                first_bad_move = bot_move_count

            illegal_moves.append(1)

            # try re-sampling 5 times
            success = False
            for i in range(5):
                bot_move = get_prediction(game_str, gpt_model, stoi, itos, sample=True)

                try:
                    board.push_san(bot_move)
                    success = True
                    break
                except ValueError:
                    pass

            if not success:
                final_illegal += 1
                bot_move = engine.play(board, chess.engine.Limit(time=0.05))
                bot_move_str = board.san(bot_move.move)
                board.push(bot_move.move)
                bot_move = bot_move_str

        game_str = game_str + bot_move + ' '
        if board.is_checkmate():
            winner = "BOT"
            break

    return (game_str, illegal_moves, first_bad_move, final_illegal, winner)


def display_results(num_illegal_moves,
                    first_illegal_move,
                    total_black_moves,
                    final_illegal_moves,
                    winners,
                    num):

    z = np.array(first_illegal_move)
    curated_first_illegal = z[z != -1]

    print(f'Analyzed {num + 1} games...')
    print('On average, ChePT made:')
    print(f'\t\t\t{int(np.mean(total_black_moves))} moves per game.')
    print(f'\t\t\tFirst illegal move on move {int(np.mean(curated_first_illegal))}.')
    print(f'\t\t\t{int(np.mean(num_illegal_moves))} illegal moves per game.')
    print(f'\t\t\t{int(np.mean(final_illegal_moves))} final illegal moves per game.')

    print('')
    percent = np.round(np.mean(np.array(num_illegal_moves) / np.array(total_black_moves)), 4) * 100
    print(f'ChePT makes an illegal move {percent}% of the time')

    n_bot_wins = np.sum(np.array(winners) == 'BOT')

    print(f'\nChePT managed to win {n_bot_wins} games.')
    engine.quit()


def main(gpt_model, stoi, itos):

    num_illegal_moves = []
    first_illegal_move = []
    total_black_moves = []
    final_illegal_moves = []
    winners = []

    n_games = 5
    print(f'\nEvaluating {n_games} games')
    for i in tqdm(range(n_games)):
        game_str, illegal_moves, first_bad_move, final_illegal, winner = bot_vs_stockfish('',
                                                                                          gpt_model,
                                                                                          stoi,
                                                                                          itos)
        winners.append(winner)
        final_illegal_moves.append(final_illegal)
        black_moves = int(len(game_str.split()) / 2)
        total_black_moves.append(black_moves)
        first_illegal_move.append(first_bad_move)
        num_illegal_moves.append(sum(illegal_moves))

    display_results(num_illegal_moves,
                    first_illegal_move,
                    total_black_moves,
                    final_illegal_moves,
                    winners,
                    i)


if __name__ == '__main__':

    # TODO: smart grabbing of checkpoints + options
    # TODO: Evaluation of move ratio in addition to invalids

    # get ckpt
    ckpt = torch.load('ckpts/finetune_default/iter_18000.pt',
                      map_location=torch.device(device))
    model_config = ckpt['model_config']
    itos = ckpt['itos']
    stoi = ckpt['stoi']

    # build model config
    mconf = model.GPTConfig(
        vocab_size=len(itos),
        args_dict=model_config.__dict__
    )

    # load model weights
    gpt_model = model.GPT(mconf)
    gpt_model = gpt_model.to(device)

    gpt_model.load_state_dict(ckpt['state_dict'])

    main(gpt_model, stoi, itos)
