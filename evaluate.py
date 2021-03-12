import chess
import argparse
import questionary
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import model
import utils
import pickle
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


def bot_vs_stockfish(game_str, gpt_model, stoi, itos, args):
    winner = None
    final_illegal = 0
    first_bad_move = -1
    board = chess.Board()
    illegal_moves = []
    bot_scores = []
    comp_scores = []
    bot_move_count = 0

    while True:
        comp_move = engine.play(board, chess.engine.Limit(time=0.0005))
        game_str += board.san(comp_move.move) + ' '
        board.push(comp_move.move)

        if board.is_stalemate() or board.is_insufficient_material():
            winner = "DRAW"
            break

        if board.is_checkmate():
            winner = "STOCKFISH"
            break

        # bot turn
        # handle cases where game str is larger than block size
        true_bot_move = None
        if len(game_str) >= 504:
            break
        bot_move = get_prediction(game_str, gpt_model, stoi, itos)
        bot_move_count += 1

        try:
            board.push_san(bot_move)
            true_bot_move = bot_move
            illegal_moves.append(0)
        except ValueError:
            if first_bad_move == -1:
                first_bad_move = bot_move_count

            illegal_moves.append(1)

            # try re-sampling 5 times
            success = False
            for i in range(args.n_tries):
                bot_move = get_prediction(game_str, gpt_model, stoi, itos, sample=True)

                try:
                    board.push_san(bot_move)
                    true_bot_move = bot_move
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

        if true_bot_move:
            bot_score_tup = engine.analyse(board, chess.engine.Limit(time=0.1), game='key1')
            board.pop()
            comp_move = engine.play(board, chess.engine.Limit(time=0.05))
            board.push(comp_move.move)
            comp_score_tup = engine.analyse(board, chess.engine.Limit(time=0.1), game='key2')
            board.pop()
            board.push_san(true_bot_move)

            bot_score, comp_score = bot_score_tup['score'], comp_score_tup['score']
            if not bot_score.is_mate() and not comp_score.is_mate():
                bot_int = bot_score.black().score()
                comp_int = comp_score.black().score()
                bot_scores.append(bot_int)
                comp_scores.append(comp_int)

        game_str = game_str + bot_move + ' '

        if board.is_stalemate() or board.is_insufficient_material():
            winner = "DRAW"
            break

        if board.is_checkmate():
            winner = "BOT"
            break

    return (game_str, illegal_moves, first_bad_move, final_illegal, bot_scores, comp_scores, winner)


def save_results(results, args):

    save_dir = 'eval_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_file = args.save_name + '.json'
    save_path = os.path.join(save_dir, save_file)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def eval_moves(bot_arr, comp_arr):
    # TODO: eval by thirds (early, middle, end)

    assert len(bot_arr) == len(comp_arr)

    normalized = []

    for i in range(len(bot_arr)):
        bot_scores = np.array(bot_arr[i])
        comp_scores = np.array(comp_arr[i])

        diff_arr = bot_scores - comp_scores
        norm_factor = np.mean(np.abs(comp_scores))
        normalized.append(diff_arr / norm_factor)
    
    avg_scores = [np.mean(x) for x in normalized]
    total_avg = np.mean(avg_scores)
    return total_avg

def display_results(num_illegal_moves,
                    first_illegal_move,
                    total_black_moves,
                    final_illegal_moves,
                    winners,
                    bot_arr,
                    comp_arr,
                    num,
                    args):

    z = np.array(first_illegal_move)
    curated_first_illegal = z[z != -1]
    n_without_illegal = z.shape[0] - curated_first_illegal.shape[0]

    move_evals = eval_moves(bot_arr, comp_arr)

    with open('bot_scores.pkl', 'wb') as f:
        pickle.dump(bot_arr, f)
    
    with open('comp_scores.pkl', 'wb') as f:
        pickle.dump(comp_arr, f)

    print(f'Analyzed {num + 1} games...')
    print('On average, ChePT made:')
    print(f'\t\t\t{int(np.mean(total_black_moves))} moves per game.')
    print(f'\t\t\tFirst illegal move on move {int(np.mean(curated_first_illegal))}.')
    print(f'\t\t\t{int(np.mean(num_illegal_moves))} attempted illegal moves per game.')
    print(f'\t\t\t{int(np.mean(final_illegal_moves))} final illegal moves per game.')

    print('')
    print(f'ChePT had {n_without_illegal} games with no illegal moves.')
    attempt_percent = np.round(np.mean(np.array(num_illegal_moves) / np.array(total_black_moves)) * 100, 3)
    final_percent = np.round(np.mean(np.array(final_illegal_moves) / np.array(total_black_moves)) * 100, 3)
    print(f'ChePT attempts to make an illegal move {attempt_percent}% of the time')
    print(f'ChePT finally makes an illegal move {final_percent}% of the time')

    n_bot_wins = np.sum(np.array(winners) == 'BOT')
    n_draws = np.sum(np.array(winners) == 'DRAW')

    print(f'\nChePT managed to win {n_bot_wins} games.')
    print(f'ChePT managed to draw {n_draws} games.')

    results = {'Number of games': num + 1,
               'Number of moves': int(np.mean(total_black_moves)),
               'First illegal move': int(np.mean(curated_first_illegal)),
               'Attempted illegal moves': int(np.mean(num_illegal_moves)),
               'Games without illegal moves': n_without_illegal,
               'Final illegal moves': int(np.mean(final_illegal_moves)),
               'Percent attempted illegal moves': float(attempt_percent),
               'Percent final illegal moves': float(final_percent),
               'Wins': int(n_bot_wins),
               'Draws': int(n_draws),
               'Average move evaluation': float(move_evals)}

    save_results(results, args)

    engine.quit()


def main(gpt_model, stoi, itos, args):

    num_illegal_moves = []
    first_illegal_move = []
    total_black_moves = []
    final_illegal_moves = []
    bot_arr = []
    comp_arr = []
    winners = []

    print(f'\nEvaluating {args.n_games} games')
    for i in tqdm(range(args.n_games)):
        game_str, illegal_moves, first_bad_move, final_illegal, bot_scores, comp_scores, winner = bot_vs_stockfish('',
                                                                                                 gpt_model,
                                                                                                 stoi,
                                                                                                 itos,
                                                                                                 args)
        winners.append(winner)
        bot_arr.append(bot_scores)
        comp_arr.append(comp_scores)
        final_illegal_moves.append(final_illegal)
        black_moves = int(len(game_str.split()) / 2)
        total_black_moves.append(black_moves)
        first_illegal_move.append(first_bad_move)
        num_illegal_moves.append(sum(illegal_moves))

    print('\nCalculating eval metrics\n')
    display_results(num_illegal_moves,
                    first_illegal_move,
                    total_black_moves,
                    final_illegal_moves,
                    winners,
                    bot_arr,
                    comp_arr,
                    i,
                    args)


def get_recent_ckpt(ckpt_dir):

    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"Default checkpoint dir at {ckpt_dir} missing!")

    files = os.listdir(ckpt_dir)
    if 'best_loss.pt' in files:
        answer = questionary.confirm("File best_loss.pt found. Use this file?").ask()
        if answer:
            return os.path.join(ckpt_dir, 'best_loss.pt')
    epoch_list = [x for x in files if 'epoch' in x]
    if len(epoch_list) > 0:
        answer = questionary.confirm("Epoch files found. Use best epoch file?").ask()
        if answer:
            epoch_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            return os.path.join(ckpt_dir, epoch_list[0])

    iter_list = [x for x in files if 'iter' in x]
    iter_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)

    return os.path.join(ckpt_dir, iter_list[0])


if __name__ == '__main__':

    # TODO: auto-grab new ckpnts?
    # TODO: Evaluation of move ratio in addition to invalids

    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint to evaluate')
    parser.add_argument('--n_games', type=int, default=5,
                        help='Numer of games to evaluate')
    parser.add_argument('--n_tries', type=int, default=5,
                        help='Number of retries to give ChePT')
    args = parser.parse_args()

    if not args.ckpt:
        ckpt_path = get_recent_ckpt('ckpts/finetune_default')
        print("\nWARNING: NO CHECKPOINT GIVEN")
        print(f"Using {ckpt_path}")
    else:
        ckpt_path = args.ckpt
    args.save_name = ckpt_path.split('/')[1]
    # get ckpt
    ckpt = torch.load(ckpt_path,
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

    main(gpt_model, stoi, itos, args)
