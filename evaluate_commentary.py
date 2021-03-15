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
import nltk

MASK_CHAR = u"\u2047"
engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


def get_prediction(game_str, gpt_model, vocabs, masks, size=10, sample=False):

    stoi = vocabs['stoi']
    itos = vocabs['itos']
    x = game_str + MASK_CHAR if masks else game_str
    x = torch.tensor([stoi[s] for s in x], dtype=torch.long)[None,...].to(device)

    pred = utils.sample(gpt_model, x, size, sample=sample)[0]
    completion = ''.join([itos[int(i)] for i in pred])
    if masks:
        return completion.split(MASK_CHAR)[1]
    else:
        return completion[len(game_str):]


def bot_vs_stockfish(game_str, comm_model, chept_model, comm_vocabs, chept_vocabs, args):

    commentaries = []
    board = chess.Board()
    while True:
        comp_move = engine.play(board, chess.engine.Limit(time=0.0005))
        game_str += board.san(comp_move.move) + ' '
        board.push(comp_move.move)

        if board.is_stalemate() or board.is_insufficient_material():
            break

        if board.is_checkmate():
            break

        # bot turn
        # handle cases where game str is larger than block size
        if len(game_str) >= 504:
            break
        bot_move = get_prediction(game_str, chept_model, chept_vocabs args.masks).split(' ')[0]

        try:
            board.push_san(bot_move)

        except ValueError:

            # try re-sampling
            success = False
            for i in range(args.n_tries):
                bot_move = get_prediction(game_str, chept_model, chept_vocabs, args.masks, sample=True).split(' ')[0]

                try:
                    board.push_san(bot_move)
                    success = True
                    break
                except ValueError:
                    pass

            if not success:
                bot_move = engine.play(board, chess.engine.Limit(time=0.05))
                bot_move_str = board.san(bot_move.move)
                board.push(bot_move.move)
                bot_move = bot_move_str

        game_str = game_str + bot_move + ' '

        # get commentary:
        commentary = get_prediction(game_str, comm_model, comm_vocabs, args.masks, size=args.comm_size)
        commentaries.append(game_str + commentary)

        if board.is_stalemate() or board.is_insufficient_material():
            break

        if board.is_checkmate():
            break

    return commentaries


def save_results(results, args, scenario):

    save_dir = 'commentary_results'
    save_dir = os.path.join(save_dir, scenario)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = args.save_name + '.json'
    save_path = os.path.join(save_dir, save_file)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def eval_scores(references, hypotheses):

    BLEUscores = []
    for i in range(references):
        reference, hypothesis = references[i], hypotheses[i]
        # TODO: weights based on lengths of sentences
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
        BLEUscores.append(BLEUscore)

    results_dict = {'BLEU Score': BLEUscores}
    return results_dict


def main(comm_model, chept_model, test_file, comm_vocabs, chept_vocabs, args):

    if chept_model and chept_vocabs:
        print(f'\nPlaying {args.n_games} games of ChePT vs. Stockfish')

        results = {}
        for i in tqdm(range(args.n_games)):
            result = bot_vs_stockfish('',
                                      comm_model,
                                      chept_model,
                                      comm_vocabs,
                                      chept_vocabs,
                                      args)
            results[i] = result
        save_results(results, args)
    else:
        test_scenarios = open(test_file).readlines()
        print(f'\nEvaluating {len(test_scenarios)} test scenarios.')
        references = []
        hypotheses = []
        for test in test_scenarios:
            split = test.split(MASK_CHAR)
            references.append(split[1].split(' '))
            commentary = get_prediction(split[0], comm_model, comm_vocabs, args.masks, size=args.comm_size)
            hypotheses.append(commentary.split(' '))

        # eval results (such as bleu score)
        results = eval_scores(references, hypotheses)
        save_results(results, args)


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

    parser = argparse.ArgumentParser()

    parser.add_argument('comm_ckpt', type=str, default='ckpts/commentary_final',
                        help='Path to commentary model to use')
    parser.add_argument('--chept_ckpt', type=str, default='ckpts/finetune_late/iter_152000.pt',
                        help='Path to ChePT model to use')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Test inputs to evaluate')
    parser.add_argument('--n_games', type=int, default=5,
                        help='Numer of games to evaluate')
    parser.add_argument('--n_tries', type=int, default=5,
                        help='Number of retries to give ChePT')
    parser.add_argument('--comm_size', type=int, default=200,
                        help='Number of samples to take for commentary generation')
    parser.add_argument('--masks', action='store_false',
                        help='Toggle masks OFF')

    args = parser.parse_args()

    comm_ckpt, chept_ckpt, test_file = args.comm_ckpt, args.chept_ckpt, args.test_file

    if os.path.isdir(comm_ckpt):
        comm_ckpt = get_recent_ckpt(comm_ckpt)
        print(f'Using {comm_ckpt} for commentary model')

    if chept_ckpt and test_file:
        chept_ckpt = None
        print('Evaluating model to find BLEU score.')
        assert os.path.isfile(test_file)
    else:
        assert os.path.isfile(chept_ckpt)

    suffix = '_with_chept' if chept_ckpt else '_score_eval'
    args.save_name = comm_ckpt.split('/')[1] + suffix
    # get ckpt
    comm_ckpt = torch.load(comm_ckpt, map_location=torch.device(device))
    comm_model_config = comm_ckpt['model_config']
    comm_itos = comm_ckpt['itos']
    comm_stoi = comm_ckpt['stoi']

    comm_vocab = {'itos': comm_itos,
                  'stoi': comm_stoi
                  }

    # build model config
    comm_mconf = model.GPTConfig(
        vocab_size=len(comm_itos),
        args_dict=comm_model_config.__dict__
    )

    # load model weights
    comm_model = model.GPT(comm_mconf)
    comm_model = comm_model.to(device)

    comm_model.load_state_dict(comm_ckpt['state_dict'])

    if chept_ckpt:
        chept_ckpt = torch.load(chept_ckpt, map_location=torch.device(device))
        chept_model_config = chept_ckpt['model_config']
        chept_itos = chept_ckpt['itos']
        chept_stoi = chept_ckpt['stoi']

        chept_vocab = {'itos': chept_itos,
                       'stoi': chept_stoi
                       }

        # build model config
        chept_mconf = model.GPTConfig(
            vocab_size=len(chept_itos),
            args_dict=chept_model_config.__dict__
        )

        # load model weights
        chept_model = model.GPT(chept_mconf)
        chept_model = chept_model.to(device)

        chept_model.load_state_dict(chept_ckpt['state_dict'])
    else:
        chept_model = None
        chept_vocab = None

    main(comm_model, chept_model, test_file, args)
    engine.quit()
