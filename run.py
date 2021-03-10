import argparse
import dataset
import model
import trainer
import utils
import questionary
import os


def main(data_path, config_args, train_args, func):

    # load pretrain dataset
    games = open(data_path).read()

    # pretrain
    print('\nProcessing dataset...')
    if func == 'pretrain':
        pretrain_dataset = dataset.PretrainDataset(games,
                                                   block_size=config_args['block_size'])

        breakpoint()

        print('reeee')

        # load model
        mconf = model.GPTConfig(
            vocab_size=pretrain_dataset.vocab_size,
            args_dict=config_args
        )
        gpt_model = model.GPT(mconf)

        train_config = trainer.TrainerConfig(args_dict=train_args)

        model_trainer = trainer.Trainer(gpt_model, pretrain_dataset,
                                        config=train_config)
        model_trainer.train()
    else:
        pretrain_dataset = dataset.PretrainDataset(games,
                                                   block_size=config_args['block_size'])
        raise NotImplementedError('REEEE')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('function', type=str,
                        help='Pretrain or finetune model.',
                        choices=["pretrain", "finetune"])
    parser.add_argument('--data_path', type=str,
                        help='Dataset to use.')
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save checkpoints.')

    # definitely use pretrain params when finetuning
    parser.add_argument('--pretrain_params', type=str,
                        help='Path to model params (use for finetune).')
    parser.add_argument('--args_path', type=str,
                        help='Path to JSON training args.')
    parser.add_argument('--block_size', type=int,
                        help='Super config arg.')
    parser.add_argument('--n_layer', type=int,
                        help='Super config arg.')
    parser.add_argument('--n_head', type=int,
                        help='Super config arg.')
    parser.add_argument('--n_embed', type=int,
                        help='Super config arg.')
    parser.add_argument('--max_epochs', type=int,
                        help='Super train arg.')
    parser.add_argument('--batch_size', type=int,
                        help='Super train arg.')
    parser.add_argument('--learning_rate', type=float,
                        help='Super train arg.')
    parser.add_argument('--num_workers', type=int,
                        help='Super train arg.')

    # WARNING: individual args superceded ARGS file
    args = parser.parse_args()

    # Double check args
    data_path = args.data_path
    save_dir = args.save_dir
    func = args.function

    if not data_path or (func == 'finetune' and not data_path):
        answer = questionary.confirm('Use default data--kingbase_cleaned.txt?').ask()
        if answer:
            data_path = 'data/datasets-cleaned/kingbase_cleaned.txt'
            assert os.path.isfile(data_path), 'DEFAULT DATA PATH NOT FOUND'
        else:
            raise FileExistsError('Must provide a dataset for training!')

    if not save_dir:
        save_dir = os.path.join('ckpts', func + '_default')

        answer = questionary.confirm(f'Use save directory at {save_dir}?').ask()
        if not answer:
            save_dir = questionary.text('Enter checkpoint save directory: ').ask()

    assert not os.path.isfile(save_dir), 'Directory cannot be a file!'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # TODO: Use state dict/params for finetuning
    # TODO: Ensure no conflict for args and pretrain for all
    if func == 'pretrain' and args.pretrain_params:
        assert questionary.confirm('Pretrain is provided with pretrain params. Continue?').ask(), \
            'Must provide a dataset for training!'
    if func == 'finetune' and not args.pretrain_params:
        raise ValueError('Cannot finteune without a pretrained model!')

    # Check config args
    meta_args = ['data_path', 'save_dir', 'function', 'pretrain_params']
    super_config_train_args = {key: val for key, val in vars(args).items() if key not in meta_args}     

    default_config_args = utils.default_config_args
    default_train_args = utils.default_train_args

    # No provided args
    if len(set(super_config_train_args.values())) == 1 and not set(super_config_train_args.values()).pop() and not args.args_path:
        print('NO ARGS PROVIDED. USING DEFAULT ARGS\n')
        print("Config Args:", default_config_args)
        print("Train Args:", default_train_args)

    # get separate updated config and train args
    arguments = utils.TrainArgs(args.args_path, super_config_train_args)
    config_args, train_args = arguments()

    main(data_path, config_args, train_args, func)
