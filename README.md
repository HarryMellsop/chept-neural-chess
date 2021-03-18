# ChePT

A Deep Transformer-Based Neural Chess Engine.  Baseline implementation using minGPT-2 architecture.  Improved architecture based on 12-layer 512 block-size 32-head attention custom neural transformer architecture inspired by GPT.  You can play the bot!

## Table of Contents

- [ChePT Data](#chept-data)
- [Commentary Data](#commentary-data)
- [Training](#training)
- [Evaluation](#evaluation)

## ChePT Data
### Install the relevant pretraining and finetuning datasets.  You'll need Kaggle and GSUtil.

    $ chmod +x install_datasets.sh
    $ ./install_datasets.sh

Then, you'll need to preprocess the datasets:

    $ python ./data/process-dataset.py

## Commentary Data
### Obtain the raw data using a data crawler.

Clone this repository and follow the instructions to use their [``data/crawler``](https://github.com/harsh19/ChessCommentaryGeneration) tools to obtain data.

You will recieve four types of output files:
1. ```train.single.che```
2. ```train.multi.che```
3. ```train.single.en```
4. ```train.multi.en```

Then, you will need to use our ``process.ipynb`` to process the data. This notebook will allow you to:

* Analyze the size of examples and remove those larger than block size
* Remove badly formatted examples
* Removes examples in languages other than English
* Convert the data format to be comptable with our model
* Save the new data to a directory

Now you're ready to go!

## Training
Training can be achieved through

    $ python run.py
    
There are many different flags to utilize depending on whether you are pretraining or finetuning.
* ``function``: Pretrain or finetune model (pretrain, finetune])
* ``--version``: Finetune version
* ``--data_path``: Dataset to use
* ``--save_dir``: Directory to save checkpoints
* ``--pretrain_params``: Path to model params (use for finetune)
* ``--args_path``: Path to JSON training args
* ``--block_size``: Super config arg
* ``--n_layer``: Super config arg
* ``--n_head``: Super config arg
* ``--n_embed``: Super config arg
* ``--max_epoch``: Super train arg
* ``--batch_size``: Super train arg
* ``--learning_rate``: Super train arg
* ``--num_workers``: Super train arg

Any argument denoted with the word ``super`` will overwite arguments loaded in the pretrain_params file and the args_path file.
    
## Evaluation
TODO -- gui to see (with some pics/gifs), evaluate script (by model type), analyzing results (notebook)
Inference - and playing the bot with a gui - can be achieved through the interactive notebook 

    $ jupyter notebook gui_inference.ipynb
    
Note that you will need to `brew install stockfish` in order to run comparative inference of the neural bot against Stockfish 12.  The parameter file is currently not in git because of how large it is.  TODO: enable Git LFS, or provide another way for users to download the parameter file so that they don't need to train it.
