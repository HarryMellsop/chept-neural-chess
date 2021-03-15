# ChePT

A Deep Transformer-Based Neural Chess Engine.  Baseline implementation using minGPT-2 architecture.  Improved architecture based on 12-layer 512 block-size 32-head attention custom neural transformer architecture inspired by GPT.  You can play the bot!

# ChePT Data
## To begin, install the relevant pretraining and finetuning datasets.  You'll need Kaggle and GSUtil.

    $ chmod +x install_datasets.sh
    $ ./install_datasets.sh

Then, you'll need to preprocess the datasets:

    $ python ./data/process-dataset.py

# Commentary Data
## First, you will need to obtain the raw data.

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

Training can be achieved through

    $ python run.py
    
Inference - and playing the bot with a gui - can be achieved through the interactive notebook 

    $ jupyter notebook gui_inference.ipynb
    
Note that you will need to `brew install stockfish` in order to run comparative inference of the neural bot against Stockfish 12, and provide move support where the model otherwise might struggle (we're working on this!).  The parameter file is currently not in git because of how large it is.  TODO: enable Git LFS, or provide another way for users to download the parameter file so that they don't need to train it.
