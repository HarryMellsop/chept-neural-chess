# ChePT

A Deep Transformer-Based Neural Chess Engine.  Baseline implementation using minGPT-2 architecture.  Improved architecture based on 12-layer 512 block-size 32-head attention custom neural transformer architecture inspired by GPT.  You can play the bot!

To begin, install the relevant pretraining and finetuning datasets.  You'll need Kaggle and GSUtil, among other dependencies which we should really wrap into a `requirements.txt`.

    $ chmod +x install_datasets.sh
    $ ./install_datasets.sh

Then, you'll need to preprocess the datasets:

    $ python ./data/process-dataset.py
    
Now you're ready to go!

Training can be achieved through

    $ python run.py
    
Inference - and playing the bot with a gui - can be achieved through the interactive notebook 

    $ jupyter notebook gui_inference.ipynb
    
Note that you will need to `brew install stockfish` in order to run comparative inference of the neural bot against Stockfish 12, and provide move support where the model otherwise might struggle (we're working on this!).  The parameter file is currently not in git because of how large it is.  TODO: enable Git LFS, or provide another way for users to download the parameter file so that they don't need to train it.
