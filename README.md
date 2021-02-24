# ChePT

A Deep Transformer-Based Neural Chess Engine.  Baseline implementation using minGPT-2 architecture, and custom architectures.

To begin, install the relevant pretraining and finetuning datasets.  You'll need Kaggle and GSUtil, among other dependencies which we should really wrap into a `requirements.txt`.

    $ chmod +x install_datasets.sh
    $ ./install_datasets.sh

Then, you'll need to preprocess the datasets:

    $ python ./data/process-dataset.py
    
Now you're ready to go!

Training can be achieved through

    $ python run.py
    
Inference - and playing the bot with a gui - can be achieved through the interactive notebook `gui_inference.ipynb`.  Note that you will need to `brew install stockfish` in order to run comparative inference of the neural bot against Stockfish, and provide move support where the model otherwise might struggle (we're working on this!)
