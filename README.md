# ChePT

A Deep Transformer-Based Neural Chess Engine.  Baseline implementation using minGPT-2 architecture, and custom architectures.

To begin, install the relevant pretraining and finetuning datasets.  You'll need Kaggle and GSUtil, among other dependencies which we should really wrap into a `requirements.txt`.

    $ chmod +x install_datasets.sh
    $ ./install_datasets.sh

Then, you'll need to preprocess the datasets:

    $ python ./data/process-dataset.py
    
Now you're ready to go!
