# ChePT

A Deep Transformer-Based Neural Chess Engine.  Baseline implementation using minGPT-2 architecture, and custom architectures.

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
