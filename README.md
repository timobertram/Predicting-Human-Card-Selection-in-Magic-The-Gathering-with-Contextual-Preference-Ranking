# Predicting Human Decision-Making with Contextual Preference Learning in Magic: The Gathering
Code and explanation for IEEE CoG paper "Predicting Human Decision-Making with Contextual Preference Learning in Magic: The Gathering"


# Necessary packages
- Pytorch 
- Sklearn


# Dataset 
Due to GitHub file size limits and copyright, we are not able to provide the dataset used. If you want to get the original dataset, visit https://draftsim.com/draft-data/.

# How to run
If you want to reproduce our work, proceed as follows:
1) Clone this repository
2) Download the associated dataset
3) Run preprocess.py with the path to the train.csv and test.csv file, e.g. "python preprocess.py Data/train.csv Data/test.csv"
4) Run training.py. If you did not change the default values, then no additional parameters are needed.

This will run the whole training for one epoch and regularly output current process, while saving the network.


