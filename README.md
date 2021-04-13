# Predicting Human Card Selection in Magic: The Gathering with Contextual Preference Ranking
Code and explanation for IEEE CoG paper "Predicting Human Card Selection in Magic: The Gathering with Contextual Preference Ranking"


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

# Notes
Currently this code is only used for the previously mentioned dataset. If you want to make this work for a different dataset, here are some guidelines:
1) If you want to run the code as is, you need to have three files: One with training data, one with testing data, and where all possible choices are visible. E.g. for this example, these are the 'training_datasets', 'test_datasets', and 'picks' folders
2) Training and test data are triplets, where the first element is a one-hot encoding of the positive, the second is a one-hot of the negative, and the third is a representation of the anchor.  Picks are similar, but the negative examples are a list of possibilities
3) Feel free to reach out if you have any questions


