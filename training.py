import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
import csv
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import random
import models
import utils
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.manifold import TSNE
import pickle


def compute_siamese_embedding_error(dataloader,network,loss_fn, plot = False):
    losses = list()
    network.eval()
    with torch.no_grad():
        i = 0
        for anchor,positive,negative in dataloader:
            if i == 10000:
                break
            out1 = network(anchor)
            out2 = network(positive)
            out3 = network(negative)
            loss = loss_fn(out1,out2,out3).item()
            losses.append(loss)
            i += 1
        if plot:
            X = np.array([out1.detach().squeeze().numpy(),out2.detach().squeeze().numpy(),out3.detach().squeeze().numpy()])
            X = TSNE(n_components=2).fit_transform(X)
            plt.scatter(X[0][0],X[0][1], label = 'Anchor')
            plt.scatter(X[1][0],X[1][1], label = 'Positive')
            plt.scatter(X[2][0],X[2][1], label = 'Negative')
            plt.legend()
            plt.show()
    return np.mean(losses)

def train_siamese(files,eval_dataloader,epochs,network,pick_path):
    train_errors = list()
    test_errors = list()
    comp_percentages = list()
    average_distances = list()
    correct_decisions = list()
    loss_fn = torch.nn.TripletMarginLoss()
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    with torch.no_grad():
        network = network.eval()
        test_errors.append(compute_siamese_embedding_error(eval_dataloader,network,loss_fn, plot = False))
        correct_picks,pick_distance = utils.correct_picks(network,pick_path)
        print('Correct decisions: ' + str(correct_picks))
        print('Average distance: ' + str(pick_distance))
        average_distances.append(pick_distance)
        correct_decisions.append(correct_picks)
    for epoch in range(epochs):
        network.train()
        epoch_training_loss = list()
        network = network.train()
        for l,file in enumerate(files):
            print('Subepoch:' + str(l))
            with open(file, 'rb') as f:
                dataloader = DataLoader(pickle.load(f), batch_size= 128)
                f.close()
            for anchor,positive,negative in dataloader:
                optimizer.zero_grad()
                out1 = network(anchor)
                out2 = network(positive)
                out3 = network(negative)
                loss = loss_fn(out1,out2,out3)
                loss.backward()
                optimizer.step()
                epoch_training_loss.append(loss.item())
            network = network.eval()
            with torch.no_grad():
                test_errors.append(compute_siamese_embedding_error(eval_dataloader,network,loss_fn, plot = False))
                train_errors.append(np.mean(epoch_training_loss))
                print('Training error: '+ str(train_errors[-1].item()))
                print('Test error: '+ str(test_errors[-1].item()))
                correct_picks,pick_distance = utils.correct_picks(network,pick_path)
                print('Correct decisions: ' + str(correct_picks))
                print('Average distance: ' + str(pick_distance))
                average_distances.append(pick_distance)
                correct_decisions.append(correct_picks)
                with open('training_progress.csv', 'w') as res:
                    writer = csv.writer(res)
                    writer.writerow(correct_decisions)
                    writer.writerow(average_distances)
                comp_percentage = evaluate_preferences(eval_dataloader,network)
                print(comp_percentage)
                comp_percentages.append(comp_percentage)
                torch.save(network.state_dict(),'network.pt')
                del dataloader
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(len(test_errors)),test_errors, label = 'Evaluation Error', color = 'orange')
    ax1.plot(range(len(train_errors)),train_errors,label = 'Training Error', color = 'blue')
    ax2.plot(range(len(correct_decisions)),correct_decisions,label = 'Correct picks', color = 'green')
    ax1.set_ylabel('Loss') 
    ax2.set_ylabel('Correct comparisons')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc = 0)
    plt.xlim(0,100)
    plt.show()
    return network

def load_network(features,output_dim,path):
    network = models.Siamese(features,output_dim)
    network.load_state_dict(torch.load(path))
    return network

def evaluate_preferences(dataset,net):
    comparisons = 0
    correct_comparisons = 0
    net.eval()
    for anchor,positive,negative in dataset:
        if comparisons == 100000:
            break
        comparisons += anchor.shape[0]
        out1 = net(anchor)
        out2 = net(positive)
        out3 = net(negative)
        for i in range(anchor.shape[0]):
            if models.get_distance(out1[i,:].view(1,-1),out2[i,:].view(1,-1)) < models.get_distance(out1[i,:].view(1,-1),out3[i,:].view(1,-1)):
                correct_comparisons += 1
    return correct_comparisons/comparisons

def main(training_path,test_path,pick_path):
    num_feature = 265
    epochs = 50
    output_dim = 256
    siamese = models.Siamese(num_feature,output_dim)

    training_files = list()
    for file in os.listdir(training_path):
        training_files.append(training_path+file)
    with open(test_path+np.random.choice(os.listdir(test_path)), 'rb') as f:
        eval_dataset = pickle.load(f)

    siamese = train_siamese(files = training_files,eval_dataloader= DataLoader(eval_dataset,batch_size=64),epochs = epochs, network = siamese,pick_path = pick_path)

if __name__ == "__main__":
    training_path = 'E:/training_mtg_datasets_2/'
    test_path = os.path.dirname(os.path.realpath(__file__)) + '/Data//test_datasets/'
    pick_path = os.path.dirname(os.path.realpath(__file__)) + '/Data//picks/'
    main(training_path,test_path,pick_path)










