import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch.nn as nn
import models as networks
import pickle
import os
import random
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import collections

def compute_pick_chance(path):
    counts = np.zeros(265)
    picks = np.zeros(265)

    for filename in os.listdir(path):
        with open(path+filename, 'rb') as f:
            data = pickle.load(f)
            last_deck = None
            last_pick = None
            for line in data:
                line = line.split(';')
                deck = line[2]
                if deck != last_deck or line[0] != last_pick:
                    counts[int(line[0])] += 1
                    picks[int(line[0])] += 1
                
                if line[1]:
                    counts[int(line[1])] += 1
                last_deck = deck
                last_pick = line[0]
    with open('pickrates.csv', 'w') as outfile:
        for i in range(265):
            outfile.write(str(picks[i]/counts[i])+';'+str(picks[i])+'\n')

def compute_firstpick_chance(path):
    counts = np.zeros(265)
    picks = np.zeros(265)

    for filename in os.listdir(path):
        with open(path+filename, 'rb') as f:
            data = pickle.load(f)
            last_deck = None
            last_pick = None
            for line in data:
                line = line.split(';')
                deck = line[2]
                if not deck:
                    if deck != last_deck or line[0] != last_pick:
                        counts[int(line[0])] += 1
                        picks[int(line[0])] += 1
                    
                    if line[1]:
                        counts[int(line[1])] += 1
                    last_deck = deck
                    last_pick = line[0]
    with open('pickrates.csv', 'w') as outfile:
        for i in range(265):
            outfile.write(str(picks[i]/counts[i])+';'+str(picks[i])+'\n')
                
def pickrates_to_cards(dict_file,score_file):
    with open(dict_file,'rb') as file:
        card_dict = pickle.load(file)
    for key in [k for k in card_dict.keys()]:
        card_dict[card_dict[key]] = key
    with open(score_file, 'r') as infile:
        with open('named_'+score_file,'w') as outfile:
            i = 0
            for line in infile:
                outfile.write(card_dict[i]+';'+line)
                i += 1

def point_cloud(network,embedding = None):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}
    
    cards = list()
    if embedding is not None:
        tsne_embedding = embedding
    embeddings = list()
    labels = list()
    colors = list()

    distances = list()
    with open('Data/named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        distances.append(0)
        embeddings.append(anchor.squeeze().numpy())
        colors.append('lime')
        border_colors = list()
        border_colors.append('lime')
        color_map = {'Colourless' : 'purple', 'Red': 'red', 'Blue':'blue', 'Green':'green', 'White':'lightgrey','Black':'black'}
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            embeddings.append(output.squeeze().numpy())
            distances.append(networks.get_distance(anchor,output).item())
            c = lines[card][4]
            c = c.split(',')
            if len(c) == 1:
                colors.append(color_map[c[0]])
                border_colors.append(color_map[c[0]])
            elif len(c) == 2:
                colors.append(color_map[c[0]])
                border_colors.append(color_map[c[1]])
            else:
                colors.append('gold')
                border_colors.append('gold')

    if embedding is None:
        tsne_embedding = TSNE(n_components = 2).fit_transform(embeddings)
    tsne_distances = [networks.get_distance(torch.Tensor(tsne_embedding[0]).view(1,-1),torch.Tensor(tsne_embedding[i]).view(1,-1)) for i in range(1,len(tsne_embedding))]
    max_distance = np.max(distances)
    plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1], c =colors, edgecolors = border_colors)
    plt.annotate('Anchor',(tsne_embedding[0,0],tsne_embedding[0,1]))
    plt.savefig('plots/point_cloud.pdf', bbox_inches='tight' )
    plt.show()
    return tsne_embedding

def point_cloud_rarity(network,embedding = None):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}
    
    cards = list()
    if embedding is not None:
        tsne_embedding = embedding
    embeddings = list()
    labels = list()
    colors = list()

    distances = list()
    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        distances.append(0)
        embeddings.append(anchor.squeeze().numpy())
        colors.append('lime')
        border_colors = list()
        border_colors.append('lime')
        color_map = {'C' : 'black', 'U': 'silver', 'R':'gold', 'M':'brown'}
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            embeddings.append(output.squeeze().numpy())
            distances.append(networks.get_distance(anchor,output).item())
            c = lines[card][3]
            c = c.split(',')
            if len(c) == 1:
                colors.append(color_map[c[0]])
                border_colors.append(color_map[c[0]])
            elif len(c) == 2:
                colors.append(color_map[c[0]])
                border_colors.append(color_map[c[1]])
            else:
                colors.append('gold')
                border_colors.append('gold')

    if embedding is None:
        tsne_embedding = TSNE(n_components = 2).fit_transform(embeddings)
    tsne_distances = [networks.get_distance(torch.Tensor(tsne_embedding[0]).view(1,-1),torch.Tensor(tsne_embedding[i]).view(1,-1)) for i in range(1,len(tsne_embedding))]
    max_distance = np.max(distances)
    plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1], c =colors, edgecolors = border_colors)
    plt.annotate('Anchor',(tsne_embedding[0,0],tsne_embedding[0,1]))

    with open('plot_understanding.csv','w') as tmp:
        writer = csv.writer(tmp)
        for i in range(len(tsne_embedding)-1):
            lines[i].extend(tsne_embedding[i+1][:])
            writer.writerow(lines[i])
    rank = construct_ranking(distances,reverse = False)
    for card in rank[1:]:
        print(inv_card_dict[card-1])
    print(np.argmax(pickrates))
    plt.savefig('plots/point_cloud_rarity.pdf', bbox_inches='tight' )
    plt.show()
    return tsne_embedding

def plot_distance_against_fpr(network):
    network = network.eval()
    distances = list()
    with open('named_firstpickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            distances.append(networks.get_distance(anchor,output).item())

    elements = [(pickrates[i],distances[i]) for i in range(265)]
    elements.sort(key = lambda v: v[0])

    plt.scatter([element[0] for element in elements],[element[1] for element in elements], alpha = 0.6)
    plt.xlabel('First-pick rate')
    plt.ylabel('Distance to empty set')
    plt.savefig('plots/fpr_distance_2.pdf', bbox_inches='tight' )

def point_cloud_anchors(network,embedding = None):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}
    
    cards = list()
    if embedding is not None:
        tsne_embedding = embedding
    embeddings = list()
    labels = list()
    colors = list()

    distances = list()
    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        distances.append(0)
        embeddings.append(anchor.squeeze().numpy())
        colors.append('lime')
        border_colors = list()
        border_colors.append('lime')
        color_map = {'Colourless' : 'purple', 'Red': 'red', 'Blue':'blue', 'Green':'green', 'White':'beige','Black':'black'}
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            embeddings.append(output.squeeze().numpy())
            distances.append(networks.get_distance(anchor,output).item())
            colors.append('black')
            border_colors.append('black')

        for i in range(50):
            tmp = torch.zeros((1,265))
            for _ in range(45):
                tmp[0,np.random.randint(265)] = 1
            embeddings.append(network(tmp).squeeze().numpy())
            colors.append('red')
            border_colors.append('red')
        
        for i in range(50):
            with open('E:/picks_2/'+np.random.choice(os.listdir('E:/picks_2/')), 'rb') as pick_file:
                some_picks = np.load(pick_file, allow_pickle = True)
                anchor = 0
                while anchor != 44:
                    anchor_array = some_picks[np.random.randint(some_picks.shape[0])]
                    anchor = sum(anchor_array[1])
                input = anchor_array[1]+anchor_array[2]
                tmp = torch.Tensor(input)
                embeddings.append(network(tmp).squeeze().numpy())
                colors.append('blue')
                border_colors.append('blue')
        
            

    if embedding is None:
        tsne_embedding = TSNE(n_components = 2).fit_transform(embeddings)
    tsne_distances = [networks.get_distance(torch.Tensor(tsne_embedding[0]).view(1,-1),torch.Tensor(tsne_embedding[i]).view(1,-1)) for i in range(1,len(tsne_embedding))]
    max_distance = np.max(distances)
    plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1], c =colors, edgecolors = border_colors)
    plt.annotate('Anchor',(tsne_embedding[0,0],tsne_embedding[0,1]))
    plt.savefig('plots/point_cloud_sets.pdf', bbox_inches='tight' )
    plt.show()
    return tsne_embedding

def point_cloud_anchors_colors(network,embedding = None):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}
    
    cards = list()
    if embedding is not None:
        tsne_embedding = embedding
    embeddings = list()
    labels = list()
    colors = list()

    distances = list()
    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        distances.append(0)
        embeddings.append(anchor.squeeze().numpy())
        colors.append('lime')
        border_colors = list()
        border_colors.append('lime')
        color_map = {'Colourless' : 'purple', 'Red': 'red', 'Blue':'blue', 'Green':'green', 'White':'beige','Black':'black'}
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            embeddings.append(output.squeeze().numpy())
            distances.append(networks.get_distance(anchor,output).item())
            colors.append('grey')
            border_colors.append('black')
        
        for i in range(50):
            with open('E:/picks_2/'+np.random.choice(os.listdir('E:/picks_2/')), 'rb') as pick_file:
                some_picks = np.load(pick_file, allow_pickle = True)
                anchor = 0
                c = False
                while anchor != 44 or not c:
                    anchor_array = some_picks[np.random.randint(some_picks.shape[0])]
                    anchor = sum(anchor_array[1])
                    input = anchor_array[1]+anchor_array[2]
                    if anchor == 44:
                        c = matching_colors(lines,input,'Red')                 
                        if c:
                            
                            tmp = torch.Tensor(input)
                            embeddings.append(network(tmp).squeeze().numpy())
                            colors.append(color_map[c])
        
            

    if embedding is None:
        tsne_embedding = TSNE(n_components = 2).fit_transform(embeddings)
    tsne_distances = [networks.get_distance(torch.Tensor(tsne_embedding[0]).view(1,-1),torch.Tensor(tsne_embedding[i]).view(1,-1)) for i in range(1,len(tsne_embedding))]
    max_distance = np.max(distances)
    plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1], c =colors)
    plt.annotate('Anchor',(tsne_embedding[0,0],tsne_embedding[0,1]))
    plt.savefig('plots/point_cloud_sets_colored.pdf', bbox_inches='tight' )
    plt.show()
    return tsne_embedding

def matching_colors(file,deck,match_color):
    color_count = collections.defaultdict(int)
    for i,card in enumerate(deck):
        for j in range(int(card)):
            color_count[file[i][4]] += 1
    out = {k: v for k, v in sorted(color_count.items(), key=lambda item: item[1], reverse = True)}
    items = list(out.keys())
    if items[0] != match_color and items[1] != match_color:
        return False
    else:
        if items[0] == match_color:
            return items[1]
        else:
            return items[0]

def point_cloud_anchors_distances(network,embedding = None):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}
    
    cards = list()
    if embedding is not None:
        tsne_embedding = embedding
    embeddings = list()
    labels = list()
    random_colors = list()
    deck_colors = list()
    tensor_embeddings = list()

    distances = list()
    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        distances.append(0)
        embeddings.append(anchor.squeeze().numpy())
    
        border_colors = list()
        border_colors.append('lime')
        color_map = {'Colourless' : 'purple', 'Red': 'red', 'Blue':'blue', 'Green':'green', 'White':'beige','Black':'black'}
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            tensor_embeddings.append(output)
            embeddings.append(output.squeeze().numpy())
            distances.append(networks.get_distance(anchor,output).item())

        for i in range(1):
            tmp = torch.zeros((1,265))
            for _ in range(45):
                tmp[0,np.random.randint(265)] = 1
            out = network(tmp)
            random_distances = [networks.get_distance(out,elem).item() for elem in tensor_embeddings]
            random_colors.extend(random_distances)
            embeddings.append(out.squeeze().numpy())
        
        for i in range(1):
            with open('E:/picks_2/'+np.random.choice(os.listdir('E:/picks_2/')), 'rb') as pick_file:
                some_picks = np.load(pick_file, allow_pickle = True)
                anchor = 0
                while anchor != 44:
                    anchor_array = some_picks[np.random.randint(some_picks.shape[0])]
                    anchor = sum(anchor_array[1])
                input = torch.Tensor(anchor_array[1]+anchor_array[2])
                
                out = network(input)
                deck_distances = [networks.get_distance(out,elem).item() for elem in tensor_embeddings]
                deck_colors.extend(deck_distances)
                embeddings.append(out.squeeze().numpy())
        
            

    if embedding is None:
        tsne_embedding = TSNE(n_components = 2).fit_transform(embeddings)
    plt.scatter(tsne_embedding[1:-2,0],tsne_embedding[1:-2,1], c =random_colors ,cmap = plt.cm.rainbow)
    plt.scatter(tsne_embedding[0,0],tsne_embedding[0,1], c = 'lime')
    plt.scatter(tsne_embedding[-2,0],tsne_embedding[-2,1], c = 'black')
    plt.annotate('Anchor',(tsne_embedding[0,0],tsne_embedding[0,1]))
    plt.colorbar()
    plt.savefig('plots/point_cloud_sets_random.pdf', bbox_inches='tight' )
    plt.show()

    
    plt.scatter(tsne_embedding[1:-2,0],tsne_embedding[1:-2,1], c =deck_colors, cmap = plt.cm.rainbow)
    plt.scatter(tsne_embedding[0,0],tsne_embedding[0,1], c = 'lime')
    plt.scatter(tsne_embedding[-1,0],tsne_embedding[-1,1], c = 'black')
    plt.annotate('Anchor',(tsne_embedding[0,0],tsne_embedding[0,1]))
    plt.colorbar()
    plt.savefig('plots/point_cloud_sets_deck.pdf', bbox_inches='tight' )
    plt.show()
    return tsne_embedding


def point_cloud_distance_colour(network,embedding = None):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}
    
    cards = list()

    if embedding:
        tsne_embedding = embedding
    else:
        embeddings = list()
    labels = list()
    colors = list()

    distances = list()
    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        distances.append(0)
        embeddings.append(anchor.squeeze().numpy())
        colors.append('lime')
        border_colors = list()
        border_colors.append('lime')
        color_map = {'Colourless' : 'purple', 'Red': 'red', 'Blue':'blue', 'Green':'green', 'White':'lightgrey','Black':'black'}
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            embeddings.append(output.squeeze().numpy())
            distances.append(networks.get_distance(anchor,output).item())
            c = lines[card][4]
            c = c.split(',')
            if len(c) == 1:
                colors.append(color_map[c[0]])
                border_colors.append(color_map[c[0]])
            elif len(c) == 2:
                colors.append(color_map[c[0]])
                border_colors.append(color_map[c[1]])
            else:
                colors.append('gold')
                border_colors.append('gold')

    if not embedding:
        tsne = TSNE(n_components = 2)
        tsne_embedding = TSNE(n_components = 2).fit_transform(embeddings)
    tsne_distances = [networks.get_distance(torch.Tensor(tsne_embedding[0]).view(1,-1),torch.Tensor(tsne_embedding[i]).view(1,-1)) for i in range(1,len(tsne_embedding))]
    max_distance = np.max(distances)
    plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1], c =distances,cmap = plt.cm.coolwarm_r)
    plt.annotate('Anchor',(tsne_embedding[0,0],tsne_embedding[0,1]))
   
    print(construct_ranking(distances,reverse = False))
    print(np.argmax(pickrates))
    cbar = plt.colorbar()
    cbar.set_ticks([0,np.max(distances)])
    cbar.set_ticklabels(['Min', 'Max'])
    plt.savefig('plots/point_cloud_distances.pdf', bbox_inches='tight' )
    plt.show()
    return tsne_embedding


def complete_draft(picks,network):
    network.eval()
    every_choice = list()
    every_chosen = list()
    every_anchor = list()
    random_choices = list()
    random_anchors = list()
    siamese_distance = 0
    random_distance = 0
    with torch.no_grad():
        picked  = torch.zeros((1,265))
        random_picked = torch.zeros((1,265))
        for pick in picks:
            anchor = network(picked)
            random_anchor = network(random_picked)
            every_anchor.append(anchor.squeeze().detach().numpy())
            random_anchors.append(random_anchor.squeeze().detach().numpy())
            choices = pick[0]
            every_choice.append(choices)
            distances = list()
            for choice in choices:
                tmp = torch.zeros((1,265))
                tmp[0,choice] = 1
                distances.append(networks.get_distance(anchor,network(tmp)).item())
            chosen_card = choices[np.argmin(distances)]
            siamese_distance += np.min(distances)
            random_index = np.random.randint(0,len(choices))
            randomly_chosen_card = choices[random_index]
            indx = list(np.where(choices == randomly_chosen_card))[0]
            random_distance += distances[random_index]
            picked[0,chosen_card] += 1
            random_picked[0,randomly_chosen_card] += 1
            every_chosen.append(chosen_card)
            random_choices.append(randomly_chosen_card)
        every_anchor.append(anchor.squeeze().detach().numpy())
    return every_choice,every_chosen,every_anchor,random_choices,random_anchors,siamese_distance,random_distance





def point_cloud_path(network):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}

    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    
    cards = list()

    embeddings = list()
    cards = list()
    labels = list()
    colors = list()

    pick_path = 'E:/picks_2/'
    with open(pick_path+os.listdir(pick_path)[0], 'rb') as f:
        picks = np.load(f, allow_pickle = True)
        p1_picks = list()
        for i in range(45):
            p1_picks.append(picks[i*8])

    with torch.no_grad():
        for pick in p1_picks:
            embeddings.append(network(torch.Tensor(pick[1])).view(1,-1).squeeze().numpy())
        
        every_choice,every_chosen,every_anchor,random_choices,random_anchors = complete_draft(p1_picks,network)
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            cards.append(output.squeeze().numpy())
        
        all_things = list()
        all_things.extend(cards)
        all_things.extend(every_anchor)

    tsne_embedding = TSNE(n_components = 2).fit_transform(all_things)
    route = list()
    choices = list()
    for choice in every_choice:
        choices.append([tsne_embedding[i] for i in choice])
    chosens = [tsne_embedding[i] for i in every_chosen]


    for i in range(len(chosens)):
        x = [row[0] for row in choices[i]]
        y = [row[1] for row in choices[i]]
        plt.scatter(x,y, c = 'blue')
        plt.scatter(chosens[i][0],chosens[i][1], c = 'red', s = 10, alpha = 0.5)
        plt.scatter(tsne_embedding[265+i,0],tsne_embedding[265+i,1], c = 'green')
        plt.show()
    network = network.train()
    return tsne_embedding


def point_cloud_path_all_cards(network):
    network = network.eval()
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}

    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    
    cards = list()

    embeddings = list()
    cards = list()
    labels = list()
    colors = list()

    pick_path = 'E:/picks_2/'
    with open(pick_path+os.listdir(pick_path)[0], 'rb') as f:
        picks = np.load(f, allow_pickle = True)
        p1_picks = list()
        for i in range(45):
            p1_picks.append(picks[i*8])

    with torch.no_grad():
        for pick in p1_picks:
            embeddings.append(network(torch.Tensor(pick[1])).view(1,-1).squeeze().numpy())
        
        every_choice,every_chosen,every_anchor,random_choices,random_anchors,siamese_distance,random_distance = complete_draft(p1_picks,network)
        print('Siamese distance: ' +str(siamese_distance))
        print('Random distance: ' + str(random_distance))
        for card in range(265):
            tmp = torch.zeros((1,265))
            tmp[0,card] = 1
            output = network(tmp)
            cards.append(output.squeeze().numpy())
            c = lines[card][4]
            if c == 'Colourless':
                colors.append('purple')
            elif c == 'Red':
                colors.append('red')
            elif c == 'Blue':
                colors.append('blue')
            elif c == 'Green':
                colors.append('green')
            elif c == 'Black':
                colors.append('black')
            elif c == 'White':
                colors.append('lavenderblush')
            else:
                colors.append('gold')
        
        all_things = list()
        all_things.extend(cards)
        all_things.extend(every_anchor)
        all_things.extend(random_anchors)

    tsne_embedding = TSNE(n_components = 2).fit_transform(all_things)
    route = list()
    chosens = [tsne_embedding[i] for i in every_chosen]
    random_chosens = [tsne_embedding[i] for i in random_choices]

    plt.scatter(tsne_embedding[:265,0],tsne_embedding[:265,1], c = colors)
    plt.plot(tsne_embedding[265:265+46,0],tsne_embedding[265:265+46,1], c = 'blue', label = 'Siamese path') 
    for i,choice in enumerate(chosens):
        plt.plot([tsne_embedding[265+i,0],choice[0]],[tsne_embedding[265+i,1],choice[1]],'--',c = 'blue', alpha = 0.2)
    for i,choice in enumerate(random_chosens):
        plt.plot([tsne_embedding[265+46+i,0],choice[0]],[tsne_embedding[265+46+i,1],choice[1]],'--',c = 'green', alpha = 0.2)
    plt.plot(tsne_embedding[-46:,0],tsne_embedding[-46:,1], c = 'green', label = 'Random path')
    plt.legend()
    plt.show()
    network = network.train()
    return tsne_embedding


def plot_random_embeddings(network):
    card_dict = load_card_dict('card_dict.pt')
    inv_card_dict = {v:k for k,v in card_dict.items()}
    
    cards = list()
    for i in range(265):
        tmp = np.zeros(265)
        tmp[i] = 1
        cards.append(torch.Tensor(tmp))

    embeddings = list()
    labels = list()
    colors = list()

    distances = list()
    with open('named_pickrates.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        lines = [line for line in reader]
        indizes = [i for i in range(265)]
        pickrates = [float(lines[index][1]) for index in indizes]
    with torch.no_grad():
        empty = torch.zeros(1,265)
        anchor = network(empty)
        embeddings.append(anchor.squeeze().numpy())
        labels.append('Empty set')
        colors.append('r')
        for card in cards:
            output = network(card)
            embeddings.append(output.squeeze().numpy())
            distances.append(networks.get_distance(anchor,output))
      #      labels.append(inv_card_dict[card.item()])
            colors.append('b')
                

    tsne_embedding = TSNE(n_components = 2).fit_transform(embeddings)
    tsne_distances = [networks.get_distance(torch.Tensor(tsne_embedding[0]).view(1,-1),torch.Tensor(tsne_embedding[i]).view(1,-1)) for i in range(1,len(tsne_embedding))]
    plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1], c = colors)
    #for i in range(1,11):
    #    plt.plot([tsne_embedding[0][0],tsne_embedding[i][0]],[tsne_embedding[0][1],tsne_embedding[i][1]])
    #    plt.annotate(str(pickrates[i-1]),((tsne_embedding[i,0]+tsne_embedding[0,0])/2,(tsne_embedding[i,1]+tsne_embedding[0,1])/2))
    #for i,label in enumerate(labels):
    #    plt.annotate(label, (tsne_embedding[i,0],tsne_embedding[i,1]))
    print(distances)
    print(pickrates)
    plt.show()
    pickrate_ranking = construct_ranking(pickrates,True)
    print(pickrate_ranking)
    distance_ranking = construct_ranking(distances,False)
    print(distance_ranking)
    tsne_distances_ranking = construct_ranking(tsne_distances,False)
    print(tsne_distances_ranking)
    print('Embedded correlation: '+ str(sc.stats.kendalltau(pickrate_ranking,distance_ranking)[0]))
    print('TSNE correlation: '+ str(sc.stats.kendalltau(pickrate_ranking,tsne_distances_ranking)[0]))
    return tsne_embedding

def construct_ranking(values,reverse):
    indizes = range(len(values))
    return [x for _,x in sorted(zip(values,indizes),reverse = reverse)]


def correct_picks(network,folder_name):
    pick_number = 0
    correct_picks = 0
    pick_distance = list()
    pick_numbers = list()
    files = os.listdir(folder_name)
    with open(folder_name+np.random.choice(files),'rb') as f:
        data = np.load(f, allow_pickle = True)
        for pick in data[np.random.choice(data.shape[0], 10000, replace=False), :]:
            pick_number += 1
            choices = pick[0]
            pick_numbers.append(len(choices))
            anchor = pick[1]
            correct_choice = np.argmax(pick[2])
            with torch.no_grad():
                out1 = network(torch.Tensor(anchor).view(1,-1))

                distances = list()
                for i in range(len(choices)):
                    elem = torch.zeros((1,265))
                    elem[0,choices[i]] = 1.0
                    distances.append(networks.get_distance(out1,network(elem)).item())

                ranking = [x for _,x in sorted(zip(distances,[y for y in range(len(distances))]))]
                if choices[ranking[0]] == correct_choice:
                    correct_picks += 1
                    pick_distance.append(0)
                else:
                    pick_distance.append(ranking.index(np.where(choices == correct_choice)[0]))
        f.close()
        del data
    print(np.mean(pick_numbers))
    return correct_picks/pick_number,np.mean(pick_distance)

def plot_pickrate(filenames):
    x = [list() for i in range(len(filenames))]
    for i,filename in enumerate(filenames):
        with open(filename, 'r') as f:
            for line in f:
                line = line.split(';')
                x[i].append(float(line[1]))

    x[0].sort()
    x[1].sort()
    plt.plot(range(len(x[0])),x[0], label = 'First-pick rate')
    plt.plot(range(len(x[1])),x[1], label = 'Pick rate', linestyle = 'dashed')
    plt.ylabel('Rate')
    plt.xlabel('Ranked cards')
    plt.legend()
    plt.xlim(0,264)
    plt.ylim(0,1)
    
    plt.savefig('plots/firstpickrate_combiend_2.pdf', bbox_inches='tight' )
    plt.show()                
                

def plot_pickrate_ratings(filename):
    colours = ['y','b','k','r','g','cyan','m']
    x = [list() for _ in range(7)]
    y = [list() for _ in range(7)]
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(';')
            if line[5] == 'None':
                continue
            if line[4] == 'White':
                x[0].append(float(line[1]))
                y[0].append(float(line[3].replace(',','.')))
            elif line[4] == 'Blue':
                x[1].append(float(line[1]))
                y[1].append(float(line[3].replace(',','.')))
            elif line[4] == 'Black':
                x[2].append(float(line[1]))
                y[2].append(float(line[3].replace(',','.')))
            elif line[4] == 'Red':
                x[3].append(float(line[1]))
                y[3].append(float(line[3].replace(',','.')))
            elif line[4] == 'Green':
                x[4].append(float(line[1]))
                y[4].append(float(line[3].replace(',','.')))
            elif line[4] == 'Colourless':
                x[5].append(float(line[1]))
                y[5].append(float(line[3].replace(',','.')))
            else:
                x[6].append(float(line[1]))
                y[6].append(float(line[3].replace(',','.')))
    reg = LinearRegression()
    for i in range(7):
        x[i] = np.array(x[i])
        y[i] = np.array(y[i])
        plt.scatter(x[i],y[i],s = 2, label = 'Cards', color = colours[i])
        reg.fit(x[i].reshape(-1,1),y[i].reshape(-1,1))
        plt.plot(np.arange(0,1,0.0001).reshape(-1,1),reg.predict(np.arange(0,1,0.0001).reshape(-1,1)) ,label = 'Regression '+colours[i], c = colours[i])
    plt.xlabel('Pick-percentage')
    plt.ylabel('Individual card rating')
    plt.title('Correlation between pick-percentage and individual card rating')
    plt.xlim(0,1)
    plt.ylim(0,5)
  #  plt.legend()
    plt.show()

def plot_pickrate_against_rarity(filename):
    colours = ['black','silver','gold','brown']
    labels = ['Common', 'Uncommon', 'Rare', 'Mythic']
    x = [list() for _ in range(4)]
    y = [list() for _ in range(4)]
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(';')
            if line[3] == 'None':
                continue
            rarity = line[3].strip('\n')
            if rarity == 'C':
                x[0].append(float(line[1]))
                y[0].append(1)
            elif rarity == 'U':
                x[1].append(float(line[1]))
                y[1].append(2)
            elif rarity == 'R':
                x[2].append(float(line[1]))
                y[2].append(3)
            elif rarity == 'M':
                x[3].append(float(line[1]))
                y[3].append(4)
    vio = plt.violinplot(x, showextrema= False, showmeans = True, bw_method= 0.2)
    for i,pc in enumerate(vio['bodies']):
        pc.set_facecolor(colours[i])
        pc.set_edgecolor('black')
    vio['cmeans'].set_edgecolor('black')
    # for i in range(4):
    #     x[i] = np.array(x[i])
    #     print(np.mean(x[i]))
    #     y[i] = np.array(y[i])
    #     print(np.mean(y[i]))
    #     # plt.violinplot(x[i],y[i],s = 100, label = labels[i], color = colours[i], alpha = 0.5)
    #     plt.violinplot(y[i],labels[i], color = colours[i])
    plt.xlabel('Rarity')
    plt.ylabel('Pick rate')
    plt.xticks([1,2,3,4],['Common','Uncommon','Rare','Mythic'])
    plt.ylim(0,1)
  #  plt.ylim(0,5)
 #   plt.legend()
    
    plt.savefig('plots/firstpick_rarity_2.pdf', bbox_inches='tight' )
    plt.show()

def plot_pickrate_rarity(filename):
    colours = ['black','silver','gold','brown']
    labels = ['Common', 'Uncommon', 'Rare', 'Mythic']
    x = [list() for _ in range(4)]
    y = [list() for _ in range(4)]
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(';')
            if line[5] == 'None':
                continue
            rarity = line[5].strip('\n')
            if rarity == 'C':
                x[0].append(float(line[1]))
                y[0].append(float(line[3].replace(',','.')))
            elif rarity == 'U':
                x[1].append(float(line[1]))
                y[1].append(float(line[3].replace(',','.')))
            elif rarity == 'R':
                x[2].append(float(line[1]))
                y[2].append(float(line[3].replace(',','.')))
            elif rarity == 'M':
                x[3].append(float(line[1]))
                y[3].append(float(line[3].replace(',','.')))
    reg = LinearRegression()
    for i in range(4):
        x[i] = np.array(x[i])
        print(np.mean(x[i]))
        y[i] = np.array(y[i])
        print(np.mean(y[i]))
        plt.scatter(x[i],y[i],s = 100, label = labels[i], color = colours[i], alpha = 0.5)
        reg.fit(x[i].reshape(-1,1),y[i].reshape(-1,1))
       # plt.plot(np.arange(0,1,0.0001).reshape(-1,1),reg.predict(np.arange(0,1,0.0001).reshape(-1,1)) , c = colours[i])
    plt.xlabel('Pick-percentage')
    plt.ylabel('Individual card rating')
    plt.title('Correlation between pick-percentage and individual card rating by rarity')
 #   plt.xlim(0,1)
 #   plt.ylim(0,5)
    plt.legend()
    plt.show()


def plot_accuracy_per_pick(network):
    with open('exact_correct.tsv') as f:
        reader = csv.reader(f, delimiter = "\t")
        lines = [line for line in reader]
    accuracies = [[list() for _ in range(45)] for __ in range(5)]
    for line in lines[1:]:
        for i,elem in enumerate(line[3:]):
            accuracies[i][int(line[1])-1].append(int(elem))
    pick_number = [0 for _ in range(45)]
    correct_picks = [0 for _ in range(45)]
    pick_distance = list()
    pick_numbers = list()
    folder_name = 'E:/picks_2/'
    with open(folder_name+np.random.choice(os.listdir(folder_name)),'rb') as f:
        data = np.load(f, allow_pickle = True)
        for pick in data:
            chosen_num = int(sum(pick[1]))
            pick_number[chosen_num] += 1
            choices = pick[0]
            pick_numbers.append(len(choices))
            anchor = pick[1]
            correct_choice = np.argmax(pick[2])
            with torch.no_grad():
                out1 = network(torch.Tensor(anchor).view(1,-1))

                distances = list()
                for i in range(len(choices)):
                    elem = torch.zeros((1,265))
                    elem[0,choices[i]] = 1.0
                    distances.append(networks.get_distance(out1,network(elem)).item())

                ranking = [x for _,x in sorted(zip(distances,[y for y in range(len(distances))]))]
                if choices[ranking[0]] == correct_choice:
                    correct_picks[chosen_num] += 1
        f.close()
        del data
    plt.plot([j for j in range(15)],[correct_picks[i]/pick_number[i] for i in range(15)], c = 'lightslategrey', label = 'SiameseBot')
    plt.plot([j for j in range(15,30)],[correct_picks[i]/pick_number[i] for i in range(15,30)], c = 'lightslategrey')
    plt.plot([j for j in range(30,45)],[correct_picks[i]/pick_number[i] for i in range(30,45)], c = 'lightslategrey')

    plt.plot([j for j in range(15)],[sum(accuracies[0][i])/len(accuracies[0][i]) for i in range(15)], c = 'red', label = 'RandomBot')
    plt.plot([j for j in range(15,30)],[sum(accuracies[0][i])/len(accuracies[0][i]) for i in range(15,30)], c = 'red')
    plt.plot([j for j in range(30,45)],[sum(accuracies[0][i])/len(accuracies[0][i]) for i in range(30,45)], c = 'red')

    plt.plot([j for j in range(15)],[sum(accuracies[1][i])/len(accuracies[1][i]) for i in range(15)], c = 'blue', label = 'RaredraftBot')
    plt.plot([j for j in range(15,30)],[sum(accuracies[1][i])/len(accuracies[1][i]) for i in range(15,30)], c = 'blue')
    plt.plot([j for j in range(30,45)],[sum(accuracies[1][i])/len(accuracies[1][i]) for i in range(30,45)], c = 'blue')

    plt.plot([j for j in range(15)],[sum(accuracies[2][i])/len(accuracies[2][i]) for i in range(15)], c = 'green', label = 'DraftsimBot')
    plt.plot([j for j in range(15,30)],[sum(accuracies[2][i])/len(accuracies[2][i]) for i in range(15,30)], c = 'green')
    plt.plot([j for j in range(30,45)],[sum(accuracies[2][i])/len(accuracies[2][i]) for i in range(30,45)], c = 'green')

    plt.plot([j for j in range(15)],[sum(accuracies[3][i])/len(accuracies[3][i]) for i in range(15)], c = 'purple', label = 'BayesBot')
    plt.plot([j for j in range(15,30)],[sum(accuracies[3][i])/len(accuracies[3][i]) for i in range(15,30)], c = 'purple')
    plt.plot([j for j in range(30,45)],[sum(accuracies[3][i])/len(accuracies[3][i]) for i in range(30,45)], c = 'purple')

    plt.plot([j for j in range(15)],[sum(accuracies[4][i])/len(accuracies[4][i]) for i in range(15)], c = 'orange', label = 'NNetBot')
    plt.plot([j for j in range(15,30)],[sum(accuracies[4][i])/len(accuracies[4][i]) for i in range(15,30)], c = 'orange')
    plt.plot([j for j in range(30,45)],[sum(accuracies[4][i])/len(accuracies[4][i]) for i in range(30,45)], c = 'orange')
    plt.xlim(0,45)
    plt.ylim(0,1)
    plt.xlabel('Pick')
    plt.ylabel('Accuracy')
    print(sum(correct_picks)/sum(pick_number))
    plt.legend(ncol = 2, loc = 4)
    plt.savefig('plots/accuracy_per_pick_comp.pdf', bbox_inches='tight' )
    plt.show()

def load_card_dict(path):
    with open(path,'rb') as file:
        card_dict = pickle.load(file)
    return card_dict

def plot_embedding_dimension(path):
    cmap = plt.get_cmap('jet_r')
    fig, ax1 = plt.subplots()
    for j,file in enumerate(os.listdir(path)):
        with open(path+file) as f:
            reader = csv.reader(f, delimiter = ',')
            lines = [line for line in reader]
            comparisons = [float(i) for i in lines[0]]
            distances = [float(i) for i in lines[2]]
            label = file[file.find('_')+1:file.find('d')]
            ax1.plot(range(len(comparisons)),comparisons,c = cmap(float(j)/len(os.listdir(path))), label = label)
    ax1.set_ylabel('Correct picks')
    plt.xlabel('Subepoch')
    plt.xlim(0,50)
    plt.ylim(0,1)
    handles, labels = ax1.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0]),reverse = True))
    ax1.legend(handles, labels, ncol = 3, loc = 4)
    plt.savefig('plots/embedding_comparison.pdf', bbox_inches='tight' )
    plt.show()


#plot_embedding_dimension('revamp/Dimension_comparison/')

#preprocess_integer_dataset('clean_logs/integer_logs_baseline2/')
#preprocess_feature_dataset('clean_logs/feature_logs_baseline2_ordered/')
#pickrates_to_cards('card_dict.pt', 'firstpickrates.csv')
# plot_pickrate(['named_firstpickrates.csv','named_pickrates.csv'])
# plot_pickrate_against_rarity('named_firstpickrates.csv')
#compute_firstpick_chance('E:/training_mtg/')
#plot_pickrate_ratings_rarity('named_pickrates.csv')