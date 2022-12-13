import csv
import json
import numpy as np
import os
import pickle
import torch
from models import mtg_dataset
import utils
import sys

def reconstruct_pack(picks):
    packs = [[list() for _ in range(len(picks))] for __ in range(3)]
    for i in range(8):
        for k in range(3):
            for j in range(15):
                if k % 2 == 1:
                    packs[k][i].append(picks[(i+(-j%8))%8][j+k*15])
                else:
                    packs[k][i].append(picks[(i+j)%8][j+k*15])
    counter = 0
    for round in packs:
        for pack in round:
            try:
                assert len(pack) == len(set(pack))
            except:
                return False
    return packs


def create_dict(dir):
    with open(dir , 'r') as f:
        reader = csv.reader(f)
        output = [line for line in reader]
        all_cards = set()
        for draft in output:
            for player_pool in draft[2:]:
                cards = player_pool.split(',')
                last_card = None
                for card in cards:
                    if card[0] == '_':
                        all_cards.remove(last_card)
                        all_cards.add(last_card+','+card)
                    else:
                        all_cards.add(card)
                        last_card = card
        card_to_num = dict()
        i = 0
        all_cards = list(all_cards)
        all_cards.sort()
        last_card_was_basic = False
        for card in all_cards:
            if 'Plains' in card:
                card_to_num[card] = i 
                last_card_was_basic = True
            elif 'Forest' in card:
                card_to_num[card] = i 
                last_card_was_basic = True
            elif 'Swamp' in card:
                card_to_num[card] = i 
                last_card_was_basic = True
            elif 'Island' in card:
                card_to_num[card] = i 
                last_card_was_basic = True
            elif 'Mountain' in card:
                card_to_num[card] = i 
                last_card_was_basic = True
            else:
                if last_card_was_basic:
                    i += 1
                    last_card_was_basic = False
                card_to_num[card] = i 
                i += 1
    
        return card_to_num

def write_preferences(picks,packs, card_dict):
    output = list()
    for k in range(3):
        for i in range(15):
            for j in range(8):
                pick = picks[j][k*15+i] 
                if k %  2 == 1:
                    unpicked = [card for card in packs[k][(j+i)%8] if card !=  pick]
                else:
                    unpicked = [card for card in packs[k][(j-i)%8] if card !=  pick]
                assert len(unpicked) == 15-1-i
                if k == 0 and i == 0:
                    drafted_num = ''
                else:
                    drafted = picks[j][:k*15+i]
                    drafted_num = ','.join([str(card_dict[i]) for i in drafted])
                if unpicked:
                    for card in unpicked:
                        output.append(str(card_dict[pick]) + ';' + str(card_dict[card])+';' + drafted_num)
                else:
                    output.append(str(card_dict[pick]) + ';' +';' + drafted_num)
                if k % 2 == 1:
                    packs[k][(j+i)%8].remove(pick)
                else:
                    packs[k][(j-i)%8].remove(pick)
    return output


def preprocess_data(card_dict, dir, outfile):
    with open(dir , 'r') as f:
        reader = csv.reader(f)
        file_num = 0
        file_data = list()
        for line in reader:
            splitted_line = [player.split(',') for player in line[2:]]
            for i,player in enumerate(splitted_line):
                for j,pick in enumerate(player):
                    if pick[0] == '_':
                        player[j-1] = player[j-1] + ',' + pick
                        player[j] = None
                splitted_line[i] = [splitted_line[i][j] for j in range(len(splitted_line[i])) if splitted_line[i][j]]

            packs = reconstruct_pack(splitted_line)
            if packs:
                output = write_preferences(splitted_line,packs,card_dict)
                file_data.extend(output)
            if len(file_data) >= 997920:
                with open(outfile+'train_data'+str(file_num)+'.pt','wb') as w:
                    pickle.dump(file_data[:997920],w)
                    file_num += 1
                    file_data = file_data[997920:]
                    w.close()

def preprocess_data_into_dataset(inpath,outpath):
    for file in os.listdir(inpath):
        dataset = mtg_dataset(inpath+file, 265)
        with open(outpath+file, 'wb') as f:
            pickle.dump(dataset,f)
            f.close()

def preprocess_evaldata_into_picks(inpath,outpath):
    j = 0
    for file in os.listdir(inpath):
        picks = list()
        with open(inpath+file,'rb') as f:
            data = pickle.load(f)
            cards_in_pack = list()
            last_positive = data.positives[0].numpy()
            last_anchor = data.anchors[0].numpy()  
            cards_in_pack.append(np.argmax(data.negatives[0].numpy()))
            last_picks = [None for _ in range(8)]
            for i in range(1,len(data.anchors)):
                anchor = data.anchors[i].numpy()
                pick = data.positives[i].numpy()
                if sum(data.negatives[i].numpy()) == 0:
                    negative = None
                else:
                    negative = np.argmax(data.negatives[i].numpy())
                if (anchor == last_anchor).all() and (pick == last_positive).all():
                    cards_in_pack.append(negative)
                else:
                    cards_in_pack.append(np.argmax(last_positive))
                    cards_in_pack = np.array(cards_in_pack)
                    tmp = [cards_in_pack,last_anchor,last_positive]
                    tmp = np.array(tmp)
                    picks.append(tmp)

                    last_positive = pick
                    if negative:
                        cards_in_pack = [negative]
                    else:
                        cards_in_pack = list()
                    last_anchor = anchor
        #fix io error
        with open(outpath+str(j)+'.npy', 'wb') as outfile:
            picks = np.array(picks)
            np.save(outfile,picks)
            picks = list()
            j += 1

def main(training_path,test_path):
    print('Preprocessing training data')
    card_dict = utils.load_card_dict('Data/card_dict.pt')
    training_out = 'E:/MtgBase/training_data/'
    if not os.path.exists(training_out):
        os.mkdir(training_out)
   # preprocess_data(card_dict,training_path,training_out)
    print('Preproessing training data into datasets')
    training_dataset_out = 'E:/MtgBase/training_datasets/'
    if not os.path.exists(training_dataset_out):
        os.mkdir(training_dataset_out)
#    preprocess_data_into_dataset(training_out,training_dataset_out)
    
    print('Preprocessing test data')
    test_out = 'E:/MtgBase/test_data/'
    if not os.path.exists(test_out):
        os.mkdir(test_out)
   # preprocess_data(card_dict,test_path,test_out)
    print('Preproessing test data into datasets')
    test_dataset_out ='E:/MtgBase/test_datasets/'
    if not os.path.exists(test_dataset_out):
        os.mkdir(test_dataset_out)
    preprocess_data_into_dataset(test_out,test_dataset_out)
    print('Creating pick files')
    pick_out = 'E:/MtgBase/picks/'
    if not os.path.exists(pick_out):
        os.mkdir(pick_out)
    preprocess_evaldata_into_picks(test_dataset_out,pick_out)


if __name__ == "__main__":
    training_path = 'E:/train.csv'
    test_path = 'E:/test.csv'
    main(training_path,test_path)