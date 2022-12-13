import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from torch.utils.data import Dataset, DataLoader



class mtg_dataset(Dataset):
    def __init__(self, dir_path, max_id):
        self.path = dir_path
        self.max_id = max_id
        self.positives = list()
        self.negatives = list()
        self.anchors = list()
        with open(dir_path, 'rb') as f:
            data = pickle.load(f)
            f.close()
        for line in data:
            line = line.split(';')
            pos = torch.zeros(self.max_id)
            pos[int(line[0])] = 1
            neg = torch.zeros(self.max_id)
            if line[1]:
                neg[int(line[1])] = 1
            anch = torch.zeros(self.max_id)
            if line[2]:
                for card in line[2].split(','):
                    anch[int(card)] += 1
            self.anchors.append(anch)
            self.positives.append(pos)
            self.negatives.append(neg)

        self.anchors = torch.stack(self.anchors)
        self.positives = torch.stack(self.positives)
        self.negatives = torch.stack(self.negatives)
        del data
                            
    def __len__(self):
        return len(self.anchors)

    def __getitem__(self,index):
        return self.anchors[index],self.positives[index],self.negatives[index]

class mtg_metadataset(Dataset):
    def __init__(self,path):
        self.files = [path+file for file in os.listdir(path)]
        self.len = 0
        for file in self.files:
            with open(file,'rb') as f:
                dataset = pickle.load(f)
                self.len += dataset.__len__()
        print('done')

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        print('error')


class Siamese(nn.Module):
    def __init__(self,input_size,output_dim):
        super(Siamese,self).__init__()
        self.input_size = input_size

        
        self.hidden_1 = nn.Sequential(
            nn.Linear(input_size,512),
            nn.Dropout(0.5),
            nn.ELU()
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(512,128),
            nn.Dropout(0.5),
            nn.ELU()
        )
        self.hidden_3 = nn.Sequential(
            nn.Linear(128,64),
            nn.Dropout(0.5),
            nn.ELU()
        )
        self.hidden_4 = nn.Sequential(
            nn.Linear(64,32),
            nn.Dropout(0.5),
            nn.ELU()
        )
        self.hidden_5 = nn.Sequential(
            nn.Linear(32,16),
            nn.Dropout(0.5),
            nn.ELU()
        )
        self.out = nn.Sequential(
            nn.Linear(128,output_dim),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x


def get_distance(positive,negative):
    return torch.sum(torch.pow(positive-negative,2),dim=1)

        
