from pathlib import Path
import os
import numpy as np
import json
import time
import copy
import argparse
from tqdm import tqdm
import easydict


from sklearn.feature_extraction import FeatureHasher

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from vocabulary import Vocabulary, make_vocabulary
from dictionary import make_dictionary

#Loading Data
print("Loading Data")
with open('traindata.json') as jsonfile:
    traindata = json.load(jsonfile)
    
with open('testdata.json') as jsonfile:
    testdata = json.load(jsonfile)

print("Finished Loading Data!")
print("Making Vocab and Dictionary")
#Making Vocabulary&Dictionary
train_vocab = Vocabulary()
train_vocab = make_vocabulary(traindata['train_X'],train_vocab)

train_dict = make_dictionary(traindata['train_X'])
test_dict = make_dictionary(testdata['test_X'])

print("Finished Making Vocab and Dict")
print("Feature Hashing")
#Feature Hashing
VOCAB_SIZE = min(train_vocab.size(),10000)
Hasher = FeatureHasher(n_features=VOCAB_SIZE) #논문에 10M 이라 되어있음
Hasher.fit(train_dict)
print("Finished Feature Hashing")
#Argument Parsing
args = easydict.EasyDict({
        "hidden_dim": 10,
        "num_epoch": 20,
        "learning_rate": 1e-2,
        "batch_size": 32,
        "momentum": 0.9,
        "val_split": 0.2
})

#Model
class TrainDataset(Dataset):
    
    def __init__(self):

        self.H = Hasher.transform(train_dict)
        self.y_data = torch.from_numpy(np.array(traindata['train_Y'])).type(torch.long)
        
    def __getitem__(self,index):
        return torch.from_numpy(self.H[index].toarray()[0]), self.y_data[index]
    
    def __len__(self):
        return self.H.shape[0]
    
class TestDataset(Dataset):
    
    def __init__(self):

        self.H = Hasher.transform(test_dict)
        self.y_data = torch.from_numpy(np.array(testdata['test_Y'])).type(torch.long)
        
    def __getitem__(self,index):
        return torch.from_numpy(self.H[index].toarray()[0]), self.y_data[index]
    
    def __len__(self):
        return self.H.shape[0]  
    
def split_Data(dataset,val_split,batch_size):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, sampler = train_sampler)
    valid_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, sampler = valid_sampler)

    dataloaders= {'train':train_loader,'val':valid_loader}

    return dataloaders

class TextClassifier(nn.Module):
    def __init__(self,vocab_size,hidden_dim,num_class):
        super(TextClassifier,self).__init__()
        #self.embedding = nn.EmbeddingBag(vocab_size, hidden_dim, sparse=True)
        self.input_layer = nn.Linear(vocab_size,hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, num_class)
        self.output_layer = nn.Softmax(dim=1)
        
    def forward(self,text):
        embedded = self.input_layer(text)
        output = self.hidden_layer(embedded)
        return self.output_layer(output)
    
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train','val']:
            if phase == 'train':
                print("train phase")
                model.train()
            else:
                print("validation phase")
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            count = 0
            for inputs,labels in dataloaders[phase]:
                #now = time.time()
                #print('{:.0f}m {:.0f}s'.format( (now-since) // 60, (now-since) % 60))
                count += len(inputs)
                inputs,labels = inputs.to(device),labels.to(device)
                inputs,labels = Variable(inputs.float()), Variable(labels)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase =='train':
                        loss.backward()
                        optimizer.step()
                        
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                            
            epoch_loss = running_loss / count
            epoch_acc = running_corrects.double() / count
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            #scheduler.step()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        time_elapsed = time.time() - since
        print('Training Epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HIDDEN_DIM = args.hidden_dim
NUM_CLASS = len(set(testdata['test_Y']))
NUM_EPOCH = args.num_epoch
LEARN_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
MOMENTUM = args.momentum
VAL_SPLIT = args.val_split

dataloaders= split_Data(TrainDataset(),VAL_SPLIT,BATCH_SIZE)

model = TextClassifier(VOCAB_SIZE,HIDDEN_DIM,NUM_CLASS).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARN_RATE, momentum = MOMENTUM)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_model = train_model(model, criterion, optimizer, scheduler, NUM_EPOCH)

model_name = 'gimotee_revised' + '_model.pth'
torch.save(best_model.state_dict(), os.path.join('model',model_name))
