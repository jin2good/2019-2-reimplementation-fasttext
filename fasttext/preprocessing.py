import pandas as pd
import json
import nltk
from pathlib import Path

datapath = Path('/data/ag_news_csv')
test_csv = datapath / 'test_csv'
train_csv = datapath / 'train_csv'

def tokenize(txt):
    tokens = [word for sentence in nltk.sent_tokenize(txt)
              for word in nltk.word_tokenize(sentence)]
    #print(type(tokens))
    return tokens


def bigram(tokens):
    
    bigram =[]
    
    for i in range(len(tokens) -1):
        bigram.append(tokens[i] + " " +tokens[i+1])
    return bigram

datapath = Path('data/ag_news_csv')

test_data = pd.read_csv(datapath / 'test.csv',names = ["class","title","body"])
train_data = pd.read_csv(datapath / 'train.csv',names = ["class","title","body"])


train_X,train_X_bi,train_Y = [],[],[]


test_X, test_X_bi, test_Y = [],[],[]

for i, row in test_data.iterrows():
    #print(type(tokenize(row['title'])))
    tokens = tokenize(row['title']) + tokenize(row['body'])
    
    test_X.append(tokens)
    test_X_bi.append(bigram(tokens))
    test_Y.append(row['class']-1)

for i, row in train_data.iterrows():
    #print(type(tokenize(row['title'])))
    tokens = tokenize(row['title']) + tokenize(row['body'])
    
    train_X.append(tokens)
    train_X_bi.append(bigram(tokens))
    train_Y.append(row['class']-1)

datadict = {"train":{"train_X":train_X, "train_X_bigram":train_X_bi,   "train_Y":train_Y },
            "test":{"test_X":test_X,    "test_X_bigram":test_X_bi,  "test_Y":test_Y}}


with open('data.json','w') as outfile:
    json.dump(datadict,outfile,indent=4)

"""
with open('data.txt','w') as outfile:
    json.dump(datadict,outfile,indent=4)
"""
