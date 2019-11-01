import pandas as pd

test_data = pd.read_csv('data/ag_news_csv/test.csv',names = ["class","title","body"])
train_data = pd.read_csv('data/ag_news_csv/train.csv',names = ["class","title","body"])

import nltk

def tokenize(txt):
    tokens = [word for sent in nltk.sent_tokenize(txt)
              for word in nltk.word_tokenize(sent)]
    #print(type(tokens))
    return tokens

def bigram(tokens):
    
    bigram =[]
    
    for i in range(len(tokens) -1):
        bigram.append(tokens[i] + " " +tokens[i+1])
    return bigram

train_X,train_Y = [],[]

test_X, test_Y = [],[]

for i, row in test_data.iterrows():
    #print(type(tokenize(row['title'])))
    tokens = tokenize(row['title']) + tokenize(row['body'])
    
    test_X.append(tokens)
    test_Y.append(row['class']-1)
    

for i, row in train_data.iterrows():
    #print(type(tokenize(row['title'])))
    tokens = tokenize(row['title']) + tokenize(row['body'])
    
    train_X.append(tokens)
    train_Y.append(row['class']-1)
    
