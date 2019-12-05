import pandas as pd
import json
from pathlib import Path
import time
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#nltk.download('stopwords')

def Preprocessing(txt):
    txt = str(txt).replace('\\',' ')
    
    #Tokenize
    tokens = [word for sentence in nltk.sent_tokenize(txt)
              for word in nltk.word_tokenize(sentence)]
    

    #Removing non Alphabet
    tokens = [word for word in tokens if word.isalpha()]
    
    #Removing Stopwords
    stopword = stopwords.words('english')
    tokens = [word for word in tokens if word not in stopword]
    
    #Lowercasing
    tokens = [word.lower() for word in tokens]
    
    #Stemming
    stemmer = PorterStemmer()
    tokens = [ stemmer.stem(word) for word in tokens]
    
    return tokens

def bigram(tokens):
    
    bigram =[]
    
    for i in range(len(tokens) -1):
        bigram.append(tokens[i] + " " +tokens[i+1])
    return bigram

datapath = Path('data/ag_news_csv')

test_data = pd.read_csv(datapath / 'test.csv',names = ["class","title","body"])
train_data = pd.read_csv(datapath / 'train.csv',names = ["class","title","body"])


train_X, train_Y = [],[]
test_X,  test_Y = [],[]


print("Preprocessing")
since = time.time()

for i, row in tqdm(test_data.iterrows()):
    #print(type(tokenize(row['title'])))
    tokens = Preprocessing(row['title']) + bigram(Preprocessing(row['title'])) + Preprocessing(row['body']) +bigram(Preprocessing(row['body']))
    #tokens = Preprocessing(row['title']) + Preprocessing(row['body'])
    
    test_X.append(tokens)
    test_Y.append(row['class']-1)

for i, row in tqdm(train_data.iterrows()):
    #print(type(tokenize(row['title'])))
    tokens = Preprocessing(row['title']) + bigram(Preprocessing(row['title'])) + Preprocessing(row['body']) + bigram(Preprocessing(row['body']))
    #tokens = Preprocessing(row['title']) + Preprocessing(row['body'])
    
    train_X.append(tokens)
    train_Y.append(row['class']-1)
    
print("Finished Preprocessing")

time_elapsed = time.time() - since
print("{:.0f}m {:.0f}".format(time_elapsed //60, time_elapsed % 60))

train_data = {"train_X":train_X, "train_Y":train_Y }
test_data  = {"test_X":test_X, "test_Y":test_Y}

with open('traindata.json','w') as outfile:
    json.dump(train_data,outfile,indent=4)

with open('testdata.json','w') as outfile:
    json.dump(test_data,outfile,indent=4)
