from tqdm import tqdm

UNK = '<UNK>'

class Vocabulary():
    """
    Class containing vocabulary.
    
        Attribute:
        | word2index : Dictionary whose key: word(word), value: index
        | index2word : key: index, value: word
        
        Methods:
        | new_word: Adds new word to vocabulary.
        | get_word: Retrieves word corresponding to given index
        | get_index: Retrieves index corresponding to given word
    """
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        
    def new_word(self,word):
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
            
        return self.word2index, self.index2word
    
    def get_word(self, index):
        return self.index2word[index]

    def get_index(self, word):
        return self.word2index[word]
    
    def print_word(self):
        print(self.word2index.keys())
        
    def size(self):
        return len(self.word2index)

def make_vocabulary(data, vocab):
    """
    vocab : Vocabulary
    data  : List(sentence) of list(word)
    """
    vocab.new_word(UNK)
    for sentence in tqdm(data):
        for word in sentence:
            vocab.new_word(word)
    
    return vocab
