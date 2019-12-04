from tqdm import tqdm

UNK = '<UNK>'

class Dictionary():
    """
    class containing dictionary.
    
        Attribute:
        | dictionary : Dictionary whose key: word, value: number of occurrence
        
        Methods:
        | add_words: Adds new words to dictionary.
        | how_many: Retrieves number of occurrence corresponding to given token
    """
    def __init__(self):
        self.dictionary = {}
    
    def add_words(self,word):
        if word not in self.dictionary:
            self.dictionary[word] = 1
        else:
            self.dictionary[word] += 1
    
    def how_many(self,word):
        if word not in self.dictionary:
            return 0
        else:
            return self.dictionary[word]
    def set_unknown(self):
        if UNK in self.dictionary:
            print("<UNK> already exists")
        else:
            self.dictionary[UNK] = 0

def make_dictionary(data):
    """
    data : list(sentence) of list(word)
    """
    dictionary_list = []
    for sentence in tqdm(data):
        dic = dict()
        for word in sentence:
            try:
                dic[word] += 1
            except:
                dic[word] = 1
        dictionary_list.append(dic)
    return dictionary_list
    
