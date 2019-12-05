import numpy as np

class Glove():
    """
        input: pretrained glove file
        
        attribute:
            model : dictionary whose input is a word and outputs vector(list)
        
        methods:
            loadGloveModel : creates GloveModel
            Transform : takes dictionary list as input where dictionary represents word occurence in sentence
            
    """
    
    def __init__(self,file):
        self.model = self.loadGloveModel(file)
    
    def loadGloveModel(self,gloveFile):
        print ("Loading Glove Model")
    
        with open(gloveFile, encoding="utf8" ) as f:
           content = f.readlines()
        model = {}
        for line in content:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]]).reshape(300,)
            model[word] = embedding
        print ("Done.",len(model)," words loaded!")
        return model
    
    def Transform(self,dict_list):
        mat = np.zeros(shape = (len(dict_list),300), dtype = np.float64)
        for sentence_num, sentence in enumerate(dict_list):
            sent_vec = np.zeros((300,),dtype=np.float64)
            sent_length = 0
            for word in sentence:
                if word in self.model:
                    sent_vec += sentence[word] * self.model[word]
                    sent_length += sentence[word]
                else:
                    continue
            sent_vec = sent_vec/sent_length
            mat[sentence_num] += sent_vec
        return mat
        
