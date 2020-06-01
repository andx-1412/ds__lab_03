import numpy as np

VOCAB_SIZE = 0
with open(file = 'dict_idf.txt') as f:
    VOCAB_SIZE =len( f.read().splitlines())

class data_reader:
    def __init__(self, vocab_size, batch_size, file_path):
        self.start = 0
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.data = np.array([[0.0 for _ in range(self.vocab_size)]])
        self.label = np.array([[0]])
        with open(file = file_path) as f:
            lines = f.read().splitlines()
            for line in lines:
                vector = [0.0 for _ in range(self.vocab_size)]
                feature = line.split('<ffff>')
                label = int(feature[0])
                for token in feature[2].split():
                    index, value = token.split(':')
                    index = int(index)
                    value = float(value)
                    vector[index] = value
                self.data = np.concatenate((self.data, np.array([vector])))
                self.label = np.concatenate((self.label, np.array([[label]])))
        self.data= self.data[1:]
        self.label = self.label[1:]
    def next_batch(self):
        start = self.start
        batch_data = self.data[start:start+ self.batch_size]
        batch_label = self.label[start:start+self.batch_size]
        if(start + self.batch_size > self.data.shape[0]):
            self.start = 0
            indice = np.array(range(self.data.shape[0]))
            self.data= self.data[indice]
            self.label = self.label[indice]
        else:
            self.start = start+ self.batch_size
       
        return batch_data,batch_label