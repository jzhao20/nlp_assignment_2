# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class DAN(nn.Module):
    def __init__(self,input, hidden,output,embedding):
        super(DAN,self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.word_indexer=embedding.word_indexer
        weights=torch.FloatTensor(embedding.vectors)
        self.embedded=nn.Embedding.from_pretrained(weights,padding_idx=0).requires_grad_(False)
        self.V=nn.Linear(input,hidden)
        self.g=nn.Tanh()
        self.W=nn.Linear(hidden,output)
        self.log_softmax=nn.LogSoftmax(dim=1)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
    def forward(self,x,batch_size=1):
        #convert this list to a np list of indexes and then convert to tensor
        #longest sentence to create input
        max_length=0
        for batch in x:
            max_length=max(len(batch),max_length)
        input=np.zeros((max_length,batch_size),dtype=np.int64)
        for batch in range(0,batch_size):
            for i in range(0,len(x[batch])):
                index=self.word_indexer.index_of(x[batch][i])
                #contains the index into the embedding matrix
                input[i][batch]=index if index!=-1 else self.word_indexer.index_of("UNK")
        input=self.embedded(torch.LongTensor(input).to(self.device))
        input=input.mean(0)
        input=self.V(input)
        input=self.W(self.g(input))
        return self.log_softmax(input)

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1



class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self,dan):
        self.model=dan
    def predict(self, dev_ex:List[str]):
        dan=self.model
        #make it a list so it'll work with forward
        log_probs=dan.forward([dev_ex])
        return torch.argmax(log_probs)

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    num_epochs=10
    input_size=word_embeddings.get_embedding_length()
    num_classes=2
    hidden_size=50
    batch_size=1
    dan=DAN(input_size,hidden_size,num_classes,word_embeddings)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dan.to(device)
    initial_learning_rate=.001
    optimizer=optim.Adam(dan.parameters(),lr=initial_learning_rate)
    for epoch in range(0,num_epochs):
        total_loss=0
        random.shuffle(train_exs)
        for i in range(0,len(train_exs)-batch_size+1,batch_size):
            #row vector which will be later converted to a matrix
            x=[train_exs[j].words for j in range(i,i+batch_size)]
            #replaced scatter with this 
            y=[[train_exs[j].label==0,train_exs[j].label==1] for j in range(i,i+batch_size)]
            y_onehot=torch.from_numpy(np.asarray(y,dtype=np.int64)).to(device).float()
            dan.zero_grad()
            log_probs=dan.forward(x,batch_size)
            #the main diagnol of the matrix is equal to the dot products 
            loss_matrix=torch.matmul(y_onehot,torch.neg(log_probs).t())
            loss=torch.sum(torch.diagonal(loss_matrix,0))
            total_loss+=loss
            loss.backward()
            optimizer.step()
        print(total_loss)
    return NeuralSentimentClassifier(dan)


