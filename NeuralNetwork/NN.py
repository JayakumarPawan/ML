import numpy as np
from random import random,shuffle
from pandas import read_csv

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        np.random.seed(2)
        self.weights = [np.random.randn(layers[i+1],layers[i]) for i in range(self.num_layers-1)]
        self.biases = [np.zeros((layers[i],1)) for i in range(1,self.num_layers)]
        self.learning_rate = .1 # .1 3 epoch and 128 128 hiddens for .56 accuracy on mnist
        self.batch_size = 64
        self.num_epochs = 5
        print("net initalized")

    @staticmethod
    def sigmoid(i,deriv = False):
        """The sigmoid function."""
        if deriv:
            return Network.sigmoid(i)*(1-Network.sigmoid(i))
        return 1.0/(1.0+np.exp(-i))
    @staticmethod
    def oneHot(x):
        zm = np.zeros(10)
        zm[x] = 1
        return np.array(zm)
    def feed_forward(self,X):
        for i in range(len(self.weights)):
            X = Network.sigmoid(np.dot(self.weights[i],X)+self.biases[i])
        return X
    def error(self,X,y):
        X = self.feed_forward(X)
        error = (.5*(X - y)**2)
        return error

    def backprop(self,X,y):
        X_l =[]
        X_l.append(X)
        for i in range(len(self.weights)):
            X = np.dot(self.weights[i],X)+self.biases[i]
            X = np.array([Network.sigmoid(i) for i in X])
            X_l.append(X)
        deltas = []
        sig_w_X = np.array([Network.sigmoid(i) for i in np.dot(self.weights[-1],X_l[-2])])
        delta_last = np.multiply(X_l[-1] - y,sig_w_X)
        deltas.append(delta_last)
        delta_i = delta_last
        for i in range(1,len(self.weights)):
            sig_prime_w_x = np.array(Network.sigmoid(np.dot(self.weights[-1-i],X_l[-2-i]),True))
            delta_i =np.multiply(np.dot(self.weights[-i].T,delta_i), sig_prime_w_x)
            deltas.append(delta_i)
        deltas.reverse()
        delta_w = [np.array(self.learning_rate*np.dot(deltas[i],X_l[i].T)) for i in range(len(deltas))]
        delta_b = [self.learning_rate*i for i in deltas]
        return(delta_w,delta_i)

    def train(self,training_data):
        for epoch in range(self.num_epochs):
            num = 0
            np.random.shuffle(training_data)
            batches = [training_data[i:i+self.batch_size] for i in range(int(len(training_data)/self.batch_size))]
            for batch in batches:
                X = [i[0] for i in batch]
                y = [i[1] for i in batch]
                e = []
                delta_w = [np.zeros((self.layers[i+1],self.layers[i])) for i in range(self.num_layers-1)]
                delta_b = [np.zeros((self.layers[i],1)) for i in range(1,self.num_layers)]
                for i in range(len(X)):
                    e.append(self.error(X[i],y[i]))
                    delta_wi,delta_bi = self.backprop(X[i],y[i])
                    for l in range(self.num_layers-1):
                        delta_w[l]+=delta_wi[l]
                        delta_b[l]+=delta_bi[l]
                if num%3==0:
                    print("Batch error: {}".format(sum(sum(e))))
                num+=1
                for l in range(len(delta_w)):
                    self.weights[l] -=self.learning_rate*delta_w[l]
                    self.biases[l] -=self.learning_rate*delta_b[l]
            print("Epoch {} out of {} completed.".format(epoch+1,self.num_epochs))
            self.learning_rate /=10
    def predict(self,x):
        b = self.feed_forward(x)
        return np.argmax(b)
training_data = read_csv('train.csv',sep=',',header=None).as_matrix()
feature = np.array([np.array(i[1:]).reshape(784,1)/155 for i in training_data])
label = [i[0] for i in training_data]
label = np.array([Network.oneHot(i).reshape(10,1) for i in label])
training_data = [list(a) for a in zip(feature,label)]

print('training data loaded')
net = Network([784,256,128,64,10])
net.train(training_data)
test_data = read_csv('test.csv',sep=',',header=None).as_matrix()
test_data = np.array([i[1:] for i in test_data])

test_data = np.array([i.reshape(784,1)/155 for i in test_data])

with open("sample2.csv",'w') as fout:
    fout.write("id,number\n")
    counter = 1
    answers = [net.predict(i) for i in test_data]
    for answer in answers:
        fout.write(str(counter)+","+str(answer)+'\n')
        counter+=1
