import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.io as sio
from keras.layers import Input, Dense, Lambda, Dropout,merge
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from tensorflow import mul
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, Word2Vec
from random import shuffle
from deepwalk import graph
import gensim
import random
import gensim.utils as ut
import scipy.io as sio

def trainDoc2Vec(doc_list=None, buildvoc=1, passes=20, dm=0,
                 size=100, dm_mean=0, window=8, hs=1, negative=5, min_count=1, workers=4):
    model = Doc2Vec(dm=dm, size=size, dm_mean=dm_mean, window=window,
                    hs=hs, negative=negative, min_count=min_count, workers=workers) #PV-DBOW
    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):
       
        shuffle(doc_list)  # shuffling gets best results
        model.train(doc_list)

    return model

NetworkSentence=namedtuple('NetworkSentence', 'words tags index')
def readNetworkData(dir, stemmer=0): #dir, directory of network dataset
    allindex={}
    alldocs = []
    labelset = set()
    with open(dir + '/new_ab.txt') as f1:
        for l1 in f1:
            #tokens = ut.to_unicode(l1.lower()).split()
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split()

            words = tokens[1:]
            tags = [tokens[0]] # ID of each document, for doc2vec model
            index = len(alldocs)
            allindex[tokens[0]] = index # A mapping from documentID to index, start from 0
            alldocs.append(NetworkSentence(words, tags, index))

    return alldocs, allindex,



n=30422
directory = 'dataset/dblp_new'
network=np.load("dataset/dblp_new/dblp.npy")
# data = sio.loadmat("../dataset/dblp/dblp_label.mat")
group =np.load("dataset/dblp_new/label.npy")

m =300

dm = 0
passes =10
cores = 4
alldocs, allsentence = readNetworkData(directory)
doc_list = alldocs[:]  # for reshuffling per pass
tridnr_model = trainDoc2Vec(doc_list, workers=cores, size=m, dm=dm, passes=passes, min_count=3)
vecs = [tridnr_model.docvecs[ut.to_unicode(str(j))] for j in range(n)]
network=np.concatenate([network,vecs],axis=1)
n=n+m
hidden_dim =1000
hid_dim=5000
nb_epoch =1
epsilon_std = 1
use_loss = 'xent' # 'mse' or 'xent'
decay = 1e-4 # weight decay, a.k. l2 regularization
bias = True

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(m,), mean=0.,std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

x = Input(shape=(n,))
h_encoded1 = Dense(hid_dim, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias, activation='tanh')(x)
h_encoded = Dense(hidden_dim, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias, activation='tanh')(h_encoded1)

z_mean = Dense(m, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias)(h_encoded)
z_log_var = Dense(m, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias)(h_encoded)
z = Lambda(sampling, output_shape=(m,))([z_mean, z_log_var])

z_mean2 = Dense(m, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias)(h_encoded)
z_log_var2 = Dense(m, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias)(h_encoded)
z2 = Lambda(sampling, output_shape=(m,))([z_mean2, z_log_var2])

con=merge([z,z2],mode='concat')

h_decoder = Dense(hidden_dim, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias, activation='tanh')
decoder_h1=h_decoder(con)
decoder_h = Dense(hid_dim, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias, activation='tanh')(decoder_h1)

x_hat = Dense(n, W_regularizer=l2(decay), b_regularizer=l2(decay), bias=bias, activation='sigmoid')(decoder_h)

def vae_loss(x, x_hat):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss2 = - 0.5 * K.sum(1 + z_log_var2 - K.square(z_mean2) - K.exp(z_log_var2), axis=-1)
    xent_loss = n * objectives.binary_crossentropy(x, x_hat)
    return xent_loss + kl_loss+kl_loss2
vae = Model(input=x, output=x_hat)
vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(network,network,
            shuffle=False,
            nb_epoch=nb_epoch,
            batch_size=1000)


encoder = Model(input=x,output=con)
embed=encoder.predict(network)
fileresult="result/dblp/"+str(m)+"emb"
np.save(fileresult,embed)
fout = open(fileresult, 'wb')
for t in range(4):
    train_size=t*0.2+0.1
    fout.write('\ntrain size:{}\n'.format(train_size))
    for l in range(10):
        random_state = l
        train, test, y_train, y_test = train_test_split(embed, group, train_size=train_size, random_state=random_state)
        classifier = LinearSVC()
        classifier.fit(train, y_train)
        y_pred = classifier.predict(test)
        macro_f1 = f1_score(y_test, y_pred, pos_label=None, average='macro')
        micro_f1 = f1_score(y_test, y_pred, pos_label=None, average='micro')
        print('Classification macro_f1=%f, micro_f1=%f' % (macro_f1, micro_f1))
fout.close()
