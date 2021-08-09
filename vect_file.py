from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import gensim
filename = 'GoogleNews-vectors-negative300.bin.gz'
w2vModel = KeyedVectors.load_word2vec_format(filename, binary=True, limit=50000)

# w2vModel = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=50000)

def w2v(X):
    # words=list(w2vModel.vocab)
    # words = list(w2vModel.vocab)
    words = list(w2vModel.index_to_key)
    x = []
    vector = np.zeros(300) # as word vectors are of zero length
    for word in X.split(): # for each word in a review/sentence
        if word in words:
            vector += w2vModel[word]
            x.append(vector)

    x = np.array(x)
    x = x.mean(axis=0)
    return x
