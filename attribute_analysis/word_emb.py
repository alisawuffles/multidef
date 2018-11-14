from collections import defaultdict
import numpy as np
from numpy import linalg as la


def read_word_emb():
    global word_norm_dict
    word_norm_dict = defaultdict()
    emb = np.load('files/50k_emb.npy')
    i = 0
    with open('files/50k_vocab.txt', 'r') as ifp:
        for word in ifp:
            word_norm_dict[word[:-1]] = la.norm(emb[i], 2)
            i += 1


def get_word_norm(word):
    return word_norm_dict[word]
