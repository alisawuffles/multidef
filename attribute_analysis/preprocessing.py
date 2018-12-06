import word_freq, ground, partofspeech, diversity, word_emb, atom_weight, training
import numpy as np
classes = {'I': 1.0, 'II': 0.6, 'III': 0.3, 'IV': 0.0}
c_perfect = set(['E'])
c_fluency = set(['R', 'S', 'P'])
c_semantic = set(['U', 'N', 'B', 'O', 'M', 'C'])
c_wrong = set(['W'])


def get_data(parsed_data):
    '''
    :param parsed_data: dictionary of word : list of tuples
    :return: W_data: list of lists, each list contains attributes and 0/1 for W
             s_data: list of lists, each list contains attributes and score
             good_data: list of lists, each list contains attributes and 0/1 for non-W labels, all have W = 0
             groups: list of words corresponding to W_data, S_data
             good_groups: list of words corresponding to good_data
    '''

    # create dictionaries
    word_freq.read_word_freq()
    ground.read_ground()
    diversity.read_definitions()
    word_emb.read_word_emb()
    atom_weight.read_atom_weight()
    partofspeech.read_pos(parsed_data)

    # initialize data lists
    W_data = []
    good_data = []
    groups = []
    good_groups = []
    scores = []

    # for each word
    for word in parsed_data:
        es = parsed_data[word]                 # es = list of duple ([labels], definition)
        div = diversity.get_diversity(word)
        word_norm = word_emb.get_word_norm(word)

        # for each output definition
        for output in es:               # output = duple ([labels], output_def)
            labels = output[0]          # labels = [labels]
            output_def = output[1]
            weight = atom_weight.get_atom_weight(word, output_def)

            E = 1 if 'E' in labels else 0
            R = 1 if 'R' in labels else 0
            S = 1 if 'S' in labels else 0
            C = 1 if 'C' in labels else 0
            P = 1 if 'P' in labels else 0
            U = 1 if 'U' in labels else 0
            N = 1 if 'N' in labels else 0
            B = 1 if 'B' in labels else 0
            O = 1 if 'O' in labels else 0
            M = 1 if 'M' in labels else 0
            W = 1 if 'W' in labels else 0

            W_data.append([div, word_norm, weight, W])
            groups.append(word)
            s = score(labels)
            scores.append([s])

            if W == 0:
                good_data.append([div, word_norm, weight, E, R, S, C, P, U, N, B, O, M])
                good_groups.append(word)

    n = training.n
    s_data = np.concatenate(([row[:n] for row in W_data], scores), axis=1)

    return W_data, s_data, groups, good_data, good_groups


def score(labels):
    '''
    :param labels: list of alpha labels
    :return: score
    '''
    if c_perfect.intersection(labels):
        return classes['I']
    if c_wrong.intersection(labels):
        return classes['IV']
    if c_fluency.intersection(labels) and c_semantic.intersection(labels):
        return classes['III']
    else:
        return classes['II']