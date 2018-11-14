from collections import defaultdict


def parse():
    '''
    read data from text file
    :return: return word : list of tuples ([labels], definition) for each model
    '''

    code = get_code()
    ev_ruimin = get_ev('ruimin')
    ev_downey = get_ev('downey')
    ev_nor = get_ev('nor')

    b1, gs1, gsp1, gspr1 = partition_ev(ev_ruimin, code)
    b2, gs2, gsp2, gspr2 = partition_ev(ev_downey, code)
    b3, gs3, gsp3, gspr3 = partition_ev(ev_nor, code)

    b = union_labels(b1, b2, b3)
    gs = union_labels(gs1, gs2, gs3)
    gsp = union_labels(gsp1, gsp2, gsp3)
    gspr = union_labels(gspr1, gspr2, gspr3)

    return b, gs, gsp, gspr


def get_code():
    '''
    :return: dictionary of word : list of models
    '''

    code = defaultdict(list)

    with open('data/inspect.baseline+gs-soft.code.txt', 'r') as ifp:
        for line in ifp:
            line = line.strip().split('\t')
            if len(line) == 2:  # line looks like [word] word
                word = line[-1]
            if line[0] == '[word]' or line[0] == '[outputs]' or line[0] == '':
                continue
            code[word].append(line[0])  # line contains model

    return code


def get_ev(annotator):
    '''
    :return: ev: dictionary of word : list of tuples
                 tuple has form ([labels], definition)
                 one list entry per output definition
    '''

    ev = defaultdict(list)
    output = False

    if annotator == 'ruimin':
        dataset = 'data/inspect_ruimin.baseline+gs-soft.blind.txt'
    elif annotator == 'downey':
        dataset = 'data/inspect_downey.baseline+gs-soft.blind.txt'
    elif annotator == 'nor':
        dataset = 'data/inspect_nor.baseline+gs-soft.blind.txt'

    with open(dataset, 'r') as ifp:
        for line in ifp:
            line = line.strip().split('\t')                 # list of strings separated by tab
            if len(line) == 2 and line[0] == '[word]':      # line looks like [word] word
                word = line[-1]                             # assign word
                continue
            if line[0] == '[outputs]':                      # line looks like [outputs]
                output = True                               # start reading outputs
                continue
            if line[0][0:14] == '[ground-truth]':
                output = False
                continue
            if output == True and len(line) == 2:           # we are reading outputs
                e = line[0]
                definition = line[1]
                if e == '[W][g]':
                    ev[word].append((['W'], definition))
                else:
                    e = e.replace('[', ',').replace(']', ',').split(',')    # list of labels, numbers, empty strings
                    e = [x for x in e if len(x) != 0]                       # remove empty strings from list
                    labels = [x for x in e if x.isalpha() and x != 'g']
                    ev[word].append((labels, definition))

    return ev


def partition_ev(ev, code):
    '''
    :param ev: dictionary of word : list of tuples for each output definition
    :param code: dictionary of word : list of models for each output definition
    :return: ev partitioned into four dictionaries by model
    '''

    b = defaultdict(list)  # dictionary of word : list of tuples ([labels],[group number])
    gs = defaultdict(list)  # for b, list has only one tuple
    gsp = defaultdict(list)
    gspr = defaultdict(list)

    for w in ev:                                # for every word
        es = ev[w]                              # es = list of tuples
        cs = code[w]                            # cs = list of models
        for e, c in zip(es, cs):                # zip is a list of tuples that pairs elements in es and cs
            if c == 'baseline':
                b[w].append(e)
            elif c == 'gs-soft':
                gs[w].append(e)
            elif c == 'gs-soft+pos':
                gsp[w].append(e)
            else:
                gspr[w].append(e)

    return b, gs, gsp, gspr


def union_labels(data1, data2, data3):
    '''
    :param data1: dictionary of word : list of tuples for annotator 1
    :param data2: dictionary of word : list of tuples for annotator 2
    :param data3: dictionary of word : list of tuples for annotator 3
    :return: dictionary of word : list of tuples where labels in each tuple are combined
    '''

    data = defaultdict(list)
    for word in data1:                                  # for all words
        for i in range(0, len(data1[word])):            # for all definitions
            labels = data1[word][i][0]                  # labels = annotator 1's labels for the i-th definition
            labels.extend(data2[word][i][0])            # add annotator 2's labels
            labels.extend(data3[word][i][0])            # add annotator 3's labels
            labels = list(set(labels))                  # remove duplicates

            if len(labels) != 1 and 'W' in labels:      # if labels contain W and other labels, discard W
                labels.remove('W')
            data[word].append((labels, data1[word][i][1]))

    return data