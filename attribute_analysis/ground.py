def read_ground():
    global ground
    ground = {}

    with open('data/inspect_ruimin.baseline+gs-soft.blind.txt', 'r') as ifp:
        for line in ifp:
            line = line.strip().split('\t')                             # list of strings separated by tab
            if len(line) == 2 and line[0] == '[word]':                  # line looks like [word] word
                word = line[-1]                                         # assign word
                continue
            if line[0][0:14] == '[ground-truth]':                       # line looks like [ground-truth][groups]
                n = float(line[0].replace(']', ',').replace('[', ',').split(',')[-2])
                ground[word] = n


def get_ground(word):
    return ground[word]