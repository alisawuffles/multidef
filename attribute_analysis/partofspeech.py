from collections import defaultdict


def read_pos(parsed_data):
    global pos_dict
    pos_dict = defaultdict()
    for word in parsed_data:
        for output in parsed_data[word]:
            atom = output[1]
            with open('data/_pruned.gs-soft.reg.pos.txt', 'r') as ifp:
                for line in ifp:
                    line = line.strip().split('\t')
                    if line[0] == word and line[3] == atom:
                        pos = line[2]
                        break
            pos_dict[(word, atom)] = pos


def get_pos(word, atom):
    return pos_dict[(word, atom)]