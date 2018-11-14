from collections import defaultdict
from ast import literal_eval as make_tuple


def read_atom_weight():
    global atom_id_dict
    atom_id_dict = defaultdict(list)               # atom: list of (word, output) generated from that atom
    texts = ['data/_pruned.gs-soft.txt', 'data/_pruned.gs-soft.pos.txt', 'data/_pruned.gs-soft.reg.pos.txt']
    for text in texts:
        with open(text, 'r') as ifp:
            for line in ifp:
                line = line.strip().split('\t')
                word = line[0]
                atom_id = int(line[1])
                output = line[3]
                atom_id_dict[atom_id].append((word, output))

    global atom_weight_dict
    atom_weight_dict = defaultdict()                # (word, atom id) : weight of atom
    with open('data/word_atom.txt', 'r') as ifp:
        for line in ifp:
            line = line.strip().split('\t')
            word = line[0]
            pairs = line[1:]
            for pair in pairs:
                pair = make_tuple(pair)
                atom_id = pair[0]
                weight = pair[1]
                atom_weight_dict[(word, atom_id)] = weight


def get_atom_weight(word, output):
    for id in atom_id_dict:
        for pair in atom_id_dict[id]:
            t_word = pair[0]
            t_output = pair[1]
            if word == t_word and output == t_output:
                return atom_weight_dict[(word, id)]
