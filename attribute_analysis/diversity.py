from collections import defaultdict

function_words = ["a", "be", "in", "is", "it", "of", "or",
                  "than", "that", "the", "their", "theirs", "them", "themselves",
                  "then", "there", "therefore", "these", "they", "this", "those", "though",
                  "through", "thus", "till", "to", "together", "too", "towards"]


def read_diversity():
    global diversity_dict
    diversity_dict = defaultdict(int)
    ground = False
    with open('data/inspect_ruimin.baseline+gs-soft.blind.txt', 'r') as ifp:
        for line in ifp:
            line = line.strip().split('\t')                     # list of strings separated by tab
            if len(line) == 2 and line[0] == '[word]':          # line looks like [word] word
                if 'word' in locals():                          # if we are not reading the first word
                    for function_word in function_words:        # remove function words
                        if function_word in defs:
                            defs.remove(function_word)
                    diversity_dict[word] = len(set(defs))/len(defs)         # before moving on to next word, store value
                word = line[-1]                                             # assign word
                defs = []                                   # stores all words in ground-truth definitions
                continue
            if '[ground-truth]' in line[0]:
                ground = True                               # start reading ground-truth definitions
                continue
            if ground == True and len(line) == 3:           # we are reading ground-truth definitions
                definition = line[2].strip().split(' ')
                defs.extend(definition)


def get_diversity(word):
    return diversity_dict[word]