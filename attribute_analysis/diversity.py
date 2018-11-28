from collections import defaultdict
import numpy as np

function_words = ["a", "about", "above", "after", "after", "again", "against", "ago", "ahead", "all", "almost",
                  "almost", "along", "already", "also", "although", "always", "am", "among", "an", "and", "any",
                  "are", "aren't", "around", "as", "away",
                  "backward", "backwards", "be", "because", "before", "behind", "below", "beneath", "beside", "between",
                  "both", "but", "by", "can", "cannot", "can't", "cause", "'cos", "could", "couldn't",
                  "despite", "did", "didn't", "do", "does", "doesn't", "don't", "down", "during",
                  "each", "either", "even", "ever", "every", "except", "for", "forward", "from",
                  "had", "hadn't", "has", "hasn't", "have", "haven't", "he", "her", "here", "hers","herself", "him",
                  "himself", "his", "how", "however", "I", "if", "in", "inside", "inspite", "instead", "into", "is",
                  "isn't", "it", "its", "itself", "just", "least", "less", "like",
                  "many", "may", "mayn't", "me", "might", "mightn't", "mine", "more", "most", "much", "must", "mustn't",
                  "my", "myself", "near", "need", "needn't", "needs", "neither", "never", "no", "none", "nor", "not",
                  "now", "of", "off", "often", "on", "once", "only", "onto", "or", "ought", "oughtn't", "our", "ours",
                  "ourselves", "out", "outside", "over", "past", "perhaps", "quite", "rather",
                  "than", "that", "the", "their", "theirs",
                  "them", "themselves", "then", "there", "therefore", "these", "they", "this", "those", "though",
                  "through", "thus", "till", "to", "together", "too", "towards"]
punct = [',', '.']


def read_definitions():
    global definitions_dict                                     # dictionary of word : list of definitions

    definitions_dict = defaultdict(list)
    ground = False
    with open('data/inspect_ruimin.baseline+gs-soft.blind.txt', 'r') as ifp:
        for line in ifp:
            line = line.strip().split('\t')                     # list of strings separated by tab
            if len(line) == 2 and line[0] == '[word]':          # line looks like [word] word
                word = line[-1]                                 # assign word
                continue
            if '[ground-truth]' in line[0]:
                ground = True                                   # start reading ground-truth definitions
                continue
            if ground == True and len(line) == 3:               # line contains ground-truth definitions
                definition = line[2].strip()
                definitions_dict[word].append(definition)


def get_diversity(word):
    def_words = []  # words across all ground-truth definitions
    definitions = definitions_dict[word]
    for definition in definitions:
        definition = definition.strip().split(' ')
        def_words.extend(definition)

    for y in function_words:  # remove function words and punctuation
        if y in def_words:
            def_words[:] = [x for x in def_words if x != y]
    for y in punct:
        if y in def_words:
            def_words[:] = [x for x in def_words if x != y]

    return len(set(def_words)) / len(def_words)


def get_avg_def_length(word):
    definitions = definitions_dict[word]
    def_lengths = []

    for definition in definitions:
        definition = definition.strip().split(' ')
        def_lengths.append(len(definition))

    return np.mean(def_lengths)