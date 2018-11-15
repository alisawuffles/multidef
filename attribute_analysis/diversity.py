from collections import defaultdict

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


def read_diversity():
    global diversity_dict
    global definitions_dict

    diversity_dict = defaultdict(int)
    definitions_dict = defaultdict(list)
    ground = False
    with open('data/inspect_ruimin.baseline+gs-soft.blind.txt', 'r') as ifp:
        for line in ifp:
            line = line.strip().split('\t')                     # list of strings separated by tab
            if len(line) == 2 and line[0] == '[word]':          # line looks like [word] word
                if 'word' in locals():                          # if we are not reading the first word
                    function_words.extend(punct)
                    for y in function_words:                    # remove function words and punctuation
                        if y in defs:
                            defs[:] = [x for x in defs if x != y]
                    diversity_dict[word] = len(set(defs))/len(defs)         # before moving on to next word, store value
                word = line[-1]                                             # assign word
                defs = []                                   # stores all words in ground-truth definitions
                continue
            if '[ground-truth]' in line[0]:
                ground = True                               # start reading ground-truth definitions
                continue
            if ground == True and len(line) == 3:           # we are reading ground-truth definitions
                definition = line[2].strip()
                definitions_dict[word].append(definition)
                defs.extend(definition.split(' '))


def get_diversity(word):
    return diversity_dict[word]