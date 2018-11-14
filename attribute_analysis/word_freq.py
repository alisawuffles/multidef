from collections import defaultdict
import csv


def read_word_freq():
    global word_freq_dict
    word_freq_dict = defaultdict()
    with open('files/word_freq_list.csv') as word_freq_list:
        reader = csv.reader(word_freq_list, delimiter = ',')
        next(reader, None)
        for row in reader:
            word, = {row[1]}
            rank, = {row[0]}
            rank = int(rank)
            word_freq_dict[word] = rank


def get_freq(word):
    if word not in word_freq_dict:
        return 5001
    else:
        return word_freq_dict[word]

