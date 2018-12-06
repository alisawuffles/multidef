import atom_weight, preprocessing, training, word_emb, diversity, ground
import numpy as np


def pos_table(s_data):
    pos_data = [row[-5:] for row in s_data]
    format_string = "{:<30}{:<30}{:<30}"
    print(format_string.format('POS', 'Count', 'Average Score'))

    a = []
    a_ct = 0
    n = []
    n_ct = 0
    r = []
    r_ct = 0
    v = []
    v_ct = 0

    for ex in pos_data:
        if ex[0] == 1:
            a.append(ex[4])
            a_ct += 1
        elif ex[1] == 1:
            n.append(ex[4])
            n_ct += 1
        elif ex[2] == 1:
            r.append(ex[4])
            r_ct += 1
        elif ex[3] == 1:
            v.append(ex[4])
            v_ct += 1

    a_avg = np.mean(a)
    n_avg = np.mean(n)
    r_avg = np.mean(r)
    v_avg = np.mean(v)
    print(format_string.format('a', a_ct, a_avg))
    print(format_string.format('n', n_ct, n_avg))
    print(format_string.format('r', r_ct, r_avg))
    print(format_string.format('v', v_ct, v_avg))


def show_examples(parsed_data):
    # random_words = random.sample(parsed_data.keys(), 10)

    for word in parsed_data.keys():
        word_norm = word_emb.get_word_norm(word)
        num_ground = ground.get_ground(word)
        div = diversity.get_diversity(word)

        display_results(parsed_data, word, div, word_norm, num_ground)


def display_results(parsed_data, word, div, word_norm, num_ground):
    outputs = parsed_data[word]
    big_atom = 0
    small_atom = 0
    for output in outputs:
        output_def = output[1]

        weight = atom_weight.get_atom_weight(word, output_def)
        if weight > 1.4:
            big_atom = 1
        elif weight < 0.6:
            small_atom = 1

    if big_atom == 0 or small_atom == 0:
        return

    print('\nword: ' + word)
    print('\n\tword norm: ' + str(word_norm))
    print('\tdef div: ' + str(div))
    print('\tnum ground: ' + str(num_ground))
    # print('\tground-truth definitions: ')
    # definitions = diversity.definitions_dict[word]
    # for definition in definitions:
    #     print('\t\t' + definition)

    for output in outputs:
        labels = output[0]
        output_def = output[1]
        sc = preprocessing.score(labels)

        weight = atom_weight.get_atom_weight(word, output_def)
        labels_for_print = []

        for label in labels:
            if label == 'W':
                meaning = 'wrong'
            else:
                label_idx = training.good_labels.index(label)
                meaning = str(training.label_meanings[label_idx])

            labels_for_print.append(label + ': ' + meaning)

        print('\n\t\tatom weight: ' + str(weight))
        print('\t\toutput: ' + output_def)
        print('\t\tlabels: ' + str(labels_for_print))
        print('\t\tscore: ' + str(sc))
