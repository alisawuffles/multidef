import atom_weight, preprocessing, training, word_emb, diversity, ground, partofspeech
import numpy as np


def pos_table(pos_data):
    format_string = "{:<10}{:<10}{:<15}{:<15}"
    print(format_string.format('POS', 'Count', 'Avg Score', 'Avg Def Length'))

    a = []
    a_len = []
    a_ct = 0
    n = []
    n_len = []
    n_ct = 0
    r = []
    r_len = []
    r_ct = 0
    v = []
    v_len = []
    v_ct = 0

    for ex in pos_data:
        if ex[0] == 1:              # it's an adjective
            a_len.append(ex[4])     # output definition length
            a.append(ex[5])         # score
            a_ct += 1
        elif ex[1] == 1:            # it's a noun
            n_len.append(ex[4])
            n.append(ex[5])
            n_ct += 1
        elif ex[2] == 1:            # it's an adverb
            r_len.append(ex[4])
            r.append(ex[5])
            r_ct += 1
        elif ex[3] == 1:            # it's a verb
            v_len.append(ex[4])
            v.append(ex[5])
            v_ct += 1

    print(format_string.format('a', a_ct, round(np.mean(a), 3), round(np.mean(a_len), 3)))
    print(format_string.format('n', n_ct, round(np.mean(n), 3), round(np.mean(n_len), 3)))
    print(format_string.format('r', r_ct, round(np.mean(r), 3), round(np.mean(r_len), 3)))
    print(format_string.format('v', v_ct, round(np.mean(v), 3), round(np.mean(v_len), 3)))


def atoms(s_data):
    weights = []
    scores = []
    big_atom_scores = []
    for ex in s_data:
        weight = ex[0]
        sc = ex[1]
        weights.append(weight)
        scores.append(sc)
        if weight > 1.48:
            big_atom_scores.append(sc)

    print(np.percentile(weights, 90))
    print(np.mean(scores))
    print(np.mean(big_atom_scores))


def show_examples(parsed_data):
    # random_words = random.sample(parsed_data.keys(), 10)

    for word in parsed_data.keys():
        word_norm = word_emb.get_word_norm(word)
        num_ground = ground.get_ground(word)
        div = diversity.get_diversity(word)

        display_results(parsed_data, word, div, word_norm, num_ground)


def display_results(parsed_data, word, div, word_norm, num_ground):
    outputs = parsed_data[word]
    # big_atom = 0
    # small_atom = 0
    # for output in outputs:
    #     output_def = output[1]
    #
    #     weight = atom_weight.get_atom_weight(word, output_def)
    #     if weight > 1.4:
    #         big_atom = 1
    #     elif weight < 0.6:
    #         small_atom = 1
    #
    # if big_atom == 0 or small_atom == 0:
    #     return

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
        pos = partofspeech.get_pos(word, output_def)
        labels_for_print = []

        for label in labels:
            if label == 'W':
                meaning = 'wrong'
            else:
                label_idx = training.good_labels.index(label)
                meaning = str(training.label_meanings[label_idx])

            labels_for_print.append(label + ': ' + meaning)

        print('\n\t\tatom weight: ' + str(weight))
        print('\n\t\tpart of speech: ' + pos)
        print('\t\toutput: ' + output_def)
        print('\t\tlabels: ' + str(labels_for_print))
        print('\t\tscore: ' + str(sc))
