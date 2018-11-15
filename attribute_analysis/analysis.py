import random
import atom_weight, preprocessing, training, word_emb, diversity


def show_examples(parsed_data):
    random_words = random.sample(parsed_data.keys(), 10)
    random_words.append('cabinet')

    for word in random_words:
        outputs = parsed_data[word]
        print('\nword: ' + word)
        word_norm = word_emb.get_word_norm(word)
        div = diversity.get_diversity(word)
        print('\n\tword norm: ' + str(word_norm))
        print('\tdef div: ' + str(div))
        print('\tground-truth definitions: ')
        definitions = diversity.definitions_dict[word]
        for definition in definitions:
            print('\t\t' + definition)

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

