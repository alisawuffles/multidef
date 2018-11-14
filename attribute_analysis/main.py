import parse, preprocessing, training
import warnings


def main():
    [b, gs, gsp, gspr] = parse.parse()
    W_data, s_data, groups, good_data, good_groups = preprocessing.get_data(gspr)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        training.train_alpha(W_data, groups, 'W')

        for label in training.good_labels:
            training.train_alpha(good_data, good_groups, label)

        training.train_s(s_data, groups)


if __name__ == "__main__":
    main()
