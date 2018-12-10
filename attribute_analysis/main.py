import parse, preprocessing, training, analysis
import warnings


def main():
    [b, gs, gsp, gspr] = parse.parse()
    s_data, groups, pos_data = preprocessing.get_data(gspr)
    print(s_data)
    print(pos_data)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        training.train_s(s_data, groups)

    # analysis.atoms(gspr)
    analysis.pos_table(pos_data)


if __name__ == "__main__":
    main()