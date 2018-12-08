import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score
from sklearn import preprocessing as preproc
from sklearn.model_selection import KFold
from scipy import stats

good_labels = ['E', 'R', 'S', 'C', 'P', 'U', 'N', 'B', 'O', 'M']
label_meanings = ['exact', 'redundancy', 'self-reference', 'semantically close', 'wrong POS', 'under-defined',
                        'over-defined', 'partially wrong', 'opposite', 'mixture of two or more meanings']
attributes = ['def div', 'word norm', 'atom wgt']
# attributes = ['def div']
n = len(attributes)


def train_alpha(data, groups, label):
    X = [row[0:n] for row in data]                  # X = m x n matrix containing attribute values

    if label == 'W':
        y = [row[n] for row in data]
        meaning = 'wrong'
    else:
        label_idx = good_labels.index(label)
        y = [row[n + label_idx] for row in data]
        meaning = str(label_meanings[label_idx])

    bias, coefficients, loss, acc, precision, recall, f1, baseline_loss, baseline_acc, p = train(X, y, groups)

    display_results(label, meaning, bias, coefficients, loss, acc, precision,
                    recall, f1, baseline_loss, baseline_acc, p)


def train_s(s_data, groups):
    weights = []
    X = []
    y = []
    for row in s_data:
        X.append(row[:n])
        y.append([1])
        weights.append(row[n])

        X.append(row[:n])
        y.append([0])
        weights.append(1-row[n])

    double_groups = []
    for row in groups:
        double_groups.append(row)
        double_groups.append(row)

    bias, coefficients, loss, acc, precision, recall, f1, baseline_loss, baseline_acc, p \
        = train(X, y, double_groups, weights=weights)

    label = '0-1'
    meaning = 'numerical measure of correctness'

    display_results(label, meaning, bias, coefficients, loss, acc, precision,
                    recall, f1, baseline_loss, baseline_acc, p)


def train(X, y, groups, weights=None):
    X = preproc.scale(X)
    X = np.array(X)
    y = np.array(y)
    if weights is not None:
        weights = np.array(weights)

    kfold = KFold(n_splits=5, shuffle=True)

    bias = []
    coefficients = []
    loss = []
    acc = []
    precision = []
    recall = []
    f1 = []
    baseline_loss = []
    baseline_acc = []
    p = []

    for train_index, test_index in kfold.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        np.ravel(y_train)
        np.ravel(y_test)

        weights_train = None if weights is None else weights[train_index]
        weights_test = None if weights is None else weights[test_index]

        if all(output == 0 for output in y_train) or all(output == 0 for output in y_test):
            continue

        logreg = LogisticRegression(solver='liblinear')

        clf = logreg.fit(X_train, y_train, sample_weight=weights_train)
        bias.append(clf.intercept_[0])
        coefficients.append(clf.coef_[0])

        # log loss
        loss.append(log_loss(y_test, clf.predict_proba(X_test), sample_weight=weights_test))

        # accuracy
        acc.append(clf.score(X_test, y_test, sample_weight=weights_test))

        # precision, recall, f1 score
        precision.append(precision_score(y_test, clf.predict(X_test), sample_weight=weights_test))
        recall.append(recall_score(y_test, clf.predict(X_test), sample_weight=weights_test))
        f1.append(f1_score(y_test, clf.predict(X_test), sample_weight=weights_test))

        # baseline log loss
        perc = sum(y_test) / len(y_test) if weights is None else np.average([a*b for a,b in zip(y_test, weights_test) if a[0] == 1])
        baseline_pred = [perc] * len(y_test)
        baseline_loss.append(log_loss(y_test, baseline_pred, sample_weight=weights_test))

        # zeroR accuracy
        if weights is not None:
            # prediction = stats.mode(y_train)[0]
            pairs = zip(y_train, weights_train)
            prediction = 1 if sum([1 for pair in pairs if pair[1] > 0.5 and pair[0][0] == 1]) > len(y_train)/2 else 0
        else:
            prediction = stats.mode(y_train)[0]

        zeroR_pred = [prediction] * len(y_test)
        baseline_acc.append(np.average([1 if x == y else 0 for x, y in zip(y_test, zeroR_pred)],
                                       weights=weights_test))

        # get p-values for the fitted model
        denom = (2.0 * (1.0 + np.cosh(clf.decision_function(X_train))))
        F_ij = np.dot((X_train / denom[:, None]).T, X_train)        # Fisher Information Matrix
        if np.linalg.det(F_ij) == 0:
            continue
        Cramer_Rao = np.linalg.inv(F_ij)                            # Inverse Information Matrix
        sigma_estimates = np.array(
            [np.sqrt(Cramer_Rao[i, i]) for i in range(Cramer_Rao.shape[0])])    # sigma for each coefficient
        z_scores = clf.coef_[0] / sigma_estimates                               # z-score for each model coefficient
        p_values = [stats.norm.sf(abs(x)) * 2 for x in z_scores]                # two tailed test for p-values
        p.append(p_values)

    return bias, coefficients, loss, acc, precision, recall, f1, baseline_loss, baseline_acc, p


def display_results(label, meaning, bias, coefficients, loss, acc, precision,
                    recall, f1, baseline_loss, baseline_acc, p):
    print('\n')
    print('Label ' + label + ': ' + meaning)
    format_string = "{:<30}{:<30}{:<30}{:<30}"
    print(format_string.format('Attribute', 'Coefficient', 'p-value', 'p<0.1'))
    print(format_string.format('bias', str(round(np.mean(bias), 3)), '', ''))

    for i in range(n):  # for every attribute
        attribute = attributes[i]
        coefficient = np.mean([row[i] for row in coefficients])
        p_value = np.mean([row[i] for row in p])

        flag = ''
        if p_value < 0.1:
            flag = 'X'

        print(format_string.format(attribute, round(coefficient, 5), round(p_value, 5), flag))

    print('---------------------------')
    print('log loss: ' + str(round(np.mean(loss), 5)))
    print('accuracy: ' + str(round(np.mean(acc), 5)))
    print('precision: ' + str(round(np.mean(precision), 5)))
    print('recall: ' + str(round(np.mean(recall), 5)))
    print('f1 score: ' + str(round(np.mean(f1), 5)))

    print('baseline log loss: ' + str(round(np.mean(baseline_loss), 5)))
    print('zeroR accuracy: ' + str(round(np.mean(baseline_acc), 5)))

