from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from random import shuffle


def names():
    with open("nafnalisti.csv", "r") as nf:
        header_lines = 1
        names = [line.split(",")[:-1] for line in nf.readlines()[header_lines:]]
    return names


def filter_approved(names):
    return [n for n in names if n[1] == "Sam"]


def filter_sex(names, tag="DR"):
    return [n for n in names if n[3] == tag]


def filter(names, **kwargs):
    pass


def ngram(names, n=(2, 3)):
    cv = CountVectorizer(ngram_range=n, analyzer="char", lowercase=False)
    X = cv.fit_transform(names)
    return cv, X


def train(names, y):
    cv, X = ngram(names)
    clf = svm.LinearSVC()
    #clf = DecisionTreeClassifier()
#    clf = svm.SVC(kernel="precomputed")
#    from ssk import build_gram_matrix as bgm
#    print("start ssk")
#    X = bgm(names, names, 0.05, 3)
#    print("finish ssk")
    clf.fit(X, y)
    return clf, cv


def split_data(n, ratio=.9):
    shuffle(n)
    n = n
    split = int(len(n) * ratio)
    train_n, test_n = n[:split], n[split:]
    x = [i[0] for i in train_n]
    #y = [i[1] == "Sam" for i in train_n]
    y = [i[3][-2:] for i in train_n]

    X = [i[0] for i in test_n]
    #Y = [i[1] == "Sam" for i in test_n]
    Y = [i[3][-2:] for i in test_n]
    return x, y, X, Y


def generator(names):
    ng, X = ngram([" %s " % n[0] for n in names], n=(2, 3))
    starts = set(i[:2] if len(i) == 3 else i[:1] for i in ng.get_feature_names())
    from collections import defaultdict
    nexts = defaultdict(dict)
    for start in starts:
        for i, feature in enumerate(ng.get_feature_names()):
            if feature.startswith(start) and len(feature) == len(start) + 1:
                nexts[start][feature[-1]] = np.sum(X[:, i])
    return nexts


def next_letter(nexts, seed):
    letters = []
    count = []
    for key, value in nexts[seed[-2:]].items():
        letters.append(key)
        count.append(value)

    count = np.array(count)
    total = np.sum(count)
    i = np.random.choice(np.arange(len(letters)), 1, p=count/total)[0]
    return letters[i]


def generate(nexts, seed=""):
    name = " " + seed
    while True:
        name += next_letter(nexts, name[-2:])
        if name[-1] == " ":
            return name.strip().title()


if __name__ == "__main__":
    n = names()
    nset = set(i[0] for i in n)
    x, y, X, Y = split_data(n, ratio=.95)
    clf, cv = train(x, y)
    #from ssk import build_gram_matrix as bgm
    #py = clf.predict(bgm(X, x, 0.05, 3))
    py = clf.predict(cv.transform(X))
    for predicted, true in zip(py, Y):
        print(predicted, true, predicted == true)

    print(f1_score(Y, py, average="macro"))
    print(recall_score(Y, py, average="micro"))
    print(precision_score(Y, py, average="micro"))

    g = generator(filter_sex(filter_approved(n), tag="MI"))
    print("New names:")
    nn = [generate(g) for i in range(100)]

    pcat = clf.predict(cv.transform(nn))
    #from ssk import build_gram_matrix as bgm
    #pcat = clf.predict(bgm(nn, x, 0.05, 3))

    for name, cat in zip(nn, pcat):
        print(name, cat)
