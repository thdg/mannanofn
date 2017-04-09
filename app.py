from flask import Flask, request, jsonify
from collections import defaultdict

app = Flask(__name__)

from names import names, generator, generate, \
    filter_approved, train, split_data

n = names()
g = generator(filter_approved(n))

x, y, X, Y = split_data(n, ratio=1)
clf, cv = train(x, y)


def predict(x):
    return clf.predict(cv.transform(x))


@app.route('/mannanofn/', methods=['GET'])
def mannanofn():
    error = None

    try:
        n = int(request.args.get('n', '1'))
        if not 0 < n <= 100:
            error = "n must satisfy 0 < n <= 100."
    except ValueError:
        error = "n should be integer."

    seed = request.args.get('seed', '').title()
    try:
        classify = bool(request.args.get('classify', False))
    except ValueError:
        error = "Classfy should be boolean."

    try:
        name_list = list(set(generate(g, seed=seed) for i in range(n)))
        name_list.sort()
    except ValueError:
        error = "Can't find names for seed \"%s\"." % seed

    if error:
        return jsonify({"error": error})

    if classify:
        y = predict(name_list)
        names = defaultdict(list)
        for name, group in zip(name_list, y):
            names[group].append(name)
    else:
        names = name_list

    return jsonify({"names": names})


if __name__ == "__main__":
    app.run()
