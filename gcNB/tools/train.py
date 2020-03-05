from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from gcNB.tools.preprocess import preprocess


class NaiveBayesClassifier:
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred

    def accuracy(self, X, Y):
        return self.model.score(X, Y)


def main(data):
    X, Y = preprocess(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)

    model = NaiveBayesClassifier()
    model.train(X_train, Y_train)
