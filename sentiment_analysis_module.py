"""
TEXT CLASSIFICATION

Get sentiment / opinion

Create good binary filters

"""
import pickle
import nltk
import random

from nltk import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

# MultinomialNB distribution | not binary but more features | and all sorts of other classifiers They come with
# general params, you can adjust them for each classifier to get a more custom experience and improve the success my
# tweaking params

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def votes(self, features):
        votes = []
        for classifier in self._classifiers:
            try:
                votes.append(classifier.classify(features))
            except Exception as e:
                print(e)
                pass
        return votes

    def classify(self, features):
        votes = self.votes(features)
        return mode(votes)

    def confidence(self, features):
        votes = self.votes(features)
        choice_votes = votes.count(mode(votes))
        confidence = choice_votes / len(votes)
        return confidence


def find_features(document, word_features):
    # DOCUMENT IS THE LIST OF WORDS / SET IS UNIQUE WORD
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features


def main():
    print("TEXT CLASSIFICATION")

    short_positive = open('data/positive.txt', 'r').read()
    short_negative = open('data/negative.txt', 'r').read()
    documents = []
    for review in short_positive.split("\n"):
        documents.append((review, 'pos'))

    for review in short_negative.split("\n"):
        documents.append((review, 'neg'))

    all_words = []

    short_positive_words = word_tokenize(short_positive)
    short_negative_words = word_tokenize(short_negative)

    for w in short_positive_words:
        all_words.append(w.lower())

    for w in short_negative_words:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)

    word_features = list(all_words.keys())[:5000]

    feature_sets = [(find_features(rev, word_features), category) for (rev, category) in documents]

    random.shuffle(feature_sets)

    # print(feature_sets)
    print("NATIVE BAYES ALGO")

    # POSITIVE DATA

    # TRAIN THE ALGO | FIRST 10k training
    training_set = feature_sets[:10000]
    # TEST THE ACCURACY OF THE ALGO | All after 10k testing
    testing_set = feature_sets[10000:]

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Original Accuracy Naive Bayes Algo Accuracy:", nltk.classify.accuracy(classifier, testing_set) * 100)

    """
        Adding SciKitLearn Algos besides NaiveBayes
    """
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier Algo Accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set) * 100)

    # GaussianNB_classifier = SklearnClassifier(GaussianNB())
    # GaussianNB_classifier.train(training_set)
    # print("MNB_classifier Algo Accuracy:", nltk.classify.accuracy(GaussianNB_classifier, testing_set) * 100)

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier Algo Accuracy:", nltk.classify.accuracy(BernoulliNB_classifier, testing_set) * 100)
    LogisticRegression_classifier = None
    try:
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=100000))
        LogisticRegression_classifier.train(training_set)
        print("LogisticRegression_classifier Algo Accuracy:",
              nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100)
    except Exception as e:
        print(e)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier Algo Accuracy:",
          nltk.classify.accuracy(SGDClassifier_classifier, testing_set) * 100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    print("SVC_classifier Algo Accuracy:", nltk.classify.accuracy(SVC_classifier, testing_set) * 100)

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print("NuSVC_classifier Algo Accuracy:", nltk.classify.accuracy(NuSVC_classifier, testing_set) * 100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier Algo Accuracy:", nltk.classify.accuracy(LinearSVC_classifier, testing_set) * 100)

    voted_classifier = VoteClassifier(
        LinearSVC_classifier,
        NuSVC_classifier,
        SVC_classifier,
        SGDClassifier_classifier,
        LogisticRegression_classifier,
        BernoulliNB_classifier,
        MNB_classifier,
        classifier
    )

    print("voted_classifier Algo Accuracy:", nltk.classify.accuracy(voted_classifier, testing_set) * 100)

    # for i in range(4):
    #     print("Classification : ", voted_classifier.classify(testing_set[i][0]), "Confidence %: ",
    #           voted_classifier.confidence(testing_set[i][0]) * 100)
