"""
TEXT CLASSIFICATION

Get sentiment / opinion

Create good binary filters

"""
import pickle
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

# MultinomialNB distribution | not binary but more features | and all sorts of other classifiers They come with
# general params, you can adjust them for each classifier to get a more custom experience and improve the success my
# tweaking params

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


def find_features(document, word_features):
    # DOCUMENT IS THE LIST OF WORDS / SET IS UNIQUE WORD
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features


def main():
    print("TEXT CLASSIFICATION")

    documents = [
        (list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]

    random.shuffle(documents)

    all_words = []
    for word in movie_reviews.words():
        all_words.append(word.lower())

    all_words = nltk.FreqDist(all_words)

    # print("MOST COMMON WORDS", all_words.most_common(15))
    # print("FREQUENCY OF WORD: STUPID", all_words['stupid'])

    word_features = list(all_words.keys())[:3000]

    # features = find_features(movie_reviews.words('neg/cv000_29416.txt'), word_features)

    # print(features)

    feature_sets = [(find_features(rev, word_features), category) for (rev, category) in documents]

    # print(feature_sets)

    print("NATIVE BAYES ALGO")

    # TRAIN THE ALGO
    training_set = feature_sets[:1900]

    # TEST THE ACCURACY OF THE ALGO
    testing_set = feature_sets[1900:]

    # posterior = prior occurrences * likelihood / evidence | likelihood for +/-
    # Scalable and easy to understand

    # print("Training the NaiveBayesClassifier")

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Original Accuracy Naive Bayes Algo Accuracy:", nltk.classify.accuracy(classifier, testing_set) * 100)

    # classifier.show_most_informative_features(15)

    """
        How to save a trained classifier
    """

    # save_classifier = open("naivebayes.pickle", "wb")
    # pickle.dump(classifier, save_classifier)
    # save_classifier.close()

    # print("Loading from disk")

    # classifier_file = open("naivebayes.pickle", 'rb')
    # classifier = pickle.load(classifier_file)
    # classifier_file.close()

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

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier Algo Accuracy:",
          nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100)

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
