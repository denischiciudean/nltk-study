from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

"""
# STEMMING

getting the root of a word, from a lot of varriations which do not change the meaning : run ran running -> all mean run

A lot of algos, most commonly used is PorterStemmer, around since 1970

You can also train your custom stemmer using nltk

"""

example_sentence = 'It is very important to be pythonly while you are pythoning with python. ALl pythoners have ' \
                   'pythoned poorly before at least once. '


def main():
    words = ["python", "pythoner", "pythoning", "python's", "pythoned"]

    port_stemmer = PorterStemmer()

    stems = [port_stemmer.stem(word) for word in words]

    print("PYTHON STEMS")
    print(stems)

    sentence_words = word_tokenize(example_sentence)

    sentence_stems = [port_stemmer.stem(word) for word in sentence_words]
    print("\n\nSENTENCE STEMS")
    print(sentence_stems)
