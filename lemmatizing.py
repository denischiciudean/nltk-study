import nltk
from nltk.stem import WordNetLemmatizer

"""
# Lemmatizing | get the contextual meaning, the root meaning of the word, sort of a more improved stemming and more powerful
"""


def main():
    print("Named Entity Recognition")

    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("cats"))
    print(lemmatizer.lemmatize("cacti"))
    print(lemmatizer.lemmatize("geese"))
    print(lemmatizer.lemmatize("rocks"))
    print(lemmatizer.lemmatize("python"))
    print("------------------------------")

    print(lemmatizer.lemmatize("better"))
    print(lemmatizer.lemmatize("best", pos='a'))

    print(lemmatizer.lemmatize("run"))
    print(lemmatizer.lemmatize("run", pos='v'))

    # POS DEFAULT IS NOUN => SHOULD ANALYZE WITH CHUNKING/PARTS OF SPEECH AND MAP TO PASS IN THE CORRECT PART OF SPEECH
