import nltk
from nltk import PunktSentenceTokenizer

"""
# Chinking | removal of things | you can chunk everything and just chink (except) stuff
"""


def process_content(content):
    try:
        for i in content:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                }<VB.?|IN|DT>+{"""

            chunk_parser = nltk.RegexpParser(chunkGram)
            chunked = chunk_parser.parse(tagged)

            chunked.draw()

            print(chunked)

    except Exception as e:
        print(str(e))


def main():
    print("CHINKING")
    train_test = nltk.corpus.state_union.raw('2005-GWBush.txt')
    test_text = nltk.corpus.state_union.raw('2006-GWBush.txt')

    # Train tokenizer on previous data from 2005 and after we tokenize from 2006
    tokenizer = PunktSentenceTokenizer(train_test)

    tokenized_speech = tokenizer.tokenize(test_text)

    process_content(tokenized_speech)
