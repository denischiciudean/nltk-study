import nltk
from nltk import PunktSentenceTokenizer

"""
# Chunking | grouping of things

noun phrases | descriptive words describing the noun
"""


def process_content(content):
    try:
        for i in content:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?<DT>}"""

            chunk_parser = nltk.RegexpParser(chunkGram)
            chunked = chunk_parser.parse(tagged)

            chunked.draw()

            print(chunked)

    except Exception as e:
        print(str(e))


def main():
    train_test = nltk.corpus.state_union.raw('2005-GWBush.txt')
    test_text = nltk.corpus.state_union.raw('2006-GWBush.txt')

    # Train tokenizer on previous data from 2005 and after we tokenize from 2006
    tokenizer = PunktSentenceTokenizer(train_test)

    tokenized_speech = tokenizer.tokenize(test_text)

    process_content(tokenized_speech)
