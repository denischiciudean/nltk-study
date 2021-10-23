import nltk
from nltk import PunktSentenceTokenizer

"""
# Named Entity Recognition

NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian

Like chunking but grouping stuff together | Such as names, George W Bush -> Is chunked as 1 item as it's 1 meaning and not 3 separate words as in basic chunking
"""


def process_content(content):
    try:
        for i in content:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            named_entity = nltk.ne_chunk(tagged, binary=True)

            named_entity.draw()

    except Exception as e:
        print(str(e))


def main():
    print("Named Entity Recognition")
    train_test = nltk.corpus.state_union.raw('2005-GWBush.txt')
    test_text = nltk.corpus.state_union.raw('2006-GWBush.txt')

    # Train tokenizer on previous data from 2005 and after we tokenize from 2006
    tokenizer = PunktSentenceTokenizer(train_test)

    tokenized_speech = tokenizer.tokenize(test_text)

    process_content(tokenized_speech)
