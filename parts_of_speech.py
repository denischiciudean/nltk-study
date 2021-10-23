from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import nltk

"""
# Parts of Speech

---- PunktSentenceTokenizer ---- 

Unsupervised ml tokenizer | pretrained, you can retrain
    
    
---- PARTS OF SPEECH TAGS ----

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when


---- Issues ----
might missfire, not really detecting nouns I guess | cause the data is not always correctly formatted
"""


def process_content(content):
    try:
        for i in content:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            print(tagged)

    except Exception as e:
        print(str(e))


def main():
    train_test = state_union.raw('2005-GWBush.txt')
    test_text = state_union.raw('2006-GWBush.txt')

    # Train tokenizer on previous data from 2005 and after we tokenize from 2006
    tokenizer = PunktSentenceTokenizer(train_test)

    tokenized_speech = tokenizer.tokenize(test_text)

    process_content(tokenized_speech)
