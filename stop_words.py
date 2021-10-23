from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

"""
# STOP WORDS
Useless words for NLTK | NLP such as the, sarcasm, filelr words   
"""
example_text = 'Hello there Mr. Smith, how are you doing today? The weather is great and python is awesome. The ski is ' \
               'pinkish-blue. You should not eat cardboard. '


def main():
    common_english_stop_words = stopwords.words('english')

    word_tokens = word_tokenize(example_text)
    print("FULL TOKENIZING")
    print(word_tokens)

    filtered_sentence = [word for word in word_tokens if word not in common_english_stop_words]
    print("\n\nREMOVED STOP WORDS")
    print(filtered_sentence)
