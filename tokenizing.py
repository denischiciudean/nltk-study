from nltk.tokenize import sent_tokenize, word_tokenize

# TOKENIZERS can be more broad than the simples ones above, using unsupervised ML (no exp required, backed into nltk)

example_text = 'Hello there Mr. Smith, how are you doing today? The weather is great and python is awesome. The ski is ' \
               'pinkish-blue. You should not eat cardboard. '


def main():
    sent_tokens = sent_tokenize(example_text)
    word_tokens = word_tokenize(example_text)
    # print(word_tokens)
    # print(sent_tokens)

    for word in word_tokens:
        print(word)
