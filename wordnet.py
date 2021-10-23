from nltk.corpus import wordnet


def noob():
    print("WORD : program")
    program_synonimes = wordnet.synsets("program")
    print(program_synonimes)
    print("\nLEMMAS : ", program_synonimes[0].lemmas())
    print("LEMMAS[0].name : ", program_synonimes[0].lemmas()[0].name())

    print("\nDEFINITIONS")
    print("SYNONYMS[0].name : ", program_synonimes[0].definition())

    print("\nEXAMPLES")
    print("SYNONYMS[0].name : ", program_synonimes[0].examples())

    synonyms = []
    antonyms = []

    for syn in wordnet.synsets("good"):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    print("\n\nSyns and Antonyms")
    print("SYN:")
    print(set(synonyms))
    print("ANS:")
    print(set(antonyms))


def semantic_similarity():
    """
        SEMANTIC SIMILARITY

        wup = Wo? and Palmer

        use to rewrite stuff
    """
    word1 = wordnet.synset("ship.n.01")
    word2 = wordnet.synset("boat.n.01")
    print(word1.wup_similarity(word2) * 100)

    word1 = wordnet.synset("ship.n.01")
    word2 = wordnet.synset("car.n.01")
    print(word1.wup_similarity(word2) * 100)

    word1 = wordnet.synset("ship.n.01")
    word2 = wordnet.synset("cat.n.01")
    print(word1.wup_similarity(word2) * 100)

    word1 = wordnet.synset("ship.n.01")
    word2 = wordnet.synset("cactus.n.01")
    print(word1.wup_similarity(word2) * 100)


def main():
    print("WORDNET")

    # noob()

    semantic_similarity()
