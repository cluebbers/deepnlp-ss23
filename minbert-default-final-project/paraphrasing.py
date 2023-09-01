import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from random import choice

def init_nltk():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def replace_adjectives_with_synonyms(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    replaced_sentence = []

    for word, pos in tagged_words:
        if not (pos.startswith('JJ') or pos.startswith('NN')):
            replaced_sentence.append(word)
            continue
        
        synsets = wordnet.synsets(word, pos=get_wordnet_pos(pos))
        if not synsets:
            replaced_sentence.append(word)
            continue

        synonyms = synsets[0].lemmas()
        if not synonyms:
            replaced_sentence.append(word)
            continue

        while synonyms:
            synonym = choice(synonyms)
            if synonym.name() == word:
                synonyms.remove(synonym)
                continue
            replaced_sentence.append(synonym.name())
            break
        else:
            replaced_sentence.append(word)
            continue

    return ' '.join(replaced_sentence)


def replace_adjectives_with_antonyms(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    replaced_sentence = []

    for word, pos in tagged_words:
        if not (pos.startswith('JJ') or pos.startswith('NN')):
            replaced_sentence.append(word)
            continue
        
        synsets = wordnet.synsets(word, pos=get_wordnet_pos(pos))
        if not synsets:
            replaced_sentence.append(word)
            continue

        synonyms = synsets[0].lemmas()
        if not synonyms:
            replaced_sentence.append(word)
            continue

        while synonyms:
            print(choice(synonyms).antonyms())
            synonym = choice(synonyms)
            if synonym.name() == word:
                synonyms.remove(synonym)
                continue
            replaced_sentence.append(synonym.name())
            break
        else:
            replaced_sentence.append(word)
            continue

    return ' '.join(replaced_sentence)

def paraphrase_sentiment(sentence: str, sentiment: int, max_sentences: int = 5):
    new_sentence = replace_adjectives_with_synonyms(sentence)
    return {new_sentence: sentiment}

if __name__ == '__main__':
    init_nltk()

    sentence = "A warm , funny , engaging film"
    print(f'{sentence = }')
    paraphased_sentences = paraphrase_sentiment(sentence, 4)
    
    for new_sentence, sentiment in paraphased_sentences.items():
        print(f'{new_sentence = }, {sentiment = }')



# for lemma in synsets[0].lemmas():
# for antonym in lemma.antonyms():
# antonyms.append(antonym.name())
# if antonyms:
# replaced_sentence.append(antonyms[0]) # Use the first antonym
# else:
# replaced_sentence.append(word)
# else:
# replaced_sentence.append(word)
# else:
# # For nouns, attempt to replace with related words
# synsets = wordnet.synsets(word, pos=get_wordnet_pos(pos))
# if synsets and synsets[0].lemmas():
# related_word = synsets[0].lemmas()[0].name()
# replaced_sentence.append(related_word)
# else:
# replaced_sentence.append(word)

# return ' '.join(replaced_sentence)

# # Example sentence
# sentence = "The quick brown fox jumps over the lazy dog."
# replaced_sentence = replace_words_with_antonyms(sentence)

# print("Original Sentence:", sentence)
# print("Replaced Sentence:", replaced_sentence)