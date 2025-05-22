import nltk

class PoSStripper:

    def __init__(self):
        # download required resources
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('punkt_tab')

    def strip(self, text):
        stripped_text = ""
        # tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)
        # tokenize each sentence into words and perform POS tagging
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            for pos_tag in pos_tags:
                if pos_tag[1] not in ['CD', 'JJ', 'JJR', 'JJS', 'LS', 'NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    continue
                else:
                    stripped_text += pos_tag[0] + " "
        return stripped_text
