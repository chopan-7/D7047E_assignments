import torch
import string
class Vocabulary:
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0
        for ch in string.printable:
            self.add_word(ch)

    def add_word(self, word):
        word = word + " "
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def word_tensor(self, listOfWords):                                
        tensor = torch.zeros(len(listOfWords)).long()                  
        for i, word in enumerate(listOfWords):
            word = word + " "
            try:
                tensor[i] = self.to_index(word)
            except:
                continue
        return tensor

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]