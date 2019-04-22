class Vocabulary:

    def __init__(self, default_indexes={}):
        self.default_indexes = {**default_indexes}
        self.init()


    def init(self):
        self.index_to_word = {**self.default_indexes}
        self.word_to_index = {}
        self.word_counts = {}
        self.num_words = len(self.default_indexes)
        for idx, word in self.index_to_word.items():
            self.word_to_index[word] = idx


    def index_words(self, word_list):
        for word in word_list:
            self.index_word(word)


    def index_word(self, word, cnt=None):
        if word not in self.word_to_index:
            self.index_to_word[len(self.index_to_word)] = word
            self.word_to_index[word] = len(self.word_to_index)
            if cnt is None:
                self.word_counts[word] = 1
                self.num_words += 1
            else:
                self.word_counts[word] = cnt
                self.num_words += cnt
        else:
            if cnt is None:
                self.word_counts[word] += 1
            else:
                self.word_counts[word] += cnt

    def get_words(self, indices):
        return [self.index_to_word[i] if i in self.index_to_word else None for i in indices ]



if __name__ == '__main__':

    vocabulary = Vocabulary(default_indexes={0: '<pad>', 1: '<unk>'})
    print(vocabulary.index_to_word)
    vocabulary.index_word('test')
    print(vocabulary.index_to_word)