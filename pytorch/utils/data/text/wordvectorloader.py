import numpy as np
import pandas as pd
import csv
import timeit
import datetime


class WordVectorLoader:

    def __init__(self, embed_dim):
        self.embed_index = {}
        self.embed_dim = embed_dim


    def load_glove(self, file_name):
        df = pd.read_csv(file_name, header=None, sep=' ', encoding='utf-8', quoting=csv.QUOTE_NONE)
        for index, row in df.iterrows():
            word = row[0]
            coefs = np.asarray(row[1:], dtype='float32')
            self.embed_index[word] = coefs
        try:
            self.embed_dim = len(coefs)
        except:
            pass



    def create_embedding_matrix(self, embeddings_file_name, word_to_index, max_idx, sep=' ', init='zeros', print_each=10000, verbatim=False):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embed_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embed_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')

        start = timeit.default_timer()
        with open(embeddings_file_name) as infile:
            for idx, line in enumerate(infile):
                elem = line.split(sep)
                word = elem[0]

                if verbatim is True:
                    if idx % print_each == 0:
                        print('[{}] {} lines processed'.format(datetime.timedelta(seconds=int(timeit.default_timer() - start)), idx), end='\r')

                if word not in word_to_index:
                    continue

                word_idx = word_to_index[word]

                if word_idx <= max_idx:
                    embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')


        if verbatim == True:
            print()

        return embed_mat


    def generate_embedding_matrix(self, word_to_index, max_idx, init='zeros'):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embed_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embed_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')

        for word, i in word_to_index.items():
            if i > max_idx:
                continue
            embed_vec = self.embed_index.get(word)
            if embed_vec is not None:
                embed_mat[i] = embed_vec

        return embed_mat


    def generate_centroid_embedding(self, word_list, avg=False):
        centroid_embedding = np.zeros((self.embed_dim, ))
        num_words = 0
        for word in word_list:
            if word in self.embed_index:
                num_words += 1
                centroid_embedding += self.embed_index.get(word)
        # Average embedding if needed
        if avg is True:
            if num_words > 0:
                centroid_embedding /= num_words
        return centroid_embedding




if __name__ == '__main__':

    word_vector_loader = WordVectorLoader()

    word_vector_loader.create_embedding_matrix('/home/vdw/data/dumps/glove/glove.6B.100d.txt', verbatim=True)