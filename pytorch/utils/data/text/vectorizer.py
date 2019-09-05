import numpy as np
import operator
import sys

from pytorch.utils.data.text.vocabulary import Vocabulary

class Vectorizer:

    def __init__(self, **kwargs):
        self.default_indexes = kwargs['default_indexes'] if 'default_indexes' in kwargs else {}
        self.vocabulary = Vocabulary(self.default_indexes)
        self.vocab_size = len(self.default_indexes)
        self.idx_offset = len(self.default_indexes)


    def save_dictionary(self, dictionary_file_name):
        with open(dictionary_file_name, 'w') as f:
            print(self.vocabulary.index_to_word, file=f)


    def load_dictionary(self, file_name, word_col, idx_col=None, sep=' '):
        offset = 1
        with open(file_name) as infile:
            for idx, line in enumerate(infile):
                elem = line.split(sep)
                word = elem[word_col].strip()
                if idx_col is None:
                    index = idx + self.idx_offset # Keep special indexes
                else:
                    index = int(elem[idx_col].strip()) + self.idx_offset
                self.vocabulary.word_to_index[word] = index
                self.vocabulary.index_to_word[index] = word
                self.vocab_size += 1


    def fit_on_texts(self, texts, separator=' '):
        for text in texts:
            self.vocabulary.index_words(text.split(separator))
        # Sort words by frequencies
        words_sorted = sorted(self.vocabulary.word_counts.items(), key=operator.itemgetter(1), reverse=True)
        #
        self.vocabulary.init()
        for word, cnt in words_sorted:
            self.vocabulary.index_word(word, cnt=cnt)
        
        self.vocab_size = len(self.vocabulary.index_to_word)
        

    def texts_to_onehot(self, texts, num_words=None, separator=' '):
        if num_words is not None:
            max_valid_index = num_words + len(self.vocabulary.default_indexes)

        onehot_vectors = []

        for text in texts:
            # Create onehot vector of vocabulary size and initalize all items with 0
            onehot_vector = [0] * num_words
            for word in text.split(separator):
                if word in self.vocabulary.word_to_index:
                    try:
                        onehot_vector[self.vocabulary.word_to_index[word]] = 1
                    except:
                        pass
            onehot_vectors.append(np.array(onehot_vector))

        return np.array(onehot_vectors)


    def texts_to_sequences(self, texts, num_words=None, separator=' ', min_len=1, max_len=None, padding='post', truncate='post', padding_idx=0, unknown_idx=None, auto_padding=False, return_lengths=False):
        max_valid_index = sys.maxsize
        if num_words is not None:
            max_valid_index = num_words + len(self.vocabulary.default_indexes)

        sequences, lengths, valid_indices, max_seq_len = [], [], [], 0

        for i, text in enumerate(texts):
            seq = []
            for word in text.split(separator):
                if word in self.vocabulary.word_to_index:
                    idx = self.vocabulary.word_to_index[word]
                    if num_words is not None and idx > max_valid_index:
                        if unknown_idx is not None:
                            seq.append(unknown_idx)
                            continue
                        else:
                            continue # If there is a word limit, ignore infrequent words (derived from indexes)
                    seq.append(idx)
                else:
                    if unknown_idx is not None:
                        seq.append(unknown_idx)

            net_length = len(seq)
            # Postprocess sequences: padding or truncating of a max sequence length is given
            if max_len is not None:
                if len(seq) > max_len: # Shorten sequence
                    if truncate == 'post':
                        seq = seq[:max_len]
                    elif truncate == 'pre':
                        seq = seq[-max_len:]
                    else:
                        raise Exception('Unknown value for parameter truncate.')
                    # Update net sequence length (shorten to max_len)
                    net_length = max_len
                elif len(seq) < max_len: # Pad sequence
                    if auto_padding == True:
                        if padding == 'post':
                            seq = seq + [padding_idx] * (max_len - len(seq))
                        elif padding == 'pre':
                            seq = [padding_idx] * (max_len - len(seq)) + seq
            # Add sequence to result list
            if net_length >= min_len:
            #if len(seq) >= min_len:
                sequences.append(seq)
                lengths.append(net_length)
                valid_indices.append(i)
                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)
        # If no max_len is given, pad sequences w.r.t the max_seq_len (depends on data)
        if auto_padding == True and max_len is None:
            sequences = np.array([ seq + [padding_idx]*(max_seq_len - len(seq)) for seq in sequences ])
        else:
            sequences = [  np.array(seq) for seq in sequences ]

        if return_lengths is True:
            return np.array(sequences), np.array(valid_indices), np.array(lengths)
        else:
            return np.array(sequences), np.array(valid_indices)


    def prepare_sequences(self, seq_list, num_words=None, separator=' ', min_len=1, max_len=None, padding='post', truncate='post', padding_idx=0, unknown_idx=None, auto_padding=False, return_lengths=False):
        max_valid_index = sys.maxsize
        if num_words is not None:
            max_valid_index = num_words + len(self.vocabulary.default_indexes)

        sequences, lengths, valid_indices, max_seq_len = [], [], [], 0

        for i, sequence in enumerate(seq_list):
            seq = []
            for idx in sequence:
                if num_words is not None and idx > max_valid_index:
                    if unknown_idx is not None:
                        seq.append(unknown_idx)
                        continue
                    else:
                        continue # If there is a word limit, ignore infrequent words (derived from indexes)
                seq.append(idx + self.idx_offset-1)

            net_length = len(seq)
            # Postprocess sequences: padding or truncating of a max sequence length is given
            if max_len is not None:
                if len(seq) > max_len: # Shorten sequence
                    if truncate == 'post':
                        seq = seq[:max_len]
                    elif truncate == 'pre':
                        seq = seq[-max_len:]
                    else:
                        raise Exception('Unknown value for parameter truncate.')
                    # Update net sequence length (shorten to max_len)
                    net_length = max_len
                elif len(seq) < max_len: # Pad sequence
                    if auto_padding == True:
                        if padding == 'post':
                            seq = seq + [padding_idx] * (max_len - len(seq))
                        elif padding == 'pre':
                            seq = [padding_idx] * (max_len - len(seq)) + seq
            # Add sequence to result list
            if net_length >= min_len:
            #if len(seq) >= min_len:
                sequences.append(seq)
                lengths.append(net_length)
                valid_indices.append(i)
                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)
        # If no max_len is given, pad sequences w.r.t the max_seq_len (depends on data)
        if auto_padding == True and max_len is None:
            sequences = np.array([  seq + [padding_idx]*(max_seq_len - len(seq)) for seq in sequences ])
        else:
            sequences = [  np.array(seq) for seq in sequences ]

        if return_lengths is True:
            return np.array(sequences), np.array(valid_indices), np.array(lengths)
        else:
            return np.array(sequences), np.array(valid_indices)


    def sequence_to_text(self, indices):
        return [ self.vocabulary.index_to_word[i] for i in indices ]