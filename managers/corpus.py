from scripts.utils import *
from pathlib import Path
import numpy as np
import random as rnd
# import streamlit as st
import re

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

class CorpusManager:
    def preprocess(self, path):
        pass

    def split_corpus(self, n):
        pass


class eHealthCorpusManager(CorpusManager):
    def __init__(self, tags):
        self.tags = tags
        self.max_word = 0
        self.words = set()
        self.initial_set = Collection()
        self.collection = Collection()
        self._indexes_to_choose = set()
        self._chosen_indexes = set()

    def preprocess(self, path, words=None):
        """
        Preprocess the eHealth corpus from eHealth Knowledge Discovery Challenge at IberLEF 2019.

        ## Arguments
        * `path`: Path where the corpus is allocated. The path must meet all the requirements describe in the challenge's specification
        * `words`: Set of words. If specified, it is assume that the vectors must be build using only the words that appears in it
        """

        if words is not None: self.words = words

        path = Path(path)

        collection = Collection()
        collection = collection.load(path)

        for sentence in collection.sentences:
            sentence_ob = Sentence(sentence.text)

            sentence_by_words = text_to_word_sequence(sentence.text, filters='!#$%&*+-./<=>?@[\\]^_`{|}~\t\n', lower=False)

            if words is None:
                self.words.update(sentence_by_words)

            start = 0
            for i, word in enumerate(sentence_by_words):
                process_word = text_to_word_sequence(word, lower=False)[0]
                
                span = re.search(process_word, word).span()
                span = (start + span[0], start + span[1])

                label = self._find_keyphrase_by_start(sentence, span[0])
                keyphrase = Keyphrase(sentence, label, i, [span])
                sentence_ob.keyphrases.append(keyphrase)

                start += len(word) + 1 # TODO plus one for the space PARCHE!!
            self.collection.sentences.append(sentence_ob)

        self.initial_set = self.collection.clone()
        self.max_word = max(map(len, self.words))
        self._indexes_to_choose = set(range(len(self.collection.sentences)))

    def split_corpus(self, n):
        """
        Get samples of sentences from the corpus.

        ## Arguments
        * `n`: Integer.

        ## Returns
        A list of new `eHealthCorpusManager`'s instances is returned. To access to the samples store during the call, use the property `samples` of each corpus manager.

        ## Notes
        For simplicity reasons this method returns new instances of `eHealthCorpusManager`. The property `samples` of the new instance
        stores a tuple of `(x_vectors, y_vectors, words_amount, max_word)`.

        The `x_vectors` is a list where each element is a tuple of the form `(words_vector, char_vectors)`. Which are the two inputs of 
        the NER algorithm.
        """
        total = len(self.initial_set.sentences)

        # Base corpus
        corpus = self._choose_indexes_to_split(n)
        sentences = [self.initial_set.sentences[i] for i in corpus]
        samples = self._get_vectors(sentences)
        base_manager = self._build_manager(samples, sentences)

        # Rest of the corpus
        corpus = set(range(total)).difference(corpus)
        sentences = [self.initial_set.sentences[i] for i in corpus]
        samples = self._get_vectors(sentences)
        rest_manager = self._build_manager(samples, sentences)
        
        return [base_manager, rest_manager]

    # @st.cache
    def get_data_frame(self):
        import pandas as pd
        data = [ [i, keyphrase.spans[0][0], keyphrase.spans[0][1], keyphrase.text, keyphrase.label] for i, sentence in enumerate(self.collection.sentences) for keyphrase in sentence.keyphrases]
        return pd.DataFrame(data, columns=['Index', 'Start', 'End', 'Word', 'Tag'])

    def _choose_indexes_to_split(self, n):
        total = len(self.initial_set.sentences)
        return set(rnd.sample(range(total), n))

    def _build_manager(self, samples, sentences, words=None, max_word=None):
        manager = eHealthCorpusManager(self.tags)

        manager.words = words if words else self.words
        manager.max_word = max_word if max_word else self.max_word
        manager.samples = samples
        manager.initial_set = self.initial_set
        manager.collection = Collection(sentences)

        return manager

    def _find_keyphrase_by_start(self, sentence, start):
        for keyphrase in sentence.keyphrases:
            spans = sorted(keyphrase.spans)
            for i, (start_keyphrase, _) in enumerate(spans):
                if start_keyphrase == start:
                    if i == 0: return f'B-{keyphrase.label}'
                    else: return f'I-{keyphrase.label}'
        return 'O'

    def _get_vectors(self, data=None, words=None, max_word=None):
        if data is None: data = self.initial_set.sentences

        x_vectors = [self._get_x_vector(sentence, words, max_word) for sentence in data]
        x_vectors = np.array(x_vectors)

        y_vectors = [self._get_y_vector(sentence, x_vector[0]) for sentence, x_vector in zip(data, x_vectors)]
        y_vectors = np.array(y_vectors)
        
        return x_vectors, y_vectors

    def _get_x_vector(self, sentence, words=None, max_word=None):        
        if words is None: words = self.words
        if max_word is None: max_word = self.max_word

        word_to_index = dict(zip(words, range(1, len(words) + 1)))
        sentence_by_words = text_to_word_sequence(sentence.text, lower=False)

        sentence_by_index = [word_to_index[word] if word in word_to_index else 0 for word in sentence_by_words]

        words_by_index = [list(map(ord, word)) for word in sentence_by_words]
        words_by_index = pad_sequences(words_by_index, maxlen=max_word, padding='post', truncating='post', value=-1)

        return np.array(sentence_by_index), words_by_index

    def _get_y_vector(self, sentence, x_vector):
        tags_amount = len(self.tags)
        tags_to_index = dict(zip(self.tags, range(tags_amount)))

        y_vector = [None for _ in range(len(x_vector))]
        for keyphrase in sentence.keyphrases:
            tag_vector = [0] * tags_amount
            tag_vector[tags_to_index[keyphrase.label]] = 1

            y_vector[keyphrase.id] = tag_vector

        return y_vector


class eHealthClusterCorpusManager(eHealthCorpusManager):
    def _choose_indexes_to_split(self, n):
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        k = 10

        sentences = [sentence.text for sentence in self.collection.sentences]

        weights = TfidfVectorizer().fit_transform(sentences)
        clusters = KMeans(k).fit_predict(weights)

        brotherhoods = [set((clusters == i).nonzero()[0]) for i in range(k)]

        per_cluster = [0] * k
        extra, in_clusters = n, list(range(k))

        while extra:
            temp_in = in_clusters
            amount = extra // len(in_clusters)

            for i in temp_in:
                sum_amount = min(amount, len(brotherhoods[i]))
                per_cluster[i] += sum_amount
                extra -= sum_amount

                if amount >= len(brotherhoods[i]):
                    in_clusters.remove(i)

        indexes = []
        for i, c in enumerate(per_cluster):
            indexes += rnd.sample(brotherhoods[i], amount)

        return set(indexes)
