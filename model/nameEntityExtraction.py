import numpy as np
from model.base import NER
from managers.corpus import eHealthCorpusManager

import warnings
warnings.filterwarnings('ignore')

# KERAS
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Concatenate, Embedding, TimeDistributed, Dense

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

# SKLEARN
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, classification_report
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report

class SimpleNER(NER):
    def __init__(self, words_amount, max_word, tags=None, embeddings=50):
        self.words_amount = words_amount
        self.tags = ['B-Concept', 'B-Action', 'B-Predicate', 'B-Reference', 'I-Concept', 'I-Action', 'I-Predicate', 'I-Reference', 'O'] if tags is None else tags
        self.tags_amount = len(self.tags)
        self.embeddings = embeddings
        self.max_word = max_word
        self.model = self._build_model()

    def _build_model(self):
        input_word_layer = Input(shape=(None,), name='Input_word')
        embedding_layer = Embedding(input_dim=self.words_amount + 1, output_dim=self.embeddings, name='Embedding')(input_word_layer)

        input_char_layer = Input(shape=(None, self.max_word), name='Input_char')
        char_lstm = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.1), name='Bidirectional_char')(input_char_layer)
        char_time = TimeDistributed(Dense(50, activation='relu'))(char_lstm)

        concat_layer = Concatenate(name='Concatenate')([embedding_layer, char_time])
        bi_lstm = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.1), name='Bidirectional')(concat_layer)
        crf_layer = CRF(self.tags_amount, name='CRF')(bi_lstm)

        model = Model(input=[input_word_layer, input_char_layer], output=crf_layer)
        model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_viterbi_accuracy])

        return model

    def fit(self, x_train, y_train, epochs=1, validation_data=None):
        n, n_digits = len(y_train), len(str(len(y_train)))
        history = {}
        epochs_logs = []

        for epoch in range(1, epochs + 1):
            epoch_log = { 'epoch' : epoch,
                          f'{self.model.metrics_names[0]}' : [],
                          f'{self.model.metrics_names[1]}' : []}

            print(f'\nEpoch {epoch}/{epochs}')

            loss, accuracy = self.model.metrics_names[0], self.model.metrics_names[1]
            val_loss, val_accurancy = 'val_' + loss, 'val_' + accuracy

            for i, ((x_words, x_char), y) in enumerate(zip(x_train, y_train)):
                score = self.model.train_on_batch({'Input_word' : np.array([x_words]), 'Input_char': np.array([x_char])}, np.array([y]))

                epoch_log[loss] += [score[0]]
                epoch_log[accuracy] += [score[1]]

                percent = ((i + 1) * 100) // (n * 10)
                equals = 3 * percent - 1
                points = 29 - equals
                i_digits = n_digits - len(str(i + 1))
                print(f'{" " * i_digits}{i + 1}/{n} [{"=" * equals}{">" if percent > 0 else ""}{"." * points}] - {loss}: {np.mean(epoch_log[loss]):0.5} - {accuracy}: {np.mean(epoch_log[accuracy]):0.5}')

            epoch_log[loss] = np.mean(epoch_log[loss])
            epoch_log[accuracy] = np.mean(epoch_log[accuracy])

            if validation_data:
                epoch_log[val_loss] = []
                epoch_log[val_accurancy] = []
                
                for (x_words, x_char), y in zip(*validation_data):
                    score = self.model.test_on_batch({'Input_word' : np.array([x_words]), 'Input_char': np.array([x_char])}, np.array([y]))

                    epoch_log[val_loss] += [score[0]]
                    epoch_log[val_accurancy] += [score[1]]

                epoch_log[val_loss] = np.mean(epoch_log[val_loss])
                epoch_log[val_accurancy] = np.mean(epoch_log[val_accurancy])
                print(f'{val_loss}: {epoch_log[val_loss]:0.5} - {val_accurancy}: {epoch_log[val_accurancy]:0.5}')
            
            epochs_logs += [epoch_log]

        history[loss] = [epoch_log[loss] for epoch_log in epochs_logs]
        history[accuracy] = [epoch_log[accuracy] for epoch_log in epochs_logs]
        if validation_data is not None:
            history[val_loss] = [epoch_log[val_loss] for epoch_log in epochs_logs]
            history[val_accurancy] = [epoch_log[val_accurancy] for epoch_log in epochs_logs]

        return self

    def predict(self, x):
        predictions = []

        for x_words, x_char in x:
            predictions.append(self.model.predict({'Input_word' : np.array([x_words]), 'Input_char': np.array([x_char])})[0])
        return predictions

    def get_params(self, deep=False):
        return dict(zip(['words_amount', 'max_word', 'tags', 'embeddings'], [self.words_amount, self.max_word, self.tags, self.embeddings]))

    def score(self, X, y, sample_weight=None):
        y_true = np.array([np.nonzero(i)[1] for i in y])
        y_predict = np.array([np.nonzero(i)[1] for i in self.predict(X)])

        return flat_f1_score(y_true, y_predict, sample_weight=sample_weight, average='weighted')

        # return np.mean([accuracy_score(y_true_i, y_predict_i, sample_weight=sample_weight) for y_true_i, y_predict_i in zip(y_true, y_predict)])


tags = ['B-Concept', 'B-Action', 'B-Predicate', 'B-Reference', 'I-Concept', 'I-Action', 'I-Predicate', 'I-Reference', 'O']