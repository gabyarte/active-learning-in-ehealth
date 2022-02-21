import numpy as np
from model.nameEntityExtraction import NER
from scores.core import Score


class DataManager:
    def __init__(self, initial_length, total_length, corpus_manager):
        self.__initial_length = initial_length
        self.__total_length = total_length
        self.samples = corpus_manager.split_corpus(initial_length)
        x_tagged, y_tagged = self.samples[0].samples
        x_untagged, y_untagged = self.samples[1].samples

        self.tagged_x = x_tagged.tolist()
        self.tagged_y = y_tagged.tolist()
        self.untagged_x = list(
            zip(x_untagged.tolist(), range(len(x_untagged))))
        self.untagged_y = list(
            zip(y_untagged.tolist(), range(len(y_untagged))))

        self.__new_x = None
        self.__new_y = None
        self.__committee = None
        self.score_algorithm = None

    def __remove_from_untagged(self, index):
        for i in range(len(self.untagged_x)):
            if self.untagged_x[i][1] == index:
                self.untagged_x.pop(i)
                self.untagged_y.pop(i)
                break

    def __get_score_rank(self, amount):
        tuples_list = []
        for i in range(len(self.untagged_x)):
            tuples_list.append((self.score_algorithm.get_score(
                self.untagged_x[i][0]), (self.untagged_x[i][0], self.untagged_y[i][0]), i))

        tuples_list.sort(key=lambda x: x[0])

        for i in range(amount):
            self.__remove_from_untagged(tuples_list[i][-1])

        new_tuples_list = tuples_list[:amount]
        return np.array([item[1] for item in new_tuples_list])

    def define_score_algorithm(self, algorithm):
        self.score_algorithm = algorithm

    def get_active_learning_data_with_retrain(self):
        while len(self.tagged_x) < self.__total_length:

            sentences = self.__get_score_rank(
                min(self.__total_length-len(self.tagged_x), self.score_algorithm.window_size))

            self.__new_x = [sentence[0] for sentence in sentences]
            self.__new_y = [sentence[1] for sentence in sentences]

            self.score_algorithm.retrain_model(self.__new_x, self.__new_y)

            self.tagged_x += self.__new_x
            self.tagged_y += self.__new_y

        return (np.array(self.tagged_x), np.array(self.tagged_y))

    def get_active_learning_data(self):
        sentences = self.__get_score_rank(
            self.__total_length-len(self.tagged_x))
        data_x = self.tagged_x.copy()
        data_y = self.tagged_y.copy()

        for sentence in sentences:
            data_x.append(sentence[0])
            data_y.append(sentence[1])

        return (data_x, data_y)

    def get_pasive_learning_data(self):
        data_x = self.tagged_x.copy()
        data_y = self.tagged_y.copy()

        for i in range(self.__total_length-len(data_x)):
            data_x.append(self.untagged_x[i][0])
            data_y.append(self.untagged_y[i][0])

        return (data_x, data_y)
