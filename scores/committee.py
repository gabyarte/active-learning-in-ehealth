import numpy as np

from scores.core import Score
from scores.committee_metrics.metrics import leo_metric, vote_entropy
from model.nameEntityExtraction import SimpleNER


class CommitteeScore(Score):

    def __init__(self, tagged_x, tagged_y, num_of_classifiers, samples_by_classifier, samples_by_classifier_by_step, window_size, model_params):
        self.__tagged_x = tagged_x
        self.__tagged_y = tagged_y
        self.__num_of_classifiers = num_of_classifiers
        self.__samples_by_classifier = samples_by_classifier
        self.__samples_by_classifier_by_step = samples_by_classifier_by_step
        self.window_size = window_size
        self.__model_params = model_params
        self.__committee = self.__build_committee()

    def __create_random_sample(self, x, y, n):
        index = np.random.randint(0, len(x), n)
        x_samples = []
        y_samples = []
        for i in range(len(index)):
            x_samples.append(x[index[i]])
            y_samples.append(y[index[i]])

        return (x_samples, y_samples)

    def __build_committee(self):
        classifiers_list = []

        # Se va a asumir que los conjuntos no son disjuntos, se deberia probar también en un futuro
        # los resultados tomandos conjuntos disjuntos.

        for i in range(self.__num_of_classifiers):
            index = np.random.randint(
                0, len(self.__tagged_x), self.__samples_by_classifier)

            x_sample, y_sample = self.__create_random_sample(
                self.__tagged_x, self.__tagged_y, self.__samples_by_classifier)

            model = SimpleNER(
                self.__model_params['word_amount'], self.__model_params['max_word'])
            model.fit(x_sample, y_sample,
                      epochs=self.__model_params['train_epochs'])

            classifiers_list.append(model)

        return classifiers_list

    def retrain_model(self, x, y):
        for classifier in self.__committee:
            classifier.fit(x, y,
                           epochs=self.__model_params['retrain_epochs'])

    def get_score(self, x):
        prediction_list = []
        for classifier in self.__committee:
            prediction_list.append(classifier.predict([x])[0])

        return vote_entropy(x, 9, np.array(prediction_list))
