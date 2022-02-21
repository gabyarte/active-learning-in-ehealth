from model.nameEntityExtraction import *
from managers.corpus import *
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report

# from managers.data import DataManager
# from scores.committee import CommitteeScore


tags = ['B-Concept', 'B-Action', 'B-Predicate', 'B-Reference',
        'I-Concept', 'I-Action', 'I-Predicate', 'I-Reference', 'O']

# data_manager = DataManager(300, 600, tags)

# model_params = {'word_amount': data_manager.word_amount,
#                 'max_word': data_manager.max_word, 'train_epochs': 5, 'retrain_epochs': 1}


# score_algorithm = CommitteeScore(
#     data_manager.tagged_x, data_manager.tagged_y, 5, 150, 1, 1, model_params)

# data_manager.define_score_algorithm(score_algorithm)

# passive_learning_data = data_manager.get_pasive_learning_data()
# active_learning_data = data_manager.get_active_learning_data_with_retrain()

# ner = NER(data_manager.word_amount, data_manager.max_word)

# plot_active_passive_learning_curve(
#     ner, passive_learning_data, active_learning_data)

# from managers.corpus import CorpusManager
# from utils.reflection import *


# modules = load_modules_from_directory('managers/')
# print(get_classes(modules, CorpusManager))

training = eHealthCorpusManager(tags)
training.preprocess('data/training/input_training.txt')
training = training.split_corpus(len(training.initial_set.sentences))[0]
x_train, y_train = training.samples

test = eHealthCorpusManager(tags)
test.preprocess('data/development/input_develop.txt', words=training.words)
test = test.split_corpus(len(test.initial_set.sentences))[0]
x_test, y_test = test.samples

ner = SimpleNER(len(training.words), training.max_word)
ner.fit(x_train, y_train, epochs=8)

y_true = np.array([np.nonzero(i)[1] for i in y_test])
y_predict = np.array([np.nonzero(i)[1] for i in ner.predict(x_test)])

print(flat_classification_report(y_true, y_predict, target_names=tags))
