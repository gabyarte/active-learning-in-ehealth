from managers.data import DataManager
from managers.corpus import eHealthCorpusManager, eHealthClusterCorpusManager
from scores.committee import CommitteeScore
from model.nameEntityExtraction import SimpleNER
from utils.plot import plot_active_passive_learning_curve

tags = ['B-Concept', 'B-Action', 'B-Predicate',
        'B-Reference', 'I-Concept', 'I-Action', 'I-Predicate', 'I-Reference', 'O']

corpus_manager1 = eHealthCorpusManager(tags)
corpus_manager1.preprocess('data/training/input_training.txt')

# corpus_manager2 = eHealthClusterCorpusManager(tags)
# corpus_manager2.preprocess('data/training/input_training.txt')

data_manager1 = DataManager(300, 600, corpus_manager1)
# data_manager2 = DataManager(300, 600, corpus_manager2)

model_params = {'word_amount': len(corpus_manager1.words),
                'max_word': corpus_manager1.max_word, 'train_epochs': 5, 'retrain_epochs': 1}


score_algorithm1 = CommitteeScore(
    data_manager1.tagged_x, data_manager1.tagged_y, 5, 150, 10, 10, model_params)

# score_algorithm2 = CommitteeScore(
#     data_manager2.tagged_x, data_manager2.tagged_y, 5, 150, 5, 5, model_params)

data_manager1.define_score_algorithm(score_algorithm1)

# data_manager2.define_score_algorithm(score_algorithm2)


passive_learning_data = data_manager1.get_pasive_learning_data()
active_learning_data = data_manager1.get_active_learning_data_with_retrain()

ner = SimpleNER(len(corpus_manager1.words), corpus_manager1.max_word)

plt = plot_active_passive_learning_curve(
    ner, passive_learning_data, active_learning_data)

plt.show()