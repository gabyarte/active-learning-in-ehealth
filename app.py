import streamlit as st
import pandas as pd
import numpy as np
from managers.corpus import CorpusManager
from model.base import NER
from utils.reflection import *
from utils.plot import *
from managers.data import DataManager
from scores.committee import CommitteeScore
from sklearn_crfsuite.metrics import flat_classification_report


def main():
    st.title("Active Learning in eHealth")

    st.sidebar.header('Manager Selection')
    modules_manager = load_modules_from_directory('managers/')
    choose_managers = get_classes(modules_manager, CorpusManager)

    manager_key = st.sidebar.selectbox("Choose the corpus manager", list(choose_managers.keys()))

    tags = ['B-Concept', 'B-Action', 'B-Predicate', 'B-Reference', 'I-Concept', 'I-Action', 'I-Predicate', 'I-Reference', 'O']
    manager = choose_managers[manager_key](tags)
    manager.preprocess('data/training/input_training.txt')

    st.sidebar.header('Algorithm Selection')

    modules_models = load_modules_from_directory('model/')
    choose_models = get_classes(modules_models, NER)

    models_key = st.sidebar.selectbox("Choose the estimator", list(choose_models.keys()))

    st.sidebar.header('Performance measure')
    curve_option = 'Learning curve'
    passive_option = 'Classification report for passive learning'
    active_option = 'Classification report for active learning'
    metrics_option = st.sidebar.multiselect('Choose performance measure', [curve_option, passive_option, active_option])

    total_corpus = len(manager.initial_set.sentences)

    st.sidebar.header('Parameters')
    start_corpus = st.sidebar.number_input('Initial amount of tagged data', min_value=1, max_value=total_corpus, value=int(total_corpus/2))
    final_corpus = st.sidebar.number_input('Final amount of tagged data', min_value=start_corpus, max_value=total_corpus, value=total_corpus)
    train_epochs = st.sidebar.number_input('Epochs to train each classifier', min_value=1, value=5)
    retrain_epochs = st.sidebar.number_input('Epochs to retrain each classifier', min_value=1)
    committee_amount = st.sidebar.number_input('Number of classifiers', min_value=1, value=5)
    train_committee = st.sidebar.number_input('Amount of tagged data to train each committee', min_value=10, max_value=start_corpus, value=min(150, start_corpus))
    retrain_committee = st.sidebar.number_input('Amount of tagged data to retrain each committee', min_value=1, max_value=total_corpus - start_corpus, value=10)

    if st.checkbox('Show initial data'):
        data = manager.get_data_frame()
        st.write(data)

    model_params = {'word_amount': len(manager.words), 'max_word': manager.max_word, 'train_epochs': train_epochs, 'retrain_epochs': retrain_epochs}

    if st.button('Start running'):
        passive_learning_data, active_learning_data = None, None
        with st.spinner('Wait while the committee decides...'):
            data_manager = DataManager(start_corpus, final_corpus, manager)
            score_algorithm = CommitteeScore(data_manager.tagged_x, data_manager.tagged_y, committee_amount, train_committee, retrain_committee, retrain_committee, model_params)

            data_manager.define_score_algorithm(score_algorithm)
            passive_learning_data = data_manager.get_pasive_learning_data()
            active_learning_data = data_manager.get_active_learning_data_with_retrain()

        st.success('Now we have an **active corpus**!!!!')
        
        ner = choose_models[models_key](len(manager.words), manager.max_word)

        if curve_option in metrics_option:
            st.subheader(curve_option)
            with st.spinner('Wait while we compute the learning curve...'):
                plt = plot_active_passive_learning_curve(ner, passive_learning_data, active_learning_data)
                st.pyplot(plt)

        if (passive_option in metrics_option) or (active_option in metrics_option):
            # st.info('We are sorry. Currently, our app does not support this measures.')
            test_manager = choose_managers[manager_key](tags)
            test_manager.preprocess('data/development/input_develop.txt', words=manager.words)
            test_manager = test_manager.split_corpus(len(test_manager.initial_set.sentences))[0]
            
            y_true = [[tags[j] for j in np.nonzero(i)[1]] for i in test_manager.samples[1]]

            if passive_option in metrics_option:
                st.subheader(passive_option)
                ner = choose_models[models_key](len(manager.words), manager.max_word)
                ner.fit(passive_learning_data[0], passive_learning_data[1], epochs=8)
                y_predict = [[tags[j] for j in np.nonzero(i)[1]] for i in ner.predict(test_manager.samples[0])]
                report = flat_classification_report(y_true, y_predict)
                st.text(report)

            if active_option in metrics_option:
                st.subheader(active_option)
                ner = choose_models[models_key](len(manager.words), manager.max_word)
                ner.fit(active_learning_data[0], active_learning_data[1], epochs=8)
                y_predict = [[tags[j] for j in np.nonzero(i)[1]] for i in ner.predict(test_manager.samples[0])]
                report = flat_classification_report(y_true, y_predict)
                st.text(report)

main() 