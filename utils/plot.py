import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def learning_curve(estimator, X, y, cv):
    cv_

def plot_active_passive_learning_curve(estimator, passive, active, cv=None):
    passive_ner = estimator
    active_ner = estimator.__class__(**(estimator.get_params()))

    passive_score = learning_curve(passive_ner, passive[0], passive[1], cv=cv)
    active_score = learning_curve(active_ner, active[0], active[1], cv=cv)

    plt.figure()

    plt.title('Learning curves for active and passive learning')

    plt.subplot(2, 1, 1)
    plot_learning_curve(passive_score, 'passive', plot_test=False)
    plot_learning_curve(active_score, 'active', plot_test=False, c=['b', 'k'])

    plt.subplot(2, 2, 3)
    plot_learning_curve(passive_score, 'passive')

    plt.subplot(2, 2, 4)
    plot_learning_curve(active_score, 'active', c=['b', 'k'])

    return plt

def plot_learning_curve(data, label, plot_test=True, c=['r', 'g']):
    train_sizes, train_scores, test_scores = data

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color=c[0])
    plt.plot(train_sizes, train_scores_mean, 'o-', color=c[0], label=f"Training score ({label})")
    
    if plot_test:
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color=c[1])
        plt.plot(train_sizes, test_scores_mean, 'o-', color=c[1], label=f"Cross-validation score ({label})")

    plt.legend(loc="best")
    return plt

def plotly_learning_curve(data):
    import plotly.express as px
    import pandas as pd

    train_sizes, train_scores, test_scores, learning = data

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    df = pd.DataFrame({'sizes' : train_sizes, 'mean' : train_scores_mean, 'std' : train_scores_std, 'learning' : learning})
    
    return px.line(df, x='sizes', y='mean', labels=dict(sizes='Training examples', mean='Mean accuracy', std='Std'), hover_name='learning', color='learning')

def plotly_active_passive_learning_curve(estimator, passive, active, cv=None):
    passive_ner = estimator
    active_ner = estimator.__class__(**(estimator.get_params()))

    passive_score = learning_curve(passive_ner, passive[0], passive[1], cv=cv, train_sizes=np.linspace(0.1, 1.0, 2))
    active_score = learning_curve(active_ner, active[0], active[1], cv=cv, train_sizes=np.linspace(0.1, 1.0, 2))

    train_sizes = passive_score[0].extend(active_score[0])
    train_scores = passive_score[1].extend(active_score[1])
    test_scores = passive_score[2].extend(active_score[2])
    learning = np.array(['passive'] * len(passive_score[1]) + ['active'] * len(active_score[1]))

    return plotly_learning_curve((train_sizes, train_scores, test_scores, learning))

