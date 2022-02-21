import numpy as np


def vote_entropy(x, classes, prediction_list):
    result = 0
    for i in range(len(x[0])):
        for j in range(classes):
            votes = np.sum(prediction_list[:, i, j])
            result += (votes/prediction_list.shape[0] *
                       np.log(votes/prediction_list.shape[0]))

    return result/len(x[0])


def leo_metric(x, classes, prediction_list):
    result_vectors = np.zeros((len(x[0]), classes))
    for i in range(len(x[0])):
        for j in range(len(prediction_list)):
            result_vectors[i, :] += np.array(prediction_list[j][i])

    result_vectors[:, :-1] = result_vectors[:, :-1]**3
    for i in range(len(x[0])):
        result_vectors[i, -1] = (result_vectors[i, -1]
                                 * 2) - 1 if result_vectors[i, -1] != 0 else 0

    return np.sum(result_vectors)
