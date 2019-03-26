from samples.wireframe.database_actions import get_known_encodings
import numpy as np


def knn(embedding):
    """
    Function that returns the nearest neighbor to an image encoding

    :param encodings: Vector of encoding for an image (,128)
    :return: Predicted label
    """
    matrix_embeddings, labels = get_known_encodings()
    dist_vector = euclidean_distance(embedding, matrix_embeddings)
    norm_dist_vector = dist_vector / np.linalg.norm(dist_vector)
    closest_indices = np.argsort(dist_vector)[0:2]

    results = []
    for index in closest_indices:
            results.append((labels[index], norm_dist_vector[index]))
    return results


def euclidean_distance(vector, matrix):
    dif_matrix = np.subtract(np.transpose(matrix), vector)
    return np.linalg.norm(dif_matrix, axis=1)


def manhattan_distance(vector, matrix):
    dif_matrix = np.abs(np.subtract(np.transpose(matrix), vector))
    return dif_matrix.sum(axis = 1)

