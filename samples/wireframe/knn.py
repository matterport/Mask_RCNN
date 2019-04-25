from database_actions import get_known_encodings
#from samples.wireframe.database_actions import get_known_encodings
import numpy as np


def knn(embedding):
    """
    Function that returns the nearest neighbor to an image encoding

    :param encodings: Vector of encoding for an image (,128)
    :return: Predicted label
    """
    matrix_embeddings, labels = get_known_encodings("Database.db")
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


def overlaps(rois):
    """
    :param rois: regions of interest in format (y1, x1, y2, x2)
    :return: List of objects


    Example:
    [[ 99,325,135,363], [ 54,229,88,264], [ 53,230,94,266], [ 93,321,132,361]]
        -> [1, 2, 2, 1]
    """
    n_objects = np.zeros(len(rois))
    objects = 1
    for i in range(len(rois) - 1):
        if n_objects[i] != 0:
            continue
        l_1_y1 = rois[i][0]
        l_1_x1 = rois[i][1]
        r_1_y2 = rois[i][2]
        r_1_x2 = rois[i][3]
        n_objects[i] = objects
        for j in range(i+1, len(rois)):
            if n_objects[j] != 0:
                continue
            l_2_y1 = rois[j][0]
            l_2_x1 = rois[j][1]
            r_2_y2 = rois[j][2]
            r_2_x2 = rois[j][3]
            if l_1_x1 > r_2_x2 or l_2_x1 > r_1_x2:
                continue
            elif l_1_y1 > r_2_y2 or l_2_y1 > r_1_y2:
                continue
            else:
                n_objects[j] = objects
        objects += 1
    return n_objects

def overlaps_bool(pred_roi, bbox):
    """
    :param rois: regions of interest in format (y1, x1, y2, x2)
    :return: List of objects


    Example:
    [[ 99,325,135,363], [ 54,229,88,264], [ 53,230,94,266], [ 93,321,132,361]]
        -> [1, 2, 2, 1]
    """
    l_1_y1 = pred_roi[0]
    l_1_x1 = pred_roi[1]
    r_1_y2 = pred_roi[2]
    r_1_x2 = pred_roi[3]
    l_2_y1 = bbox[0]
    l_2_x1 = bbox[1]
    r_2_y2 = bbox[2]
    r_2_x2 = bbox[3]
    if l_1_x1 > r_2_x2 or l_2_x1 > r_1_x2:
        return False
    elif l_1_y1 > r_2_y2 or l_2_y1 > r_1_y2:
        return False
    else:
        return True

def overlapsTrueAndPredicted(roisTrue, roisPred):
    """
    :param rois: regions of interest in format (y1, x1, y2, x2)
    :return: List of objects


    Example:
    ([[ 99,325,135,363], [ 54,229,88,264]], [[ 53,230,94,266], [ 93,321,132,361]])
        -> [2, 1]
    """
    n_objects = np.zeros(len(roisTrue))
    for i in range(len(roisTrue) - 1):
        if n_objects[i] != 0:
            continue
        l_1_y1 = roisTrue[i][0]
        l_1_x1 = roisTrue[i][1]
        r_1_y2 = roisTrue[i][2]
        r_1_x2 = roisTrue[i][3]
        for j in range(len(roisPred)):
            l_2_y1 = roisPred[j][0]
            l_2_x1 = roisPred[j][1]
            r_2_y2 = roisPred[j][2]
            r_2_x2 = roisPred[j][3]
            if l_1_x1 > r_2_x2 or l_2_x1 > r_1_x2:
                continue
            elif l_1_y1 > r_2_y2 or l_2_y1 > r_1_y2:
                continue
            else:
                n_objects[i] = j
    return n_objects
