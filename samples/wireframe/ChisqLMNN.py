import numpy as np
from sklearn.utils.extmath import softmax
import sklearn.decomposition as sk_decomp
import database_actions as db

#Initial embeddings and labels
embeddings, labels = db.get_known_encodings()

#Reduce the dimensionality
pca = sk_decomp.PCA(n_components=3)
pca.fit(embeddings.T)
embeddings = pca.transform(embeddings.T)
embeddings = abs(embeddings / np.linalg.norm(embeddings))

print("Shape of trained embeddings is: {}".format(np.shape(embeddings)))
print("Explained variance of low dimensional data is: {}".format(sum(pca.explained_variance_ratio_)))


#Initialize variables
#It's very important to set the margin very low as the data points are quite densly packed
l = 1.05e-05
mu = 0.5
K = 2
alpha = 0.001

def chi_square_distance(xi, xj):
    """
    Chi square distance

    :param xi: Embedding       (1, D)
    :param xj: Target Neighbor (1, D)
    :return: Distance
    """
    return 1 / 2 * np.nansum(np.square(xi - xj) / (xi + xj))


def distance(xi, X, L):
    """
    Chi square distance from one point xi, to all other points

    :param xi: Embedding       (1, D)
    :param X: Data             (N, D)
    :return: Distances         (1, N)

    """
    N, K = np.shape(X)
    Distances = np.zeros(N)
    for i in range(N):
        Distances[i] = chi_square_distance(L @ xi, L @ X[i, :])
    return Distances


def find_target_neighbors(X, Y, L):
    """
    Find target neighbours for all points
    :param X: Data Matrix      (N, D)
    :param Y: Labels           (1, N)
    :return: TN_lookup_table   (N, K)
    :return: TN_distance_table (N, K)
    """

    global TN_lookup_table
    global TN_distance_table

    N, _ = np.shape(X)
    TN_lookup_table = np.zeros((N, K))
    TN_distance_table = np.zeros((N, K))

    for i in range(N):
        xi = X[i, :]
        yi = Y[i]

        # Find distance from xi to all other points
        TN_Distances = distance(xi, X, L)
        TN_Indicies = np.argsort(TN_Distances)
        j = k = 0

        # Loop to add indicies of target neighbours to lookup table
        for j in range(K):
            # if yi and target neighbour have the same label AND it is not the same point
            if Y[TN_Indicies[k]] == yi and TN_Indicies[k] != i:
                # Add to lookup table and distance table
                TN_lookup_table[i, j] = TN_Indicies[k]
                TN_distance_table[i, j] = TN_Distances[TN_Indicies[k]]
                j += 1
            k += 1
    TN_lookup_table = TN_lookup_table.astype(int)
    return TN_lookup_table, TN_distance_table


def find_imposters(i, X, Y, L):
    """
    Find imposters for a given point in index i

    :param i: Index in data matrix
    :param X: Data Matrix    (N, D)
    :param Y: Labels         (1, N)
    :return: imposters       (?, D)
    :return: imposters_dist  (1, D)
    """

    # Max margin = max distance of a target neighbour for that point + l
    max_dist = 0
    for TN in TN_lookup_table[i, :]:

        # We need to update the distance to our target neighbours and compute the max distance
        dist = chi_square_distance(L @ X[i], L @ X[TN])
        if dist > max_dist:
            max_dist = dist

    # Max dist + margin l
    max_margin = max_dist + l

    # Find distances
    distances = distance(X[i, :], X, L)
    imposters = np.array([0])
    for j, index in enumerate(np.argsort(distances)):
        if distances[j] > max_margin:
            break
        elif Y[i] != Y[index]:
            imposters = np.append(imposters, index)
    imposters.astype(int)
    return imposters[1:]

#Tau Function
def tau_function(X_Matrix, L_Matrix, i, j, alpha):
    N, D = np.shape(X_Matrix)
    numerator = 0
    denominator = 0
    for k in range(D):
        numerator +=   L_Matrix[alpha, k] * (X_Matrix[i, k] - X_Matrix[j, k])
        denominator += L_Matrix[alpha, k] * (X_Matrix[i, k] + X_Matrix[j, k])
    return numerator / denominator

#Check if the impostor is the margin of the target neighbor + marginal distance l
def check(L, xi, xj, xk):
    return chi_square_distance(L @ xi, L @ xj) + l >= chi_square_distance(L @ xi, L @ xk)


def gradient_and_loss_function(X, Y, L_Matrix):
    D, D = np.shape(L_Matrix)
    gradient_matrix = np.zeros((D, D))
    for alpha in range(D):
        for beta in range(D):
            gradient_matrix[alpha, beta], loss = gradient_and_loss_element(X, Y, L_Matrix, alpha, beta)
    return gradient_matrix, loss


def gradient_and_loss_element(X_Matrix, Y, L_Matrix, alpha, beta):
    global mu
    N, _ = np.shape(X_Matrix)
    gradient = 0
    outer_sum = 0
    Inner_sum = 0
    loss = 0
    for i in range(N):
        imposters = find_imposters(i, X_Matrix, Y, L_Matrix)
        Pull = 0
        for j in TN_lookup_table[i, :]:
            tauij = tau_function(X_Matrix, L_Matrix, i, j, alpha)
            Lij = 2 * tauij * (X_Matrix[i, beta] - X_Matrix[j, beta]) - (tauij ** 2) * (
                        X_Matrix[i, beta] + X_Matrix[j, beta])
            outer_sum += Lij
            for l_imposter in imposters:
                if check(L_Matrix, X_Matrix[i],  X_Matrix[j],  X_Matrix[l_imposter]):
                    tauik = tau_function(X_Matrix, L_Matrix, i, l_imposter, alpha)
                    Lik = 2 * tauik * (X_Matrix[i, beta] - X_Matrix[l_imposter, beta]) - (tauik ** 2) * (
                                X_Matrix[i, beta] + X_Matrix[l_imposter, beta])
                    Inner_sum += Lij - Lik
                else:
                    print(i, j, l_imposter)
            # Calculate loss
            loss += (1 - mu) * pull_loss(X_Matrix, L_Matrix, i, j) + mu * push_loss(X_Matrix, L_Matrix, i, j, imposters)

    gradient = (1 - mu) * outer_sum + mu * Inner_sum
    return gradient, loss


# Loss for pull
def pull_loss(X_Matrix, L, i, j):
    return chi_square_distance(L @ X_Matrix[i], L @ X_Matrix[j])


# Loss for push
def push_loss(X_Matrix, L, i, j, imposters):
    loss = 0
    for l_imposter in imposters:
        loss += max(0, l + chi_square_distance(L @ X_Matrix[i], L @ X_Matrix[j]) - chi_square_distance(L @ X_Matrix[i],
                                                                                                       L @ X_Matrix[
                                                                                                           l_imposter]))
    return loss