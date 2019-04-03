from sklearn.manifold import TSNE
import numpy as np
import samples.wireframe.database_actions as db
import matplotlib.pyplot as plt

def t_sne():
    """
        Does TSNE of dataset in the database
    """
    matrix_encodings, labels = db.get_known_encodings()
    matrix_encodings = np.transpose(matrix_encodings)
    n_components = 3

    x = (matrix_encodings - np.mean(matrix_encodings, 0)) / np.std(matrix_encodings, 0)
    tsne = TSNE(n_components, init='pca', random_state=0)
    x_r = tsne.fit_transform(x)
    return x_r, labels


def plot3D(data, labels):
    """
    Plots PCA with Labels

    :param data: PCA Data
    :param labels: Labels
    """
    scatter_x = data[:, 0]
    scatter_y = data[:, 1]
    scatter_z = data[:, 2]
    group = labels
    cdict = {b'Cross': 'red', b'Done': 'blue', b'h': 'green', b'Heart': 'black', b'Home': 'orange', 
             b'Menu': 'pink', b'More': 'yellow', b'Search': 'magenta', b'Wifi': 'purple', b'H': 'brown'}


    plt.subplots()
    ax = plt.axes(projection='3d')
    for g in np.unique(group):
        ix = [i for i, x in enumerate(group) if x == g]
        ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplots()
    ax = plt.axes(projection='3d')
    for g in np.unique(group):
        ix = [i for i, x in enumerate(group) if x == g]
        ax.scatter(scatter_x[ix], scatter_z[ix], scatter_y[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.show()