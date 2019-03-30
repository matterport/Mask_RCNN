import matplotlib.pyplot as plt
import numpy as np
from samples.wireframe.database_actions import get_known_encodings



def feature_maps(embeddings, labels):
    """

    Plots 4 feature maps taken from the database.
    """

    _, N = np.shape(embeddings)
    e_map = embeddings.reshape((32, 32, N))
    fig = plt.figure(figsize=(12, 12))
    columns = 2
    rows = 2
    for i in range(1, columns * rows + 1):
        img = e_map[:, :, i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.gca().set_title(labels[i - 1].decode("utf-8"))
        plt.colorbar()
    plt.show()


def histogram_plot(embeddings, labels):
    """
    Outputs a plot of a few of the embeddings in histogram form.


    """

    DIMENSIONS, N_plots = np.shape(embeddings)
    fig = plt.figure()
    for i in range(N_plots):
        ax = fig.add_subplot(N_plots, 1, i + 1)
        label = labels[i]
        ax.set_title("{}".format(label), size=12)
        ax.bar(list(range(DIMENSIONS)), embeddings[:, i])
    plt.show()