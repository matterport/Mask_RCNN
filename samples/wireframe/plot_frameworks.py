import matplotlib.pyplot as plt
import numpy as np

# Plot the dimensionality reduction frameworks.
def plot_2D(encodings, labels, title):
    encodings = (encodings - np.mean(encodings, 0)) / np.std(encodings, 0)

    plt.figure()
    scatter_x = encodings[:, 0]
    scatter_y = encodings[:, 1]
    group = labels
    cdict = {b'Cross': 'red', b'Done': 'blue', b'Heart': 'black', b'Home': 'orange',
             b'Menu': 'pink', b'More': 'yellow', b'Search': 'magenta', b'Wifi': 'purple', b'H': 'green'}

    ax = plt.axes()
    for g in np.unique(group):
        ix = [i for i, x in enumerate(group) if x == g]
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    if title is not None:
        plt.title(title)

