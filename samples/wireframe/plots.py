import matplotlib.pyplot as plt
import numpy as np
from samples.wireframe.database_actions import get_known_encodings
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


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


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm