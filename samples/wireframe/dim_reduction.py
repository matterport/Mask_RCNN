import sklearn.decomposition as deco
from sklearn import manifold
import samples.wireframe.plot_frameworks as plf
import matplotlib.pyplot as plt

def dim_reduction(embeddings, labels):
    # Neighbors
    n_neighbors = 21

    # Initiate the list of algorithms
    algorithms = ["x_pca", "x_iso", "x_lle", "x_mlle", "x_hes", "x_ltsa", "x_mds", "x_se", "x_tsne"]

    algorithmDef = {
        "x_pca": deco.PCA(n_components = 12).fit(embeddings).transform(embeddings),
        "x_iso": manifold.Isomap(n_neighbors, n_components=2).fit_transform(embeddings),
        "x_lle": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard').fit_transform(embeddings),
        "x_mlle": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified').fit_transform(embeddings),
        "x_hes": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian').fit_transform(embeddings),
        "x_ltsa": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa').fit_transform(embeddings),
        "x_mds": manifold.MDS(n_components=2, n_init=1, max_iter=100).fit_transform(embeddings),
        "x_se": manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack").fit_transform(embeddings),
        "x_tsne": manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(embeddings)
    }

    # Define the titles
    title = {
        "x_pca": "Principal Components projection of the digits",
        "x_iso": "Isomap projection of the digits",
        "x_lle": "Locally Linear Embedding of the digits",
        "x_mlle": "Modified Locally Linear Embedding of the digits",
        "x_hes": "Hessian Locally Linear Embedding of the digits",
        "x_ltsa": "Local Tangent Space Alignment of the digits",
        "x_mds": "MDS embedding of the digits",
        "x_se": "Spectral embedding of the digits",
        "x_tsne": "t-SNE embedding of the digits"
    }

    # Plot the data points for which different algorithms are applied
    for algo in algorithms:
        plf.plot_2D(algorithmDef[algo], labels, title[algo])
        plt.show()