import sklearn.cluster as skclus
from sklearn import metrics
import pandas as pd

def metricFunction(embeddings, labels, n_clusters):
    algorithms = []
    algorithms.append(skclus.KMeans(n_clusters=n_clusters, random_state=1))
    algorithms.append(skclus.SpectralClustering(n_clusters=n_clusters, random_state=1,
                                         affinity='nearest_neighbors'))
    algorithms.append(skclus.AgglomerativeClustering(n_clusters=n_clusters))

    data = []
    for algo in algorithms:
        algo.fit(embeddings)
        data.append(({
            'ARI': metrics.adjusted_rand_score(labels, algo.labels_),
            'AMI': metrics.adjusted_mutual_info_score(labels, algo.labels_),
            'Homogenity': metrics.homogeneity_score(labels, algo.labels_),
            'Completeness': metrics.completeness_score(labels, algo.labels_),
            'V-measure': metrics.v_measure_score(labels, algo.labels_),
            'Silhouette': metrics.silhouette_score(embeddings, algo.labels_)}))

    results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity',
                                               'Completeness', 'V-measure',
                                               'Silhouette'],
                           index=['K-means', 'Spectral', 'Agglomerative'])
    return results
