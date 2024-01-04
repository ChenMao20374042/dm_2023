from json import load
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_road_flow_speed(feature_dir, standardize=True, decomposition=None):

    feature_names = ['flow_day', 'flow_hour', 'speed_day', 'speed_hour']
    features = []
    for name in feature_names:
        f = np.load(os.path.join(feature_dir, name+'.npy'))
        # mask = (f - np.mean(f)) > 20*np.std(f)
        # f[mask] = np.mean(f)
        # print(np.sum(mask))
        if standardize:
            f = (f - np.mean(f)) / np.std(f)
        features.append(f)

    features = np.concatenate(features, axis=-1)
    if decomposition is not None:
        pca = PCA(n_components=decomposition)
        features = pca.fit_transform(features)
    return features

def kmeans_cluster(features, k):

    cluster = KMeans(n_clusters=k)
    cluster.fit(features)
    label = cluster.labels_
    score = silhouette_score(features, label)
    return label, score


if __name__ == '__main__':
    scores = []
    features = load_road_flow_speed('../data/', standardize=True, decomposition=2)
    label, score = kmeans_cluster(features, 6)
    print(label.tolist())
    print(score)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(6):
        cluster_points = features[label == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}')
    plt.legend()
    plt.show()
