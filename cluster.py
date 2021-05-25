import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from path import preprocessed_dir
from helper import load_csv_as_df
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class AREIX_Cluster(BaseEstimator, ClusterMixin):
    def __init__(self, eps = 0.5, min_samples=10):
        self.eps = eps #epsilon value used in DBSCAN
        self.min_samples = min_samples #minimum point counts in the neighborhood of a point to be considered as a core point in DBSCAN

    def fit(self, X, y=None):
        # Load processed csv
        self.df = X

        # Save the standard scaler as a instance variable (to be called later)
        self.ss = StandardScaler()
        self.ss.fit(self.df)
        self.scale() # normalization
        self.PCA() # perform dimensionality reduction by PCA
        self.cluster() # perform cluster algorithm
        self.get_centroid() # human-readable unscaled values of centroids of the clusters

        return self

    def score(self, X=None):
        # Number of non-noise clusters in labels
        n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        if n_clusters_ < 4 or n_clusters_ > 7:
            return 0 # not fulfilling requirement: 4~7 clusters

        # Number of noise counts
        n_noise_ = list(self.labels_).count(-1)

        # Calculate the Silhouette Score
        sil = silhouette_score(self.df_PCA, self.labels_)
        print("Estimated number of clusters: {}".format(n_clusters_))
        print("Estimated number of noise points: {}".format(n_noise_))
        print("Silhouette Coefficient: {:.3f}".format(sil))
        return sil

    def scale(self):
        # Scale the values using a standard scaler, as a standard normalization approach before clustering
        scaled_data = self.ss.transform(self.df)
        self.df_scaled = pd.DataFrame(scaled_data, columns=self.df.columns)

    def PCA(self):
        self.pca = PCA(n_components=2)
        self.df_PCA = self.df_scaled # copy
        self.df_PCA[['PC1', 'PC2']] = self.pca.fit_transform(self.df_scaled)

    def plot_before_cluster(self):
        self.df_PCA.plot.scatter(x='PC1', y= 'PC2')

    def plot_cluster_results(self):
        outliers = self.df_PCA['label'] == -1
        # plot outliers
        sns.scatterplot(data=self.df_PCA[outliers], hue='label', x='PC1', y= 'PC2',
                        palette = ["red"], marker = 'x', s=20)

        # plot clusters
        sns.scatterplot(data=self.df_PCA[~outliers], hue='label', x='PC1', y= 'PC2',
                        palette = "pastel")
        sns.scatterplot(data=self.centroids[self.centroids.index != -1], hue='label', x='PC1', y= 'PC2',
                        palette = "dark", marker = 'o', s=25, legend=False)
        plt.title('Scatter plot of Clustering Results')
        plt.legend()

    def cluster(self, kind='DBSCAN'):
        self.df_res = self.df_PCA # copy
        if kind == 'DBSCAN':
            clf = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.clf = clf.fit(self.df_PCA[['PC1', 'PC2']])
            self.df_res['label'] = self.clf.labels_
            self.labels_ = self.clf.labels_
        else:
            raise NotImplemented('Only DBSCAN is implemented at the moment.')

    def get_centroid(self):
        # calculate the centroids of the clusters
        self.centroids = self.df_res.groupby('label').mean()[['PC1', 'PC2']]

        # calculate the original unscaled values of fields, represented by the centroids of the clusters
        unscaled_vals = pd.DataFrame([
            self.ss.inverse_transform(
                np.dot(self.pca.components_.T, self.centroids.values[i])
            ) for i in range(len(self.centroids))
        ], columns=self.df.columns)

        return pd.concat([pd.Series(self.centroids.index), unscaled_vals], axis=1 )

if __name__ == '__main__':
    # context manager for better df display
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
        # load preprocessed df
        df_processed = load_csv_as_df('Data_processed.csv', dir=preprocessed_dir).set_index('psid')

        # define the cluster instance
        clf = AREIX_Cluster()

        # set up a dictionary of params to be passed in grid-search cross validation
        params = {
            'eps': np.linspace(0.05, 0.5, 10),
            'min_samples': np.linspace(6, 10, 5)
        }

        # search through the parameters to find the best set of params
        search = GridSearchCV(
            estimator=clf,
            param_grid=params,
            verbose=1,
            n_jobs=1,
            error_score=0
        )
        search.fit(df_processed)

        # print out searching results
        print('Searching complete')
        print('Best parameters: {}'.format(search.best_params_))
        print('Best score: {}'.format(search.best_score_))

        # get the best model
        model = search.best_estimator_
        model.fit_predict(df_processed)
        model.plot_cluster_results()