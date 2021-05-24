import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from path import preprocessed_dir
from helper import load_csv_as_df
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AREIX_Cluster(object):
    def __init__(self):
        # Load processed csv
        self.df = load_csv_as_df('Data_processed.csv', dir=preprocessed_dir)

        # Save the standard scaler as a instance variable (to be called later)
        self.ss = StandardScaler()
        self.ss.fit(self.df)

    def run(self):
        self.scale() # normalization
        self.PCA() # perform dimensionality reduction by PCA
        self.cluster() # perform cluster algorithm
        self.get_centroid()

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
        sns.scatterplot(data=self.df_PCA, hue='label', x='PC1', y= 'PC2',
                        palette = "pastel")
        sns.scatterplot(data=self.centroids, hue='label', x='PC1', y= 'PC2',
                        palette = "dark", marker = 'x', s=20, legend=False)
        plt.title('Scatter plot of Clustering Results')
        plt.legend()

    def cluster(self, kind='DBSCAN'):
        if kind == 'DBSCAN':
            self.clf = DBSCAN().fit(self.df_PCA[['PC1', 'PC2']])
            self.df_PCA['label'] = self.clf.labels_
        else:
            raise NotImplemented('Only DBSCAN is implemented at the moment.')

    def get_centroid(self):
        # calculate the centroids of the clusters
        self.centroids = self.df_PCA.groupby('label').mean()[['PC1', 'PC2']]

        # calculate the original non-scaled values of fields, represented by the centroids of the clusters
        return pd.DataFrame([
            self.ss.inverse_transform(np.dot(self.pca.components_.T, self.centroids.value[i]))
            for i in range(len(self.centroids))
        ], columns=self.df.columns)

if __name__ == '__main__':
    clf = AREIX_Cluster()
    clf.run()