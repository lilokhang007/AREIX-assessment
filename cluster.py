import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from path import preprocessed_dir, output_dir
from helper import load_csv_as_df
from matplotlib.cbook import boxplot_stats
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def boxplot(df, label):
    # data visualization: plot the boxplot of the given label
    plt.boxplot(df[label])
    dct = boxplot_stats(df, label)[0] # get statistics, such as q1, q2, q3, etc.
    return dct

class AREIX_Cluster(BaseEstimator, ClusterMixin):
    def __init__(self, eps = 0.5, min_samples=10, n_components=3):
        self.eps = eps #epsilon value used in DBSCAN
        self.min_samples = min_samples #minimum point counts in the neighborhood of a point to be considered as a core point in DBSCAN
        self.n_components = n_components #dimensions in PCA

    def fit(self, X, y=None):
        # Load processed csv
        self.df = X
        self.scale() # normalization
        self.PCA() # perform dimensionality reduction by PCA
        self.cluster() # perform cluster algorithm
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
        # Save the standard scaler as a instance variable (to be called later)
        self.ss = StandardScaler()
        self.ss.fit(self.df)

        # Scale the values using a standard scaler, as a standard normalization approach before clustering
        scaled_data = self.ss.transform(self.df)
        self.df_scaled = pd.DataFrame(scaled_data, columns=self.df.columns, index=self.df.index)

    def PCA(self):
        self.pca = PCA(n_components=self.n_components)
        self.df_PCA = self.df_scaled # copy
        self.df_PCA[['PC1', 'PC2', 'PC3']] = self.pca.fit_transform(self.df_scaled)

    def plot_before_cluster(self):
        if self.n_components == 2:
            self.df_PCA.plot.scatter(x='PC1', y= 'PC2')

        elif self.n_components == 3:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(projection='3d')
            sns.set(style="darkgrid")
            x = self.df_PCA['PC1']
            y = self.df_PCA['PC2']
            z = self.df_PCA['PC3']
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.scatter(x,y,z)

        plt.title('Scatter plot of PCA Results')
        plt.savefig(output_dir + 'before_cluster_results.png')

    def plot_cluster_results(self):
        # 2D plot
        if self.n_components == 2:
            outliers = self.df_PCA['label'] == -1
            # plot outliers
            sns.scatterplot(data=self.df_PCA[outliers], hue='label', x='PC1', y= 'PC2',
                            palette = ["red"], marker = 'x', s=20)
            # plot clusters
            sns.scatterplot(data=self.df_PCA[~outliers], hue='label', x='PC1', y= 'PC2',
                            palette = "pastel")
            sns.scatterplot(data=self.centroids[self.centroids.index != -1], hue='label', x='PC1', y= 'PC2',
                            palette = "dark", marker = 'o', s=25, legend=False)

        # 3D plot
        elif self.n_components == 3:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(projection='3d')
            sns.set(style="darkgrid")
            x = self.df_PCA['PC1']
            y = self.df_PCA['PC2']
            z = self.df_PCA['PC3']
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            for l in sorted(self.df_PCA['label'].unique()):
                marker = 'o' if l != -1 else 'x'
                ax.scatter(x[self.df_PCA['label'] == l],
                           y[self.df_PCA['label'] == l],
                           z[self.df_PCA['label'] == l],
                           label=l,
                           marker=marker)

        else:
            raise NotImplemented('No visualizations for dimensions > 3.')

        plt.title('Scatter plot of Clustering Results')
        plt.legend()
        plt.savefig(output_dir + 'cluster_results.png')

    def cluster(self, kind='DBSCAN'):
        self.df_res = self.df_PCA # copy
        if kind == 'DBSCAN':
            clf = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.clf = clf.fit(self.df_PCA[['PC1', 'PC2', 'PC3']])
            self.df_res['label'] = self.clf.labels_
            self.labels_ = self.clf.labels_
        else:
            raise NotImplemented('Only DBSCAN is implemented at the moment.')

        # calculate the centroids of the clusters
        self.centroids = self.df_res.groupby('label').mean()[['PC1', 'PC2', 'PC3']]

    def get_cluster_counts(self):
        return self.df_res.groupby('label')['label'].rename('count')

    def unscale(self):
        self.centroid_mean = model.df_res.groupby('label').mean().drop(['PC1', 'PC2', 'PC3'], axis=1)
        # calculate the original unscaled values of fields, represented by the centroids of the clusters
        unscaled_vals = pd.DataFrame(
            model.ss.inverse_transform(self.centroid_mean),
            columns=self.df.columns
        ).round(3) # get rid of long decimals

        unscaled_df = pd.concat([pd.Series(self.centroids.index), unscaled_vals], axis=1)
        unscaled_df.to_csv(output_dir + 'unscale.csv')
        return unscaled_df

    def unscale_to_categorical(self):
        # print human-readable unscaled values of centroids of the clusters
        self.df_interpret= self.unscale()
        # loop through features in unscaled df
        # convert numbers to category labels
        for feature in self.unscale():
            # skip for 'label'
            if feature == 'label':
                continue
            # no need to be negated
            elif feature == 'Investment Income':
                labels = ['big loss', 'loss', 'profit', 'big profit']
                pass
            elif feature in ('Income', 'Investment Income', 'bal_sum', 'saved_amount_sum'):
                labels = ['negatively big', 'negatively small', 'small', 'big']
                pass
            # negate some columns to expenditures
            elif feature in areix_categories:
                labels = ['negatively big', 'negatively small', 'small', 'big'] # for expenditures
                self.df_interpret[feature] = self.df_interpret[feature] * -1

            fmin = min(self.df_interpret[feature])
            fmax = max(self.df_interpret[feature])
            fdiff = abs(fmax - fmin)
            mmin = fmin - fdiff * 0.01
            mmax = fmax + fdiff * 0.01
            if fmin > 0:
                bins = [0, mmax/2, mmax]
                labels = labels[2:]
            else:
                bins = [mmin, mmin/2, 0, mmax/2, mmax]
            print(bins)
            self.df_interpret[feature] = pd.cut(self.df_interpret[feature],
                                           bins=bins,
                                           labels=labels)
            self.df_interpret.to_csv(output_dir + 'interpret.csv')

if __name__ == '__main__':
    # load categories columns
    with open(output_dir + 'areix_categories.json', 'rb') as fp:
        areix_categories = json.load(fp)

    # context manager for better df display
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
        # load preprocessed df
        df_processed = load_csv_as_df('Data_processed.csv', dir=preprocessed_dir).set_index('psid')

        # define the cluster instance
        clf = AREIX_Cluster()

        # set up a dictionary of params to be passed in grid-search cross validation
        params = {
            'n_components': [3],
            'eps': np.linspace(0.1, 1, 10),
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

        # get the best model and perform clustering
        model = search.best_estimator_
        model.fit_predict(df_processed)
        model.plot_before_cluster()
        model.plot_cluster_results()
        model.unscale_to_categorical()
