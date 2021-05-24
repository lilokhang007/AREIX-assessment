from sklearn.cluster import KMeans
from path import preprocessed_dir
from helper import load_csv_as_df
from sklearn.decomposition import PCA
df_res = load_csv_as_df('Data_scaled.csv', dir=preprocessed_dir)
pca = PCA(n_components=2)
df_res[['PC1', 'PC2']] = pca.fit_transform(df_res)
kmeans = KMeans(n_clusters=7, random_state=0).fit(df_res[['PC1', 'PC2']])
df_res['label'] = kmeans.labels_