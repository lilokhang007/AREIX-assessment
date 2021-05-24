#import datetime
import pandas as pd
from helper import convert_to_HKD, load_csv_as_df
from sklearn.preprocessing import StandardScaler

df_acc = load_csv_as_df('data/Data_account.csv')
df_bud = load_csv_as_df('data/Data_budget.csv')
df_tra = load_csv_as_df('data/Data_Transaction.csv')
# ----------------------------------------------------------------------------------------------------------------
# extract account data and convert currencies
df_acc['balance_in_HKD'] = df_acc.apply(lambda row: convert_to_HKD(
    row, convert_field='balance'), axis=1)

# get unique sum of balances from each customers
df_acc_bal = df_acc['balance_in_HKD'].\
    groupby(df_acc['psid']).unique().\
    apply(lambda ls: sum(ls)).\
    apply(pd.Series)
df_acc_bal.columns = ['bal_sum']

# extract transaction data and convert currencies
df_tra['amount_in_HKD'] = df_tra.apply(lambda row: convert_to_HKD(row), axis=1)
df_tra = df_tra[['psid', 'amount_in_HKD', 'areix_category']]

# distribute 'amount_in_HKD' column by areix_category, using a pivot table
df_tra_pivot = pd.pivot_table(df_tra, index='psid', columns='areix_category', values='amount_in_HKD')
df_tra = pd.concat([df_tra_pivot, df_tra], axis=1) # concatenate to original dataframe
df_tra = df_tra.join(df_acc_bal, on='psid') # join to get all fields
df_tra = df_tra.fillna(0) # fill all nan with 0

# scale the values using a standard scaler, as a standard normalization approach before clustering
ss = StandardScaler()
df_res = df_tra.drop(['areix_category', 'psid'], axis=1)
df_res = pd.DataFrame(ss.fit_transform(df_res),columns = df_res.columns)



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0).fit(df_res)
df_res['label'] = kmeans.labels_

