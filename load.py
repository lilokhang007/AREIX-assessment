#import datetime
import pandas as pd
from path import preprocessed_dir, output_dir
from helper import convert_to_HKD, load_csv_as_df
import json

# load all the csvs into dfs
df_acc = load_csv_as_df('Data_account.csv')
df_bud = load_csv_as_df('Data_budget.csv')
df_tra = load_csv_as_df('Data_Transaction.csv')
# ----------------------------------------------------------------------------------------------------------------
# 1. Get sum of assets of all accounts from each customer
# extract account data and convert currencies
df_acc['balance_in_HKD'] = df_acc.apply(lambda row: convert_to_HKD(
    row, convert_field='balance'), axis=1)

# get unique sum of balances from each customers
df_acc_bal = df_acc['balance_in_HKD'].\
    groupby(df_acc['psid']).unique().\
    apply(lambda ls: sum(ls)).\
    apply(pd.Series)
df_acc_bal.columns = ['bal_sum']

# 2. Get sum of saved amounts of all goals from each customer
# extract budget data
df_bud['saved_amount'] = df_bud.apply(lambda row: convert_to_HKD(
    row, convert_field='saved_amount', from_currency_field='currency'), axis=1)
df_bud = df_bud[df_bud['status'] != 'Failed'] # eliminate failed budgets

df_bud_amt = df_bud['saved_amount'].\
    groupby(df_bud['psid']).unique().\
    apply(lambda ls: sum(ls)).\
    apply(pd.Series)
df_bud_amt.columns = ['saved_amount_sum']

# 3. Get sum of transactions divided into different categories
# extract transaction data and convert currencies
df_tra['amount_in_HKD'] = df_tra.apply(lambda row: convert_to_HKD(row), axis=1)

# extract income and non-income from Financials
df_tra.loc[(df_tra['areix_category'] == 'Financials') &
           (df_tra['category'] == 'income'), 'areix_category'] = 'Income'
df_tra.loc[(df_tra['areix_category'] == 'Financials') &
           (df_tra['category'] != 'income'), 'areix_category'] = 'Investment Income' # rename as "investment" for any "Financials" not "Income"
df_tra = df_tra[['psid', 'amount_in_HKD', 'areix_category']] # only remain useful fields

# distribute 'amount_in_HKD' column by areix_category, using a pivot table
df_tra_pivot = pd.pivot_table(df_tra, index='psid', columns='areix_category', values='amount_in_HKD')
with open(output_dir + 'areix_categories.json', 'w') as fp:
    json.dump(list(df_tra_pivot.columns), fp)

# combine fields from other dataframes
df_tra_pivot = df_tra_pivot.join(df_acc_bal, on='psid')
df_tra_pivot = df_tra_pivot.join(df_bud_amt, on='psid')
df_tra_pivot = df_tra_pivot.fillna(0) # fill all nan with 0

# sort the index correctly
sorted_index = sorted(df_tra_pivot.index, key=lambda x: int(str(x).split('_')[1]))
df_tra_pivot = df_tra_pivot.reindex(sorted_index)

# output the file as a processed csv
df_tra_pivot.to_csv(preprocessed_dir + 'Data_processed.csv')