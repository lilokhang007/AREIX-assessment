#import datetime
import pandas as pd
from path import preprocessed_dir
from helper import convert_to_HKD, load_csv_as_df

df_acc = load_csv_as_df('Data_account.csv')
df_bud = load_csv_as_df('Data_budget.csv')
df_tra = load_csv_as_df('Data_Transaction.csv')
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

# extract budget data
df_bud['saved_amount'] = df_bud.apply(lambda row: convert_to_HKD(
    row, convert_field='saved_amount', from_currency_field='currency'), axis=1)
df_bud = df_bud[df_bud['status'] != 'Failed'] # eliminate failed budgets

df_bud_amt = df_bud['saved_amount'].\
    groupby(df_bud['psid']).unique().\
    apply(lambda ls: sum(ls)).\
    apply(pd.Series)
df_bud_amt.columns = ['saved_amount_sum']

# extract transaction data and convert currencies
df_tra['amount_in_HKD'] = df_tra.apply(lambda row: convert_to_HKD(row), axis=1)
df_tra = df_tra[['psid', 'amount_in_HKD', 'areix_category']]

# distribute 'amount_in_HKD' column by areix_category, using a pivot table
df_tra_pivot = pd.pivot_table(df_tra, index='psid', columns='areix_category', values='amount_in_HKD')
df_tra_pivot = df_tra_pivot.join(df_acc_bal, on='psid')
df_tra_pivot = df_tra_pivot.join(df_bud_amt, on='psid')
df_tra_pivot = df_tra_pivot.fillna(0) # fill all nan with 0

# sort the index correctly
sorted_index = sorted(df_tra_pivot.index, key=lambda x: int(str(x).split('_')[1]))
df_tra_pivot = df_tra_pivot.reindex(sorted_index)

# output the file as a processed csv
df_tra_pivot.to_csv(preprocessed_dir + 'Data_processed.csv')




