import pandas as pd
from currency_converter import CurrencyConverter
c = CurrencyConverter(fallback_on_missing_rate=True)

# function to load data into dataframes
def load_csv_as_df(path_to_csv):
    with open(path_to_csv, encoding="UTF-8") as csvfile:
        return pd.read_csv(csvfile)

# given a field name and original currency code, convert the value of that field to HKD
def convert_to_HKD(
    row,
    convert_field='amount',
    from_currency_field='currency_code',
    #datetime_field='made_on',
    #datetime_format_field='%d/%m/%Y',
):
    # convert the amount to HKD
    return c.convert(
        row[convert_field],
        row[from_currency_field],
        'HKD',
        # the datetime parameter might be useful for long-term project, since currencies conversion vary over time
        # for simplicity, I just demonstrate that it can be calculated from the datetime
        # we can safely assume that the currencies conversion are constant as the given data is quite recent
        #datetime.datetime.strptime(row[datetime_field], datetime_format_field)
    )