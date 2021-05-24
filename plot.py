import matplotlib.pyplot as plt
# some plotting functions for data exploration, to at least get some idea what's the data shape is like

def plot_all_pairs(df):
    '''
    Plot all pairs of the columns of a dataframe
    :param df: dataframe with columns of numeric value datatypes
    '''

    N = len(df.columns)
    fig, ax = plt.subplots(nrows=N, ncols=N)
    for i, row in enumerate(ax):
        for j, cell in enumerate(row):
            x = df.columns[i]
            y = df.columns[j]

            if i < j + 1:
                df.plot.scatter(x=x, y=y, ax=ax[i, j], title='')
            else:
                pass
                ax[i, j].axis('off')

            if i == j:
                cell.set_xlabel(i)
                cell.set_ylabel(j)
            else:
                cell.set_xlabel('')
                cell.set_ylabel('')

            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
