import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy.stats
import probscale


def summary(path, c=1.0):
    files_ls = os.listdir(path)
    files_ls = [_ for _ in files_ls if _.endswith('.xlsx')]
    files_ls = [_ for _ in files_ls if _.startswith('MR-A16-')]
    # files_paths = files_paths[10:20]

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax6 = ax5.twinx()

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    idx = 0
    for i, file in enumerate(files_ls):
        # read excel
        print(file.replace('.xlsx', ''))

        try:
            df = pd.read_excel(path + file, sheet_name='power up ')
            df.columns = np.arange(df.shape[1])  # rename column
            df = df.drop(np.arange(0, np.where(np.array(df) == 'I (A)')[1][0]), axis=1)  # drop I (A) 之前的column
            df.columns = df.loc[np.where(np.array(df) == 'I (A)')[0][0]]  # rename column as 有 "I (A)" 的那一個row
            df = df.drop([np.where(np.array(df) == 'I (A)')[0][0]], axis=0)  # drop 有 "I (A)" 的那一個row
            df = df[df['I (A)'].notna()]  # drop I (A) is nan 的row
            df = df[df['I (A)'].astype(str).str.isnumeric()]  # drop I (A) is not numeric
            df = df.drop_duplicates(subset='I (A)', keep='first')  # drop duplicate I (A) and keep the first one
            df = df.reset_index(drop=True)
            df = df.astype('float64')
            df['P (W)'] = df['P (W)'] * c
            df['PCE'] = df['PCE'] * c

        except:
            print(file + ' is not loaded')

        # build DataFrame (Derek)
        for n in range(len(df)):
            df2 = df2.append(pd.DataFrame({
                'Serial Number': file.replace('.xlsx', ''),
                'I (A)': df.loc[n, 'I (A)'],
                'P (W)': df.loc[n, 'P (W)'],
                'V (V)': df.loc[n, 'V (V)'],
                'PCE': df.loc[n, 'PCE'],
                'P in 0.114NA': df.loc[n, 'Percent Power in 0.114NA'],
                'P in 0.123NA': df.loc[n, 'Percent Power in 0.123NA'],
                'CT': df.loc[n, 'CT']
            }, index=[idx]), ignore_index=True)
            idx += 1

        # Calculate values at P = 335W
        Fp = [df['I (A)'], df['V (V)'], df['PCE'], df['Percent Power in 0.114NA'], df['Percent Power in 0.123NA']]
        Op_335W = []
        for fp in Fp:
            xp = df['P (W)']
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xp[-2:], fp[-2:])
            op_335W = slope * 335 + intercept
            Op_335W.append(op_335W)

        # build DataFrame
        df1 = df1.append(pd.DataFrame({
            'Serial Number': file.replace(path, '').replace('.xlsx', ''),
            'P@2A': df.loc[df[df['I (A)'] == 2].index.tolist()[0], 'P (W)'],
            'P@5A': df.loc[df[df['I (A)'] == 5].index.tolist()[0], 'P (W)'],
            'P@10A': df.loc[df[df['I (A)'] == 10].index.tolist()[0], 'P (W)'],
            'P@15A': df.loc[df[df['I (A)'] == 15].index.tolist()[0], 'P (W)'],
            'P@20A': df.loc[df[df['I (A)'] == 20].index.tolist()[0], 'P (W)'],
            'P@23A': df.loc[df[df['I (A)'] == 23].index.tolist()[0], 'P (W)'],
            'P@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'P (W)'],
            'V@2A': df.loc[df[df['I (A)'] == 2].index.tolist()[0], 'V (V)'],
            'V@5A': df.loc[df[df['I (A)'] == 5].index.tolist()[0], 'V (V)'],
            'V@10A': df.loc[df[df['I (A)'] == 10].index.tolist()[0], 'V (V)'],
            'V@15A': df.loc[df[df['I (A)'] == 15].index.tolist()[0], 'V (V)'],
            'V@20A': df.loc[df[df['I (A)'] == 20].index.tolist()[0], 'V (V)'],
            'V@23A': df.loc[df[df['I (A)'] == 23].index.tolist()[0], 'V (V)'],
            'V@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'V (V)'],
            'PCE@2A': df.loc[df[df['I (A)'] == 2].index.tolist()[0], 'PCE'],
            'PCE@5A': df.loc[df[df['I (A)'] == 5].index.tolist()[0], 'PCE'],
            'PCE@10A': df.loc[df[df['I (A)'] == 10].index.tolist()[0], 'PCE'],
            'PCE@15A': df.loc[df[df['I (A)'] == 15].index.tolist()[0], 'PCE'],
            'PCE@20A': df.loc[df[df['I (A)'] == 20].index.tolist()[0], 'PCE'],
            'PCE@23A': df.loc[df[df['I (A)'] == 23].index.tolist()[0], 'PCE'],
            'PCE@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'PCE'],
            'P in 0.114NA@5A': df.loc[df[df['I (A)'] == 5].index.tolist()[0], 'Percent Power in 0.114NA'],
            'P in 0.114NA@10A': df.loc[df[df['I (A)'] == 10].index.tolist()[0], 'Percent Power in 0.114NA'],
            'P in 0.114NA@15A': df.loc[df[df['I (A)'] == 15].index.tolist()[0], 'Percent Power in 0.114NA'],
            'P in 0.114NA@20A': df.loc[df[df['I (A)'] == 20].index.tolist()[0], 'Percent Power in 0.114NA'],
            'P in 0.114NA@23A': df.loc[df[df['I (A)'] == 23].index.tolist()[0], 'Percent Power in 0.114NA'],
            'P in 0.114NA@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'Percent Power in 0.114NA'],
            'P in 0.123NA@5A': df.loc[df[df['I (A)'] == 5].index.tolist()[0], 'Percent Power in 0.123NA'],
            'P in 0.123NA@10A': df.loc[df[df['I (A)'] == 10].index.tolist()[0], 'Percent Power in 0.123NA'],
            'P in 0.123NA@15A': df.loc[df[df['I (A)'] == 15].index.tolist()[0], 'Percent Power in 0.123NA'],
            'P in 0.123NA@20A': df.loc[df[df['I (A)'] == 20].index.tolist()[0], 'Percent Power in 0.123NA'],
            'P in 0.123NA@23A': df.loc[df[df['I (A)'] == 23].index.tolist()[0], 'Percent Power in 0.123NA'],
            'P in 0.123NA@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'Percent Power in 0.123NA'],
            'CT@0A': df.loc[df[df['I (A)'] == 0].index.tolist()[0], 'CT'],
            'CT@2A': df.loc[df[df['I (A)'] == 2].index.tolist()[0], 'CT'],
            'CT@5A': df.loc[df[df['I (A)'] == 5].index.tolist()[0], 'CT'],
            'CT@10A': df.loc[df[df['I (A)'] == 10].index.tolist()[0], 'CT'],
            'CT@15A': df.loc[df[df['I (A)'] == 15].index.tolist()[0], 'CT'],
            'CT@20A': df.loc[df[df['I (A)'] == 20].index.tolist()[0], 'CT'],
            'CT@23A': df.loc[df[df['I (A)'] == 23].index.tolist()[0], 'CT'],
            'CT@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'CT'],
            'HT1@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'HT1'],
            'HT2@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'HT2'],
            'HT3@25A': df.loc[df[df['I (A)'] == 25].index.tolist()[0], 'HT3'],
            'I@335W': Op_335W[0],
            'V@335W': Op_335W[1],
            'PCE@335W': Op_335W[2],
            'P in 0.114NA@335W': Op_335W[3],
            'P in 0.123NA@335W': Op_335W[4]
        }, index=[i]), ignore_index=True)

        # plot
        df = df.drop(0, axis=0)
        ax1.plot(df['I (A)'], df['P (W)'], 'o-', label=file.replace('MR-A16-', '').replace('.xlsx', ''))
        ax2.plot(df['I (A)'], df['V (V)'], 'o-', label=file.replace('MR-A16-', '').replace('.xlsx', ''))
        ax3.plot(df['I (A)'], df['PCE'], 'o-', label=file.replace('MR-A16-', '').replace('.xlsx', ''))
        ax4.plot(df['I (A)'], df['Percent Power in 0.114NA'], 'o-',
                 label=file.replace('MR-A16-', '').replace('.xlsx', ''))
        ax5.plot(df['I (A)'], df['P (W)'], 'o-', label=file.replace('MR-A16-', '').replace('.xlsx', ''))
        ax6.plot(df['I (A)'], df['PCE'] * 100, 'o-', label=file.replace('MR-A16-', '').replace('.xlsx', ''))

    try:
        os.mkdir(path + 'Summary')
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")

    df1.to_csv(path + 'Summary/Summary.csv', index=False, header=True)
    df2.to_csv(path + 'Summary/Summary - Derek.csv', index=False, header=True)

    ax1.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    ax2.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    ax3.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    ax4.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    ax5.legend(fontsize='small', bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    ax1.set_xlabel('Iop (A)')
    ax2.set_xlabel('Iop (A)')
    ax3.set_xlabel('Iop (A)')
    ax4.set_xlabel('Iop (A)')
    ax5.set_xlabel('Iop (A)')
    ax1.set_ylabel('Power (W)')
    ax2.set_ylabel('Voltage (V)')
    ax3.set_ylabel('PCE')
    ax4.set_ylabel('Ratio')
    ax5.set_ylabel('Power (W)')
    ax6.set_ylabel('Conversion Efficiency (%)')
    ax1.set_title('LIV')
    ax2.set_title('Voltage')
    ax3.set_title('PCE')
    ax4.set_title('Power in 0.114NA')
    ax5.set_title('LIV & Conversion Efficiency (%)')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()
    # ax1.axvline(x=23, linewidth=1, ls='--', color='r')
    # ax2.axvline(x=23, linewidth=1, ls='--', color='r')
    # ax3.axvline(x=23, linewidth=1, ls='--', color='r')
    # ax4.axvline(x=23, linewidth=1, ls='--', color='r')
    ax1.axhline(y=335, linewidth=1, ls='--', color='r')
    ax1.axhline(y=314, linewidth=1, ls='--', color='r')
    ax2.axhline(y=28.4, linewidth=1, ls='--', color='r')
    ax2.axhline(y=27.4, linewidth=1, ls='--', color='r')
    ax2.axhline(y=26.4, linewidth=1, ls='--', color='r')
    ax3.axhline(y=0.42, linewidth=1, ls='--', color='r')
    ax4.axhline(y=0.82, linewidth=1, ls='--', color='r')
    ax4.axhline(y=0.90, linewidth=1, ls='--', color='r')
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig1.savefig(path + 'Summary/LIV.png')
    fig2.savefig(path + 'Summary/Voltage.png')
    fig3.savefig(path + 'Summary/PCE.png')
    fig4.savefig(path + 'Summary/Power in 0.114NA.png')
    fig5.savefig(path + 'Summary/LIV and PCE.png')

    # Plot NA vs Power in NA
    x = [0.114, 0.123]
    y_23 = [df1['P in 0.114NA@23A'].mean() * 100, df1['P in 0.123NA@23A'].mean() * 100]
    y_25 = [df1['P in 0.114NA@25A'].mean() * 100, df1['P in 0.123NA@25A'].mean() * 100]

    fig7, ax7 = plt.subplots(figsize=(8, 6))
    ax7.plot(x, y_23, 'o-', label='23A')
    ax7.plot(x, y_25, 'o-', label='25A')
    ax7.set_xlabel('NA')
    ax7.set_ylabel('Power in NA (%)')
    ax7.set_title('Average Power in NA')
    ax7.axhline(y=90, linewidth=1, ls='--', color='r')
    ax7.legend()
    ax7.grid()
    fig7.savefig(path + 'Summary/Power in NA.png')

    return df1, df2, df


def summary_fix_current(I, df1):
    x = [df1['P@%sA' % I], df1['V@%sA' % I], df1['PCE@%sA' % I],
         df1['P in 0.114NA@%sA' % I], df1['CT@%sA' % I]]
    df = pd.DataFrame()

    for j, x in enumerate(x):
        df = df.append(pd.DataFrame({
            'mean': np.mean(x),
            'std': np.std(x),
            'min': np.min(x),
            'max': np.max(x)
        }, index=[j]), ignore_index=True)
    df.index = ['P@%sA' % I, 'V@%sA' % I, 'PCE@%sA' % I, 'PR@%sA' % I, 'CT@%sA' % I]
    df['Target Min'] = [314, 26.4, 0.442, 0.82, np.nan]
    df['Target Mean'] = [335, 27.4, 0.489, 0.9, 35]
    df['Target Max'] = [np.nan, 28.4, np.nan, np.nan, np.nan]
    df['Y with min'] = [sum([_ > 314 for _ in df1['P@%sA' % I]]) / len(df1['P@%sA' % I]),
                        sum([_ > 26.4 for _ in df1['V@%sA' % I]]) / len(df1['V@%sA' % I]),
                        sum([_ > 0.442 for _ in df1['PCE@%sA' % I]]) / len(df1['PCE@%sA' % I]),
                        sum([_ > 0.82 for _ in df1['P in 0.114NA@%sA' % I]]) / len(df1['P in 0.114NA@%sA' % I]),
                        np.nan]
    df['Y with mean'] = [sum([_ > 335 for _ in df1['P@%sA' % I]]) / len(df1['P@%sA' % I]),
                         np.nan,
                         sum([_ > 0.489 for _ in df1['PCE@%sA' % I]]) / len(df1['PCE@%sA' % I]),
                         sum([_ > 0.9 for _ in df1['P in 0.114NA@%sA' % I]]) / len(df1['P in 0.114NA@%sA' % I]),
                         np.nan]
    df['Statistical Yield'] = [np.nan,
                               scipy.stats.norm(df.iloc[1, 0], df.iloc[1, 1]).cdf(28.4)
                               - scipy.stats.norm(df.iloc[1, 0], df.iloc[1, 1]).cdf(26.4),
                               np.nan, np.nan, np.nan]
    df = df.round(3)
    return df


def probability_plot(df, current, string):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig = probscale.probplot(df.loc[df['I (A)'] == current, string].values, ax=ax, probax='y',
                             bestfit=True, estimate_ci=True, line_kws={'label': 'Fitting Line', 'color': 'b'},
                             scatter_kws={'label': 'Observations'}, problabel='Probability (%)')
    ax.legend(loc='lower right')
    ax.grid(ls='--', c='lightgrey')
    ax.set_title('Probability Plot @ I = %.0fA' % current)
    ax.set_xlabel(string)
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path + 'Summary/Probability - %s.png' % string.replace(' ', ''))


times1 = dt.datetime.now()
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 13})

# path = "//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Derek/MR & EL/0511/Shipment temp/"
path = 'C:/Users/cha75794/Desktop/MR & EL/MR/MR 0622/'
# path = "C:/Users/cha75794/Desktop/MR & EL/MR Summary/Test Data/Shipment 5/"
df1, df2, df_temp = summary(path, 0.91)
Current = [23, 25]
for current in Current:
    df3 = summary_fix_current(current, df1)
    df3.to_csv(path + 'Summary/Summary %sA.csv' % current)

# probability plot
current = 25
probability_plot(df2, current, 'P (W)')
probability_plot(df2, current, 'V (V)')
probability_plot(df2, current, 'PCE')
probability_plot(df2, current, 'P in 0.114NA')

times2 = dt.datetime.now()
T = times2 - times1
T = T.total_seconds()
print('Spending time: %.2fsec' % T)
