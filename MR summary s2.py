import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy.stats
import probscale
from stats import Cpk


def plot_lines(path, x, y, xlabel, ylabel, title, temperature, power_coefficient):
    fig, ax = plt.subplots()
    for i, file in enumerate(files):
        data = TestData(path, file, power_coefficient)
        if data.Temp != temperature:
            continue

        ax.plot(data.df[x], data.df[y], '.-', label=data.SN)
    ax.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = title + ' at %.0fC' % temperature
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    fig.savefig(path + title + '.png')
    plt.close('all')


class TestData:
    def __init__(self, path, file, c=1.0):
        self.path = path
        self.file = file
        self.df, self.df_info = self.read_file(c)
        self.Date = self.df_info.loc[self.df_info.item == 'Date', 'value'].values[0]
        self.SN = self.df_info.loc[self.df_info.item == 'SerialNumber', 'value'].values[0]
        self.Temp = float(self.df_info.loc[self.df_info.item == 'TempSetpoint', 'value'].values[0][:-1])
        self.FWHM = float(self.df_info.loc[self.df_info.item == 'FWMH', 'value'].values[0][:-2])
        self.TP_power = float(self.df_info.loc[self.df_info.item == 'Thermal-Pile-Power(W)', 'value'].values[0][:-1])
        self.TP_temp = float(self.df_info.loc[self.df_info.item == 'Thermal-Pile-Temp(C)', 'value'].values[0][:-1])
        self.P_NA = float(self.df_info.loc[self.df_info.item == 'N/A-1', 'value'].values[0][:-1])

        self.Pop = float(self.df_info.loc[self.df_info.item == 'POP', 'value'].values[0][:-1])
        self.Iop = self.cal_op(self.df['I'])
        self.Vop = self.cal_op(self.df['V'])
        self.PCEop = self.cal_op(self.df['PCE'])
        self.PIBop = self.cal_op(self.df['PIB'])
        self.WLop = self.cal_op(self.df['WL'])

    def read_file(self, c):
        path = self.path
        file = self.file
        with open(path + file) as f:
            data = f.readlines()

        split_idx = [i for i, _ in enumerate(data) if _.startswith('Current')][0]
        df_info = pd.DataFrame(data[:split_idx])[0].str.split('=', expand=True).replace('\n', '', regex=True)
        df = pd.DataFrame(data[split_idx:])[0].str.split('\t', expand=True).replace('\n', '', regex=True)
        f.close()

        df_info.columns = ['item', 'value']
        df.columns = df.iloc[0].values
        df = df.iloc[1:].astype('float').reset_index(drop=True)
        df_info.item = df_info.item.str.replace(' ', '')
        df_info.value = df_info.value.str.replace(' ', '')
        df.columns = ['I', 'P', 'V', 'PCE', 'P_corrected', 'PIB', 'WL']

        df.P = df.P * c
        df.PCE = df.PCE * c
        df.P_corrected = df.P_corrected * c

        return df, df_info

    def cal_op(self, y):
        x = self.df['P_corrected']
        if x.max() < self.Pop:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[-3:], y[-3:])
            op = slope * self.Pop + intercept
        else:
            op = np.interp(self.Pop, x, y)
        return op


class MultiTest:
    def __init__(self, path, c=1.0):
        self.path = path
        self.files = self.file_lst()
        self.df1, self.df2 = self.summary(c)

    def file_lst(self):
        files = os.listdir(self.path)
        files = [_ for _ in files if _.endswith('.txt')]
        return files

    def summary(self, c):
        path = self.path
        files = self.files

        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        idx = 0
        for i, file in enumerate(files):
            data = TestData(path, file, c)

            # I vs wavelength
            x = np.array(data.df['I']).reshape(-1)
            y = np.array(data.df['WL']).reshape(-1)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            y1 = slope * x + intercept
            # fig, ax = plt.subplots()
            # ax.scatter(x, y, s=10)
            # ax.grid()
            # ax.set_xlabel('Ramp Current (mA)')
            # ax.set_ylabel('Wavelength (nm)')
            # # ax.plot(x, y1, linestyle='-')
            # # ax.set_title('%s at %sC; slope=%.2f' % (data.SN, data.Temp, slope))
            # fig.savefig(path + 'WL %s at %.0fC.png' % (data.SN, data.Temp))
            # plt.close('all')

            df1 = df1.append(pd.DataFrame({
                'SN': data.SN,
                'Temp(C)': data.Temp,
                'Thermal_pile_temp(C)': data.TP_temp,
                'Thermal_pile_power(W)': data.TP_power,
                'Power_in_NA(%)': data.P_NA,
                'I_op(A)': data.Iop,
                'V_op(V)': data.Vop,
                'Wavelength_op(nm)': data.WLop,
                'PCE_op(%)': data.PCEop,
                'PIB_op(%)': data.PIBop,
                'P(non NA corrected)@25A': data.df.loc[data.df['I'] == 25, 'P'].values[0],
                'P@25A': data.df.loc[data.df['I'] == 25, 'P_corrected'].values[0],
                'V@25A': data.df.loc[data.df['I'] == 25, 'V'].values[0],
                'PCE@25A': data.df.loc[data.df['I'] == 25, 'PCE'].values[0],
                'PIB@25A': data.df.loc[data.df['I'] == 25, 'PIB'].values[0],
                'WL@25A': data.df.loc[data.df['I'] == 25, 'WL'].values[0],
                'WL slope': slope,
                'WL r_value': r_value,
                'WL std_err': std_err,
                'FWHM': data.FWHM,
                'Date': data.Date
            }, index=[i]), ignore_index=True)

            for n in range(len(data.df)):
                df2 = df2.append(pd.DataFrame({
                    'SN': data.SN,
                    'Temp(C)': data.Temp,
                    'I(A)': data.df.loc[n, 'I'],
                    'P(W)': data.df.loc[n, 'P_corrected'],
                    'V(V)': data.df.loc[n, 'V'],
                    'PCE(%)': data.df.loc[n, 'PCE'],
                    'PIB(%)': data.df.loc[n, 'PIB'],
                    'WL(nm)': data.df.loc[n, 'WL']
                }, index=[idx]), ignore_index=True)
                idx += 1

        df1.to_csv(path + 'summary1.csv', index=False, header=True)
        df2.to_csv(path + 'summary2.csv', index=False, header=True)

        return df1, df2

    def summary3(self, temp):
        df1 = self.df1
        df1 = df1[df1['Temp(C)'] == temp]
        x = [df1['P@25A'], df1['V@25A'], df1['PCE@25A'], df1['Power_in_NA(%)'], df1['Thermal_pile_temp(C)']]

        df3 = pd.DataFrame()
        for j, x in enumerate(x):
            df3 = df3.append(pd.DataFrame({
                'mean': np.mean(x),
                'std': np.std(x),
                'min': np.min(x),
                'max': np.max(x)
            }, index=[j]), ignore_index=True)
        df3.index = ['P@25A', 'V@25A', 'PCE@25A', 'P_in_NA@25A', 'Temp@25A']
        df3['Target Min'] = [314, 26.4, 0.442, 0.82, np.nan]
        df3['Target Mean'] = [335, 27.4, 0.489, 0.9, 35]
        df3['Target Max'] = [np.nan, 28.4, np.nan, np.nan, np.nan]
        df3['Y with min'] = [sum([_ > 314 for _ in df1['P@25A']]) / len(df1['P@25A']),
                             sum([_ < 28.4 for _ in df1['V@25A']]) / len(df1['V@25A']),
                             sum([_ > 44.2 for _ in df1['PCE@25A']]) / len(df1['PCE@25A']),
                             sum([_ > 82 for _ in df1['Power_in_NA(%)']]) / len(df1['Power_in_NA(%)']),
                             np.nan]
        df3['Cpk'] = [Cpk(df1['P@25A'], 314, np.nan), Cpk(df1['V@25A'], np.nan, 28.4),
                      Cpk(df1['PCE@25A'], 44.2, np.nan), Cpk(df1['Power_in_NA(%)'], 82, np.nan), np.nan]

        df3 = df3.round(3)
        df3.to_csv(path + 'summary3.csv', index=True, header=True)

    def plot_lines(self, x, y, temp):
        SNs = self.df2['SN'].unique()

        fig, ax = plt.subplots()
        for SN in SNs:
            df = self.df2[self.df2['Temp(C)'] == temp]
            df = df[df['SN'] == SN]
            if len(df) == 0:
                continue
            ax.plot(df[x], df[y], '.-', label=SN)

        if y == 'P(W)':
            ax.axhline(y=314, linewidth=1, ls='--', color='r')
        elif y == 'V(V)':
            ax.axhline(y=28.4, linewidth=1, ls='--', color='r')
            ax.axhline(y=26.4, linewidth=1, ls='--', color='r')
        elif y == 'PCE(%)':
            ax.axhline(y=44.2, linewidth=1, ls='--', color='r')

        ax.legend(fontsize='small', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        title = y + ' at %.0fC' % temp
        ax.set_title(title)
        ax.grid()
        fig.tight_layout()
        fig.savefig(path + title + '.png')
        plt.close('all')

    def plot_prob_25A(self, temp, string):
        df = self.df1
        fig, ax = plt.subplots()
        df = df[df['Temp(C)'] == temp]

        fig = probscale.probplot(df[string].values, ax=ax, probax='y',
                                 bestfit=True, estimate_ci=True, line_kws={'label': 'Fitting Line', 'color': 'b'},
                                 scatter_kws={'label': 'Observations'}, problabel='Probability (%)')

        ax.legend(loc='lower right')
        ax.grid(ls='--', c='lightgrey')
        ax.set_title('I = 25A, T = %.0fC; mean = %.1f, std = %.1f'
                     % (temp, df[string].mean(), df[string].std()))
        ax.set_xlabel(string)
        fig.tight_layout()
        sns.despine(fig)
        fig.savefig(self.path + 'Probability - %s @ I = 25A; T = %.0fC.png' % (string, temp))
        plt.close('all')


matplotlib.use('Agg')
path = 'C:/Users/cha75794/Box/derek.chang/MR & EL/MR/MR lot12/'
# path = '//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Derek/MR & EL/lot 10 (final)/'
# file = 'MantarayMR-A16-147-10-07132020-035930_07132020_035939.txt'

files = os.listdir(path)
files = [_ for _ in files if _.endswith('.txt')]

A = MultiTest(path, 1)
A.summary3(35)

lst1 = ['P(W)', 'V(V)', 'PCE(%)']
lst2 = [35]
for j in lst2:
    for i in lst1:
        A.plot_lines('I(A)', i, j)
        A.plot_prob_25A(j, i[:-3] + '@25A')
    A.plot_prob_25A(j, 'Power_in_NA(%)')
    A.plot_prob_25A(j, 'I_op(A)')
