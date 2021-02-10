import os
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class singleFile:
    def __init__(self, path, file):
        self.path = path
        self.file = file
        self.df_info, self.df_coarse, self.df_peak, self.df_valley = self.read_file()
        self.ID = self.get_info('DUTID')
        self.date = self.get_info('Date')
        self.time = self.date_time()
        self.wafer = self.get_info('WaferID')
        self.assembly = self.get_info('AssemblyLot')
        self.PER_PERC = float(self.get_info('PER_PERC'))
        self.PBC_PER = float(self.get_info('PBC_PER', i=-1))

        self.Pmax_c = float(self.get_info('MaxPower'))
        self.Pmin_c = float(self.get_info('MinPower'))
        self.th_p_c = float(self.get_info('PeakAngle'))
        self.th_v_c = float(self.get_info('ValleyAngle'))
        self.PER_PERC_c = self.Pmax_c / (self.Pmax_c + self.Pmin_c) * 100
        self.PBC_PER_c = self.PER_PERC_c * np.cos(np.deg2rad(self.th_v_c - 90)) ** 2

        self.Pmax_f = float(self.get_info('MaxPower', i=-1))
        self.Pmin_f = float(self.get_info('MinPower', i=-1))
        self.th_p_f = float(self.get_info('PeakAngle', i=-1))
        self.th_v_f = float(self.get_info('ValleyAngle', i=-1))
        self.PER_PERC_f = self.Pmax_f / (self.Pmax_f + self.Pmin_f) * 100
        self.PBC_PER_f = self.PER_PERC_f * np.cos(np.deg2rad(self.th_v_f - 90)) ** 2

    def read_file(self):
        path = self.path
        file = self.file

        with open(path + file) as f:
            data = f.readlines()

        idx0 = [i for i, _ in enumerate(data) if _.startswith('Coarse Scan Raw Data')][0]
        df_info = pd.DataFrame(data[:idx0]).replace(' ', '', regex=True)
        df_info = df_info.replace('%', '', regex=True)
        df_info = df_info.replace('\(deg\)', '', regex=True)
        df_info = df_info.replace('\(mW\)', '', regex=True)
        df_info = df_info.replace('Idrive', '', regex=True)
        df_info = df_info[0].str.split(':|,|=', expand=True).replace('\n', '', regex=True)
        df = pd.DataFrame(data[idx0:]).replace(' ', '', regex=True)
        df = df[0].str.split(',', expand=True).replace('\n', '', regex=True)

        lst_s = df[df[0] == 'Angle(deg)'].index.values + 1
        lst_e = np.append(lst_s[1:] - 3, -1)
        dfs = [df.iloc[lst_s[n]:lst_e[n]] for n in range(len(lst_e))]
        for df in dfs:
            df.columns = ['Angle(deg)', 'Power(mW)']
        df_coarse = dfs[0].astype(float)
        df_peak = dfs[1].astype(float)
        df_valley = dfs[2].astype(float)

        f.close()
        return df_info, df_coarse, df_peak, df_valley

    def get_info(self, string, i=0, j=1):
        """
        :param string: 要找的目標 ex: 'WaferID'
        :param i: 想要取出找到的第幾個目標?
        :param j: 想要目標後的第幾格
        """
        df = self.df_info
        mask = np.column_stack([df[col].str.contains(string, na=False) for col in df])
        value = df.iloc[np.where(mask == True)[0][i], np.where(mask == True)[1][i] + j]
        return value

    def date_time(self):
        string = self.file.split('_')[1]
        date_time_obj = datetime.strptime(string, '%m%d%Y%H%M%S')
        return date_time_obj

    def plot_power(self, condition=True):
        if condition:
            df_coarse = self.df_coarse
            df_peak = self.df_peak
            df_valley = self.df_valley

            fig, ax = plt.subplots()
            ax.plot(df_coarse['Angle(deg)'], df_coarse['Power(mW)'], color='blue', marker='o')
            ax.plot(df_peak['Angle(deg)'], df_peak['Power(mW)'], color='orange')
            ax.plot(df_valley['Angle(deg)'], df_valley['Power(mW)'], color='orange')
            ax.set_xlabel('Angle (deg)')
            ax.set_ylabel('Power (mW)')
            title = self.ID + ' ' + self.time.strftime('%m/%d/%Y %H:%M') + ' PBC_PER(fine) = %.2f%%' % self.PBC_PER_f
            ax.set_title(title)
            ax.grid()
            title = self.ID + ' ' + self.time.strftime('%m%d%Y%H%M%S')
            fig.savefig(self.path + title + '.png')
            plt.close('all')


matplotlib.use('Agg')
# path = '//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Mantaray/COS PER/second correlation/'
# path = '//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Mantaray/COS PER/Maruwa vs Kyocera 091620/'
path = '//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Mantaray/COS PER/121520/'
# path = 'C:/Users/cha75794/Desktop/111620/'
files = os.listdir(path)
files = [_ for _ in files if '.' not in _]
A = singleFile(path, files[0])

df = pd.DataFrame()
for i, file in enumerate(files):
    data = singleFile(path, file)
    data.plot_power(not data.PBC_PER_f > 90)

    df = df.append(pd.DataFrame({
        'ID': data.ID,
        'Time': data.time,
        'test times': 0,
        'Wafer_ID': data.wafer,
        'Assembly_Lot': data.assembly,
        'PER_PERC': data.PER_PERC,
        'PER_PERC(fine)': data.PER_PERC_f,
        'PER_PERC(coarse)': data.PER_PERC_c,
        'PBC_PER': data.PBC_PER,
        'PBC_PER(fine)': data.PBC_PER_f,
        'PBC_PER(coarse)': data.PBC_PER_c,
        'Pmax(coarse)': data.Pmax_c,
        'Pmin(coarse)': data.Pmin_c,
        'PeakAngle(coarse)': data.th_p_c,
        'ValleyAngle(coarse)': data.th_v_c,
        'Pmax(fine)': data.Pmax_f,
        'Pmin(fine)': data.Pmin_f,
        'PeakAngle(fine)': data.th_p_f,
        'ValleyAngle(fine)': data.th_v_f
    }, index=[i]), ignore_index=True)

df = df.sort_values(by=['Time'])
df['test times'] = df.groupby('ID')['ID'].transform('count')
df = df.sort_values(by=['ID', 'Time'])
df.to_csv(path + 'summary.csv', index=False, header=True)
