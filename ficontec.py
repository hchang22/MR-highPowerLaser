import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def find_bw(x, y, h_percent):
    peak_y = max(y)
    h = peak_y * h_percent
    zero_crossings = np.where(np.diff(np.sign(y - h)))[0]
    max_x = x[zero_crossings[-1]]
    min_x = x[zero_crossings[0]]
    return max_x - min_x


def calc_PIB(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    max_val = max(y)
    max_idx = np.where(y == max_val)[0][0]

    # make the noise to zero
    zc = np.where(np.diff(np.sign(y)))[0]
    zc_l = zc[zc < max_idx][-1]
    zc_r = zc[zc > max_idx][0]
    y[:zc_l + 1] = 0
    y[zc_r:] = 0

    # find wavelength at peak wavelength +-1nm
    x_r, _ = closest(x, x[max_idx] + 1)
    x_l, _ = closest(x, x[max_idx] - 1)

    # calculate PIB
    PIB_ratio = sum(y[x_l:x_r]) / sum(y)
    PIB_value = sum(y[x_l:x_r])

    return PIB_ratio, PIB_value


def closest(lst, k):
    lst = np.asarray(lst)
    idx = (np.abs(lst - k)).argmin()
    return idx, lst[idx]


class ficontecData:
    def __init__(self, path, file):
        self.path = path
        self.file = file
        self.df = self.get_frame()

    def get_frame(self):
        path = self.path
        file = self.file
        path_ficontec = 'C:/Program Files/ficontec/Process-Master/DataLog/'
        df = pd.read_excel(path + file)
        df = df.dropna(how='all', axis=1)

        # replace redundant paths
        cols = df.select_dtypes(include='object').columns
        df = df.apply(lambda x: x.str.replace('\\', '/') if x.name in cols else x)
        df = df.apply(lambda x: x.str.replace(path_ficontec, path) if x.name in cols else x)
        return df

    @staticmethod
    def get_scanning(f_path):
        with open(f_path) as f:
            data = f.readlines()
        df_data = pd.DataFrame(data)
        df_data = df_data[0].str.split("\t", expand=True)
        cols = df_data.columns
        df_data[cols] = df_data[cols].apply(pd.to_numeric, errors='coerce')  # apply to_numeric to all of the columns
        df_data = df_data.dropna(axis=0, thresh=2)  # Keep only the rows with at least 2 non-NA values
        df_data = df_data.dropna(axis=1)
        df_data.columns = ['x', 'measurement', 'fitting']
        return df_data

    def plot_spectrum(self, file_path_wet, file_path_cur, module, channel):
        matplotlib.use('Agg')
        df_wet = self.get_scanning(file_path_wet)
        df_cur = self.get_scanning(file_path_cur)
        max_idx = df_wet['measurement'].idxmax()

        # find FWHM and FWPercentageM
        bw10_wet_mea = find_bw(df_wet.x, df_wet.measurement, 0.1)
        bw50_wet_mea = find_bw(df_wet.x, df_wet.measurement, 0.5)
        bw10_cur_mea = find_bw(df_cur.x, df_cur.measurement, 0.1)
        bw50_cur_mea = find_bw(df_cur.x, df_cur.measurement, 0.5)

        fig, ax = plt.subplots()
        ax.plot(df_wet.x, df_wet.measurement, color='g',
                label='wet (10%%: %.2f, 50%%: %.2fnm)' % (bw10_wet_mea, bw50_wet_mea))
        ax.plot(df_wet.x, df_wet.fitting, ls='--', color='lightgreen', label='')
        ax.plot(df_cur.x, df_cur.measurement, color='b',
                label='cure (10%%: %.2f, 50%%: %.2fnm)' % (bw10_cur_mea, bw50_cur_mea))
        ax.plot(df_cur.x, df_cur.fitting, ls='--', color='lightblue', label='')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim([df_wet.x[max_idx] + 1, df_wet.x[max_idx] - 1])
        ax.set_title(module + ' CH%s' % channel)
        ax.legend(loc='lower right')
        ax.grid()
        # fig.tight_layout()

        save_path = file_path_wet.replace(file_path_wet[file_path_wet.rfind('wet_HR 4000'):], '')
        fig.savefig(save_path + module + ' CH%s spectrum.png' % channel)
        plt.close('all')

    def plot_dry_Y(self, file_path, module, channel, final_pos):
        matplotlib.use('Agg')
        df = self.get_scanning(file_path)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, 900, len(df)), df.measurement, label='measurement')
        ax.plot(np.linspace(0, 900, len(df)), df.fitting, label='fitting')
        ax.axvline(x=final_pos, linewidth=1, ls='--', color='r')
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Amplitude')
        ax.set_title(module + ' CH%s final_pos = %.0fum' % (channel, final_pos))
        ax.legend()
        ax.grid()
        # fig.tight_layout()

        save_path = file_path.replace(file_path[file_path.rfind('Profile Scan'):], '')
        fig.savefig(save_path + module + ' CH%s Dry_Y.png' % channel)
        plt.close('all')

    def plot_spectra(self):
        df = self.df
        df = df.dropna(subset=['Spectrum_after_wet_align', 'Spectrum_after_cure']).reset_index(drop=True)
        for i in range(len(df)):
            f_spec_wet = df.Spectrum_after_wet_align.values[i]
            f_spec_cur = df.Spectrum_after_cure.values[i]
            module = df['Modul_Serial_Number'][i]
            ch = df['Laser#'][i]
            self.plot_spectrum(f_spec_wet, f_spec_cur, module, ch)
            print('plotting spectrum: ' + module + ' CH%s' % ch)

    def plot_dry_Ys(self):
        df = self.df
        df = df.dropna(subset=['Scan_Dry_Y']).reset_index(drop=True)
        for i in range(len(df)):
            f_scan_dry_Y = df.Scan_Dry_Y[i]
            module = df['Modul_Serial_Number'][i]
            ch = df['Laser#'][i]
            final_pos = df['Final_Pos_Dry_Y'][i]
            self.plot_dry_Y(f_scan_dry_Y, module, ch, final_pos)

            print('plotting dry_Y: ' + module + ' CH%s' % ch)

    def calc_spectrum(self, file_path_wet, file_path_cur):
        df_wet = self.get_scanning(file_path_wet)
        df_cur = self.get_scanning(file_path_cur)

        # find FWHM and FWPercentageM
        bw10_wet_mea = find_bw(df_wet.x, df_wet.measurement, 0.1)
        bw50_wet_mea = find_bw(df_wet.x, df_wet.measurement, 0.5)
        bw10_cur_mea = find_bw(df_cur.x, df_cur.measurement, 0.1)
        bw50_cur_mea = find_bw(df_cur.x, df_cur.measurement, 0.5)

        # find PIB
        PIB_wet_ratio, PIB_wet_value = calc_PIB(df_wet.x, df_wet.measurement)
        PIB_cur_ratio, PIB_cur_value = calc_PIB(df_cur.x, df_cur.measurement)

        lst = [bw10_wet_mea, bw50_wet_mea, bw10_cur_mea, bw50_cur_mea,
               PIB_wet_ratio, PIB_cur_ratio, PIB_wet_value, PIB_cur_value, PIB_cur_value / PIB_wet_value]
        return lst

    def EL_add_cols(self):
        df = self.df
        lst = []
        for i in range(len(df)):
            f_spec_wet = df.Spectrum_after_wet_align.values[i]
            f_spec_cur = df.Spectrum_after_cure.values[i]
            lst_temp = self.calc_spectrum(f_spec_wet, f_spec_cur)
            lst.append(lst_temp)

        cols = ['bw10_wet', 'bw50_wet', 'bw10_cur', 'bw50_cur',
                'PIB_wet', 'PIB_cur', 'PIB_wet_value', 'PIB_cur_value', 'delta_PIB']
        array = np.array(lst).reshape(-1, len(cols))
        df_temp = pd.DataFrame(data=array, columns=cols)
        df = pd.concat([df, df_temp], axis=1)

        self.df = df.loc[:, ~df.columns.duplicated()]

    def MR_add_cols(self):
        df = self.df
        df['FAC_cur_power_ratio'] = df['Peak_Power_after_FAC_Curing'] / df['Peak_Power_after_FAC_Wet_Align']
        df['Mir_wet_power_ratio'] = df['Peak_Power_after_mirror_Wet_Align'] / df['Peak_Power_after_mirror_Dry_Align']
        df['Mir_cur_power_ratio'] = df['Peak_Power_after_mirror_Curing'] / df['Peak_Power_after_mirror_Wet_Align']
        df['Mir_power_ratio'] = df['Peak_Power_after_mirror_Curing'] / df['Peak_Power_after_mirror_Dry_Align']

        self.df = df.loc[:, ~df.columns.duplicated()]

    def module_group(self):
        df = self.df
        grouped = df.groupby(['Modul_Serial_Number']).agg(
            {'Peak_Power_after_mirror_Curing': ['mean', 'min', 'max', 'sum']})
        grouped = grouped.reset_index()
        return grouped

    def correlation(self):
        df = self.df
        path = self.path
        df = df.select_dtypes(include='number')

        C = df.corr(method='pearson')
        C.to_excel(path + 'Correlation Matrix.xlsx', index=True, header=True)

        matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns_plot = sns.heatmap(C, annot=True, fmt='.1f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax).get_figure()
        sns_plot.tight_layout()
        ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
        sns_plot.savefig(path + 'correlation matrix.png')


# path1 = 'C:/Users/cha75794/Desktop/MR & EL/MR/datalog 0623/'
# path1 = '//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Derek/MR & EL/FiconTec/'
# file1 = 'data LT2.xlsx'
#
# MR = ficontecData(path1, file1)
# MR.MR_add_cols()
# MR.plot_dry_Ys()
# df_group = MR.module_group()
# df_group.to_csv(path1 + file1.replace('.xlsx', ' grouped.csv'), index=False, header=True)

path2 = '//li.lumentuminc.net/data/MIL/SJMFGING/Optics/Planar/Derek/MR & EL/FiconTec/'
file2 = 'data EL2 02032021.xlsx'
EL = ficontecData(path2, file2)
EL.EL_add_cols()
EL.plot_spectra()
EL.plot_dry_Ys()
EL.df.to_csv(path2 + file2 + ' summary.csv', index=False, header=True)

# df = pd.read_excel(path2 + 'correlation.xlsx')
# C = df.corr(method='pearson')
# C.to_csv(path2 + 'correlation.csv')
