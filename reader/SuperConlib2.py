"""
Refactering the program created by Dr. Matsunami
SuperConB -> read and analysis of Dr. Banno's measurement files
SuperConM -> read and analysis of Dr. Matsumoto's measurement files (inherited from SuperConB)

"""


import datetime
from pathlib import Path
import re

import pandas as pd

# import scipy as sp
# from scipy.optimize import leastsq
import numpy as np
from natsort import natsorted

import matplotlib.pyplot as plt


import warnings
warnings.simplefilter('ignore')
# warnings.resetwarnings()
# warnings.simplefilter('ignore', FutureWarning)

# %load_ext autoreload
# %autoreload 2

def now_datetime(type=1):
    """
    日時を文字列で返す
    type1:通常表示 "%Y-%m-%d %H:%M:%S"
    type2:"%Y%m%d%H%M%S"
    type3:"%Y%m%d_%H%M%S"
    type4:Base_file_nameで使用する形式 "%Y%m%d%H%M"
    elae:日付のみ "%Y%m%d"
    :return:
    """
    now = datetime.datetime.now()
    if type == 1:
        now_string = now.strftime("%Y-%m-%d %H:%M:%S")
    elif type == 2:
        now_string = now.strftime("%Y%m%d%H%M%S")
    elif type == 3:
        now_string = now.strftime("%Y%m%d_%H%M%S")
    elif type == 4:
        now_string = now.strftime("%Y%m%d%H%M")
    elif type == 5:
        now_string = now.strftime("%m%d_%H:%M:%S")
    elif type == 6:
        now_string = now.strftime("%Y%m%d")    
    else:
        now_string = now

    return  now_string

def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    
    return idx

plt.rcParams["font.size"] = 14

   
    
class SuperConB():
    """
    Example:
        tmp_ins = scl.SuperConB(fi)
        tmp_ins.read_file() 
        tmp_ins.estimate_n_j0()
        tmp_ins.make_plots()
        n = tmp_ins.meta["slope"]
        j0 = tmp_ins.meta["j0"]
    """
    
    def __init__(self,file):
        self.file = Path(file)
        
    def read_file(self):
        
        try:
            file_name = self.file.stem
            #ファイル名から磁場強度の読み出し
            pattern = "\d+T"
            
            field = re.search(pattern,file_name).group(0)
            self.magnetic_field = int(field.split('T')[0])
            
        except:
            self.magnetic_field = 0
        
        #端子間距離は1cmと固定
        VV_length = 1

        ave = 0
        
        #数値部の取り出し
        self.df = pd.read_csv(self.file, sep='\t',names =['current','voltage'], header=1)
        
        #データ削除（熱起電力カット）
        self.df = self.df[:-5]
        self.df = self.df.query('current > 0').reset_index(inplace=False, drop=True)
        self.df = self.df.query('index > 20')
        
        #ベースライン・オフセット補正
        ave = self.df.query('20< index < 30').mean()   
        self.df['voltage'] = self.df['voltage']-ave['voltage']
        
        #電界強度の追加    
        self.df['Electric_field_strength'] = self.df['voltage']/float(VV_length)

        # return self.df
   
    
    def estimate_n_j0(self):
        #Estimate N
        # 電圧領域として1～10V/cm範囲での電流電圧特性からn値を算出
        index_min = getNearestValue(self.df['voltage'], 1)
        index_max = getNearestValue(self.df['voltage'], 10)

        Ic = np.array(self.df['current'][index_min:index_max]).reshape(-1)
        Vc = np.array(self.df['voltage'][index_min:index_max]).reshape(-1)
        
        
        # print(Ic,Vc)
        x = np.log(Ic)
        y = np.log(Vc)
        
        # もしマイナスの値があった時にはnanが入ってしまうのでそれを避ける必要がある。
        #　np.isfinite-> Test element-wise for finiteness (not infinity and not Not a Number).
        idx = np.isfinite(x) & np.isfinite(y)

        try:
            # 直線Fitting
            # fit_y = a * x + b
            a, b = np.polyfit(x[idx], y[idx], 1)
        except:
            print('Fitting error')
            print(f'Ic:{Ic}\nVc:{Vc}') 
            print(f'slope= None, slice = None')
            a, b = np.nan, np.nan
        

        # Estimate J0
        # 臨界電流値の取得
        index = getNearestValue(self.df['Electric_field_strength'], 1)
        j0 = self.df['current'][index]
        jc = (1-(1/a))*j0
        
        self.analysis_region ={'Ic':Ic,'Vc':Vc}
        self.meta = {'slope':a,'slice':b,'j0':j0,'jc':jc,'magnetic_filed':self.magnetic_field}   
        
    def make_plots(self,title='$J_{0}$, n plot', save_path=None, save=False):
       
        nrows, ncols = 1, 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                               figsize=(ncols*6,nrows*5), 
                               squeeze=False, tight_layout=True)
        n = self.meta['slope']
        j0 = self.meta['j0']
        jc = j0*(1-(1/n))
        
        X = self.df['current'].values
        Y = self.df['voltage'].values
        select_X = self.analysis_region['Ic']
        select_Y = self.analysis_region['Vc'] 
        ax[0,0].plot(X,Y,'ro',label='Data')
        ax[0,0].plot(select_X,select_Y,'b-',label='n Fitting Region')
       
        ax[0,0].set_xlabel('Current [A]')
        ax[0,0].set_ylabel('Voltage [$\mu$V/cm]')
        # ax[0,0].tick_params(direction = "inout")
        # ax[0,0].set_title("$J_{0}$")
        # ax[0,0].grid(which = "major", axis = "both",
        #              color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
        
        ax[0,0].axhline(y=1,xmin=0.5, xmax=1, color='black',linestyle = "--")
        ax[0,0].axvline(self.meta['j0'],color='blue',)
        ax[0,0].text(self.meta['j0']+0.1,0, "$J_{0}$")
        ax[0,0].grid()
        ax[0,0].axvline(jc,0,0.5,color='black',)
        ax[0,0].text(jc+0.1,0, "$J_{c}$")
        
        
        # ax[0,0].axhline(y=np.min(select_Y), xmin=np.median(X), xmax=np.max(X))
        # ax[0,0].axhline(y=np.max(select_Y), xmin=np.median(X), xmax=np.max(X),color='g')
        # ax[0,0].text(np.median(X),np.max(select_Y)*0.9, 'n estimate range')
        ax[0,0].legend(title=f"$J_0$ = {self.meta['j0']:.2f}\n$J_c=(1-1/n) \cdot J_0$ ={jc:.2f}\nFiled = {self.meta['magnetic_filed']}T")
        
        ax[0,1].plot(np.log(X),np.log(Y),'ro',label='Data')
        # ax[0,1].plot(np.log(select_X),np.log(select_Y),'ro',label='Data')
        
        if self.meta['slope'] == np.nan:
            fit_line = 0
        else:   
            fit_line = self.meta['slope']*np.log(select_X) + self.meta['slice']
        
        ax[0,1].plot(np.log(select_X), fit_line,'b-',label='n Fitting Region')
        ax[0,1].legend(title=f"n = {self.meta['slope']:.2f}\nFiled = {self.meta['magnetic_filed']}T")
        ax[0,1].set_xlabel('ln(Current) [A]')
        ax[0,1].set_ylabel('ln(Voltage) [$\mu$V/cm]')
        # ax[0,1].tick_params(direction = "inout", which = "both")
        # ax[0,1].set_title(f"n")
        
        ax[0,1].axhline(y=np.log(1),xmin=0.5, xmax=1, color='black',linestyle = "--",)
        ax[0,1].axhline(y=np.log(np.max(Y)),xmin=0.5, xmax=1, color='black',linestyle = "--",)
        ax[0,1].grid()
        # ax[0,1].grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
        # ax[0,1].grid(which = "major", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
        # ax[0,1].set_xlim(np.log(X[1]),np.log(X[-1]))
        # ax[0,1].set_xlim(np.log(Y[1]),np.log(Y[-1]))
     
     
        fig.suptitle(f'{self.file.stem}')
        # fig.suptitle(f'{self.file.stem}\n{title}')
        plt.tight_layout()
        
        if save:
            if save_path is None:
                s_name = f'{self.file.stem}.png'
            else:
                s_path =Path(save_path)
                s_name = str(s_path/f'{self.file.stem}.png')
                # print(s_name)
            plt.savefig(s_name)
            plt.close()
            
        else:
            plt.show()
            


class SuperConM(SuperConB):
    
    def read_file(self):
        
        # ファイルの読み込みとヘッダー部の読み出し
        df_header = pd.read_csv(self.file, header=None,nrows=5, delimiter=';') 
        
        #　端子間距離の取得
        #sample = df_header[1][0]
        VV_length = float(df_header[1][1])
        self.magnetic_field = round(float(df_header[1][2]))
        #temperature = float(df_header[1][3])
        #angle = df_header[1][4]

        #　数値部の取り出し
        self.df = pd.read_csv(self.file,
                    header= 6, 
                    delim_whitespace=True, 
                    names=('time', 'current', 'voltage', 'Temp.1', 'Temp.2', 'Temp.3'))
        
        #　掃引速度の計算
        self.sweep_time = self.df.iloc[-1][1]/self.df.iloc[-1][0]
        
        #　電界強度の追加    
        self.df['Electric_field_strength'] = self.df['voltage']/float(VV_length)
        
        # Currentがマイナスの領域のIndexを見つける。マイナス領域削除
        rem_max_ind = np.where(self.df['current']< 0)[0]
        # self.df.drop(self.df.index[rem_max_ind],inplace=True)
        self.df = self.df.drop(self.df.index[rem_max_ind])
        
        # リーク電流の削除　（V値の差分で10以上）
        for i in range(len(self.df)):
            try:
                j = self.df['voltage'][i+1]-self.df['voltage'][i]
        
                if j > 10:
                    print (i,j)
                    self.df = self.df.drop(index= i+1)
            except:
                break
    
        
def estimate_n_j0(xi,yv):
    # VI測定からISOに従ったn値と臨界電流値を求める
    #Estimate N
    # 電圧領域として1～10V/cm範囲での電流電圧特性からn値を算出
    index_min = getNearestValue(yv, 1)
    index_max = getNearestValue(yv, 10)

    Ic = np.array(xi[index_min:index_max]).reshape(-1)
    Vc = np.array(yv[index_min:index_max]).reshape(-1)
    
    
    # print(Ic,Vc)
    x = np.log(Ic)
    y = np.log(Vc)
    
    # もしマイナスの値があった時にはnanが入ってしまうのでそれを避ける必要がある。
    #　np.isfinite-> Test element-wise for finiteness (not infinity and not Not a Number).
    idx = np.isfinite(x) & np.isfinite(y)

    try:
        # 直線Fitting
        # fit_y = a * x + b
        a, b = np.polyfit(x[idx], y[idx], 1)
    except:
        print('Fitting error')
        print(f'Ic:{Ic}\nVc:{Vc}') 
        print(f'slope= None, slice = None')
        a, b = np.nan, np.nan
    

    # Estimate J0
    # 臨界電流値の取得
    index = getNearestValue(yv, 1)
    j0 = xi[index]
    
    analysis_region ={'Ic':Ic,'Vc':Vc}
    meta = {'slope':a,'slice':b,'j0':j0} 
    
    return meta, analysis_region