"""
Inv-Power-Plot, Shift-log-log-plot
analysis module

"""				
import math
import time
from datetime import datetime
import re
from pathlib import Path

import scipy as sp
import scipy.optimize as optimize
from scipy.optimize import curve_fit  

from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter('ignore')
# warnings.resetwarnings()
# warnings.simplefilter('ignore', FutureWarning)

from pfit import relu_abs_fit as raf

   
class PfAnalysis():
    # べき乗の解析Class
    
    def __init__(self,xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    
    def estimate(self,power_num=2,info=False,ini_para=None, retry_r2=0.9, min_error='mae',zero_replace=False):
        # 1/n Power
        self.res_pof= const_inv_power_fit(self.xdata,
                                        self.ydata,
                                        power_num,
                                        ini_params=ini_para, 
                                        retry_r2=retry_r2,
                                        min_error= min_error,
                                        zero_replace=zero_replace,
                                        plot=info, 
                                        info=info)
        # self.res_prof(dict)のkeys
        # 'rex':re_xdata,
        # 'rey':re_ydata, 
        # 'fit':re_fit, 
        # 'popt':res_params,tuple -> Nor, Ip, bg
        # 'r2':r2,
        # 'adj_r2':自由度調整済みR2
        # 'r2_bp':Bp以降のR2,
        # 'er':error_range,
        # 'cv':変動係数

        
    def shift_estimate_by_power_scan(self, search_range=None, 
                                     zero_replace=False,
                                     min_error='mae', 
                                     likely_evaluation='r2_bp',
                                     info=False, plot=True,
                                     plot_fig_step=1):
        # 1/n Power
        self.shebypw = xpower_search(self.xdata, self.ydata, 
                                        search_range=search_range,
                                        r2_plot=plot, process_plot=plot,
                                        min_error=min_error,
                                        zero_replace=zero_replace,
                                        likely_evaluation =likely_evaluation,
                                        info=info, 
                                        every_plot=False, 
                                        plot_save=False,
                                        plot_fig_step=plot_fig_step)
        
        # shebypw:keys
        # ['r2','shift', 'power', 'rex', 'rey', 'fit','popt']

    def power_estimate_by_shift_scan(self, search_range, 
                                       fit_type='weight', 
                                       bg_num=3,
                                       lim_val=0.5, 
                                       min_error='mae',
                                       info=True, plot=True,
                                       plot_fig_step=1):
        # shift-log-log
        self.pwebysh = xshift_search(self.xdata, self.ydata, 
                                        search_range,
                                        bg_num=bg_num, 
                                        lim_val=lim_val, 
                                        fit_type=fit_type,
                                        r2_plot=plot, 
                                        process_plot=plot,
                                        min_error=min_error,
                                        info=info,
                                        every_plot=False, 
                                        plot_save=False,
                                        plot_fig_step=plot_fig_step)
        
        # pwebysh:keys
        # ['r2','shift', 'power', 'rex','rey','fit']
 
        
# ------

def m_line(x, a, b):
    """liner function

    Args:
        x (array_like): x data
        a (float): slope
        b (float): y-slice

    Returns:
        array_like: f
    """
    f = a * x  + b
    return f

def mae_line(param,x,y):
    # resouidual of Line function
    fit = m_line(x,*param)
    mae = mean_absolute_error(y,fit)
    return  mae

def y_slice(Ip, Nor, Bg):
    # for ReLu
    b = Bg - Nor*Ip
    return b

def get_nearest_value(lst: list, num: float):
    """
    Returns the element from the input list that is closest to the given number.
    
    Args:
        lst: A list of numbers.
        num: A float number.
    
    Returns:
        The index of the element in the list that is closest to the given number.
    """
    # Calculate the difference between each element in the list and the target number,
    # and get the index of the minimum difference
    idx = np.abs(np.asarray(lst) - num).argmin()
    
    return idx

def nan_inf_rm(xdata,ydata,zero_replace=False,info=False):
    """remove nan and inf values
        If you take Log
            Negative value: Nan for not defined
            0: Negative infinity because it is not defined
            If these values are included, remove them (Index) to avoid an error

    Args:
        xdata (ndarray or list): xdata
        ydata (ndarrayor list): ydata
        zero_replace(bool): True: Nan -> 0 replace  False:Nan remove. default False
        info (bool): show infomation
        
    Returns:
        trimmed data (ndarray): re_xdata, re_ydata
    """
    # inf -> nan
    # nan_inf_rm(np.log(xf),np.log(yf))
    xarray = np.array(xdata)
    yarray = np.array(ydata)
    tr_inf_x = np.where(np.isinf(xarray),np.nan,xarray)
    tr_inf_y = np.where(np.isinf(yarray),np.nan,yarray)
    
    re_xdata = tr_inf_x.copy()
    re_ydata = tr_inf_y.copy()
    
    if zero_replace:
        re_xdata = np.nan_to_num(re_xdata, copy=False)
        re_ydata = np.nan_to_num(re_ydata, copy=False)
        
    else:
        nan_ind_x = np.where(~np.isnan(tr_inf_x))
        re_xdata = re_xdata[nan_ind_x[0]]
        re_ydata = re_ydata[nan_ind_x[0]]
        
        nan_ind_y = np.where(~np.isnan(re_ydata))
        re_xdata = re_xdata[nan_ind_y[0]]
        re_ydata = re_ydata[nan_ind_y[0]]
    
    if info:
        print(f'number of Nan of (x, y) : ({np.count_nonzero(np.isnan(tr_inf_x))}, {np.count_nonzero(np.isnan(tr_inf_y))})')
        print(f'Original shape x, y: {xarray.shape}, {yarray.shape}')
        print(f'output shape x, y: {re_xdata.shape}, {re_ydata.shape}')
        
    return re_xdata, re_ydata

def adj_r2_score(y_true, y_pred):
    # 決定係数と自由度調整済み決定係数、誤差範囲を返す
    # 誤差範囲 = √(1-R^2)*標準偏差
    # 決定係数と誤差範囲はどちらも回帰分析の評価指標（異なる側面からモデルの性能を評価）。
    # 決定係数はモデルの当てはまりを、誤差範囲はモデルの精度を評価。
    
    r2 = r2_score(y_true,y_pred)
    
    # 自由度調整済み決定係数
    # https://qiita.com/bianca26neve/items/4ddcf5ca12652b652f04
    adj_r2 = 1-(1-r2)*((len(y_true)-1)/(len(y_true)-2-1)) 
    
    # 誤差範囲
    # 目的変数の予測値の平方和を求める
    y_hat_square = y_pred ** 2
    # 目的変数の予測値の数を求める
    n = len(y_true)
    # 目的変数の予測値の標準偏差を求める
    std_y_hat = np.sqrt(np.sum(y_hat_square) / n)
    # # 決定係数を求める
    # R2 = np.sum(y_pred ** 2) / np.sum((y_true ** 2)
    
    # 誤差範囲を求める 誤差範囲が小さいほど、目的変数の真の値を正確に推定
    error_range = np.sqrt(1 - r2) * std_y_hat
        
    return r2, adj_r2, error_range

def static_inf(sdata,info=True):
    
    medi_ = np.median(sdata)
    mean_ = np.mean(sdata)
    std_= np.std(sdata)
    cv_ = std_/mean_
    skew_ = sp.stats.skew(sdata)
    kurtosis_ =  sp.stats.kurtosis(sdata, bias=False)
    asymetric_ = medi_/ mean_
    # 変動係数(CV): 標準偏差を平均値で割ったもの。 
    # median =< mean-std とする　(mean-std)/mean = 1 - cv >= median/mean
    # median/mean は変動係数を用いて表すこともできる
   
    info_dict = {'median':medi_, 'mean':mean_, 'std':std_, 'cv':cv_, 'sk':skew_,'kt':kurtosis_,'asymetric':asymetric_}
    
    if info:
        for k, v in info_dict.items():
            print(k, v)
        
    return info_dict

#---- plot

# 黄金比 1：1.62 wdth:height = 4.86:3
# 白銀比 1：1.43  wdth:height = 4.3:3 
# wdth:height= 6:5

# memo figsize=(width,height) wdth:height= 4:3 
# (row,col)->figsize(col*4,row*3): (3,4)->figsize(16,9)

# width_u = 5.2
# height_u = 4
# https://qiita.com/skotaro/items/5c9893d186ccd31f459d
# cmap = plt.get_cmap("tab10") # point
# for i in range(4):
#     ax.plot(t,i*(t+1),   color=cmap(i), linestyle = '-') #tuple

# cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # point
# for i in range(4):
#     ax.plot(t,i*(t+1), color=cycle[i], linestyle = '-')
#     ax.plot(t,i*(t+1)+.3,color=cycle[i], linestyle = ':')

# w_in, h_in = plt.rcParams[cParams["figure.figsize"]
# dpi =plt.rcParams["figure.dpi"]
# print("figsize (inch)", (w_in, h_in))
# figsize (inch) (6.0, 4.0)
# print("dpi (px/inch)", dpi)
# dpi (px/inch) 72.0
# print("save size (px)", (w_in * dpi, h_in * dpi))
# save size (px) (432.0, 288.0)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def re_replace(text):
    """Remove special symbols with regular expressions 

    Args:
        text (str): text
    Returns:
        str: Text with special symbols removed
    Examples:
        text = '4 inch $\phi$=0'
        re_replace(text)
        >>> '4_inch_phi0
    Ref:
        https://qiita.com/ganyariya/items/42fc0ed3dcebecb6b117 
    """
    # code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')

    code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~]')
    cleaned_text = code_regex.sub('', text).replace(' ', '_')
    # print(cleaned_text)

    return cleaned_text

def search_r2_plot(shifts,powers,r2,msg,save=False,ylabel=None):
    """shift vs r2, power vs r2 plot

    Args:
        shifts (array like): shift data
        powers (array like): power data
        r2 (array like): r2 data
        msg (str): graph title
        save (bool, optional): figure save. Defaults to False.
    """
    
    # fig = plt.figure(figsize=(12,5), tight_layout=True)
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    
    if ylabel is None:
        ylabel = '$R^2$'
        
    fig = plt.figure(figsize=(12,5), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    
    ax1.plot(shifts,r2,'ro-')
    ax1.set_xlabel('shift')
    ax1.set_ylabel(ylabel)
    ax1.grid(which='both')
    
    ax2.plot(powers,r2,'b^-')
    ax2.set_xlabel('n')
    ax2.set_ylabel(ylabel)
    ax2.grid(which='both')
    
    # ax3.plot(r2,shifts,'ro-',label="shift")
    # ax3.plot(r2,powers,'b^-',label="power")
    # ax3.set_xlabel('$r^2$')
    # ax3.set_ylabel('')
    # ax3.legend()
    # ax3.grid(True)
    
    fig.suptitle(msg)
    
    if save:
        filename = re_replace(msg)
        fig.savefig(f'{filename}.png', dpi=300)
        
    plt.show()
    
def inv_power_plot(xdata, ydata, shift, power):
    # power 2-> 1/2 plot
    #---
    inv_power = 1/power
    fig = plt.figure(figsize=(12,5), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    ax1.plot(xdata,ydata,'ro-',label='Data')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(which='both')
    
    ax2.plot(xdata,np.power(ydata,1/power),'b^-',label='Data')
    ax2.set_xlabel('x')
    ax2.set_ylabel('$y^{{{1/power}}}$')
    ax2.legend(title=f'Shift: {shift:.2f}\nn: {power:.2f}')
    ax2.grid(which='both')
        
    plt.show()

def plot_ax(xdata, ydata, m_xdata, m_ydata, breakpoints=None, lgtitle=None, title='', axi=None):
    
    if axi is None:
        fig_ = plt.figure()
        ax_ = fig_.add_subplot(111)
    else:
        ax_ = axi
                  
    ax_.set_title(f'{title}')    
    ax_.plot(xdata, ydata,'ro',label='Data')
    ax_.plot(m_xdata, m_ydata,'bo-',label='Processed')
    
    if breakpoints is not None:
        for i, bp in enumerate(breakpoints):
            ax_.axvline(bp)
            ax_.text(bp,np.max(ydata)*0.8, f'{bp:.2f}')
        
    ax_.grid(which='both')
    
    if lgtitle is None:
        ax_.legend() 
        
    else:   
        ax_.legend(title=lgtitle)
    
    if axi is None:
        plt.show()
        return fig_
    
    return ax_   

def plot3_pw_ax(xdata, ydata, m_xdata=None, m_ydata=None, n_xdata=None, n_ydata=None, breakpoints=None, title='', axi=None):
    
    if axi is None:
        fig_ = plt.figure()
        ax_ = fig_.add_subplot(111)
    else:
        ax_ = axi
                  
    ax_.set_title(title)    
    ax_.plot(xdata, ydata, color=cycle[0], marker="o",label='Data')
    
    if m_xdata is not None:
        ax_.plot(m_xdata, m_ydata, color=cycle[1], linestyle = '-', label='Fit')
    
    if n_xdata is not None:
        ax_.plot(n_xdata, n_ydata, color=cycle[2], linestyle = '-', label='User')
        
    
    if breakpoints is not None:
        for i, bp in enumerate(breakpoints):
            ax_.axvline(bp,color=cycle[i+1] )
            ax_.text(bp,np.max(ydata)*0.1*(i+5), f'{bp:.2f}')
        
    ax_.grid()
    ax_.legend()     
    
    if axi is None:
        plt.show()
        return fig_
    
    return ax_   
   
def multi_plots(search_lists, res_lists, title='Shift evaluation', 
                nrows=None, ncols=3, 
                save=False, para_inv=False, 
                plot_fig_step=1):
    """Multiplot
    para_inv =True -> search_list= Power number
    
    Args:
        search_lists (array like): shift data -> para_inv=False
                                    power data -> para_inv=True
        res_lists (array like): if shift data, power data. if power data, shift data
        title (str, optional): graph title. Defaults to 'Shift evaluation'.
        nrows (int, optional): rows. Defaults to None.
        ncols (int, optional): cols. Defaults to 4.
        save (bool, optional): save. Defaults to False.
        para_inv (bool, optional): para_inv =True -> search_list= Power number. Defaults to False.
    """
    all_data = len(search_lists)
    total = len(search_lists[::plot_fig_step])
    
    if nrows == None:
        nrows = (len(search_lists) + ncols -1)//ncols

    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.8,nrows*3.6), squeeze=False, tight_layout=True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*5), squeeze=False, tight_layout=True)
    

    ax_list=[]    
    for i in range(nrows):
        for j in range(ncols):
            ax_list.append(ax[i,j])

    for  idx, (i_sea, i_res) in enumerate(zip(search_lists[::plot_fig_step], res_lists[::plot_fig_step])): 
        
        p_para = i_res['popt']
        ax_list[idx].plot(i_res['rex'],i_res['rey'],'ro',label='data')
        ax_list[idx].plot(i_res['rex'],i_res['fit'],'b-',label='fit')
        ax_list[idx].grid(which='both')
        
        if para_inv:
            comment = f'No.{idx*plot_fig_step+1}/{all_data}\nshift: {i_res["popt"][1]:.2f}\nn: {i_sea:.2f}\n$R^2$: {i_res["r2"]:.3f}\n$R^{2}_{{bp}}$: {i_res["r2_bp"]:.3f}'
            ax_list[idx].set_xlabel('x')
            ax_list[idx].set_ylabel(f'y${{1/{i_sea:.1f}}}$')
        else:    
            comment = f'No.{idx*plot_fig_step+1}/{all_data}\nshift: {i_sea:.2f}\nn: {i_res["popt"][0]:.2f}\nslice: {i_res["popt"][1]:.2f}\n$R^2$: {i_res["r2"]:.3f}'
            ax_list[idx].set_xlabel('$ln{x}$')
            ax_list[idx].set_ylabel('$ln{y}$')
        ax_list[idx].legend(title=comment)
        
    if len(ax_list) != total:
        for ij in range(len(ax_list)-total):
            newi= ij + total
            ax_list[newi].axis("off")


    fig.suptitle(title)
    # plt.tight_layout()
    
    if save:
        filename=re_replace(title)
        plt.savefig(f'{filename}.png', dpi=300)
        
    plt.show()
    
# ----

def const_inv_power_fit(xdata, ydata, power_num=2, ini_params=None,
                        min_error='mae',
                        retry_r2=0.9, 
                        negative=False, zero_replace=False,
                        plot=False, info=False,):
    """ReLU & MAE fitting  data^(1/power_num)

    Args:
        xdata (ndarray): xdata
        ydata (ndarray): ydata
        power_num (float, optional): number of power. Defaults to 2.
        ini_params (list, optional): None -> (Nor,Ip,Bg)=[1, 4.5, 0.0]
        min_error:(str,option): 'mae':absolute error, 'mse':squares error, defalut is 'mae'
        retry_r2(float, optional):retry r2 lower limit. Defaults to 0.9.
        
        Handling of negative values when set to a power.
        negative(bool, option): 
            After taking the logarithm of the negative value, it becomes Nan.
            After taking the absolute value of a negative value and the logarithm, 
            add a negative sign to the value.
            Ensure that a power of a negative value does not result in Nan
            Defaults to False.
        zero_replace(bool, option): if True, Replace nan to zero. Defaults to False.
        
        plot (bool, optional): plot. Defaults to False.
        info (bool, optional): info. Defaults to False.

    Returns:
        dict: {'rex':re_xdata, 'rey':re_ydata, 'fit':re_fit, 
                'popt':res_params,
                'r2':r2,'adj_r2':adj_r2,'r2_bp':r2_bp,'er':error_range,
                'cv':cv}
        
    """

    if ini_params is None:
        bg = np.nanmean(ydata[:5]) # 　バックグラウンドの推定
        # nor = np.max(ydata)/10 # 　ノーマライズのオーダーを推定
        nor = 1
        # ip = np.percentile(xdata,50)

        ipy = np.percentile(ydata,5)
        ip_ind = get_nearest_value(ydata, ipy)
        # ip = xdata[ip_ind]
        ip = 5
        ini_params = (nor, ip, bg) 
    # xxdata = np.power(xdata, power_num) 
    
    
    # 負の値のべき乗をとってもNanにならないようにする。
    # 正負を調べる、絶対値でべき乗を取り、その後に正負をかけて戻す。
    def none_negative_power(ydata,pw):
        ysing = np.sign(ydata)
        y_power = ysing*np.power(np.abs(ydata),pw)
    
        return y_power
    
    if negative:
        yydata = none_negative_power(ydata, pw=1/power_num)
    else:
        yydata = np.power(ydata, 1/power_num) 
     
       
        
    re_xdata, re_ydata = nan_inf_rm(xdata,yydata,
                                    zero_replace=zero_replace,info=info)
    re_fit = re_ydata.copy()
    st_info = static_inf(re_ydata,info=False)
    res_params = raf.fit_m_relu(re_xdata, re_ydata, ini_params,  min_error)
    re_fit = raf.m_relu(re_xdata, *res_params)
    # print(re_fit,re_ydata)
    
    # Breakpoint index
    idx = get_nearest_value(re_xdata, res_params[1])
    
    try: 
        r2, adj_r2, error_range =  adj_r2_score(re_ydata, re_fit) 
        r2_bp = r2_score(re_ydata[idx:], re_fit[idx:])
    except:
        r2, adj_r2, error_range, r2_bp = -1, -1, -1, -1

    # 実データでこの項目が必要かどうかは検討（Fittingがうまくいかなかたっら初期値のパラメータを変えてFitting）
    if r2 < retry_r2: # この値は引数に含めている
        bg = np.mean(re_ydata[:5])
        nor = np.max(re_ydata)/2
        for p in [45,55,40,60,35,65]:
        # for p in [45,55]:
            ip = np.percentile(re_xdata,p)
            res_params = raf.fit_m_relu(re_xdata, re_ydata, (nor,ip,bg), min_error)
            re_fit = raf.m_relu(re_xdata, *res_params)
            idx = get_nearest_value(re_xdata, res_params[1])
            try:
                r2, adj_r2, error_range =  adj_r2_score(re_ydata, re_fit) 
                r2_bp = r2_score(re_ydata[idx:], re_fit[idx:])
            except:
                r2, adj_r2, error_range, r2_bp = -1, -1, -1, -1
            if r2 > retry_r2 + 0.02:
                break
    
    if info:
        print(f'power(n): {power_num:.2f}, 1/power(1/n):{1/power_num:.2f}')
        print(f'$R^2$: {r2:.2f}')
        print(f'Error range: {error_range:.2f}')
        print(f'$R^2$ > threshold {r2_bp:.2f}')
        print(f'params(Nor,Ip,bg): {res_params}')
        
         
    if plot:
        fig = plt.figure(figsize=(12,5), tight_layout=True)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        ax1.plot(xdata,ydata,'ro-',label='Data')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(which='both')
        ax2= plot_ax(re_xdata, re_ydata, re_xdata, re_fit, 
                    breakpoints=[res_params[1]],
                    lgtitle=f'n: {power_num:.2f}\n1/n: {1/power_num:.2f}',
                #    title=f'power(n): {power_num:.2f}, 1/n: {1/power_num:.2f}', 
                    title='',
                    axi=ax2)
        ax2.set_xlabel('x')
        ax2.set_ylabel(f'y$^{{1/{power_num:.1f}}}$')
        plt.show()
         
    return {'rex':re_xdata, 'rey':re_ydata, 'fit':re_fit, 
            'popt':res_params,'r2':r2,'adj_r2':adj_r2, 'r2_bp':r2_bp,
            'er':error_range,
            'cv':st_info['cv']}

# --- log-log

def val_rm(xdata,ydata,lim_val,info=True):
    """Remove values smaller than the specified value along the x-axis
        
    Args:
        xdata (ndarray): x data
        ydata (ndarray): y data
        lim_val (float): specified value. ex 0.1, 0.3, 0.5
        info (bool, optional): show infomation. Defaults to True.

    Returns:
        ndarray: xdata, ydata
        
    Note:
        x軸方向で、指定された値より小さい値を削除する
        閾値付近は、データのばらつきが大きい。Shift移動させた場所からどのくらい離れた値から評価するか
        lim_val:x軸のLogを取る前の値(0.1,0.5など）
       
        -inf < ln(x) < 0  at 0 < x < 1 ->xが1以下の時Logを取るとマイナスになる。
        
        ln(lim_val)= val
        ln(1) = 0
        ln(0.1) = -2.3
        ln(0.5) = -0.69
        ln(0) = -inf
        lim_val
        
        np.e**(val)= lim_val
        np.e**(0) = 1
        np.e**(-0.69) = 0.5
        np.e**(-1) = 0.36
        np.e**(-2.3) = 0.1
        np.e**(-3) = 0.05
   
    Example:
        val_rm(np.log(xf),np.log(yf),0.3)
    """
    val = np.log(lim_val)
    tr_val_x = np.where(xdata > val)
    
    re_xdata = xdata.copy()
    re_ydata = ydata.copy()
    re_xdata = xdata[tr_val_x[0]]
    re_ydata = ydata[tr_val_x[0]]
    
    if info:
        print(f'number of remove under {val} on xdata: {np.count_nonzero(tr_val_x)}')
        print(f'shape x, y: {re_xdata.shape}, {re_ydata.shape}')

    return re_xdata, re_ydata


def log_log_fit(xdata, ydata, bg_num=None, 
                lim_val=0.36, fit_type='mae', 
                comment='', 
                info=True, plot=True):
    """estimate power and shift

    Args:
        xdata (ndarray): xdata
        ydata (ndarray): ydata
        num (int, optional): Background average point number. Defaults to 3.
        lim_val (float, optional): Remove values smaller than the specified value along the x-axis.
                        Defaults to 0.36.  np.e**(0.36) = -1 
        fit_type (str,optinal): fitting type: 
        'mae'= mean absolute error,'weight'=rmse and weight,'mse'= root mean squared error
        comment(str,optional)
        info (bool, optional): information. Defaults to True.
        plot(bool, optional):  Plots. Defaults to True.

    Returns:
        dict: {'popt':r_popt,'r2':r2,'adj_r2':adj_r2, 'rex':re_xdata, 'rey':re_ydata, 'fit':fity}
                
                popt=[slope, slice]

    """

    bg_rm_ydata = ydata
    
    if bg_num is not None:
        # バックグランドの値を差し引く
        # Xの最初から数個の値の平均値をとりその値を引く
    
        ybg = np.nanmean(ydata[:bg_num])
        bg = ybg
        bgn = ybg
        bg_rm_ydata = ydata - bg
        
    re_xdata, re_ydata = nan_inf_rm(np.log(xdata), np.log(bg_rm_ydata),zero_replace=False, info=info)
    re_xdata, re_ydata = val_rm(re_xdata, re_ydata, lim_val, info=info)
    e_re_xdata = np.exp(re_xdata)
    

    r_popt = [0,0]
    if len(re_xdata) and len(re_ydata) >= 5: # Fittigする点が5つ以上
        try:
            if fit_type == 'mae':
            #rmse
            # r_popt, _ = curve_fit(m_line, re_xdata, re_ydata, p0=(1,1))
            
                # mae
                r_popt_ = optimize.minimize(mae_line, (1,1), args=(re_xdata, re_ydata))
                r_popt = r_popt_.x
            
            # print(r_popt)
            elif fit_type == 'weight':
                # 重み付
                r_popt, _ = curve_fit(m_line, re_xdata, re_ydata, p0=(1,1),sigma=1/re_xdata)
                
            else:
                r_popt, _ = curve_fit(m_line, re_xdata, re_ydata, p0=(1,1))
                
            fity = m_line(re_xdata,*r_popt)
            r2, adj_r2, error_range =  adj_r2_score(re_ydata, fity) 
            
        except:
            r_popt[0], r_popt[1] = np.nan, np.nan
            r2, adj_r2, error_range = np.nan, np.nan, np.nan
            fity = np.nan

        # print(f'slope, slice: {r_popt}, r2: {r2:.3f}')
        
        if info:
            # print(f'number of NaN: {np.count_nonzero(np.isnan(re_ydata))}')
            print(comment)
            # print(f'estimate background: {bgn}')
            print(f'slope: {r_popt[0]:.3f}, slice: {r_popt[1]:.3f}, $R^2$: {r2:.3f}')
        
        if plot:    
            plt.plot(re_xdata, re_ydata,'ro',label='data')
            # plt.plot(np.log(xdata),m_line(np.log(xdata),*r_popt),'b-',label='fit')
            plt.plot(np.log(e_re_xdata),m_line(np.log(e_re_xdata),*r_popt),'b-',label='fit')
            # plt.xlim(val_rang,np.max(re_xdata))
            plt.grid(True)
            plt.legend(title=f'slope: {r_popt[0]:.3f}\nslice: {r_popt[1]:.3f}\n$R^2$: {r2:.3f}\nadj $R^2$: {adj_r2:.3f}')
            plt.show()
    else:
        r_popt[0],r_popt[1] = np.nan, np.nan
        r2, adj_r2, error_range = np.nan, np.nan, np.nan
        re_xdata, re_ydata,fity = np.nan, np.nan, np.nan
        
    return {'popt':r_popt,'r2':r2,'adj_r2':adj_r2,'er':error_range, 'rex':re_xdata, 'rey':re_ydata, 'fit':fity}

def create_serach_range(center, range, step):
    """create search range

    Args:
        center (float): estimated shift position
        range (flaot): search range
        step (float): step

    Returns:
        ndarray : search range

    s_range = create_serach_range(center=5, range=1, step=0.2)
    >>>[4.  4.2 4.4 4.6 4.8 5.  5.2 5.4 5.6 5.8 6. ]
    
    """
    s_range = np.arange(center-range, center+range+step, step)
    return s_range

def xshift_search(xdata, ydata, search_range,
                  bg_num=None, lim_val=0.36, fit_type='mae',
                  r2_plot=True,
                  process_plot=True, 
                  min_error='mae',
                  info=False, every_plot=False, plot_save=False,
                  plot_fig_step=1):
    """shift, power estimation using shift list
        閾値を変えながらべき乗数を見積もる
        
    Args:
        xdata (ndarray): xdata
        ydata (ndarray): ydata
        search_range (list or tuple): shift list
        r2_plot (bool, optional): View result(slope,shift,r2) information and plots. Defaults to True.
        bg_num (int, optional): Background average point number. Defaults to None.
        lim_val (float, optional): Remove values smaller than the specified value along the x-axis.
                Defaults to 0.36.  np.e**(0.36) = -1 
        fit_type (str,optinal): fitting type: 
        'mae'= mean absolute error,'weight'=rmse and weight,'mse'= root mean squared error
        r2_plot (bool, optional): shift vs r2, power vs r2  plots. Defaults to True.
        process_plot (bool, optional): fitting process plots. Defaults to True.
        plot_save(bool, optional): save. Defaults to False.

    Returns:
        dict: {'r2':likeli_r2, 'shift':likeli_shift, 'power':likeli_power,
            'rex':likeli_rex,'rey':likeli_rey, 'fit':likeli_fit}
    Examples:
    res_data = xshift_search(xdata, ydata, search_range=[3,3.5,4,4.5,5],
                  bg_num=None, lim_val=0.36, fit_type='mae',
                  r2_plot=True, process_plot=True, plot_save=False):
    """
    power = []
    r2 = []
    res_list = []
    adj_r2 = []
    error_range =[]
    applied_serach_range = []
    
    for i in search_range:
        # print(f'shift: {i:.2f}'
        res_para = log_log_fit(xdata-i,ydata,bg_num=bg_num,
                               lim_val=lim_val,
                               fit_type=fit_type, 
                               comment=f'shift: {i:.2f}',
                               info=info,
                               plot=every_plot)
        
        if math.isnan(res_para['r2']):
            pass
        
        else:
            applied_serach_range.append(i)
            power.append(res_para['popt'][0])
            r2.append(res_para['r2'])
            adj_r2.append(res_para['adj_r2'])
            res_list.append(res_para)
            error_range.append(res_para['er'])
            
    if process_plot:    
        multi_plots(applied_serach_range, res_list, title='Shift-Log-Log',save=plot_save,
                    plot_fig_step=plot_fig_step)    
    
    # 決定係数が1に最も近いものを選ぶ
    max_r2_idx = get_nearest_value(r2, 1)
    # max_r2_idx = idx_of_the_nearest(data=r2, value=1)
    likeli_power = power[max_r2_idx]
    likeli_shift = applied_serach_range[max_r2_idx]
    likeli_r2 = r2[max_r2_idx]
    likeli_er = error_range[max_r2_idx]
    likeli_rex = res_list[max_r2_idx]['rex']
    likeli_rey = res_list[max_r2_idx]['rey']
    likeli_fit = res_list[max_r2_idx]['fit']
    
   
    msg = f'likelihood shift: {likeli_shift:.3f}, power: {likeli_power:.3f}'
    
    if info:
        print(msg)
    
    if r2_plot:
        search_r2_plot(shifts=applied_serach_range,
                       powers=power,
                       r2=r2,
                       msg=msg,
                       save=plot_save)
        
        search_r2_plot(shifts=applied_serach_range,
                        powers=power,
                        r2=error_range,
                        msg=msg,
                        save=plot_save,
                        ylabel='error range')
        
        with_bg_fit = const_inv_power_fit(xdata, ydata, 
                                power_num=likeli_power,
                                ini_params=(1,likeli_shift,0), 
                                min_error=min_error,
                                retry_r2=0.9, 
                                zero_replace=True,
                                plot=True, info=False)
        
        # # Backgroundを差し引いて解析を行う
    
        # bg = with_bg_fit['popt'][2]**likeli_power
        # if bg <= 0:
        #     bg=0
        # const_inv_power_fit(xdata, ydata-bg, 
        #                     power_num=likeli_power,
        #                     ini_params=(1,likeli_shift,0), 
        #                     min_error=min_error,
        #                     retry_r2=0.9, 
        #                     zero_replace=True,
        #                     plot=True, info=False)
                
        # inv_power_plot(xdata, ydata, 
        #                shift=likeli_shift, 
        #                power=likeli_power)
          
    return {'r2':likeli_r2, 'er':likeli_er, 
            'shift':likeli_shift, 'power':likeli_power,
            'rex':likeli_rex, 'rey':likeli_rey, 'fit':likeli_fit}



def xpower_search(xdata, ydata, search_range=None,
                  r2_plot=True, process_plot=True,
                  min_error='mae',
                  zero_replace=False,
                  likely_evaluation ='r2',
                  info=True, every_plot=False, plot_save=False,
                  plot_fig_step=1):
    """べき乗数を変えながら閾値を見積もる 1/n-power-plot

    Args:
        xdata (ndarray): xdata
        ydata (ndarray): ydata
        search_range (list or tuple): Power range. Defaults to None. If None,[2,2.5,3,3.5,4]
        r2_plot (bool, optional): View result(slope,shift,r2) information and plots. Defaults to True.
        process_plot (bool, optional): View plots of each trial. Defaults to True.
        likely_evaluation (str, optional):Evaluation method. 'r2 or 'r2_bp' Defaults to 'r2'
        'r2' :all region, 'r2_bp' : after bp region
        info (bool, optional): info. Defaults to True.
        every_plot (bool, optional): each plot show. Defaults to False.
        plot_save (bool, optional): save. Defaults to False.

    Returns:
        dict :  {'r2':likeli_r2, 'shift':likeli_shift, 'power':likeli_power,
                 'rex':likeli_rex,'rey':likeli_rey, 'fit':likeli_fit}
        
    Examples:
    
    """
    shift = []
    r2 = []
    r2_bp = []
    res_list = []
    adj_r2 = []
    error_range = []
    applied_serach_range = []
    
    if search_range is None:
        search_range = [1.5,2,2.5,3,3.5,4,4.5,5]
    
    if len(search_range) == 3:
        ncols = 3
    elif len(search_range) == 2:
        ncols = 2 
    else:
        ncols = 4 

        
    for i in search_range:
        # print(f'shift: {i:.2f}'
        res_para = const_inv_power_fit(xdata, ydata, power_num=i, 
                                       ini_params=None, 
                                       min_error=min_error,
                                       zero_replace=zero_replace,
                                       plot=every_plot, info=info)
        # {'rex':re_xdata, 'rey':re_ydata, 'fit':re_fit, 
            # 'popt':res_params,'r2':r2,'adj_r2':adj_r2, 'r2_bp':r2_bp,
            # 'er':error_range,
            # 'cv':st_info['cv']}
        if math.isnan(res_para['r2']):
            pass
        
        else:
            applied_serach_range.append(i)
            shift.append(res_para['popt'][1])
            r2.append(res_para['r2'])
            adj_r2.append(res_para['adj_r2'])
            r2_bp.append(res_para['r2_bp'])
            error_range.append(res_para['er'])
            res_list.append(res_para)
            
    if process_plot:    
        multi_plots(applied_serach_range, res_list, title='1/n-Power', ncols=ncols, 
                    para_inv=True,save=plot_save,
                    plot_fig_step=plot_fig_step)    
    
    # max_r2_idx = idx_of_the_nearest(data=r2, value=1)
    # max_r2_idx = get_nearest_value(r2_bp, 1)
    if likely_evaluation == 'r2_bp':
        eva_r2 = r2_bp
        decision_eva = r'$R^{2}_{bp}$' 
    else:
        eva_r2 = r2
        decision_eva = r'$R^{2}$'
    max_r2_idx = get_nearest_value(eva_r2, 1)
 
    likeli_shift = shift[max_r2_idx]
    likeli_power = applied_serach_range[max_r2_idx]
    likeli_r2 = eva_r2[max_r2_idx]
    likeli_er = error_range[max_r2_idx]
    likeli_rex = res_list[max_r2_idx]['rex']
    likeli_rey = res_list[max_r2_idx]['rey']
    likeli_fit = res_list[max_r2_idx]['fit']
    likeli_popt = res_list[max_r2_idx]['popt']
    
    
    msg = f'{decision_eva}, likelihood shift: {likeli_shift:.3f}, power: {likeli_power:.1f}, 1/power: {1/likeli_power:.3f}'
    
    if info:
        print(msg)
    
    if r2_plot:
        search_r2_plot(shifts=shift,
                        powers=applied_serach_range,
                        r2=eva_r2,
                        msg=msg,
                        save=plot_save)
        
        with_bg_fit = const_inv_power_fit(xdata, ydata, 
                        power_num=likeli_power,
                        ini_params=(1,likeli_shift,0), 
                        zero_replace=zero_replace,
                        retry_r2=0.9, plot=True, info=False) 
        
         # Backgroundを差し引いて解析を行う
        const_inv_power_fit(xdata, ydata-with_bg_fit['popt'][2]**likeli_power, 
                            power_num=likeli_power,
                            ini_params=(1,likeli_shift,0), 
                            zero_replace=zero_replace,
                            retry_r2=0.9, 
                            plot=True, info=False)
        
        # inv_power_plot(xdata, ydata, 
        #                shift=likeli_shift, 
        #                power=likeli_power)     

    
    return {'r2':likeli_r2, 'er':likeli_er, 
            'shift':likeli_shift, 'power':likeli_power,
            'rex':likeli_rex,'rey':likeli_rey, 'fit':likeli_fit,
            'popt':likeli_popt}
