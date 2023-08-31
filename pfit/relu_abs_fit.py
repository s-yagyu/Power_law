# ReLU absolute fit
import numpy as np
from scipy.optimize import minimize
import scipy.optimize as optimize

# m_relu関数の定義
@np.vectorize
def m_relu(x, Nor, Ip, Bg):
    u = (x - Ip)
    return Nor * u * (u > 0.0) + Bg

# # 平均絶対誤差の定義
# def absolute_error(params, x, y):
#     y_pred = m_relu(x, *params)
#     return np.sum(np.abs(y - y_pred))/len(x)

# 絶対誤差の定義
def absolute_error(params, x, y):
    y_pred = m_relu(x, *params)
    return np.sum(np.abs(y - y_pred))

# # 最小平均二乗誤差の定義
# def least_squares_error(params, x, y):
#     y_pred = m_relu(x, *params)
#     return np.sum((y - y_pred) ** 2)/len(x)

# 最小二乗誤差の定義
def least_squares_error(params, x, y):
    y_pred = m_relu(x, *params)
    return np.sum((y - y_pred) ** 2)


def fit_m_relu(x, y, params_init=None, min_error='mae'):
    """Absolute fitting
    Args:
        x (ndarray): _description_
        y (ndarray): _description_
        params_init (list or ndarray, optional): _description_. Defaults to None.
        min_error:(str,option): 'mae':absolute error, 'mse':squares error

    Returns:
        float: Nor_opt, Ip_opt, Bg_opt
        
    examples:
    fit_param = fit_m_relu(xdata,ydata, params_init=[1, 4.5, 0.0])
    
    """
    
    if params_init is None:
        params_init = np.array([1, 4.5, 0.0])

    if min_error == 'mse':
        result = minimize(least_squares_error, params_init, args=(x, y))
        
    else:
        result = minimize(absolute_error, params_init, args=(x, y))

    Nor_opt, Ip_opt, Bg_opt = result.x

    return Nor_opt, Ip_opt, Bg_opt


def de_fit_m_relu(x,y,params_init=None):
    #differential_evolution(func, bounds[, args, …]) 多変数関数の大域最小値を求める。
    # para=(Nor,ip,bg)
    bounds = [(0, np.max(y)), (np.min(x), np.max(x)), (0,np.max(y))]
    # print(bounds)
    result  = optimize.differential_evolution(absolute_error, bounds, args=(x, y))
    
    Nor_opt, Ip_opt, Bg_opt = result.x

    return Nor_opt, Ip_opt, Bg_opt

def bh_fit_m_relu(x,y,params_init=None):
    # basinhopping(func, x0[, niter, T, stepsize, …]) basin-hoppingアルゴリズムを使って関数の大域最小値を求める。
    if params_init is None:
        nor_m = np.max(y)/10
        params_init = np.array([nor_m, 4.5, 0.0])

    result  = optimize.basinhopping(absolute_error, params_init, minimizer_kwargs={"method": "L-BFGS-B","args":(x, y)})
    Nor_opt, Ip_opt, Bg_opt = result.x

    return Nor_opt, Ip_opt, Bg_opt

# if __name__ == "__main__":
#     pass
    # import matplotlib.pyplot as plt
    # xx = np.arange(4,6.1,0.1)
    # yy=m_relu(xx,2,4.5,2)
    # f1=fit_m_relu(xx, yy, params_init=None)
    # f2=de_fit_m_relu(xx,yy,params_init=None)
    # # f3= bh_fit_m_relu(xx,yy,params_init=None)
    # print(f1)
    # print(f2)
    # # print(f3)
    # plt.plot(xx,yy)
    # plt.plot(xx,m_relu(xx,*f1))
    # plt.plot(xx,m_relu(xx,*f2))
    # # plt.plot(xx,m_relu(xx,*f3))

