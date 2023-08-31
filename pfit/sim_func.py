import numpy as np
import matplotlib.pyplot as plt

def power_plus(x,a,b,c,d):
    # f = a*np.power((x-b),c) + d
    u = (x-b)
    g = a*(np.power(u,c))*(u>0)+ d 
    return g

# pys function
@np.vectorize
def pys(x, Nor, Ip, T, Bg):
    '''
    pys calculation
    :param x: list of energy  or value
    :param Ip:
    :param T:
    :param Nor:
    :param Bg:
    :return: pys calculation
    example:
    -----
    >>>p=pys(np.array([4,4.2,4.3,4.5]),4.3,300,1,1)
    >>>print(p,type(p))
    [ 1.00000912  1.02078619  1.81083333 32.5716379 ] <class 'numpy.ndarray'>
    >>>p=pys(4.2,4.3,300,1,1)
    >>>print(p,type(p))
    1.020786189180001 <class 'numpy.ndarray'>
    Note:
    -----
    numpyのVectorizeを利用すると、配列を受け付けるようになる
    '''
    u = (x - Ip) / (T * 8.6171e-5)
    if u <= 0:
        f = Nor * (np.exp(u) - (np.exp(2 * u) / (2 * 2)) + (np.exp(3 * u) / (3 * 3))
                   - (np.exp(4 * u) / (4 * 4)) + (np.exp(5 * u) / (5 * 5)) - (np.exp(6 * u) / (6 * 6))) + Bg

    else:
        f = Nor * (np.pi * np.pi / 6 + (1 / 2) * u * u - (np.exp(-u)) + (np.exp(-2 * u) / (2 * 2))
                   - (np.exp(-3 * u) / (3 * 3)) + (np.exp(-4 * u) / (4 * 4)) - (np.exp(-5 * u) / (5 * 5))
                   + (np.exp(-6 * u) / (6 * 6))) + Bg

    return f

# 論文用のシミュレーション作成コード
def lnln(x,a,b,c,d):
    # f = a*np.power((x-b),c) + d
    u = (x-b)
    f = a*(u**c)*(u>0.0) + d
    return {'pw':f,'ln':np.log(f),'lnx':np.log(x)}

def pw_ln_plot(xdata,para1,para2,para3,exlim=1000,leg_title=None,axi=None,figsize=(12,5)):
    """

    Args:
        xdata (_type_): _description_
        para1 (_type_): _description_
        para2 (_type_): _description_
        para3 (_type_): _description_
        exlim (int, optional): _description_. Defaults to 1000.
        leg_title (_type_, optional): _description_. Defaults to None.
        axi (_type_, optional): _description_. Defaults to None.
        figsize (tuple, optional): _description_. Defaults to (12,5).

    Returns:
        _type_: _description_
        
    Examples:
        xx2 = np.linspace(0,10,100)
        para1b=(2,0,3,0)
        para2b=(2,-1,3,0)
        para3b=(2,1,3,0)
        pw_ln_plot(xx2,para1b,para2b,para3b,
                exlim=1000,
                leg_title='$y=a \cdot (x-b)^c + d$',
                axi=None,figsize=(12,5))
        
        para1bc=(2,0,3,0)
        para2bc=(2,0,3,1)
        para3bc=(2,0,3,5)
        pw_ln_plot(xx2,para1bc,para2bc,para3bc,
                exlim=1000,
                leg_title='$y=a \cdot (x-b)^c + d$',
                axi=None,figsize=(12,5))
    
    """
    if axi is None:
        fig_ = plt.figure(figsize=figsize, tight_layout=True)
        ax0 = fig_.add_subplot(121)
        ax1 = fig_.add_subplot(122)
        
    else:
        ax0 = axi[0]
        ax1 = axi[1]
        
    
    ax0.plot(xdata,lnln(xdata,*para1)['pw'],'ro-',label=f'{para1}')
    ax0.plot(xdata,lnln(xdata,*para2)['pw'],'gs-',label=f'{para2}')
    ax0.plot(xdata,lnln(xdata,*para3)['pw'],'b^-',label=f'{para3}')

        
    ax0.set_xlabel('$\it{x}$')
    ax0.set_ylabel('$\it{y}$')
    ax0.set_ylim(0,exlim)
    ax0.grid(True)
    
    
    ax1.plot(lnln(xdata,*para1)['lnx'],lnln(xdata,*para1)['ln'],'ro-',label=f'{para1}')
    ax1.plot(lnln(xdata,*para2)['lnx'],lnln(xdata,*para2)['ln'],'gs-',label=f'{para2}')
    ax1.plot(lnln(xdata,*para3)['lnx'],lnln(xdata,*para3)['ln'],'b^-',label=f'{para3}')

        
    ax1.set_xlabel('$ln{x}$')
    ax1.set_ylabel('$ln{y}$')
    
    def x2ln(x):
            return np.log(x)

    def ln2x(x):
        return np.exp(x)

    secax = ax1.secondary_xaxis('top', functions=(ln2x, x2ln))
    secax.set_xlabel('$\it{x}$')
    # secax.grid(True)
    # ax1.set_ylim(0,1000)
    # ax1.set_xlim(min(lnln(xdata,*para3)['lnx']),max(lnln(xdata,*para3)['lnx']))
    ax1.grid(True)


    if leg_title is None:
        ax0.legend(loc='upper left')
        ax1.legend(loc='upper left')
    else:
        # ax0.legend(title=f'{leg_title}\n(a,b,c,d)',loc='upper left')
        # ax1.legend(title=f'{leg_title}\n(a,b,c,d)',loc='upper left')
        ax0.legend(title=f'{leg_title}\n            (a,b,c,d)')
        ax1.legend(title=f'{leg_title}\n            (a,b,c,d)')
        
    if axi == None:
        plt.show()
        return  None

    else:
        return ax0,ax1