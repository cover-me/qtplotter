# pieces of code taken from project qtplot
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from urllib.request import urlopen
from scipy import ndimage
import ipywidgets as widgets
plt.rcParams['figure.facecolor'] = 'white'

# operations
def create_kernel(x_dev, y_dev, cutoff, distr):
    distributions = {
        'gaussian': lambda r: np.exp(-(r**2) / 2.0),
        'exponential': lambda r: np.exp(-abs(r) * np.sqrt(2.0)),
        'lorentzian': lambda r: 1.0 / (r**2+1.0),
        'thermal': lambda r: np.exp(r) / (1 * (1+np.exp(r))**2)
    }
    func = distributions[distr]

    hx = int(np.floor((x_dev * cutoff) / 2.0))
    hy = int(np.floor((y_dev * cutoff) / 2.0))


    x = np.zeros(1) if x_dev==0 else np.linspace(-hx, hx, hx * 2 + 1) / x_dev
    y = np.zeros(1) if y_dev==0 else np.linspace(-hy, hy, hy * 2 + 1) / y_dev

    xv, yv = np.meshgrid(x, y)

    kernel = func(np.sqrt(xv**2+yv**2))
    kernel /= np.sum(kernel)

    return kernel

def yderiv(d):
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
    y = d[1]
    z = d[2]
    dzdy0 = (z[1]-z[0])/(y[1]-y[0])
    dzdy1 = (z[-2]-z[-1])/(y[-2]-y[-1])
    z[1:-1] = (z[2:] - z[:-2])/(y[2:] - y[:-2])
    z[0] = dzdy0
    z[-1] = dzdy1

def lowpass(d, x_width=0.5, y_height=0.5, method='gaussian'):
    """Perform a low-pass filter."""
    z = d[2]
    kernel = create_kernel(x_width, y_height, 7, method)
    z[:] = ndimage.filters.convolve(z, kernel)

def scale(d,amp=[]):
    for i, ai in enumerate(amp):
        d[i] *= ai

def offset(d,off=[]):
    for i, oi in enumerate(off):
        d[i] += oi

def g_in_g2(d, rin):
    """z = z/(1-(z*Rin))/7.74809e-5. z: conductance in unit 'S', R in unit 'ohm' (SI units)"""
    G2 = 7.74809e-5#ohm^-1, 2e^2/h
    d[2] = d[2]/(1-(d[2]*rin))/G2

def autoflip(d):
    '''
    Make the order of elements in x and y good for imshow() and filters
    '''
    x = d[0]
    y = d[1]
    xa = abs(x[0,0]-x[0,-1])
    xb = abs(x[0,0]-x[-1,0])
    ya = abs(y[0,0]-y[0,-1])
    yb = abs(y[0,0]-y[-1,0])
    if (xa<xb and yb<ya) or (xa>xb and yb<ya and yb/ya<xb/xa) or (xa<xb and yb>ya and ya/yb>xa/xb):
        d = np.transpose(d, (0, 2, 1))# swap axis 1 and 2
    x = d[0]#note: x y won't unpdate after d changes. There maybe nan in last lines of x and y.
    y = d[1]
    if x[0,0]>x[0,-1]:
        d = d[:,:,::-1]
    if y[0,0]>y[-1,0]:
        d = d[:,::-1,:]
    return d
        
def get_quad(x):
    '''
    Calculate the patch corners for pcolormesh
    More discussion can be found here: https://cover-me.github.io/2019/02/17/Save-2d-data-as-a-figure.html, https://cover-me.github.io/2019/04/04/Save-2d-data-as-a-figure-II.html
    '''
    l0, l1 = x[:,[0]], x[:,[1]]
    r1, r0 = x[:,[-2]], x[:,[-1]]
    x = np.hstack((2*l0 - l1, x, 2*r0 - r1))
    t0, t1 = x[0], x[1]
    b1, b0 = x[-2], x[-1]
    x = np.vstack([2*t0 - t1, x, 2*b0 - b1])
    x = (x[:-1,:-1]+x[:-1,1:]+x[1:,:-1]+x[1:,1:])/4.  
    return x

# read data 
def _readMTX(fPath):
    '''
    read 2D .mtx files, which have the form:
        Units, Dataset name, xname, xmin, xmax, yname, ymin, ymax, zname, zmin, zmax
        nx ny nz length
        [binary data....]
    '''
    if mpl.is_url(fPath):
        open = lambda url,mode: urlopen(url)
    with open(fPath, 'rb') as f:
        line1 = f.readline().decode().rstrip('\n\t\r')
        if line1.startswith('Units'):#MTX file
            _ = line1.split(',') 
            labels = [x.strip() for x in [_[2],_[5],_[8],_[1]]]
            line2 = f.readline().decode()
            shape = [int(x) for x in line2.split(' ')]
            x = np.linspace(float(_[3]),float(_[4]),shape[0])
            y = np.linspace(float(_[6]),float(_[7]),shape[1])
            z = np.linspace(float(_[9]),float(_[10]),shape[2])
            z,y,x = np.meshgrid(z,y,x,indexing='ij')
            dtp = np.float64 if shape[3] == 8 else np.float32#data type
            shape = shape[0:3]
            w = np.frombuffer(f.read(),dtype=dtp).reshape(shape).T
            if shape[2] == 1:# only care about .mtx files with 2D data
                return x[0],y[0],w[0],[labels[0],labels[1],labels[3]]

def _readDat(fPath,cols=[0,1,3],cook=None,a3=2,a3index=0):
    '''
    read .dat files
    '''
    sizes = []# nx,ny,nz for each dimension of scan. Assume a 3D scan (1D and 2D are also kinds of 3D).
    labels = []# labels for each column
    # read comments
    with open(fPath, 'r') as f:
        for line in f:
            line = line.rstrip('\n\t\r')
            if line.startswith('#\tname'):
                labels.append(line.split(': ', 1)[1])
            elif line.startswith('#\tsize'):
                sizes.append(int(line.split(': ', 1)[1]))
            if len(line) > 0 and line[0] != '#':# where comments end
                break
    # load data
    print('File: %s, cols: %s'%(os.path.split(fPath)[1],[labels[i] for i in cols]))
    d = np.loadtxt(fPath,usecols=cols)
    
    #assume this is data from a 3D scan, we call the element of D1/D2/D3 the point/line/page
    n_per_line = sizes[0]
    n_per_page = sizes[0]*sizes[1]
    n_dp = d.shape[0]# Real number of datapoints
    n_pg = int(np.ceil(float(n_dp)/n_per_page))# number of pages, it may be smaller than sizes[2] because sometimes the scan is interrupted by a user
    
    pivot = np.full((len(cols),n_per_page*n_pg), np.nan)# initialize with na.nan
    pivot[:,:n_dp] = d.T
    pivot = pivot.reshape([len(cols),n_pg,sizes[1],sizes[0]])

    # You have a 3D scan, you want to extract a 2D slice. a3 and a3index are the parameters for slicing
    if a3 == 0:#slice with x_index=const
        pivot = pivot[:,:,:,a3index]
    elif a3 == 1:#y_ind=const
        pivot = pivot[:,:,a3index,:]
    elif a3 == 2:#z_ind=const
        pivot = pivot[:,a3index,:,:]
    
    # remove nan lines in x,y,w...
    nans = np.isnan(pivot[0,:,0])
    pivot = pivot[:,~nans,:]
    
    # Some values in the last line of x and y may be nan. Recalculate these values. Keep w untouched.
    nans = np.isnan(pivot[0,-1,:])
    pivot[:2,-1,nans] = pivot[:2,-2,nans]*2.-pivot[:2,-3,nans]
    
    # autoflip for filters and imshow()
    pivot = autoflip(pivot)
    
    if cook:
        cook(pivot)

    x,y,w = pivot[:3]
    return x,y,w,[labels[cols[i]] for i in range(3)]

def read2d(fPath,**kw):
    if fPath.endswith('.mtx'):
        x,y,w,labels = _readMTX(fPath)
    elif fPath.endswith('.dat'):
        x,y,w,labels = _readDat(fPath,kw['cols'],kw['cook'])
    else:
        return
    return x,y,w,labels
    
def plot2d(fPath,**kw):
    x,y,w,labels = read2d(fPath,**kw)
    if 'labels' not in kw:
        kw['labels'] = labels
    return _plot2d(x,y,w,**kw) 

def play2d(fPath,**kw):
    x,y,w,labels = read2d(fPath,**kw)
    x0 = x[0]
    y0 = y[:,0]
    dx = x0[1]-x0[0]
    dy = y0[1]-y0[0]
    wmin = np.min(w)
    wmax = np.max(w)
    if 'labels' not in kw:
        kw['labels'] = labels
    @widgets.interact(
        xpos=(x0[0],x0[-1],dx),
        ypos=(y0[0],y0[-1],dy),
        gamma=(-100,100,10),
        vlim=widgets.FloatRangeSlider(value=[wmin,wmax],min=wmin,max=wmax,step=(wmax-wmin)/20,description='limit'),)
    def foo(xpos,ypos,gamma,vlim):
        fig, axs = plt.subplots(1,2,figsize=(6,2),dpi=120)#main plot and h linecut
        plt.subplots_adjust(hspace=0.3,wspace=0.5)
        axs[1].yaxis.tick_right()
        axs[1].tick_params(axis='x', colors='tab:orange')
        axs[1].tick_params(axis='y', colors='tab:orange')
        axv = fig.add_axes(axs[1].get_position(), frameon=False)#vertical linecut
        axv.xaxis.tick_top()
        axv.tick_params(axis='x', colors='tab:blue')
        axv.tick_params(axis='y', colors='tab:blue')
        # plot 2D data
        kw['gamma'],kw['vmin'],kw['vmax']=gamma,vlim[0],vlim[1]
        _plot2d(x,y,w,fig=fig,ax=axs[0],**kw)
        # vlinecut
        indx = np.abs(x0 - xpos).argmin()# x0 may be a non uniform array
        axs[0].plot(x[:,indx],y0,'tab:blue')
        axv.plot(w[:,indx],y0,'tab:blue')
        # hlinecut
        indy = np.abs(y0 - ypos).argmin()
        axs[0].plot(x0,y[indy,:],'tab:orange')
        axs[1].plot(x0,w[indy,:],'tab:orange')

def _plot2d(x,y,w,**kw):
    '''Plot 2D figure'''
    #plot settings
    ps = {'labels':['','',''],'useImshow':True,'gamma':0,'gmode':'moveColor',
          'cmap':'seismic','vmin':None, 'vmax':None,'plotCbar':True}
    for i in ps:
        if i in kw:
            ps[i] = kw[i]
    if 'ax' in kw and 'fig' in kw:
        # sometimes you want to use customized axes
        fig = kw['fig']
        ax = kw['ax']
    else:
        fig, ax = plt.subplots(figsize=(3.375,2),dpi=120)

    x1 = get_quad(x)
    y1 = get_quad(y)    
    imkw = {'cmap':ps['cmap'],'vmin':ps['vmin'],'vmax':ps['vmax']}
    gamma_real = 10.0**(ps['gamma'] / 100.0)#to be consistent with qtplot
    if gamma_real != 1:
        if ps['gmode']=='moveColor':
            _ = 1024# default: 256
            cmap = mpl.cm.get_cmap(imkw['cmap'], _)
            cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 1, _)**gamma_real))
            imkw['cmap'] = cmap
        else:
            imkw['norm'] = mpl.colors.PowerNorm(gamma=gamma_real)
    
    if ps['useImshow']:#slightly different from pcolormesh, especially if saved as vector formats. Imshow is better if it works. See the links in get_quad() description.
        xy_range = (x1[0,0],x1[0,-1],y1[0,0],y1[-1,0])
        im = ax.imshow(w,aspect='auto',interpolation='none',origin='lower',extent=xy_range,**imkw)
    else:
        im = ax.pcolormesh(x1,y1,w,**imkw)

    if ps['plotCbar']:
        cbar = fig.colorbar(im,ax=ax)
        cbar.set_label(ps['labels'][2])
    else:
        cbar = None
    ax.set_xlabel(ps['labels'][0])
    ax.set_ylabel(ps['labels'][1])
    
    return fig,ax,cbar,im

def simpAx(ax,cbar,im,n=(None,None,None),pad=(-5,-15,-10)):
    _min,_max = ax.get_xlim()
    if n[0] is not None:
        a = 10**(-n[0])
        _min = np.ceil(_min/a)*a
        _max = np.floor(_max/a)*a
        ax.set_xlim(_min,_max)
    ax.set_xticks([_min,_max])
    ax.xaxis.labelpad = pad[0]

    _min,_max = ax.get_ylim()
    if n[1] is not None:
        a = 10**(-n[1])
        _min = np.ceil(_min/a)*a
        _max = np.floor(_max/a)*a
        ax.set_ylim(_min,_max)
    ax.set_yticks([_min,_max])
    ax.yaxis.labelpad = pad[1]

    #assumes a vertical colorbar
    _min,_max = cbar.ax.get_ylim()
    label = cbar.ax.get_ylabel()
    if n[2] is not None:
        a = 10**(-n[2])
        _min = np.ceil(_min/a)*a
        _max = np.floor(_max/a)*a
        im.set_clim(_min,_max)
    cbar.set_ticks([_min,_max])
    cbar.ax.yaxis.labelpad = pad[2]
    cbar.ax.set_ylabel(label)
