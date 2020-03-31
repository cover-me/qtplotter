'''
It's a simpler, easier-to-access, notebook-based version of Rubenknex/qtplot. Most of the code is grabbed from qtplot.
The project is hosted on https://github.com/cover-me/qtplotter
'''
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from urllib.request import urlopen
from scipy import ndimage
import ipywidgets as widgets
plt.rcParams['figure.facecolor'] = 'white'

# operation
class operation:
    
    # auxiliary operations (begin with '_')
    @staticmethod
    def _create_kernel(x_dev, y_dev, cutoff, distr):
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

    @staticmethod
    def _get_quad(x):
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
    
    # filters
    @staticmethod
    def yderiv(d):
        '''
        y derivation
        https://en.wikipedia.org/wiki/Finite_difference_coefficient
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
        '''
        y = d[1]
        z = d[2]
        dzdy0 = (z[1]-z[0])/(y[1]-y[0])
        dzdy1 = (z[-2]-z[-1])/(y[-2]-y[-1])
        z[1:-1] = (z[2:] - z[:-2])/(y[2:] - y[:-2])
        z[0] = dzdy0
        z[-1] = dzdy1
    
    @staticmethod
    def lowpass(d, x_width=0.5, y_height=0.5, method='gaussian'):
        """Perform a low-pass filter."""
        z = d[2]
        kernel = operation._create_kernel(x_width, y_height, 7, method)
        z[:] = ndimage.filters.convolve(z, kernel)

    @staticmethod
    def scale(d,amp=[]):
        for i, ai in enumerate(amp):
            d[i] *= ai

    @staticmethod
    def offset(d,off=[]):
        for i, oi in enumerate(off):
            d[i] += oi

    @staticmethod
    def g_in_g2(d, rin):
        """z = z/(1-(z*Rin))/7.74809e-5. z: conductance in unit 'S', R in unit 'ohm' (SI units)"""
        G2 = 7.74809e-5#ohm^-1, 2e^2/h
        d[2] = d[2]/(1-(d[2]*rin))/G2

    @staticmethod
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

# data
def read2d(fPath,**kw):
    '''
    Supported data types:
    .mtx, .dat
    1d, 2d, 3d scan. Returned data is 2d (slicing parameters are required if the file contains 3d data).
    local file, url
    '''
    if fPath.endswith('.mtx'):
        x,y,w,labels = _readMTX(fPath)
    elif fPath.endswith('.dat'):
        x,y,w,labels = _readDat(fPath,**kw)
    else:
        return
    return x,y,w,labels

# funtions load data
def _readMTX(fPath):
    '''
    read .mtx files, which have the form:
        Units, Dataset name, xname, xmin, xmax, yname, ymin, ymax, zname, zmin, zmax
        nx ny nz length
        [binary data....]
    It's a 3d data format created by Gary Steele, https://nsweb.tn.tudelft.nl/~gsteele/spyview/#mtx
    However, we hardly export data into 3d mtx files. Let's assume nz = 1 for simplicity.
    '''
    if mpl.is_url(fPath):
        open2 = lambda url,mode: urlopen(url)
    else:
        open2 = open
    with open2(fPath, 'rb') as f:
        line1 = f.readline().decode().rstrip('\n\t\r')
        if line1.startswith('Units'):#MTX file
            _ = line1.split(',') 
            labels = [x.strip() for x in [_[2],_[5],_[8],_[1]]]#xname,yname,zname,dataset name
            line2 = f.readline().decode()
            shape = [int(x) for x in line2.split(' ')]#nx ny nz element_length_in_bytes
            x = np.linspace(float(_[3]),float(_[4]),shape[0])
            y = np.linspace(float(_[6]),float(_[7]),shape[1])
            z = np.linspace(float(_[9]),float(_[10]),shape[2])
            z,y,x = np.meshgrid(z,y,x,indexing='ij')
            dtp = np.float64 if shape[3] == 8 else np.float32#data type
            shape = shape[0:3]
            w = np.frombuffer(f.read(),dtype=dtp).reshape(shape).T
            if shape[2] == 1:# only care about .mtx files nz=1 (1d or 2d data) 
                return x[0],y[0],w[0],[labels[0],labels[1],labels[3]]

def _readDat(fPath,cols=[0,1,3],cook=None,a3=2,a3index=0):
    '''
    read .dat files generated by qtlab
    If data is taken from a 3d scan, use a3 and a3index to get a 2D slice. a3 stands for "the third axis" which is perpendicular to the slice plane.
    '''
    sizes = []# nx,ny,nz for each dimension of scan. Default a 3d scan (1D and 2D are also kinds of 3D).
    labels = []
    if mpl.is_url(fPath):
        open2 = lambda url,mode: urlopen(url)
    else:
        open2 = open
    # read comments
    with open2(fPath, 'rb') as f:
        for line in f:
            line = line.decode()
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
    pivot = operation.autoflip(pivot)

    if cook:
        cook(pivot)

    x,y,w = pivot[:3]
    return x,y,w,[labels[cols[i]] for i in range(3)]


# plot
def plot2d(fPath,**kw):
    '''Generate a 2D plot with customize parameters'''
    x,y,w,labels = read2d(fPath,**kw)
    if 'labels' not in kw:
        kw['labels'] = labels
    if len(x)==1:#1d data
        _plot1d(x,y,w,**kw)
    else:
        return _plot2d(x,y,w,**kw)


def play2d(fPath,**kw):
    '''
    Generate an interactive 2D plot to play with
    x,y must be uniform spaced, autoflipped.
    '''
    # get data
    x,y,w,labels = read2d(fPath,**kw)
    if 'labels' not in kw:
        kw['labels'] = labels
    x0 = x[0]
    y0 = y[:,0]
    xmin,xmax,dx = x[0,0],x[0,-1],x[0,1]-x[0,0]
    ymin,ymax,dy = y[0,0],y[-1,0],y[1,0]-y[0,0]
    wmin,wmax = np.min(w),np.max(w)
    dw = (wmax-wmin)/20

    # UI
    sxpos = widgets.FloatSlider(value=(xmin+xmax)/2,min=xmin,max=xmax,step=dx,description='x')
    sypos = widgets.FloatSlider(value=(ymin+ymax)/2,min=ymin,max=ymax,step=dy,description='y')
    vb1 = widgets.VBox([sxpos,sypos])
    sgamma = widgets.IntSlider(value=0,min=-100,max=100,step=10,description='gamma')
    svlim = widgets.FloatRangeSlider(value=[wmin,wmax],min=wmin,max=wmax,step=dw,description='limit')
    vb2 = widgets.VBox([sgamma,svlim])
    bexpMTX = widgets.Button(description='To mtx')
    htmlexp = widgets.HTML()
    vb3 = widgets.VBox([bexpMTX,htmlexp])
    ui = widgets.Tab(children=[vb1,vb2,vb3])
    [ui.set_title(i,j) for i,j in zip(range(3), ['linecuts','color','export'])]

    # interactive funcion
    indx,indy = 0,0
    def _play2d(xpos,ypos,gamma,vlim):
        nonlocal indx,indy
        # initialize the figure
        fig, axs = plt.subplots(1,2,figsize=(7,2.5),dpi=120)#main plot and h linecut
        plt.subplots_adjust(wspace=0.4)
        axs[1].yaxis.tick_right()
        axs[1].tick_params(axis='x', colors='tab:orange')
        axs[1].tick_params(axis='y', colors='tab:orange')
        axv = fig.add_axes(axs[1].get_position(), frameon=False)#ax vertical linecut
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
    
    def _export(_):
        htmlexp.value = 'Saving...'
        fname = os.path.split(fPath)[1]
        fname = os.path.splitext(fname)[0]
        # vlincut
        fnamev = fname+'.vcut.%e.mtx'%x[0,indx]
        _saveMTX2d(fnamev,y0[np.newaxis],x[np.newaxis,:,indx],w[np.newaxis,:,indx],[labels[i] for i in [1,0,2]])
        # hlincut
        fnameh = fname+'.hcut.%e.mtx'%y[indy,0]
        _saveMTX2d(fnameh,x0[np.newaxis],y[[indy],:],w[[indy],:],labels)
        # 2d data
        fname2d = fname+'.mtx'
        _saveMTX2d(fname2d,x,y,w,labels)
        htmlexp.value = 'Files saved:<br>%s<br>%s<br>%s'%(fnamev,fnameh,fname2d)
        
    out = widgets.interactive_output(_play2d, {'xpos':sxpos,'ypos':sypos,'gamma':sgamma,'vlim':svlim})
    bexpMTX.on_click(_export)
    display(ui, out)

def _saveMTX2d(fpath,x,y,w,labels):
    with open(fpath, 'wb') as f:
        labels = [i.replace(',','_') for i in labels]#',' is forbidden
        xmin = x[0,0]
        xmax = x[0,-1]
        ymin = y[0,0]
        ymax = y[-1,0]
        ny, nx = np.shape(y)
        f.write(('Units, %s,%s, %s, %s,%s, %s, %s,None(qtplotter), 0, 1\n'%(labels[2],labels[0],xmin,xmax,labels[1],ymin,ymax)).encode())#data_label,x_label,xmin,xmax,ylabel,ymin,ymax
        f.write(('%d %d 1 %d\n'%(nx,ny,w.dtype.itemsize)).encode())#dimensions nx,ny,nz=1,data_element_size
        w.T.ravel().tofile(f)

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

    x1 = operation._get_quad(x)
    y1 = operation._get_quad(y)    
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

    if ps['useImshow']:#slightly different from pcolormesh, especially if saved as vector formats. Imshow is better if it works. See the links in operation._get_quad() description.
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

def _plot1d(x,y,w,**kw):
    '''A simple 1d plot function'''
    ps = {'labels':['','','']}
    for i in ps:
        if i in kw:
            ps[i] = kw[i]
    if 'ax' in kw and 'fig' in kw:
        fig = kw['fig']
        ax = kw['ax']
    else:
        fig, ax = plt.subplots(figsize=(3.375,2),dpi=120)
    ax.plot(x[0],w[0])
    ax.set_xlabel(ps['labels'][0])
    ax.set_ylabel(ps['labels'][1])
    

def simpAx(ax,cbar,im,n=(None,None,None),pad=(-5,-15,-10)):
    '''Simplify the ticks'''
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
