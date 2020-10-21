'''
It's a simpler, easier-to-access, notebook-based version of Rubenknex/qtplot. Most of the code is grabbed from qtplot.
The project is hosted on https://github.com/cover-me/qtplotter
'''
import os, sys, zipfile
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from urllib.request import urlopen
from scipy import ndimage
import ipywidgets as widgets
from collections import OrderedDict

# Print module versions
print('python:',sys.version)
print('matplotlib:', mpl.__version__)
print('numpy:', np.__version__)

# data operations
class Operation:
    '''
    A collection of static methods for data operation.
    Methods with names start with '_': auxiliary operations.
    The rest: filters who modify the data directly! No returned values.
    Data shape: Any shape. Usually, it looks like: [x,y,w,...] and each of x,y,w,... is a 2d matrix.
    '''
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
        y derivation, slightly different from qtplot
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
        return d
    
    @staticmethod
    def lowpass(d, x_width=0.5, y_height=0.5, method='gaussian'):
        """Perform a low-pass filter."""
        z = d[2]
        kernel = Operation._create_kernel(x_width, y_height, 7, method)
        z[:] = ndimage.filters.convolve(z, kernel)
        return d

    @staticmethod
    def scale(d,amp=[]):
        for i, ai in enumerate(amp):
            d[i] *= ai
        return d

    @staticmethod
    def offset(d,off=[]):
        for i, oi in enumerate(off):
            d[i] += oi
        return d

    @staticmethod
    def g_in_g2(d, rin):
        """z = z/(1-(z*Rin))/7.74809e-5. z: conductance in unit 'S', R in unit 'ohm' (SI units)"""
        G2 = 7.74809e-5#ohm^-1, 2e^2/h
        d[2] = d[2]/(1-(d[2]*rin))/G2
        return d

    @staticmethod
    def xy_limit(d,xmin=None,xmax=None,ymin=None,ymax=None):
        '''Crop data with xmin,xmax,ymin,ymax'''
        x = d[0]
        y = d[1]
        if not all([i is None for i in [xmin,xmax,ymin,ymax]]):
            x1 = 0 if xmin is None else np.searchsorted(x[0],xmin)
            x2 = -1 if xmax is None else np.searchsorted(x[0],xmax,'right')-1
            y1 = 0 if ymin is None else np.searchsorted(y[:,0],ymin)
            y2 = -1 if ymax is None else np.searchsorted(y[:,0],ymax,'right')-1
            return Operation.crop(d,x1,x2,y1,y2)
        else:
            return d

    @staticmethod
    def crop(d, left=0, right=-1, bottom=0, top=-1):
        """Crop data by indexes. First and last values included"""
        right = d[0].shape[1] + right + 1 if right < 0 else right + 1
        top = d[0].shape[0] + top + 1 if top < 0 else top + 1
        if (0 <= left < right <= d[0].shape[1] 
            and 0 <= bottom < top <= d[0].shape[0]):
            return d[:,bottom:top,left:right]
        else:
            raise ValueError('Invalid crop parameters: (%s,%s,%s,%s)'%(left,right,bottom,top))

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

# data loading/saving
class Data2d:
    '''
    A collection of static methods loading/saving 2d data.
    The data can be 1d, 2d or 3d.
    '''
    @staticmethod
    def myopen(fPath):
        if mpl.is_url(fPath):
            return lambda url,mode: urlopen(url)
        elif '.zip/' in fPath:
            return lambda url,mode: zipfile.ZipFile(url.split('.zip/')[0]+'.zip').open(url.split('.zip/')[1])
        else:
            return open
    
    @staticmethod
    def readMTX(fPath):
        '''
        read mtx files, which have the structure:
            Units, Dataset name, xname, xmin, xmax, yname, ymin, ymax, zname, zmin, zmax
            nx ny nz length
            [binary data....]
        mtx is created by Gary Steele, https://nsweb.tn.tudelft.nl/~gsteele/spyview/#mtx
        mtx files can be generated by spyview and qtplot
        mtx data is 3d (nx*ny*nz). However, we assume nz = 1 (which is always the case because we only get them from spyview and qtplot) for simplicity.
        '''
        open2 = Data2d.myopen(fPath)
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
                if shape[2] == 1:#assume nz=1
                    return x[0],y[0],w[0],[labels[0],labels[1],labels[3]]

    @staticmethod
    def readDat(fPath,cols=[0,1,3],cook=None,a3=2,a3index=0,**kw):#kw are uselss parameters
        '''
        read .dat files generated by qtlab. (structure: https://github.com/cover-me/qtplot#dat-file-qtlab)
        If data is taken from a 3d scan, use a3 and a3index to get a 2D slice. a3 stands for "the third axis" which is perpendicular to the slicing plane.
        '''
        sizes = []# nx,ny,nz for each dimension of scan. Default a 3d scan (1D and 2D are also kinds of 3D).
        labels = []# labelx,labely,labelw
        open2 = Data2d.myopen(fPath)
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
            # Reposition pointer at the beginning of the file
            f.seek(0, 0)
            # load data
            print('File: %s, cols: %s'%(os.path.split(fPath)[1],[labels[i] for i in cols]))
            d = np.loadtxt(f,usecols=cols)

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
        pivot = Operation.autoflip(pivot)

        if cook:
            pivot = cook(pivot)

        x,y,w = pivot[:3]
        return x,y,w,[labels[cols[i]] for i in range(3)]
    
    @staticmethod
    def saveMTX2d(fpath,x,y,z,labels,xyUniform):
        if not xyUniform:
            raise('Use MTX format only when x and y are uniformly sampled!')
        with open(fpath, 'wb') as f:
            labels = [i.replace(',','_') for i in labels]#',' is forbidden
            #make sure this is real min! Guaranteed by Operation.autoflip() when importing the data.
            xmin,xmax,ymin,ymax = x[0,0],x[0,-1],y[0,0],y[-1,0]
            ny, nx = np.shape(y)
            f.write(('Units, %s,%s, %s, %s,%s, %s, %s,None(qtplotter), 0, 1\n'%(labels[2],labels[0],xmin,xmax,labels[1],ymin,ymax)).encode())#data_label,x_label,xmin,xmax,ylabel,ymin,ymax
            f.write(('%d %d 1 %d\n'%(nx,ny,z.dtype.itemsize)).encode())#dimensions nx,ny,nz=1,data_element_size
            z.T.ravel().tofile(f)
            print('MTX data saved: %s'%fpath)

    @staticmethod
    def saveNPY2d(fpath,x,y,z,labels,xyUniform):
        if xyUniform:
            #make sure this is real min! Guaranteed by Operation.autoflip() when importing the data.
            xmin,xmax,ymin,ymax = x[0,0],x[0,-1],y[0,0],y[-1,0]
            np.save(fpath[:-3]+'z.npy',z)
            with open(fpath[:-3]+'meta.json', 'w') as f:
                metadata = {'labels':labels,'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax}
                json.dump(metadata, f)
        else:
            np.save(fpath[:-3]+'x.npy',x)
            np.save(fpath[:-3]+'y.npy',y)
            np.save(fpath[:-3]+'z.npy',z)
            with open(fpath[:-3]+'meta.json', 'w') as f:
                metadata = {'labels':labels}
                json.dump(metadata, f)
        print('NPY data saved: %s'%fpath)

    @staticmethod
    def readSettings(fpath):
        '''
        Read instrument settings. Copied and modified from Rubenknex/qtplot/qtplot/data.py.
        '''
        st = OrderedDict()
        settings_file = fpath.replace('.dat','.set')
        try:
            open2 = Data2d.myopen(settings_file)
            with open2(settings_file,'rb') as f:
                lines = f.readlines()
        except:
            print('Error opening setting file: %s'%settings_file)
            return st
        current_instrument = None
        for line in lines:
            line = line.decode()
            line = line.rstrip('\n\t\r')
            if line == '':
                continue
            if not line.startswith('\t'):
                name, value = line.split(': ', 1)
                if (line.startswith('Filename: ') or
                   line.startswith('Timestamp: ')):
                    st.update([(name, value)])
                else:#'Instrument: ivvi'
                    current_instrument = value
                    new = [(current_instrument, OrderedDict())]
                    st.update(new)
            else:
                param, value = line.split(': ', 1)
                param = param.strip()
                new = [(param, value)]
                st[current_instrument].update(new)
        return st

def read2d(fPath,**kw):
    '''
    Supported data types:
    .mtx, .dat
    1d, 2d, 3d scan. Returned data is 2d (slicing parameters are required if the file contains 3d data).
    local file, url
    '''
    if fPath.endswith('.mtx'):
        x,y,w,labels = Data2d.readMTX(fPath)
    elif fPath.endswith('.dat'):
        x,y,w,labels = Data2d.readDat(fPath,**kw)
    else:
        return
    return x,y,w,labels

def save2d(fPath,x,y,w,labels,xyUniform):
    if fPath.endswith('.mtx'):
        Data2d.saveMTX2d(fPath,x,y,w,labels,xyUniform)
    elif fPath.endswith('.npz'):
        Data2d.saveNPY2d(fPath,x,y,w,labels,xyUniform)
    else:
        raise('Format not recognized.')

class Data1d:
    '''
    A collection of static methods loading/saving 2d data.
    The data can be 1d, 2d or 3d.
    '''
    @staticmethod
    def saveNPZ1d(fPath,x,y,labels):
        np.savez(fPath,**{labels[0]:x,labels[1]:y})
        print('NPZ data saved.')

def save1d(fPath,x,y,labels):
    if fPath.endswith('.npz'):
        Data1d.saveNPZ1d(fPath,x,y,labels)
    else:
        raise('Format not recognized.')

# plot
class Painter:
    '''
    Static methods for 2d or 1d painting
    '''
    @staticmethod
    def get_default_ps():
        '''
        return a defult plot setting
        '''
        return {'labels':['','',''],'xyUniform':True,'gamma':0,'gmode':'moveColor',
              'cmap':'seismic','vmin':None, 'vmax':None,'plotCbar':True}
    @staticmethod
    def plot2d(x,y,w,**kw):
        '''
        Plot 2D figure. We need this method because plotting 2d is not as easy as plotting 1d.
        imshow() and pcolormesh() should be used in different situations. imshow() is prefered if x and y are uniformly spaced.
        For some interesting issues, check these links:
        https://cover-me.github.io/2019/02/17/Save-2d-data-as-a-figure.html
        https://cover-me.github.io/2019/04/04/Save-2d-data-as-a-figure-II.html
        '''
        #plot setting
        ps = Painter.get_default_ps()
        for i in ps:
            if i in kw:
                ps[i] = kw[i]

        #save fig data
        if 'figdatato' in kw and kw['figdatato']:
            save2d(kw['figdatato'],x,y,w,ps['labels'],ps['xyUniform'])

        if 'ax' in kw and 'fig' in kw:
            # sometimes you want to use your own ax
            fig = kw['fig']
            ax = kw['ax']
        else:
            # Publication quality first. Which means you don't want large figures with small fonts as those default figures.
            # dpi is set to 120 so the figure is enlarged for the mornitor.
            figsize = kw['figsize'] if 'figsize' in kw else (3.375,2)
            dpi = kw['dpi'] if 'dpi' in kw else 120
            fig, ax = plt.subplots(figsize=figsize,dpi=dpi)

        x1 = Operation._get_quad(x)# explained here: https://cover-me.github.io/2019/02/17/Save-2d-data-as-a-figure.html
        y1 = Operation._get_quad(y)
        imkw = {'cmap':ps['cmap'],'vmin':ps['vmin'],'vmax':ps['vmax']}
        gamma_real = 10.0**(ps['gamma'] / 100.0)# to be consistent with qtplot
        if gamma_real != 1:
            if ps['gmode']=='moveColor':# qtplot style
                imkw['cmap'] = Painter._get_cmap_gamma(imkw['cmap'],gamma_real,1024)
            else:# matplotlib default style
                imkw['norm'] = mpl.colors.PowerNorm(gamma=gamma_real)
        
        #Imshow is better than pcolormesh if it xy is uniformly spaced. See the links in operation._get_quad() description.
        if ps['xyUniform']:
            #data need to be autoflipped when imported
            xy_range = (x1[0,0],x1[0,-1],y1[0,0],y1[-1,0])
            im = ax.imshow(w,aspect='auto',interpolation='none',origin='lower',extent=xy_range,**imkw)
            #clip the image a little to set xy limits to the real numbers
            dx = x1[0,1]-x1[0,0]
            dy = y1[1,0]-y1[0,0]
            ax.set_xlim(xy_range[0]+dx/2.,xy_range[1]-dx/2.)
            ax.set_ylim(xy_range[2]+dy/2.,xy_range[3]-dy/2.)
        else:
            im = ax.pcolormesh(x1,y1,w,**imkw)

        if ps['plotCbar']:
            cbar = fig.colorbar(im,ax=ax)
            cbar.set_label(ps['labels'][2])
        else:
            cbar = None
        ax.set_xlabel(ps['labels'][0])
        ax.set_ylabel(ps['labels'][1])

    @staticmethod
    def plot1d(x,y,w,**kw):
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
    
    @staticmethod
    def _get_cmap_gamma(cname,g,n=256):
        '''Get a listed cmap with gamma'''
        cmap = mpl.cm.get_cmap(cname, n)
        cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 1, n)**g))
        return cmap
    
    @staticmethod
    def simpAx(ax=None,cbar=None,im=None,n=(None,None,None),apply=(True,True,True),pad=(-5,-15,-10)):
        '''Simplify the ticks'''
        if ax is None:
            ax = plt.gca()
        
        if apply[0]:
            _min,_max = ax.get_xlim()
            if n[0] is not None:
                a = 10**(-n[0])
                _min = np.ceil(_min/a)*a
                _max = np.floor(_max/a)*a
            ax.set_xticks([_min,_max])
            ax.xaxis.labelpad = pad[0]

        if apply[1]:
            _min,_max = ax.get_ylim()
            if n[1] is not None:
                a = 10**(-n[1])
                _min = np.ceil(_min/a)*a
                _max = np.floor(_max/a)*a
            ax.set_yticks([_min,_max])
            ax.yaxis.labelpad = pad[1]
            
        if apply[2]:
            #assumes a vertical colorbar
            if cbar is None:
                if im:
                    cbar = im.colorbar
                else:
                    ims = [obj for obj in ax.get_children() if isinstance(obj, mpl.image.AxesImage) or isinstance(obj,mpl.collections.QuadMesh)]
                    if ims:
                        im = ims[0]
                        cbar = im.colorbar
                    else:
                        im,cbar = None, None
            if cbar is not None and im is not None:
                _min,_max = cbar.ax.get_ylim()
                label = cbar.ax.get_ylabel()
                if n[2] is not None:
                    a = 10**(-n[2])
                    _min = np.ceil(_min/a)*a
                    _max = np.floor(_max/a)*a
                    #im.set_clim(_min,_max)
                cbar.set_ticks([_min,_max])
                cbar.ax.yaxis.labelpad = pad[2]
                cbar.ax.set_ylabel(label)  
            
    @staticmethod
    def formatLabel(fname,s):
        '''
        see qtplot/export.py
        '''
        conversions = {
            '<filename>': os.path.split(fname)[-1],
            '<operations>': '',
        }
        for old, new in conversions.items():
            s = s.replace(old, new)
        for key, value in Data2d.readSettings(fname).items():
            if isinstance(value, dict):
                for key_, value_ in value.items():
                    s = s.replace('<%s:%s>'%(key,key_), '%s'%value_)
        return s

def plot(fPath,**kw):
    '''Generate a 2d or 1d plot with customize parameters'''
    x,y,w,labels = read2d(fPath,**kw)
    if 'labels' not in kw:
        kw['labels'] = labels
    if len(x)==1:#1d data
        Painter.plot1d(x,y,w,**kw)
    else:
        Painter.plot2d(x,y,w,**kw)

# play
def play(path_or_url,**kw):
    Player(path_or_url,**kw)
        
class Player:
    
    # CONSTANTS
    PLAYER_STATIC = False # set it True to make Player generate static figures (for previewing in viewers).
    
    def __init__(self,path_or_url,**kw):
        # data
        self.path = path_or_url
        x,y,w,labels = read2d(path_or_url,**kw)
        if 'labels' not in kw:
            kw['labels'] = labels
        
        if len(y)==1:#1d data
            x = np.vstack([x,x,x])
            y = np.vstack([y-.5,y,y+.5])
            w = np.vstack([w,w,w])

        self.x = x
        self.y = y
        self.w = w
        self.kw = kw
        
        self.create_ui()
        self.draw(event=None)
        
        if mpl.get_backend() == 'module://ipympl.backend_nbagg':#ipympl backend. Not good at this moment. But faster
            self.s_gamma.observe(self.on_gamma_change,'value')
            self.s_vlim.observe(self.on_vlim_change,'value')
            self.c_cmap.observe(self.on_cmap_change,'value')
            self.s_xpos.observe(self.on_xpos_change,'value')
            self.s_ypos.observe(self.on_ypos_change,'value')
            self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        else:# inline mode
            self.s_gamma.observe(self.draw,'value')
            self.s_vlim.observe(self.draw,'value')
            self.c_cmap.observe(self.draw,'value')
            self.s_xpos.observe(self.draw,'value')
            self.s_ypos.observe(self.draw,'value')
        self.tb_showtools.observe(self.on_showtools_change,'value')
        self.b_expMTX.on_click(self.exportMTX)
        
    def create_ui(self):
        x,y,w = self.x,self.y,self.w
        x0 = x[0]
        y0 = y[:,0]
        xmin,xmax,dx = x[0,0],x[0,-1],x[0,1]-x[0,0]
        ymin,ymax,dy = y[0,0],y[-1,0],y[1,0]-y[0,0]
        wmin,wmax = np.min(w),np.max(w)
        dw = (wmax-wmin)/20
        
        ## Tab of tools
        self.s_xpos = widgets.FloatSlider(value=(xmin+xmax)/2,min=xmin,max=xmax,step=dx,description='x')
        self.s_ypos = widgets.FloatSlider(value=(ymin+ymax)/2,min=ymin,max=ymax,step=dy,description='y')
        vb1 = widgets.VBox([self.s_xpos,self.s_ypos])
        self.s_gamma = widgets.IntSlider(value=0,min=-100,max=100,step=10,description='gamma')
        self.s_vlim = widgets.FloatRangeSlider(value=[wmin,wmax], min=wmin, max=wmax, step=dw, description='vlim')
        self.c_cmap = widgets.Combobox(value='', placeholder='Choose or type', options=plt.colormaps(), description='cmap:', ensure_option=False, disabled=False)
        vb2 = widgets.VBox([self.s_gamma,self.s_vlim,self.c_cmap])
        self.b_expMTX = widgets.Button(description='To mtx')
        self.html_exp = widgets.HTML()
        vb3 = widgets.VBox([self.b_expMTX,self.html_exp])
        self.t_tools = widgets.Tab(children=[vb1,vb2,vb3])
        [self.t_tools.set_title(i,j) for i,j in zip(range(3), ['linecuts','color','export'])]
        self.t_tools.layout.display = 'none'
        ## A toggle button
        self.tb_showtools = widgets.ToggleButton(value=False, description='...', tooltip='Toggle Tools', icon='plus-circle')
        self.tb_showtools.layout.width='50px'
        ## Top layer ui
        ui = widgets.Box([self.t_tools,self.tb_showtools])
        self.out = widgets.Output()        
        if 'gamma' in self.kw:
            self.s_gamma.value = self.kw['gamma']
        if 'vlim' in self.kw:
            self.s_vlim.value = self.kw['vlim']
        if 'cmap' in self.kw:
            self.c_cmap.value = self.kw['cmap']
            
        if Player.PLAYER_STATIC:
            from IPython.core.display import HTML
            display(HTML('<button style="border:none;" title="For interaction, run the cell first.">+...</button>'))
        else:
            display(ui,self.out)
            
        
    def draw(self,event):
        # axs
        fig, axs = plt.subplots(1,2,figsize=(6.5,2.5),dpi=100)#main plot and h linecut
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False
        plt.subplots_adjust(wspace=0.4,bottom=0.2)
        axs[1].yaxis.tick_right()
        axs[1].tick_params(axis='x', colors='tab:orange')
        axs[1].tick_params(axis='y', colors='tab:orange')
        axv = fig.add_axes(axs[1].get_position(), frameon=False)#ax vertical linecut
        axv.xaxis.tick_top()
        axv.tick_params(axis='x', colors='tab:blue')
        axv.tick_params(axis='y', colors='tab:blue')
        self.fig = fig
        self.ax = axs[0]
        self.axv = axv
        self.axh = axs[1]

        # heatmap
        g = self.s_gamma.value
        v0,v1 = self.s_vlim.value
        cmap = self.c_cmap.value
        if cmap not in plt.colormaps():
            cmap = 'seismic'
        self.kw['gamma'],self.kw['vmin'],self.kw['vmax'],self.kw['cmap']=g,v0,v1,cmap
        Painter.plot2d(self.x,self.y,self.w,fig=self.fig,ax=self.ax,**self.kw)
        self.im = [obj for obj in self.ax.get_children() if isinstance(obj, mpl.image.AxesImage) or isinstance(obj,mpl.collections.QuadMesh)][0]

        # vlinecut
        x0 = self.x[0]
        y0 = self.y[:,0]
        xpos = self.s_xpos.value
        indx = np.abs(x0 - xpos).argmin()# x0 may be a non uniform array
        [self.linev1] = axs[0].plot(self.x[:,indx],y0,'tab:blue')
        [self.linev2] = axv.plot(self.w[:,indx],y0,'tab:blue')
        self.indx = indx

        # hlinecut
        ypos = self.s_ypos.value
        indy = np.abs(y0 - ypos).argmin()
        [self.lineh1] = axs[0].plot(x0,self.y[indy,:],'tab:orange')
        [self.lineh2] = axs[1].plot(x0,self.w[indy,:],'tab:orange')
        self.indy = indy
        if Player.PLAYER_STATIC or mpl.get_backend() == 'module://ipympl.backend_nbagg':
            plt.show()
        else:
            with self.out:
                plt.show()
                self.out.clear_output(wait=True)
        
    def on_gamma_change(self,change):
        cmpname = self.c_cmap.value
        if cmpname not in plt.colormaps():
            cmpname = 'seismic'
        g = change['new']
        g = 10.0**(g / 100.0)# to be consistent with qtplot
        if g!= 1:
            self.im.set_cmap(Painter._get_cmap_gamma(cmpname,g,1024))
        else:
            self.im.set_cmap(cmpname)

    def on_cmap_change(self,change):
        cmap = change['new']
        if cmap in plt.colormaps():
            self.im.set_cmap(cmap)
    
    def on_vlim_change(self,change):
        v0,v1 = change['new']
        self.im.set_clim(v0,v1)
    
    def on_xpos_change(self,change):
        xpos = change['new']
        x0 = self.x[0]
        indx = np.abs(x0 - xpos).argmin()# x0 may be a non uniform array
        self.linev1.set_xdata(self.x[:,indx])
        self.linev2.set_xdata(self.w[:,indx])
        self.axv.relim()
        self.axv.autoscale_view()
        self.indx = indx

    def on_ypos_change(self,change):
        ypos = change['new']
        y0 = self.y[:,0]
        indy = np.abs(y0 - ypos).argmin()# x0 may be a non uniform array
        self.lineh1.set_ydata(self.y[indy,:])
        self.lineh2.set_ydata(self.w[indy,:])
        self.axh.relim()
        self.axh.autoscale_view()
        self.indy = indy

    def on_showtools_change(self,change):
        if change['new']:
            self.t_tools.layout.display = 'block'
            self.tb_showtools.icon = 'minus-circle'
            self.tb_showtools.description = ''
        else:
            self.t_tools.layout.display = 'none'
            self.tb_showtools.icon = 'plus-circle'
            self.tb_showtools.description = '...'

    def on_mouse_click(self,event):
        x,y = event.xdata,event.ydata
        if self.s_xpos.value != x:
            self.on_xpos_change({'new':x})
        if self.s_ypos.value != y:
            self.on_ypos_change({'new':y})
    
    def exportMTX(self,_):
        self.html_exp.value = 'Saving...'
        fname = os.path.split(self.path)[1]
        fname = os.path.splitext(fname)[0]
        x = self.x
        y = self.y
        w = self.w
        x0 = x[0]
        y0 = y[:,0]
        labels = self.kw['labels']
        # vlincut
        fnamev = fname+'.vcut.%e.mtx'%x[0,self.indx]
        Data2d.saveMTX2d(fnamev,y0[np.newaxis],x[np.newaxis,:,self.indx],w[np.newaxis,:,self.indx],[labels[i] for i in [1,0,2]])
        # hlincut
        fnameh = fname+'.hcut.%e.mtx'%y[self.indy,0]
        Data2d.saveMTX2d(fnameh,x0[np.newaxis],y[[self.indy],:],w[[self.indy],:],labels)
        # 2d data
        fname2d = fname+'.mtx'
        Data2d.saveMTX2d(fname2d,x,y,w,labels)
        self.html_exp.value = 'Files saved:<br>%s<br>%s<br>%s'%(fnamev,fnameh,fname2d)
    
    @staticmethod    
    def play_inline(fPath,**kw):
        '''
        Obsolete.
        Use inline mode because at this moment ipympl is not good enough for notebook.
        Generate an interactive 2D plot to play with
        x,y must be uniform spaced, autoflipped.
        '''
        # Data
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
            fig, axs = plt.subplots(1,2,figsize=(6.5,2.5),dpi=120)#main plot and h linecut
            plt.subplots_adjust(wspace=0.4)
            axs[1].yaxis.tick_right()
            axs[1].tick_params(axis='x', colors='tab:orange')
            axs[1].tick_params(axis='y', colors='tab:orange')
            axv = fig.add_axes(axs[1].get_position(), frameon=False)#ax vertical linecut
            axv.xaxis.tick_top()
            axv.tick_params(axis='x', colors='tab:blue')
            axv.tick_params(axis='y', colors='tab:blue')
            # plot 2D data
            kw = {}
            kw['gamma'],kw['vmin'],kw['vmax']=gamma,vlim[0],vlim[1]
            Painter.plot2d(x,y,w,fig=fig,ax=axs[0],**kw)
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
            Data2d.saveMTX2d(fnamev,y0[np.newaxis],x[np.newaxis,:,indx],w[np.newaxis,:,indx],[labels[i] for i in [1,0,2]])
            # hlincut
            fnameh = fname+'.hcut.%e.mtx'%y[indy,0]
            Data2d.saveMTX2d(fnameh,x0[np.newaxis],y[[indy],:],w[[indy],:],labels)
            # 2d data
            fname2d = fname+'.mtx'
            Data2d.saveMTX2d(fname2d,x,y,w,labels)
            htmlexp.value = 'Files saved:<br>%s<br>%s<br>%s'%(fnamev,fnameh,fname2d)

        out = widgets.interactive_output(_play2d, {'xpos':sxpos,'ypos':sypos,'gamma':sgamma,'vlim':svlim})
        bexpMTX.on_click(_export)
        display(ui, out)
