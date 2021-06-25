#!/bin/env python
# -*- coding: utf-8 -*-
from sympy import LeviCivita
from scipy.optimize import curve_fit
import scipy as sp
import numpy as np
import time
PackageLost = []
try:
    import xarray as xr
except ImportError:
    print("package xarray can't find!")
    PackageLost.append("xarray")
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    print("matplotlib package not find")
    PackageLost.append("matplotlib")
try:
    import gvar as gv
    import corrfitter as cf
    import lsqfit
except ImportError:
    print("lsqfit package not find")
    PackageLost.append("lsqfit")
try:
    import bottleneck
except ImportError:
    print("bottleneck package not find")
    PackageLost.append("bottleneck")

endian = "<"

typename = np.array(["other", "x", "y", "z", "t", "d", "c", "d2",
                     "c2", "complex", "mass", "smear", "displacement",
                     "s_01", "s_02", "s_03", "s_11", "s_12", "s_13",
                     "d_01", "d_02", "d_03", "d_11", "d_12",
                     "d_13", "conf", "operator", "momentum", "direction",
                     "t2", "mass2", "column", "row",
                     "temporary", "temporary2", "temporary3", "temporary4",
                     "errorbar", "operator2", "param",
                     "fit_left", "fit_right", "jackknife", "jackknife2",
                     "jackknife3", "jackknife4", "summary",
                     "channel", "channel2", "eigen", "d_row", "d_col",
                     "c_row", "c_col", "parity", "noise",
                     "evenodd", "disp_x", "disp_y", "disp_z", "disp_t",
                     "t3", "t4", "t_source", "t_current", "t_sink",
                     "bootstrap", "nothing"])

HeadType = np.dtype(
    [('head',
      [('n_dims', 'i4'), ('one_dim',
                          [('type', 'i4'), ('n_indices', 'i4'), ('indices', 'i4', 1024)
                           ], 16)
       ])
     ]
)


def ReadIog(filename):
    """
    读入iog格式文件名，返回数据，维度坐标，维度名
    """
    head_data = np.fromfile(filename, dtype=HeadType, count=1)[0]
    DataType_ = np.dtype([
        ('head_tmp', 'S102400'),
        ('data', endian + 'f8', tuple(head_data['head']['one_dim']['n_indices']
                                      [0:head_data['head']['n_dims']]))
    ])
    data = np.fromfile(filename, dtype=DataType_)[0]['data']
    head_data = np.fromfile(filename, dtype=HeadType, count=1)[0]
    n_dims_ = head_data['head']['n_dims']
    type_ = head_data['head']['one_dim']['type'][0:n_dims_]
    typename_ = [typename[i] for i in type_]
    n_indices_ = tuple(head_data['head']['one_dim']['n_indices'][0:n_dims_])
    indices_ = head_data['head']['one_dim']['indices'][0:n_dims_]
    coords_ = {}
    for i in range(n_dims_):
        coords_[typename_[i]] = indices_[i][0:n_indices_[i]]
    return data, coords_, typename_


def ReadBin(filename, coords, dims, dtype=np.complex128):
    """filename:echo conf filename
        coords={"conf":value,...}
        dims=["conf","dim2",...]
        if file have lbin,then dims should include lbin
        """
    data_shape = [len(coords[dims[i]]) for i in range(len(dims))]
    if "lbin" in dims:
        dims.remove("lbin")
        del coords["lbin"]
        new_data_shape = [len(coords[dims[i]]) for i in range(len(dims))]
    else:
        new_data_shape = data_shape

    if type(filename) is list:
        iodata = np.prod(data_shape)*16
        data = np.zeros(new_data_shape, dtype=dtype)
        time = Time()
        time.start("readdata")
        if "lbin" in dims:
            for i in range(len(filename)):
                data[i, ...] = np.average(np.reshape(np.fromfile(filename[i], dtype),
                                                     tuple(data_shape[1:])), dims.index["lbin"]-1)
        else:
            for i in range(len(filename)):
                data[i, ...] = np.reshape(np.fromfile(
                    filename[i], dtype), tuple(data_shape[1:]))
        time.end("readdata", iodata=iodata)
    else:
        data = np.reshape(np.fromfile(filename, dtype), tuple(data_shape))
    return data, coords, dims


def Variation(data_, t0=0, t1=1, flag="glc"):
    """input:data(ncon,nt,op_i,op_j),t0
    return e_value(ncon,nt,op_i),e_vec(ncon,nt,op_i,op_j) """
    # force data to be Hermitian
    data = 0.5 * (data_.real + np.swapaxes(data_.real, -1, -2))
    if "glc" in flag:
        L = np.linalg.cholesky(np.average(data[:, t0, :, :], 0))
        g_vec = np.zeros(data.shape)
        values = np.zeros(data.shape[0:-1])
        for iconf in np.arange(data.shape[0]):
            for it in np.arange(data.shape[1]):
                values[iconf, it, ...], g_vec[iconf, it, ...] = np.linalg.eigh(np.dot(np.dot(np.linalg.inv(L[...]), data[iconf, it, ...]),
                                                                                      np.linalg.inv(np.transpose(np.conjugate(L[...])))))

    if "eig" in flag:
        if len(data.shape) == 3:
            values, g_vec = sp.linalg.eig(data[t1, ...], data[t0, ...])
        if len(data.shape) == 4:
            values, g_vec = sp.linalg.eig(np.average(
                data[:, t1, ...], 0), np.average(data[:, t0, ...], 0))
        order = np.argsort(values)
        g_vec = g_vec[..., order]
        values = values[order]

    if "eigh" in flag:
        if len(data.shape) == 3:
            values, g_vec = sp.linalg.eigh(data[t1, ...], data[t0, ...])
        if len(data.shape) == 4:
            values, g_vec = sp.linalg.eigh(np.average(
                data[:, t1, ...], 0), np.average(data[:, t0, ...], 0))
        order = np.argsort(values)
        g_vec = g_vec[..., order]
        values = values[order]

    return values, g_vec


def Jackknife(data):
    nconf = len(data.coords["conf"])
    aver = data.mean("conf")
    jdata = (nconf * aver - data) / (nconf - 1)
    error = jdata.std("conf") * (nconf - 1) ** 0.5
    return jdata, aver, error


def EffectiveMass(data, flag='log', dim="t"):
    """ effectmass exp """
    print("effectivemass use ", flag, " to calc")
    nconf = len(data.coords["conf"])
    if 'log' in flag:
        tmp = np.log(data.roll(roll_coords=False, **{dim: 1}) / data)
    elif 'cosh' in flag:
        tmp = np.arccosh((data.roll(
            roll_coords=False, **{dim: 1})+data.roll(roll_coords=False, **{dim: -1}))/(2.0*data))
    aver_mass = tmp.mean("conf")
    error_mass = tmp.std("conf") * (nconf - 1) ** (0.5)
#    error_mass = tmp.std("conf") / (nconf - 1) ** (0.5)
    return aver_mass, error_mass


def ErrorBarPlot(data, error, choice=[{}], plot_params={}, plot_dim="t", filename=None):
    plt.figure()
    plt.xlabel(plot_dim)
    plt.title('Effective Mass')
    plt.xlabel('$\\hat{%s}$' % plot_dim)
    plt.ylabel('Mass')
    for key, value in plot_params["setup"].items():
        if isinstance(value, str):
            exec("plt.%s('%s')" % (key, value))
        else:
            exec("plt.%s(%s)" % (key, value))
    j = 0
    for i in choice:
        # print(i)
        if "label" in plot_params.keys():
            plt.errorbar(data.coords[plot_dim][i[plot_dim]], data.loc[i], error.loc[i],
                         fmt='.', label=plot_params["label"][j])
        else:
            plt.errorbar(data.coords[plot_dim][i[plot_dim]], data.loc[i], error.loc[i],
                         fmt='.', label="%s" % ("".join(["%s=%s," % (key, value) for key, value in i.items() if key is not plot_dim])))
        j = j+1
    plt.grid()
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def GetDataTagFromDict(dict_):
    return "".join([str(i) for i in dict_.values()])
def remove_key(dict_,key):
    r = dict(dict_)
    del r[key]
    return r

class Time(object):

    """used to calculate codes run time """

    def __init__(self):
        """ """
        self.time = {}

    def start(self, arg1="all"):
        """TODO: Docstring for start.

        :arg1: time name
        :returns: current time

        """
        self.time[arg1] = time.time()

    def end(self, arg1="all", iodata=0, Ncal=0):
        """TODO: Docstring for end.

        :arg1: time name
        :iodata: N Byte of data
        :Ncal: N of flops
        :returns: this time name all used times.

        """
        self.time[arg1] = time.time() - self.time[arg1]
        print(20*"=", "time")
        print("%s time  used is : %fs" % (arg1, self.time[arg1]))
        if Ncal != 0:
            print("%s Gflops is : %f" % (arg1, Ncal/(1024**3)/self.time[arg1]))
        if iodata != 0:
            print("%s io is %f GB/s" %
                  (arg1, iodata/(1024**3)/self.time[arg1]))
        print(20*"=")


class Corr(xr.Dataset):
    def __init__(self):
        super(Corr, self).__init__()
        self.gvdata={}
        
    def ReadData(self, filename, coords={}, dims=[], dtype=np.complex128, keywords=None, formatflag=""):
        """ReadData from files. filename can be a strings or list
        format can be  npy,npz,io_g(io_general),b(binary),chroma ...

        :filename: TODO
        :formatflag: TODO
        :returns: TODO
        """
        if keywords is None:
            keywords = "corr"

        if type(filename) is list:
            if "iog" in formatflag or filename[0].endswith("iog"):
                tmp = xr.DataArray()
                for i in range(len(filename)):
                    data, coords_, typename_ = ReadIog(filename[i])
                    tmp = tmp.combine_first(xr.DataArray(
                        data, coords=coords_, dims=tuple(typename_)))
                self[keywords] = tmp
            elif "bin" in formatflag or filename[0].endswith(".dat") or filename[0].endswith(".bin"):
                data, coords_, typename_ = ReadBin(
                    filename, coords, dims, dtype=dtype)
                self[keywords] = xr.DataArray(
                    data, coords=coords_, dims=tuple(typename_))
        else:
            if "iog" in formatflag or filename.endswith("iog"):
                data, coords_, typename_ = ReadIog(filename)
                self[keywords] = xr.DataArray(
                    data, coords=coords_, dims=tuple(typename_))
            elif "bin" in formatflag or filename.endswith(".dat") or filename.endswith(".bin"):
                data, coords_, typename_ = ReadBin(
                    filename, coords, dims, dtype=dtype)
                self[keywords] = xr.DataArray(
                    data, coords=coords_, dims=tuple(typename_))
            elif "nc" in formatflag or filename.endswith("nc"):
                self[keywords] = xr.open_dataarray(filename)
            elif "npy" in formatflag or filename.endswith("npy"):
                data = np.load(filename).reshape(
                    [len(coords[i]) for i in dims])
                self[keywords] = xr.DataArray(data, coords=coords, dims=dims)
            else:
                print("data format not recognized!")

    def Jackknife(self, keywords=None):
        if keywords is None:
            keywords = "corr"
            jdata = "jcorr"
            aver_data = "aver_corr"
            error_data = "error_corr"
        else:
            jdata = "j"+keywords
            aver_data = "aver" + keywords
            error_data = "error" + keywords

        self[jdata], self[aver_data], self[error_data] = Jackknife(
            self[keywords])

    def EffectiveMass(self, flag, dim=None):
        if dim is None:
            dim = "t"
            jdata = "jcorr"
            aver_mass = "aver_mass_t"
            error_mass = "error_mass_t"
        else:
            jdata = "jcorr"
            aver_mass = "aver_mass_" + dim
            error_mass = "error_mass_" + dim
        self[aver_mass], self[error_mass] = EffectiveMass(
            self[jdata], flag, dim)

    def ShowData(self, plotarray="mass_t", choice=[{}], plot_params={}, plot_dim="t", filename=None):
        aver = "aver_" + plotarray
        error = "error_" + plotarray
        ErrorBarPlot(self[aver], self[error], choice,
                  plot_params, plot_dim, filename)

# 或许应该在这里写一个提前产生gvar数据的字典，每个关键字即为datatag。这样后续可以直接在该字典内添加需要拟合的数据。

    def Fit(self, fitparsC2=[{}], fitparsC3=[{}], dim="t", fitdata="corr", prior=None, filename=None,binsize=None, SVDCUT=10**-12,flag=""):
        self.gvdata.update({GetDataTagFromDict(i["data"]): gv.dataset.avg_data(
            self[fitdata].loc[i["data"]].transpose("conf", dim)) for i in fitparsC2 + fitparsC3 if "data" in i })
        [i.update({"datatag": GetDataTagFromDict(i["data"])})
         for i in fitparsC2 + fitparsC3 if "data" in i]
        models = [cf.Corr2(**remove_key(i,"data")) for i in fitparsC2 if len(i)>0] + \
            [cf.Corr3(**remove_key(i,"data")) for i in fitparsC3 if len(i) > 0]
        if "fastfit" in flag: 
            for value in self.gvdata.values():
                fastfit = cf.fastfit(G=value)
                print(30 * "=","fastfit")
                print(fastfit,"\n")

        fitter = cf.CorrFitter(models=models)
        p0 = None
        N = max([len(i) for i in prior.values()])
        for i in range(1,N+1):
            tmp_prior={"%s"%j:k[:i] for j,k in prior.items()}
            fit = fitter.lsqfit(data=self.gvdata, prior=tmp_prior, p0=p0, svdcut=SVDCUT)
            p0 = fit.pmean
            print(30 * "=","lsqfit N = %i"%i)
            print(fit.formatall())
        self.fit=fit
        if filename is not None:
            fit.show_plots(save=filename,view="std")
        else:
            fit.show_plots(view="std")
