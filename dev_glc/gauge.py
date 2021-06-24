from mpi4py import MPI
import timeit
import numpy as np
import heat as ht
import torch
import abc


def ReadLimeData(f, nt=128, nx=16, **kwargs):
    """
    需要检查偏移量是否正确,另外Lime的格式可能是Big endian
    """
    data = np.fromfile(f, **kwargs)
    num_data = nx**3 * nt * 3 * 3 * 4 * 2
    return data[-num_data-18-17:-18-17]


def CalcCorTime(files, nt, nx,ntau=30):
    """
    计算自关联时间
    """
    tmp = gauge(nt, nx)
    plaq = []
    for i in files:
        tmp.ReadClqcdConf(i)
        tmp.CalcPlaq()
        plaq.append(np.average(tmp.plaq, 0))
    plaq = np.real(np.array(plaq))
    Corr = np.array([np.average(plaq * np.roll(plaq, -i, 0), 0)-np.average(plaq, 0)
                     * np.average(np.roll(plaq, -i, 0), 0) for i in range(ntau)])
    Corr / np.expand_dims(Corr[0,...],0)
    tau = 1/2 + np.sum(Corr,0)
    return (tau,Corr) 


class gauge(object):

    def __init__(self, nt, nx, **kwargs):
        self.nc = 3
        self.ns = 1
        self.ndim = 4
        self.nri = 2
        self.nt = nt
        self.nx = nx
        self.u = np.zeros((self.nt, self.nx, self.nx, self.nx,
                           self.ndim, self.ns, self.ns, self.nc, self.nc), **kwargs)
        self.shape = self.u.shape

    def ReadClqcdConf(self, file, **kwargs):
        """
        CLQCD组态的顺序是nmu = 4，nc_col = 3，nc_row = 3，nri = 2，nt ,nz , ny, nx 
        我们的数据格式 nt,nz,ny,nx,ndim,ns,ns,nc_row,nc_col
        """
        tmp = np.fromfile(file, **kwargs).reshape(self.ndim, self.nc,
                                                  self.nc, self.nri, self.nt, self.nx, self.nx, self.nx)
        self.u = np.transpose(tmp[:, :, :, 0, ...]+1j*tmp[:, :,
                                                          :, 1, ...], (3, 4, 5, 6, 0, 2, 1)).reshape(self.u.shape)

    def ReadLimeConf(self, file, **kwargs):
        """
        Lime组态的顺序是nt ,nz , ny, nx ,nmu = 4，nc_row = 3，nc_col = 3，nri = 2， 
        且是big endian  需要传入">f8"
        我们的数据格式 nt,nz,ny,nx,ndim,ns,ns,nc_row,nc_col
        """
        tmp_data = np.reshape(ReadLimeData(file, self.nt, self.nx, **kwargs), (self.nt, self.nx,
                                                                               self.nx, self.nx, self.ndim, self.nc, self.nc, self.nri))
        tmp_data2 = tmp_data[..., 0]+1j*tmp_data[..., 1]
        self.u = tmp_data2.reshape(self.shape)

    def CalcPlaq(self):
        tmp = self.u
        self.plaq = np.zeros((self.nt, 4, 4), "c16")
        for inu in range(1, self.ndim):
            for imu in range(0, inu):
                plaq_1 = np.einsum("...ij,...jk->...ik", tmp[..., inu, :, :, :, :], np.roll(
                    tmp[..., imu, :, :, :, :], shift=-1, axis=(3 - inu)))
                tmp_dag = np.conjugate(np.moveaxis(tmp, -1, -2))
                plaq_2 = np.einsum("...ij,...jk->...ik", np.roll(
                    tmp_dag[..., inu, :, :, :, :], -1, axis=(3 - imu)), tmp_dag[..., imu, :, :, :, :])
                plaq = np.einsum("...ij,...ji->...", plaq_1, plaq_2) / 3
                self.plaq[:, inu, imu] = np.average(plaq, (1, 2, 3, 4, 5))
        print(np.sum(np.average(self.plaq,0))/6)
