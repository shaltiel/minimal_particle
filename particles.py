import sys
sys.path.append('/Developer/NVIDIA/CUDA-8.0/bin')
# coding: utf-8

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import kernels as gpu

dtype=np.float32
class particles:
    def __init__(self, n,L, CUDA=True):
        self.dtype = dtype
        self.r = np.zeros([n,3], dtype=dtype);
        self.v = np.zeros([n,3], dtype=dtype);
        self.m = np.ones([n,1], dtype=dtype);
        self.a = np.zeros([n,3], dtype=dtype);
        self.f = np.zeros([n,3], dtype=dtype);
        self.n = n
        self.nc = L
        self.L=L
        self.Lh=L/2.0
        self.max_nei=10
        self.rc= 1.0
        self.CUDA =True

        if (self.CUDA):
            self.h_r = np.zeros([n*3], dtype=dtype)
            self.h_v = np.zeros([n*3], dtype=dtype)
            self.h_m = np.zeros([n], dtype=dtype)
            self.h_a = np.zeros([n*3], dtype=dtype)
            self.h_f = np.zeros([n*3], dtype=dtype)

            self.h_cells = np.zeros([self.nc*self.nc ** 3], dtype=np.int32)
            self.h_narray = np.zeros([self.nc ** 3], dtype=np.int32)
            self.h_nei_index = np.zeros([n],dtype=np.int32)
            self.h_nei_list = np.zeros([n*self.max_nei], dtype=np.int32)

            self.d_r = cuda.mem_alloc_like(self.h_r)
            self.d_v = cuda.mem_alloc_like(self.h_v)
            self.d_m = cuda.mem_alloc_like(self.h_m)
            self.d_a = cuda.mem_alloc_like(self.h_a)
            self.d_f = cuda.mem_alloc_like(self.h_f)

            self.d_nei_index = cuda.mem_alloc_like(self.h_nei_index)
            self.d_nei_list = cuda.mem_alloc_like(self.h_nei_list )
            self.d_cells = cuda.mem_alloc_like(self.h_cells)
            self.d_narray = cuda.mem_alloc_like(self.h_narray)

            cuda.memcpy_htod(self.d_r, self.h_r)
            cuda.memcpy_htod(self.d_v, self.h_v)
            cuda.memcpy_htod(self.d_m, self.h_m)
            cuda.memcpy_htod(self.d_a, self.h_a)
            cuda.memcpy_htod(self.d_f, self.h_f)

            cuda.memcpy_htod(self.d_cells, self.h_cells)
            cuda.memcpy_htod(self.d_narray, self.h_narray)
            cuda.memcpy_htod(self.d_nei_list, self.h_nei_list)
            cuda.memcpy_htod(self.d_nei_index, self.h_nei_index)

    def exchange_af(self):
        self.a, self.f = self.f, self.a
        self.a = self.a/self.m

    def populate_random(self):
        L=self.L
        self.r = np.random.random([self.n,3])*(L)-L/2.0
        self.make_disk();

    def create_cell_list(self, CUDA=True, DEBUG=False):
        if DEBUG:
            self.cells = np.empty((self.nc**3,), dtype=object)
            for i,v in enumerate(self.cells): self.cells[i]=list()
            for i,r in zip(range(self.n),self.r):
                c = self.nc*(r+self.L/2.0)/self.L
                self.cells[int(c[0]) + self.nc*int(c[1]) + self.nc*self.nc*int(c[2])].append(i)
        if CUDA:
            gpu.kernels.create_cell_list(self)
            return

    def create_neighbour_list(self, CUDA=True, DEBUG=False):
        if CUDA:
            gpu.kernels.neighbour_list(self, rcut=1.0)
        if DEBUG:
            gpu.copy_to_host(self)
            for i in range(0,self.n):
                gpunei_list = [self.h_nei_list[i*self.max_nei+h] for h in range(0,self.h_nei_index[i])]
                cpunei_list = self.get_neigbour_list(i,1.0)
                if cpunei_list.sort() != gpunei_list.sort():
                    print("neighbour list problem")

    def initial_sphere_index(self):
        nrc = int(self.rc + 1)
        sphere_table = []
        for cx in range(-nrc, nrc):
            for cy in range(-nrc, nrc):
                for cz in range(-nrc, nrc):
                    if cx*cx + cy*cy + cz*cz>self.rc:
                        continue;
                    else:
                        sphere_table.append(int(cx[0]) + self.nc*int(cy[1]) + self.nc*self.nc*int(cz[2]))

    def update_cell_list(self):
        nc = self.nc
        rc = self.rc
        for i, v in enumerate(self.cells): self.cells[i] = list()
        for i, r in zip(range(self.n), self.r):
            c = nc * (r + self.L / 2.0) / self.L
            self.cells[int(c[0]) + nc * int(c[1]) + nc * nc * int(c[2])].append(i)

    def get_neigbour_list(self,i,d):
            list_nei=[]
            rp = self.r[i]
            nrc = int(d/self.rc+1.0)
            for cx in range(-nrc,nrc):
                for cy in range(-nrc, nrc):
                    for cz in range(-nrc, nrc):
                        rx = self.periodic(rp[0] + cx)
                        ry = self.periodic(rp[1] + cy)
                        rz = self.periodic(rp[2] + cz)
                        cell_index=int(rx + self.Lh) + self.nc * int(ry + self.Lh) + self.nc * self.nc * int(rz + self.Lh)
                        cell = self.cells[cell_index]

                        list_nei.extend(list(filter(lambda x: self.distance(rp,self.r[x])<d and i !=x, cell)))
                        for c in list_nei:
                            if list_nei.count(c)>1:
                                stop=1
            return list_nei

    def get_neigbour_list_o2(self, i, d):
        list_nei = []
        rp = self.r[i]
        for j,p in enumerate(self.r):
            if i==j: continue;
            if self.distance(rp,p)<d:
                        list_nei.append(j)
        return list_nei

    def periodic(self, r):
        if r>self.Lh:
            return r-self.L
        if r<-self.Lh:
            return self.L+r
        return r;

    def periodic3(self, r3):
        r3p = [r - self.L if r > self.Lh else self.L + r if r < -self.Lh else r for r in r3]
        return r3p

    def periodic_all(self):
        for i,r in enumerate(self.r):
            self.r[i]=self.periodic3(r)

    def make_disk(self):
        self.r[:,2] = 0.0

    def distance(self,pi,pj):
        return np.linalg.norm(self.periodic3(pi - pj))

    def distance_squared(self,pi,pj):
        a_min_b = self.periodic3(pi - pj)
        return np.einsum("i,i->", a_min_b, a_min_b), a_min_b

    def total_momentum(self):
        return np.sum(self.v*self.m,axis=0)

    def total_force(self):
        return np.sum(self.f,axis=0)








