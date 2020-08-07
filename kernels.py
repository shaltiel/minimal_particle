import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import particles

class kernels():
    load_create_cell_list = True;
    load_create_calc_pair = True;
    load_time_integral = True;
    load_neighbour_list = True;
    mod = SourceModule("")

    @classmethod
    def create_cell_list(cls, p):
        if cls.load_create_cell_list:
            cls.mod_cell_list = SourceModule("""
                            __device__ volatile int sem = 0;

                            __device__ void acquire_semaphore(volatile int *lock){
                                while (atomicCAS((int *)lock, 0, 1) != 0);
                                }
                            __device__ void release_semaphore(volatile int *lock){
                                *lock = 0;
                                __threadfence();
                                }
                               
                               __global__ void zeros_narray(int *narray, int nc3)
                               {
                                 const int i = threadIdx.x + blockIdx.x * blockDim.x ;
                                 if (i >= nc3)
                                 {
                                     return;
                                 } 
                                 narray[i]=0;
                               }
                               
                               __global__ void populate_mesh(int *cells,float *r, unsigned int *narray, int nc, float nc_2, float nc_L, int n_i)
                               {
                                 const int j = threadIdx.y;
                                 const int i = threadIdx.x + blockIdx.x * blockDim.x ;
                                 if (i >= n_i || j > 0)
                                 {
                                     return;
                                 } 
                                 const int loc = j + i*blockDim.y;
                                 float c = r[loc]*nc_L+nc_2;
                                 int binx = (int)c;
                                 c = r[loc+1]*nc_L+nc_2;
                                 int biny = (int)c;
                                 c = r[loc+2]*nc_L+nc_2;
                                 int binz = (int)c;
                                 int pos = (binx + nc*biny + nc*nc*binz);
                                 const unsigned int offset = atomicInc(&narray[pos], 0xffffffff);
                                 cells[pos*nc+offset] = i;
                                
                                // __syncthreads();
                                //if (threadIdx.x == 0)
                                 //acquire_semaphore(&sem);
                                //__syncthreads();
                                 //cells[pos*nc+narray[pos]] = i;
                                 //__syncthreads();
                                 //atomicAdd(&narray[pos],1);
                                //__threadfence(); // not strictly necessary for the lock, but to make any global updates in the critical section visible to other threads in the grid
                                //__syncthreads();
                                //if (threadIdx.x == 0)
                                 // release_semaphore(&sem);
                                //__syncthreads();         
                               }
                               """)
            cls.load_create_cell_list=False
        func = cls.mod_cell_list.get_function("populate_mesh")
        func0 = cls.mod_cell_list.get_function("zeros_narray")
        func0(p.d_narray,np.int32(p.nc**3), block=(int(p.nc**3/128+1), 1, 1), grid=(128, 1, 1))
        func(p.d_cells, p.d_r, p.d_narray,np.int32(p.nc), np.float32(p.nc/2.0), np.float32(p.nc/p.L), np.int32(p.n), block=(int(p.n/128+1), 3, 1), grid=(128, 1, 1))

    @classmethod
    def neighbour_list(cls, p, rcut):
        if cls.load_neighbour_list:
            cls.mod_full = SourceModule("""
                                      __global__ void neighbour_list(float *r, int *cells, int *narray, int *nei_l, int *nei_index, int nc, float nc_2, float L, int n_i, int max_nei, float rcut)
                                      {
                                        float Lh=L/2.0;
                                        float ri[3]  = {0.0,0.0,0.0};
                                        int d=(int)(rcut+1.0);

                                        const int j = threadIdx.y;
                                        const int i = threadIdx.x + blockIdx.x * blockDim.x ;
                                        if (i >= n_i || j > 0)
                                        {
                                            return;
                                        } 
                                        nei_index[i]=0;
                                        const int loc = i*blockDim.y+j;
                                        int s=0;
                                        for (int cx=-d;cx<d;cx++)
                                        {
                                           for (int cy=-d;cy<d;cy++)
                                           {
                                               for (int cz=-d;cz<d;cz++)
                                               {
                                                   ri[0]= r[loc]+cx;
                                                   ri[1]= r[loc+1]+cy;
                                                   ri[2]= r[loc+2]+cz;

                                                   for (int c=0;c<3;c++)
                                                   {
                                                       if (ri[c]<-Lh) 
                                                       {
                                                               ri[c]=ri[c]+L;
                                                       }
                                                       else
                                                       {
                                                           if (ri[c]>Lh) ri[c]=ri[c]-L;
                                                       }  
                                                   }

                                                   int cell_index = (int)(ri[0] + Lh) + nc * (int)(ri[1] + Lh) + nc * nc * (int)(ri[2] + Lh);

                                                   for (int k=0; k < narray[cell_index]; k++)
                                                   {
                                                       int nei_i=cells[cell_index*nc+k];
                                                       if (i==nei_i) continue;
                                                       float dr2=0;
                                                       for (int c=0;c<3;c++)
                                                       {
                                                           float dr=r[loc+c]-r[nei_i*3+c];
                                                           if (dr<=-Lh) 
                                                           {
                                                                   dr=dr+L;
                                                           }
                                                           else
                                                           {
                                                               if (dr>Lh) dr=dr-L; 
                                                           }  
                                                           dr2+=dr*dr;
                                                       }           
                                                       if (dr2<=rcut*rcut)
                                                       {
                                                           nei_l[i*max_nei+s++]=nei_i;
                                                           atomicAdd(&nei_index[i],1);
                                                       }                                            
                                                   }    
                                               }
                                           }
                                         }
                                       }
                                      """)
            cls.load_neighbour_list = False
        func = cls.mod_full.get_function("neighbour_list")
        func(p.d_r, p.d_cells, p.d_narray, p.d_nei_list, p.d_nei_index, np.int32(p.nc), np.float32(p.nc / 2.0),
             np.float32(p.L), np.int32(p.n), np.int32(p.max_nei), np.float32(rcut), block=(int(p.n / 128 + 1), 3, 1),
             grid=(128, 1, 1))

    @classmethod
    def pair_calc(cls, p, rcut=1.0, sigma=1.0, epsilon=0.7, threshold=1000.0):
        if cls.load_create_calc_pair:
            cls.modd = SourceModule("""
                                        #include <stdio.h>
                                        __global__ void pair_calc(int *nei_list,int *nei_index,float *r, float *f, int nei_max, int nc, float nc_2, float L, int n_i, float rcut, float sigma, float epsilon, float threshold)
                                            {
                                                 float Lh=L/2.0;
                                            
                                                 float rij [3] ={0.0,0.0,0.0};
                                                 float fij [3] ={0.0,0.0,0.0};
                                                 
                                                 const int j = threadIdx.y;
                                                 const int ix = threadIdx.x + blockIdx.x * blockDim.x ;

                                                 if (ix >= n_i || j > 0)
                                                 {
                                                     return;
                                                 }   
                                                 
                                                 int i = ix*3; 
                                                 int lm = ix*nei_max;     
                                                
                                                for (int k=0;k<nei_index[ix];k++)                
                                                {  
                                                    int nei_i = 3*nei_list[(k+lm)];
                                                    float dr2 =0.0;
                                                    if (nei_i==i) continue;
                                                    for (int c=0;c<3;c++)
                                                        { 
                                                        float dr=r[i+c]-r[nei_i+c];
                                                        if (dr<=-Lh) 
                                                        {
                                                            dr=dr+L;
                                                        }
                                                        else
                                                        {
                                                            if (dr>Lh) dr=dr-L; 
                                                        }  
                                                        rij[c]=dr;
                                                        dr2+=dr*dr;
                                                       
                                                        }
                                                    if (dr2<=rcut)
                                                    {
                                                        
                                                        float r2inv = sigma / dr2;
                                                        float r6inv = r2inv * r2inv * r2inv;
                                                        float F_divr=epsilon*48*r6inv*(r6inv-0.5)/dr2;
                                                        if (F_divr > threshold)
                                                          F_divr = threshold;

                                                        fij[0]+=F_divr*rij[0];
                                                        fij[1]+=F_divr*rij[1];
                                                        fij[2]+=F_divr*rij[2];
                                                    }
                                                }
                                                        atomicAdd(&f[i],fij[0]);
                                                        atomicAdd(&f[i+1],fij[1]);
                                                        atomicAdd(&f[i+2],fij[2]);  
                                            }
                                               """)
            cls.load_create_calc_pair=False
        func = cls.modd.get_function("pair_calc")
        func(p.d_nei_list,p.d_nei_index, p.d_r, p.d_f, np.int32(p.max_nei), np.int32(p.nc), np.float32(p.nc / 2.0), np.float32(p.L), np.int32(p.n),
             np.float32(rcut), np.float32(sigma), np.float32(epsilon), np.float32(threshold), block=(int(p.n/ 128 + 1), 3, 1), grid=(128, 1, 1))

    @classmethod
    def time_integral(cls,p,dt):
        if cls.load_time_integral:
            cls.verlet_2steps = SourceModule("""
                               __global__ void verlet_2steps(float *r,float *v, float *a, float *f, float *m, float dt, int n_i, float L)
                               {
                                 const int j = threadIdx.y;
                                 const int i = threadIdx.x + blockIdx.x * blockDim.x ;
                                 if (i >= n_i)
                                 {
                                     return;
                                 } 
                    
                                 const int loc = j + i*blockDim.y;
                                 if (j==2) r[loc]=0;
                                 
                                 r[loc] = r[loc] + v[loc]*dt + 0.5*a[loc]*dt*dt;
                                 
                                 if (r[loc]<=-L/2.0) 
                                    {
                                      r[loc]=r[loc]+L;
                                    }
                                 else
                                    {               
                                       if (r[loc]>L/2.0) r[loc]=r[loc]-L; 
                                    }  
                                 
                                 v[loc] = v[loc] + 0.5 * a[loc]*dt;
                                 
                                 a[loc] = f[loc]/m[i];
                                 
                                 v[loc] = v[loc] + 0.5 * a[loc]*dt;
                                 
                                 f[loc]=0.0;
                                 
                               }
                               """)
            cls.load_time_integral=False
        func = cls.verlet_2steps.get_function("verlet_2steps")
        func(p.d_r, p.d_v, p.d_a, p.d_f, p.d_m, np.float32(dt), np.int32(p.n), np.float32(p.L), block=(int(p.n/4+1), 3, 1), grid=(4, 1, 1))


def compare(h_array,d_array,np_array):
    cuda.memcpy_dtoh(h_array,d_array);
    print(h_array-np_array)


def copy_to_device(p):
    p.h_r = p.r.flatten().astype(dtype=p.dtype)
    cuda.memcpy_htod(p.d_r, p.h_r)
    p.h_v = p.v.flatten().astype(dtype=p.dtype)
    cuda.memcpy_htod(p.d_v, p.h_v)
    p.h_m = p.m
    cuda.memcpy_htod(p.d_m, p.h_m)
    p.h_a = p.a.flatten().astype(dtype=p.dtype)
    cuda.memcpy_htod(p.d_a, p.h_a)
    p.h_f = p.f.flatten().astype(dtype=p.dtype)
    cuda.memcpy_htod(p.d_f, p.h_f)

    # nei lists
    cuda.memcpy_htod(p.d_narray, p.h_narray)
    cuda.memcpy_htod(p.d_cells, p.h_cells)
    cuda.memcpy_htod(p.d_nei_index, p.h_nei_index)
    cuda.memcpy_htod(p.d_nei_list, p.h_nei_list)

def copy_to_host(p):
    cuda.memcpy_dtoh(p.h_r, p.d_r)
    p.r = p.h_r.reshape((-1,3))
    cuda.memcpy_dtoh(p.h_v, p.d_v)
    p.v = p.h_v.reshape((-1, 3))
    cuda.memcpy_dtoh(p.h_a, p.d_a)
    p.a = p.h_a.reshape((-1, 3))
    cuda.memcpy_dtoh(p.h_f, p.d_f)
    p.f = p.h_f.reshape((-1, 3))
    cuda.memcpy_dtoh(p.h_m, p.d_m)
    p.m = p.h_m

    cuda.memcpy_dtoh(p.h_nei_list, p.d_nei_list)
    cuda.memcpy_dtoh(p.h_nei_index, p.d_nei_index)
    cuda.memcpy_dtoh(p.h_narray, p.d_narray)
    cuda.memcpy_dtoh(p.h_cells, p.d_cells)

