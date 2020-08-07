
import particles
import numpy as np
import kernels as gpu

class PairForce:
    def __init__(self, particles, pair_object):
        self.particles = particles
        self.pair_object = pair_object
        self.r = particles.r

    def iter_pairs(self):
        for i,_ in enumerate(self.r):
            list_nei = self.particles.get_neigbour_list(i,1.0)
            F_i = np.zeros([3])
            for j in list_nei:
                if i==j: continue
                r_ip, dr = self.particles.distance_squared(self.r[i], self.r[j])
                FdivR = self.pair_object.calc_pair(r_ip)
                F_i += [FdivR*dr[0], FdivR*dr[1], FdivR*dr[2]]
            self.particles.f[i] = F_i.copy()

    def iter_pairs_cuda(self):
        self.pair_object.calc_pair_gpu(self.particles)

    def re_neighbour(self):
        self.particles.create_cell_list()
        self.particles.create_neighbour_list();


class LJ:
    def __init__(self, alpha, sigma, epsilon,threshold=1000.0):
        self.alpha = alpha
        self.sigma = sigma
        self.epsilon = epsilon
        self.threshold = threshold
        lj1 = (4.0 * epsilon * pow(sigma, 12.0));
        lj2 = (alpha * 4.0 * epsilon * pow(sigma, 6.0));
        self.lj1_12 = (12.0) * lj1;
        self.lj2_6 = (6.0) * lj2;
        self.potential_kernel = {"pair": gpu.kernels.pair_calc}

    def calc_pair(self, r_ip):
        r2inv = self.sigma / r_ip;
        r6inv = r2inv * r2inv * r2inv;
        F_divr=self.epsilon*48*r6inv*(r6inv-0.5)/r_ip
        if (abs(F_divr) > self.threshold):
            F_divr = self.threshold;
        return F_divr

    def calc_pair_(self, r_ip):
        r2inv = 1.0 / r_ip;
        r6inv = r2inv * r2inv * r2inv;
        F_divr = r2inv * r6inv * (self.lj1_12 * r6inv - self.lj2_6);
        if (abs(F_divr) > 100.0):
            F_divr = 100.0;
        return F_divr
        #E = r6inv * (lj1 * r6inv - lj2);

    def calc_pair_gpu(self, particles):
        self.potential_kernel["pair"](particles,particles.rc, self.sigma, self.epsilon,self.threshold)
