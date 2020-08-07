import kernels as gpu
import numpy as np

class Verlet:
    def __init__(self, particles, dt):
        self.particles = particles
        self.dt=dt
    # perform the first half step of velocity verlet
    # r(t + dt) = r(t) + v(t) * dt + (1 / 2) a(t) * dt ^ 2
    # v(t + dt / 2) = v(t) + (1 / 2) a * dt

    def half_integral_begin(self):
        r = self.particles.r
        v = self.particles.v
        a = self.particles.a
        dt = self.dt
        np.copyto(self.particles.r, r + v * dt + (1/2.0)*a*dt*dt)
        self.particles.periodic_all()
        np.copyto(self.particles.v, v+(1/2.0)*a*dt)

    def half_integral_end(self):
        v = self.particles.v
        self.particles.exchange_af()
        a = self.particles.a
        dt = self.dt
        np.copyto(self.particles.v, v  + (1/2.0)*a*dt)
        stop =1

    def integral_cuda(self):
        gpu.kernels.time_integral(self.particles, self.dt)



