from particles import *
import time_integral as timeinteg
import pair_force as pair
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

p = particles(256, 30)
p.populate_random()

pair = pair.PairForce(p, pair.LJ(0.0,0.9, 0.2))
timeint = timeinteg.Verlet(p, 1e-4)

p.make_disk()

fig, ax = plt.subplots()
ln = plt.scatter(p.r[:,1], p.r[:,0], s=50)
periods=50
relist=2

def update(frame):
    start = time.time()

    gpu.copy_to_device(p)

    for i in range(0,periods):
        if i % relist == 0: pair.re_neighbour()
        pair.iter_pairs_cuda()
        timeint.integral_cuda()

    gpu.copy_to_host(p)

    end = time.time()
    print(periods/(end - start))

    data = np.c_[p.r[:, 1], p.r[:, 0]]
    ln.set_offsets(data)
    return ln,

ani = animation.FuncAnimation(fig, update, interval=20)
plt.show()
