from __future__ import print_function

import pymp
import numpy as np
import time
import multiprocessing

I = np.random.random((4, 6))
I_f = I.flatten()
I_sort = np.sort(I_f)[::-1][:10]
argmax_val = I_f.argsort()[::-1][:10]
multi_idx = np.unravel_index(np.asarray(argmax_val), I.shape)

print(I)
print(I_sort)
print(argmax_val)
print(np.array(multi_idx))

print(np.sort(I_f)[::-1][:10])

t0 = time.time()
test_list = pymp._shared.list()
rnge = iter(range(10000))

with pymp.Parallel(multiprocessing.cpu_count()) as p:
    for i in p.iterate(rnge):
        test_list.append(i)


print(test_list)
t1 = time.time()

print(t1-t0)


test_list_control = []


for i in range(0, 10000):
    test_list_control.append(i)

print(time.time() - t1)
print(multiprocessing.cpu_count())