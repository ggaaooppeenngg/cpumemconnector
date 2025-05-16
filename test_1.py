import time
# In the first Python interactive shell
import numpy as np
a = np.array([1, 1, 2, 3, 5, 8])  # Start with an existing NumPy array
from multiprocessing import shared_memory
shm = shared_memory.SharedMemory(create=True,name="connector_shared_memory",size=a.nbytes)
# Now create a NumPy array backed by shared memory
b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
b[:] = a[:]  # Copy the original data into shared memory
print(b)
print(shm.name)
print(shm.size)
time.sleep(10000000)
