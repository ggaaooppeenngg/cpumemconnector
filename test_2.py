from multiprocessing import shared_memory
import numpy as np

# Attach to an existing shared memory block
existing_shm = shared_memory.SharedMemory(name='connector_shared_memory')
c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
print(c)

existing_shm.close()   # Close each SharedMemory instance