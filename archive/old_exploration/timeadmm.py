from pysrf import SRF
import numpy as np
import time

a = np.random.rand(2000, 2000)
a = 0.5 * (a + a.T)

shift = np.abs(np.nanmin(a)) + 1.0
a = a + shift

mask = np.triu(np.random.rand(2000, 2000) > 0.3)
mask = mask | mask.T
a[~mask] = np.nan
bounds_min = np.nanmin(a)
bounds_max = np.nanmax(a)
model = SRF(
    rank=10, max_inner=30, max_outer=10, verbose=True, bounds=(bounds_min, bounds_max)
)
start = time.time()
w = model.fit_transform(a)
end = time.time()

print(w.shape)
print(end - start)
