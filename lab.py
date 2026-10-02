import time
import numpy as np
import numba
from numba import njit, prange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAXT = numba.config.NUMBA_NUM_THREADS
print(f"Hardware Threads Detected: {MAXT}\n")


@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside += 1
    return (4.0 * inside) / n_samples

_ = monte_carlo_pi(10_000)
SAMPLES = 120_000_000
tcs = sorted(set(t for t in [1, 2, 4, 8, MAXT] if t <= MAXT))
c1 = {}
t1 = None
print(f"{'Threads':<8}|{'Time (s)':<12}|{'Speedup':<10}|{'Eff (%)':<10}")
for t in tcs:
    numba.set_num_threads(t)
    s = time.perf_counter()
    monte_carlo_pi(SAMPLES)
    el = time.perf_counter() - s
    if t == 1:
        t1 = el
    sp = t1 / el
    c1[t] = (el, sp, sp / t * 100)
    print(f"{t:<8}|{el:<12.4f}|{sp:<10.2f}|{sp/t*100:<10.1f}")
numba.set_num_threads(MAXT)


@njit(parallel=True)
def mandel_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            zr, zi = 0.0, 0.0
            it = 0
            while (zr*zr + zi*zi <= 4.0) and (it < max_iter):
                nr = zr*zr - zi*zi + cx
                zi = 2.0*zr*zi + cy
                zr = nr
                it += 1
            img[r, c] = it
    return img

@njit(parallel=True)
def mandel_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            zr, zi = 0.0, 0.0
            it = 0
            while (zr*zr + zi*zi <= 4.0) and (it < max_iter):
                nr = zr*zr - zi*zi + cx
                zi = 2.0*zr*zi + cy
                zr = nr
                it += 1
            img[r, c] = it
    return img

_ = mandel_rows(100, 100, 50); _ = mandel_cols(100, 100, 50)
H, W, MAX_IT = 2500, 2500, 1000
s = time.perf_counter(); g = mandel_rows(H, W, MAX_IT); t_rows = time.perf_counter() - s
s = time.perf_counter(); mandel_cols(H, W, MAX_IT); t_cols = time.perf_counter() - s
print(f"\nRows: {t_rows:.3f} s | Cols: {t_cols:.3f} s")
plt.figure(figsize=(8, 8))
plt.imshow(g, cmap='magma', extent=[-2.0, 0.5, -1.2, 1.2])
plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
plt.axis('off')
plt.savefig('mandelbrot_output.png', dpi=300, bbox_inches='tight')


@njit(parallel=True)
def heat_step(u, u_next, alpha=0.20):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0 * u[i, j])

def run_heat(dtype, N=1500, steps=300):
    u = np.zeros((N, N), dtype=dtype); un = np.zeros_like(u)
    u[0, :] = 100.0; u[:, 0] = 100.0; un[0, :] = 100.0; un[:, 0] = 100.0
    heat_step(u, un)
    s = time.perf_counter()
    for _ in range(steps):
        heat_step(u, un)
        u, un = un, u
    el = time.perf_counter() - s
    return el, (N * N * steps) / el / 1e6

e64, m64 = run_heat(np.float64)
e32, m32 = run_heat(np.float32)
print(f"\nHeat float64: {e64:.3f} s, {m64:.2f} Mcells/s")
print(f"Heat float32: {e32:.3f} s, {m32:.2f} Mcells/s  -> faster x{e64/e32:.2f}")


print("\n===== TABLE 1 =====")
for t in tcs:
    if t in (1, 2, 4):
        print(f"MonteCarlo {t} thr: {c1[t][0]:.4f} s | {c1[t][1]:.2f}x | {c1[t][2]:.1f}%")
print(f"MonteCarlo MAX ({MAXT}) : {c1[MAXT][0]:.4f} s | {c1[MAXT][1]:.2f}x | {c1[MAXT][2]:.1f}%")
print(f"Mandelbrot Rows : {t_rows:.3f} s")
print(f"Mandelbrot Cols : {t_cols:.3f} s")
print(f"Heat Stencil    : {e64:.3f} s ({m64:.2f} Mcells/s)")