def snmf_cyclic_bsum(m, max_iter, x0, max_time=100, verbose=False):
    r, n = x0.shape  # r, n  (mind the transpose!)
    x = x0.copy()
    xxt = x @ x.T  # r x r
    vtmp = np.sum(x**2, axis=0)  # diag(xxt)
    obj = np.linalg.norm(x.T @ x - m, "fro") ** 2

    obj_vec, grad_vec, time_vec = [], [], []
    start = time.time()
    iter_count = 0
    # handy real cube‑root helper
    cbrt = np.cbrt

    while iter_count < max_iter:
        iter_count += 1
        for i in range(r):
            for j in range(n):
                # ===== coefficient bookkeeping (identical to MATLAB) =====
                a = 4.0
                b = 12.0 * x[i, j]
                c = 4.0 * (vtmp[j] - m[j, j] + xxt[i, i] + x[i, j] ** 2)
                d = 4.0 * (xxt[i, :] @ x[:, j] - x[i, :] @ m[:, j])
                p = (3.0 * a * c - b**2) / (3.0 * a**2)
                q = (9.0 * a * b * c - 27.0 * a**2 * d - 2.0 * b**3) / (27.0 * a**3)

                # closed‑form minimiser of the 1‑D quartic upper‑bound
                if c > b**2 / (3.0 * a):  # three real roots → pick the one in (0, inf)
                    delta = math.sqrt(q**2 / 4.0 + p**3 / 27.0)
                    x_new = cbrt(delta + q / 2.0) - cbrt(delta - q / 2.0)
                else:  # one real root
                    stmp = b**3 / (27.0 * a**3) - d / a
                    x_new = cbrt(stmp)

                if x_new < 0.0:  # non neg constraint
                    x_new = 0.0

                diff = x_new - x[i, j]

                # rank‑one updates (O(r) instead of recomputing xxt)
                xxt[i, i] += diff**2
                xxt[:, i] += diff * x[:, j]
                xxt[i, :] += diff * x[:, j]
                vtmp[j] += 2.0 * diff * x[i, j] + diff**2

                # objective value change via fourth‑order Taylor surrogate
                mind = (
                    (a / 4.0) * diff**4
                    + (b / 3.0) * diff**3
                    + (c / 2.0) * diff**2
                    + d * diff
                )
                obj += mind
                x[i, j] = x_new

        elapsed = time.time() - start
        if elapsed > max_time:
            break

        if verbose:
            obj = evar(m, x.T @ x)
            print(f"it {iter_count:3d}  obj {obj: .6f}", end="\r")

        # diagnostics
        time_vec.append(elapsed)
        obj_vec.append(abs(obj))

    return x, np.array(obj_vec), np.array(time_vec)
