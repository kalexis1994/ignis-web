#!/usr/bin/env python3
"""Generate a 64x64 4-channel blue noise texture via void-and-cluster (Ulichney).

Output: bluenoise64.bin — raw RGBA8, 64*64*4 = 16384 bytes, row-major.
Each channel is an independent blue noise rank map (values 0..255).
Used by the path tracer as a low-discrepancy per-pixel sample source,
animated over frames with a golden-ratio (R2) Cranley-Patterson rotation.
"""
import numpy as np

N = 64
SIGMA = 1.9


def toroidal_gaussian(n, sigma):
    ax = np.arange(n)
    d = np.minimum(ax, n - ax).astype(np.float64)
    dx2 = d[None, :] ** 2
    dy2 = d[:, None] ** 2
    return np.exp(-(dx2 + dy2) / (2.0 * sigma * sigma))


KERNEL = toroidal_gaussian(N, SIGMA)


def energy_add(energy, y, x, sign=1.0):
    energy += sign * np.roll(np.roll(KERNEL, y, axis=0), x, axis=1)


def void_and_cluster(rng):
    total = N * N
    # 1) Random initial pattern (10%)
    binary = np.zeros((N, N), dtype=bool)
    initial = total // 10
    idx = rng.choice(total, initial, replace=False)
    binary.flat[idx] = True

    energy = np.zeros((N, N))
    for y, x in zip(*np.nonzero(binary)):
        energy_add(energy, y, x)

    # 2) Relax: move tightest cluster point into largest void until stable
    for _ in range(total * 4):
        e_on = np.where(binary, energy, -np.inf)
        cy, cx = np.unravel_index(np.argmax(e_on), e_on.shape)
        binary[cy, cx] = False
        energy_add(energy, cy, cx, -1.0)
        e_off = np.where(binary, np.inf, energy)
        vy, vx = np.unravel_index(np.argmin(e_off), e_off.shape)
        if (vy, vx) == (cy, cx):
            binary[cy, cx] = True
            energy_add(energy, cy, cx)
            break
        binary[vy, vx] = True
        energy_add(energy, vy, vx)

    rank = np.zeros((N, N), dtype=np.int32)
    # 3) Rank the initial points by removing tightest clusters
    work = binary.copy()
    e = energy.copy()
    count = int(work.sum())
    for r in range(count - 1, -1, -1):
        e_on = np.where(work, e, -np.inf)
        cy, cx = np.unravel_index(np.argmax(e_on), e_on.shape)
        work[cy, cx] = False
        energy_add(e, cy, cx, -1.0)
        rank[cy, cx] = r

    # 4) Fill the rest into the largest void
    work = binary.copy()
    e = energy.copy()
    for r in range(count, total):
        e_off = np.where(work, np.inf, e)
        vy, vx = np.unravel_index(np.argmin(e_off), e_off.shape)
        work[vy, vx] = True
        energy_add(e, vy, vx)
        rank[vy, vx] = r

    return (rank.astype(np.float64) * 256.0 / total).clip(0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(1234)
    channels = [void_and_cluster(rng) for _ in range(4)]
    rgba = np.stack(channels, axis=-1)  # (N, N, 4)
    rgba.tofile('bluenoise64.bin')
    print(f'bluenoise64.bin written: {rgba.shape}, {rgba.nbytes} bytes')
    # Quick quality check: mean should be ~127.5, all ranks distinct per channel
    for i, ch in enumerate(channels):
        print(f'  ch{i}: mean={ch.mean():.1f} unique={len(np.unique(ch))}')


if __name__ == '__main__':
    main()
