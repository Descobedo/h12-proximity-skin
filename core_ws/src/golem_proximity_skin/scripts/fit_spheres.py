"""Fit covering spheres to H1-2 links' meshes, for the sphere self-filter.

    python3 scripts/fit_spheres.py ../../../CL_Assets/mujoco_assets/scene_h1_2_magpie_pelvis_fixed.xml \
        --out config/h12_leg_spheres.yaml

GOLEM's sphere model (CL_Assets/ros_assets/h1_2_magpie_sphere.urdf) covers the
pelvis, torso and arms but not the legs, so a knee skin has nothing to filter its
self-returns against. This fits spheres to each leg link's visual meshes, in the
link's frame: k-means on the mesh's vertices, each sphere grown to cover every
vertex of its cluster, adding spheres until none is larger than --max-radius.
Every mesh vertex is inside a sphere by construction; the spheres also cover some
space beyond the surface (reported as how far the spheres reach past the mesh).

Writes {spheres: [{link, xyz, radius}]} for sphere_filter.SphereSelfFilter(extra=...).
"""
import argparse
import os

import mujoco
import numpy as np

LEG_LINKS = [f'{s}_{l}' for s in ('left', 'right') for l in
             ('hip_yaw_link', 'hip_pitch_link', 'hip_roll_link', 'knee_link', 'ankle_pitch_link', 'ankle_roll_link')]


def link_vertices(model, data, body):
    """All visual-mesh vertices of `body`, in its frame."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if b < 0:
        raise KeyError(f'no body {body!r}')
    R, p = data.xmat[b].reshape(3, 3), data.xpos[b]
    pts = []
    for g in np.nonzero(model.geom_bodyid == b)[0]:
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pts.append((w - p) @ R)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def kmeans(x, k, rng, iters=30):
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c, lab


def fit(x, max_radius, max_spheres, rng):
    """Spheres covering every point of x, each radius <= max_radius if possible."""
    for k in range(1, max_spheres + 1):
        c, lab = kmeans(x, k, rng)
        r = np.array([np.sqrt(((x[lab == j] - c[j]) ** 2).sum(-1).max()) if (lab == j).any() else 0.0
                      for j in range(k)])
        if r.max() <= max_radius or k == max_spheres:
            keep = r > 0
            return c[keep], r[keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mjcf')
    ap.add_argument('--out', required=True)
    ap.add_argument('--links', nargs='*', default=LEG_LINKS)
    ap.add_argument('--max-radius', type=float, default=0.06, help='m')
    ap.add_argument('--max-spheres', type=int, default=12, help='per link')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(args.seed)
    lines = ['# Covering spheres for the H1-2 legs (GOLEM\'s sphere model has none), fitted by',
             '# scripts/fit_spheres.py to each link\'s visual meshes, in the link frame. Regenerate,',
             f'# do not edit. max radius {args.max_radius} m, at most {args.max_spheres} per link.',
             'spheres:']
    total = 0
    for link in args.links:
        x = link_vertices(model, data, link)
        if not len(x):
            print(f'{link}: no meshes, skipped')
            continue
        c, r = fit(x, args.max_radius, args.max_spheres, rng)
        # How far the spheres reach past the mesh: sample points on each sphere's surface
        # and measure their distance to the nearest mesh vertex (an over-estimate of the gap).
        u = rng.standard_normal((200, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        shell = (c[:, None, :] + r[:, None, None] * u[None]).reshape(-1, 3)
        sub = x[rng.choice(len(x), min(len(x), 4000), replace=False)]
        reach = np.sqrt(((shell[:, None] - sub[None]) ** 2).sum(-1).min(axis=1))
        print(f'{link:24s} {len(x):6d} vertices -> {len(c):2d} spheres, radius {r.min():.3f}-{r.max():.3f} m, '
              f'reach past the mesh: median {np.median(reach) * 1000:.0f} mm, 95th pct {np.percentile(reach, 95) * 1000:.0f} mm')
        # Written to 0.1 mm: pad each radius by the worst rounding of its centre and itself
        # (sqrt(3) * 0.05 mm + 0.1 mm), so every vertex stays covered after the round trip.
        lines += [f'  - {{link: {link}, xyz: [{c_[0]:.4f}, {c_[1]:.4f}, {c_[2]:.4f}], '
                  f'radius: {np.ceil((r_ + 1.9e-4) * 1e4) / 1e4:.4f}}}' for c_, r_ in zip(c, r)]
        total += len(c)
    with open(args.out, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{total} spheres -> {args.out}')


if __name__ == '__main__':
    main()
