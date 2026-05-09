# Tetrahedral grids for DMTet

Place pre-computed tetrahedral grids here, named `<grid_size>_tets.npz`.
For example:

- `128_tets.npz`
- `256_tets.npz`

The DMTet refinement stage (`./train.sh ... -d`) loads
`tets/<tet_grid_size>_tets.npz` according to the `--tet_grid_size` argument
(default: 128).

Grids can be obtained from the
[NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec) repository
(`data/tets/`).
