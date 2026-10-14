# OSIRIS monthly zonal means

How much does `rebasis.optimal_estimate` improve OSIRIS monthly zonal-mean ozone over bin
averaging? These scripts need the ARG data volumes (paths in `data.py`, overridable with
`OSIRIS_ROOT` and `MLS_ROOT`) and `h5py`/`h5netcdf`:

```bash
uv run --with h5py --with h5netcdf python examples/osiris/sampling_study.py 2010 9 10 --knots 15
uv run --with h5py --with h5netcdf python examples/osiris/osiris_month.py 2010 9 30
examples/osiris/run_batch.sh          # several months and levels, three covariance models
uv run --with matplotlib python examples/osiris/figures.py
```

- `sampling_study.py`: MLS (dense, even sampling) gives a daily lat × lon field F at one
  pressure level. F is sampled at OSIRIS's actual locations and times, and the monthly zonal
  means from those samples, by bin averaging and by optimal estimation, are compared with F's
  exact zonal means: pure sampling error. The covariance (variance, correlation lengths in
  latitude, longitude and time, a nugget) is fitted by maximum likelihood; `--local` fits it per
  band, `--two-scale` adds a large-scale component.
- `osiris_month.py`: the same estimate for real OSIRIS ozone at one altitude (no truth), with
  the `rebasis.xarray` front-end straight from the HARMOZ file; writes a netCDF of the
  estimates.
- Files are copied to `out/cache` once, because the network shares occasionally fail reads.
