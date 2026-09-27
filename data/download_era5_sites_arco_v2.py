#!/usr/bin/env python3
"""
Fast direct-ARCO ERA5 downloader for the 2003/2010 heatwave attribution study.

This version bypasses the CDS time-series retrieval adaptor entirely and reads
the official ECMWF geo-chunked Zarr stores directly. That avoids CDS job queues
and the MultiAdaptorNoDataError seen for ERA5-Land soil-water requests.

Outputs
-------
site.csv       : Trappes
site_2010.csv  : Voronezh

Columns
-------
time : UTC
T    : ERA5-Land 2 m temperature [degC], hourly
SM   : ERA5-Land 0-100 cm depth-weighted soil moisture [m3 m-3], hourly
Z    : ERA5 500 hPa geopotential height [m], 6-hourly interpolated to hourly
H    : ERA5 850 hPa temperature [degC], 6-hourly interpolated to hourly
VPD  : optional, from 2 m temperature and dewpoint [hPa]

Examples
--------
# Small diagnostic test: JJA of the event year
python download_era5_sites_arco_v2.py --site trappes --test

# Full JJA baseline
python download_era5_sites_arco_v2.py --site all \
    --start 1979-01-01 --end 2022-12-31

# Include VPD
python download_era5_sites_arco_v2.py --site all --vpd

Requirements
------------
python -m pip install -U xarray pandas numpy dask zarr fsspec requests netcdf4

Authentication
--------------
Reads the CDS API token from either:
  1. environment variable CDSAPI_KEY
  2. ~/.cdsapirc, line beginning with "key:"

The token is never printed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SITES = {
    "trappes": {
        "lat": 48.8,
        "lon": 2.0,
        "out": "site.csv",
        "label": "Trappes / 2003 European mega-heatwave",
        "event_year": 2003,
    },
    "voronezh": {
        "lat": 51.7,
        "lon": 39.2,
        "out": "site_2010.csv",
        "label": "Voronezh / 2010 Russian mega-heatwave",
        "event_year": 2010,
    },
}

# Official ECMWF ERA5-Land geo-chunked stores.
LAND_T2M_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-007/arco/"
    "reanalysis_era5_land/sfc-2m-temperature/geoChunked.zarr"
)
LAND_SM_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-005/arco/"
    "reanalysis_era5_land/sfc-soil-water/geoChunked.zarr"
)

# Official ERA5 pressure-level geo-chunked store.
PL_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/"
    "reanalysis_era5_pressure_levels/pl/geoChunked.zarr"
)


def cds_token() -> str:
    token = os.getenv("CDSAPI_KEY")
    if token:
        return token.strip()

    cfg = Path.home() / ".cdsapirc"
    if cfg.exists():
        for raw in cfg.read_text().splitlines():
            line = raw.strip()
            if line.startswith("key:"):
                token = line.split(":", 1)[1].strip()
                if token:
                    return token

    raise SystemExit(
        "Could not find CDS API key.\n"
        "Set CDSAPI_KEY or create ~/.cdsapirc containing:\n"
        "url: https://cds.climate.copernicus.eu/api\n"
        "key: YOUR_TOKEN"
    )


def storage_options(token: str) -> dict:
    return {"headers": {"Authorization": f"Bearer {token}"}}


def open_store(url: str, token: str) -> xr.Dataset:
    print(f"  opening ARCO store: {url.split('/')[-2]}", flush=True)
    return xr.open_zarr(
        url,
        consolidated=True,
        chunks="auto",
        storage_options=storage_options(token),
    )


def pick_var(ds: xr.Dataset, *names: str) -> xr.DataArray:
    for name in names:
        if name in ds.data_vars:
            return ds[name]
    lower = {str(k).lower(): k for k in ds.data_vars}
    for name in names:
        if name.lower() in lower:
            return ds[lower[name.lower()]]
    raise KeyError(
        f"None of {names} found.\nAvailable variables: {list(ds.data_vars)}"
    )


def pressure_coord(ds: xr.Dataset) -> str:
    for name in ("pressureLevel", "pressure_level", "level", "plev"):
        if name in ds.coords or name in ds.dims:
            return name
    for name in list(ds.coords) + list(ds.dims):
        low = str(name).lower()
        if "pressure" in low or "isobaric" in low:
            return str(name)
    raise KeyError(
        f"Could not identify pressure coordinate. "
        f"coords={list(ds.coords)}, dims={list(ds.dims)}"
    )


def nearest_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """
    Select nearest native grid point. Handles -180..180 and 0..360 longitudes.
    """
    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        raise KeyError(
            f"Expected latitude/longitude coordinates; found {list(ds.coords)}"
        )

    lon_values = np.asarray(ds.longitude.values)
    use_lon = lon
    if np.nanmax(lon_values) > 180 and lon < 0:
        use_lon = lon % 360

    return ds.sel(latitude=lat, longitude=use_lon, method="nearest")


def select_time(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    # Include the full final calendar day when user passes YYYY-MM-DD.
    end_ts = pd.Timestamp(end)
    if len(end) == 10:
        end_ts = end_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ds.sel(time=slice(pd.Timestamp(start), end_ts))


def to_series(da: xr.DataArray, name: str) -> pd.Series:
    da = da.squeeze(drop=True)
    s = da.to_series()
    if isinstance(s.index, pd.MultiIndex):
        if "time" not in s.index.names:
            raise ValueError(f"Unexpected index for {name}: {s.index.names}")
        frame = s.rename(name).reset_index()
        s = frame.set_index("time")[name]
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s.name = name
    return s


def only_jja(ds: xr.Dataset) -> xr.Dataset:
    return ds.where(ds.time.dt.month.isin([6, 7, 8]), drop=True)


def load_land(
    token: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
    want_vpd: bool,
) -> xr.Dataset:
    print("  ERA5-Land: temperature/dewpoint", flush=True)
    tstore = open_store(LAND_T2M_URL, token)
    tpoint = nearest_point(tstore, lat, lon)
    tpoint = select_time(tpoint, start, end)
    tpoint = only_jja(tpoint)

    keep_t = ["t2m"]
    if "t2m" not in tpoint:
        # Defensive support for long variable names.
        keep_t = []
        for nm in ("2m_temperature",):
            if nm in tpoint:
                keep_t.append(nm)

    if want_vpd:
        if "d2m" in tpoint:
            keep_t.append("d2m")
        elif "2m_dewpoint_temperature" in tpoint:
            keep_t.append("2m_dewpoint_temperature")

    if not keep_t:
        raise KeyError(
            "Could not find 2-m temperature in temperature store. "
            f"Available: {list(tpoint.data_vars)}"
        )

    # Selecting point + JJA before load is the key speed/memory optimisation.
    tpoint = tpoint[keep_t].load()

    print("  ERA5-Land: soil water", flush=True)
    sstore = open_store(LAND_SM_URL, token)
    spoint = nearest_point(sstore, lat, lon)
    spoint = select_time(spoint, start, end)
    spoint = only_jja(spoint)

    soil_names = []
    for candidates in (
        ("swvl1", "volumetric_soil_water_layer_1"),
        ("swvl2", "volumetric_soil_water_layer_2"),
        ("swvl3", "volumetric_soil_water_layer_3"),
    ):
        da = pick_var(spoint, *candidates)
        soil_names.append(da.name)

    spoint = spoint[soil_names].load()

    print(
        "  ERA5-Land returned grid point: "
        f"{float(np.asarray(tpoint.latitude).squeeze()):.3f} N, "
        f"{float(np.asarray(tpoint.longitude).squeeze()):.3f} E",
        flush=True,
    )

    return xr.merge([tpoint, spoint], compat="override")


def load_pressure(
    token: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    print("  ERA5 pressure levels: Z500 and T850", flush=True)
    pstore = open_store(PL_URL, token)
    ppoint = nearest_point(pstore, lat, lon)
    ppoint = select_time(ppoint, start, end)
    ppoint = only_jja(ppoint)

    pcoord = pressure_coord(ppoint)

    z = pick_var(ppoint, "z", "geopotential")
    t = pick_var(ppoint, "t", "temperature")

    z500 = z.sel({pcoord: 500}, method="nearest").load() / 9.80665
    t850 = t.sel({pcoord: 850}, method="nearest").load() - 273.15

    print(
        "  ERA5 pressure returned grid point: "
        f"{float(np.asarray(ppoint.latitude).squeeze()):.3f} N, "
        f"{float(np.asarray(ppoint.longitude).squeeze()):.3f} E",
        flush=True,
    )
    return z500, t850


def build_dataframe(
    land: xr.Dataset,
    z500: xr.DataArray,
    t850: xr.DataArray,
    want_vpd: bool,
) -> pd.DataFrame:
    t2m = pick_var(land, "t2m", "2m_temperature")
    sw1 = pick_var(land, "swvl1", "volumetric_soil_water_layer_1")
    sw2 = pick_var(land, "swvl2", "volumetric_soil_water_layer_2")
    sw3 = pick_var(land, "swvl3", "volumetric_soil_water_layer_3")

    # ERA5-Land layer depths: 0-7, 7-28, 28-100 cm.
    root_sm = 0.07 * sw1 + 0.21 * sw2 + 0.72 * sw3

    sT = to_series(t2m - 273.15, "T")
    sSM = to_series(root_sm, "SM")
    sZ = to_series(z500, "Z")
    sH = to_series(t850, "H")

    df = pd.DataFrame(index=sT.index)
    df.index.name = "time"
    df["T"] = sT
    df["SM"] = sSM.reindex(df.index).interpolate(method="time")

    # Pressure-level source is 6-hourly. Interpolate only between valid
    # pressure-level samples; avoid silently propagating across long gaps.
    df["Z"] = sZ.reindex(df.index).interpolate(method="time", limit=6)
    df["H"] = sH.reindex(df.index).interpolate(method="time", limit=6)

    if want_vpd:
        d2m = pick_var(land, "d2m", "2m_dewpoint_temperature")
        tc = t2m - 273.15
        dc = d2m - 273.15
        es_t = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
        es_d = 6.112 * np.exp(17.67 * dc / (dc + 243.5))
        df["VPD"] = to_series(es_t - es_d, "VPD").reindex(df.index)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def sanity_check(df: pd.DataFrame, site: str) -> None:
    if df.empty:
        raise RuntimeError(f"{site}: no rows were returned")

    # Do not silently discard missing values before reporting them.
    miss = df[["T", "SM", "Z", "H"]].isna().sum()
    print("  missing values before final drop:")
    for k, v in miss.items():
        print(f"    {k}: {int(v)}")

    # Broad physical checks, intended to catch unit/parsing mistakes only.
    checks = {
        "T": (-80, 60),
        "SM": (0, 1),
        "Z": (4500, 6500),
        "H": (-80, 60),
    }
    for col, (lo, hi) in checks.items():
        good = df[col].dropna()
        if len(good) and not ((good >= lo) & (good <= hi)).all():
            raise RuntimeError(
                f"{site}: implausible {col} values; "
                f"range={good.min():.3f}..{good.max():.3f}"
            )


def process_site(key: str, token: str, args) -> None:
    meta = SITES[key]

    if args.test:
        start = f"{meta['event_year']}-06-01"
        end = f"{meta['event_year']}-08-31"
    else:
        start, end = args.start, args.end

    print(f"\n=== {meta['label']} ===")
    print(f"requested location: {meta['lat']:.3f} N, {meta['lon']:.3f} E")
    print(f"period: {start} .. {end} (JJA retained)")

    land = load_land(
        token,
        meta["lat"],
        meta["lon"],
        start,
        end,
        args.vpd,
    )
    z500, t850 = load_pressure(
        token,
        meta["lat"],
        meta["lon"],
        start,
        end,
    )

    df = build_dataframe(land, z500, t850, args.vpd)
    sanity_check(df, key)

    df = df.dropna(subset=["T", "SM", "Z", "H"])

    out = Path(args.out) if args.out and args.site != "all" else Path(meta["out"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp = out.with_suffix(out.suffix + ".part")
    df.to_csv(tmp)
    os.replace(tmp, out)

    print(
        f"  wrote {out}: {len(df):,} rows, "
        f"{df.index.min()} .. {df.index.max()}"
    )
    print(df[["T", "SM", "Z", "H"]].describe().round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--site",
        choices=["all", "trappes", "voronezh"],
        default="all",
    )
    ap.add_argument("--start", default="1979-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vpd", action="store_true")
    ap.add_argument(
        "--test",
        action="store_true",
        help="download only JJA of each site's event year",
    )
    args = ap.parse_args()

    if args.site == "all" and args.out:
        ap.error("--out can only be used with one site")

    token = cds_token()
    keys = list(SITES) if args.site == "all" else [args.site]

    for key in keys:
        process_site(key, token, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
