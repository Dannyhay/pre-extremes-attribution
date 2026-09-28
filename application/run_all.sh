#!/bin/sh
# Reproduces Sec. VII A and Fig. 7 of the manuscript.
# Requires site.csv (2003, Trappes) and site_2010.csv (2010, Voronezh) in this folder,
# written by ../data/download_era5_sites_arco_v2.py.
set -e
mkdir -p results
for s in trappes voronezh; do
  python attribute_event.py $s --tail linear      # primary: components continued linearly beyond the knots
  python attribute_event.py $s --tail flat        # sensitivity: components held constant beyond the knots
  python bootstrap.py $s --B 200                  # year-block refitting uncertainty
  python hindcast.py $s                           # leave-one-year-out calibration, final model
  python hindcast.py $s --no-diurnal-mod --white  # same, original specification
done
cd ../figures && python make_fig7_application.py
