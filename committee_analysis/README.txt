Committee analysis - how to run (Python 3.9+, packages: numpy pandas matplotlib)
  python3 -m pip install numpy pandas matplotlib
  python3 run_all.py            # runs everything below in order and rebuilds every output
Or step by step:
  python3 scorecard.py          # event scores, anomalies, confusion matrix (8-week link window)
  python3 scorecard.py 13       # robustness: 13-week link window (overwrites the three outputs)
  python3 alarm_analysis.py     # Geopolitical Risk Index vs disruptions -> alarm_results.json
  python3 make_charts.py        # all deck charts -> charts/*.png
Inputs (all in this folder): weekly_raw.txt (PortWatch weekly transits, 11 chokepoints),
gpr_weekly_raw.txt and gpr_country_monthly.csv (Caldara-Iacoviello GPR), suez_daily_mar_apr_2021.csv,
portwatch_hormuz_daily.csv, radar_results.json (Sentinel-1 output from Earth Engine).
