"""Run the whole committee analysis in order. Usage: python3 run_all.py"""
import subprocess, sys
steps = [["scorecard.py", "13"], ["scorecard.py", "8"], ["alarm_analysis.py"], ["make_charts.py"]]
for s in steps:
    print(f"\n=== {' '.join(s)} ===", flush=True)
    r = subprocess.run([sys.executable] + s)
    if r.returncode != 0:
        sys.exit(f"Stopped: {s[0]} failed")
print("\nDone. Outputs: events_scored.csv, anomalies.csv, confusion.json, alarm_results.json, charts/*.png")
