"""Static charts for the committee deck. Run after scorecard.py and alarm_analysis.py.
Output: charts/*.png and charts/chart_numbers.json"""
import json, re, runpy, io, contextlib, sys, os, datetime as dt
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.makedirs("charts", exist_ok=True)
INK, MUTED, GRID, ACC, ACC2, BG = "#1F2A3D", "#9AA3AE", "#E4E6EA", "#2B6CB0", "#D9822B", "#FBFAF6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
nums = {}

def save(fig, name):
    fig.savefig(f"charts/{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

sys.argv = ["scorecard.py", "8"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("scorecard.py")
S, dates, an, week_of = g["S"], g["dates"], g["an"], g["week_of"]

# 1. Ever Given at three resolutions
d = pd.read_csv("suez_daily_mar_apr_2021.csv", parse_dates=["date"])
base = d.n[:22].mean()
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.plot(d.date, d.n / base * 100, color=MUTED, lw=2, label="Daily")
wk = d.set_index("date").n.resample("W-SUN").mean()
for t, v in wk.items():
    ax.hlines(v / base * 100, t - pd.Timedelta(days=6.5), t + pd.Timedelta(days=0.5), color=ACC, lw=3)
mo = d.set_index("date").n.resample("MS").mean()
for t, v in mo.items():
    end = (t + pd.offsets.MonthEnd(1))
    ax.hlines(v / base * 100, t, end, color=ACC2, lw=3, linestyles="--")
ax.axhline(100, color=GRID, lw=1)
ax.set_ylabel("Transits, % of 1-22 March average"); ax.set_ylim(0, 165)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.text(pd.Timestamp("2021-03-25"), 6, "  daily low: -96%", color=INK, va="bottom")
ax.text(pd.Timestamp("2021-03-15"), 30, "weekly mean: -65%  ", color=ACC, ha="right")
ax.text(pd.Timestamp("2021-03-03"), 70, "monthly mean: -15%", color=ACC2, va="top")
ax.set_title("Daily (grey), weekly mean (blue), monthly mean (orange)", fontsize=14, color=INK, loc="left")
save(fig, "ever_given")
nums["ever_given"] = dict(daily=round(d.n.min() / base - 1, 3), weekly=round(d.n[21:28].mean() / base - 1, 3),
                          monthly=round(d.n[:31].mean() / base - 1, 3), four_week=round(d.n[21:49].mean() / base - 1, 3))

# 2. Recovery paths (weekly, deseasonalised)
cases = [("Hormuz 2026", "Hormuz", "2026-02-28"), ("Red Sea 2023 (Suez)", "Suez Canal", "2023-11-19"),
         ("Bosporus 2022", "Bosporus", "2022-02-24"), ("Panama 2023", "Panama Canal", "2023-07-30"),
         ("Taiwan drills 2022", "Taiwan Strait", "2022-08-02")]
fig, ax = plt.subplots(figsize=(13, 6))
for name, site, day in cases:
    x = S[site]; t = week_of(dt.date.fromisoformat(day)); b = np.nanmedian(x[t - 26:t])
    ks = [k for k in range(-8, 53) if 0 <= t + k < len(x)]
    ys = [x[t + k] / b * 100 for k in ks]
    acc = name.startswith("Red Sea")
    ax.plot(ks, ys, color=ACC if acc else MUTED, lw=3 if acc else 1.8)
    ax.text(ks[-1] + 0.8, ys[-1] - (5 if name.startswith("Panama") else 0), name, color=ACC if acc else INK, va="center", fontsize=13)
ax.axhline(90, color=INK, lw=1, ls="--"); ax.text(-7.5, 92, "90% = recovered", fontsize=12, color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-8, 66); ax.set_ylim(0, 140); ax.set_xlabel("Weeks after the event"); ax.set_ylabel("% of the 26 weeks before")
save(fig, "recovery_paths")

# 3. z vs drop scatter
ev = pd.read_csv("events_scored.csv").dropna(subset=["z_13w"])
ev["flag13"] = ((ev.chg_13w <= -0.10) & (ev.z_13w <= -2)).astype(int)
fig, ax = plt.subplots(figsize=(12, 6))
for f, col in [(0, MUTED), (1, ACC)]:
    e = ev[ev.flag13 == f]; ax.scatter(e.chg_13w * 100, e.z_13w, s=90, color=col, edgecolor=BG, lw=1.5, zorder=3)
ax.axvline(-10, color=INK, ls="--", lw=1); ax.axhline(-2, color=INK, ls="--", lw=1)
lab = {("E07", "Kerch Strait"): "Kerch 2022: -42% but z -1.8 (noisy)", ("E05", "Panama Canal"): "COVID, Panama: -9%, z -12.6",
       ("E21", "Hormuz"): "Hormuz 2026", ("E13b", "Bab el-Mandeb"): "Red Sea, Jan 2024", ("E07", "Bosporus"): "Bosporus 2022"}
for r in ev.itertuples():
    if (r.id, r.site) in lab:
        ax.annotate(lab[(r.id, r.site)], (r.chg_13w * 100, r.z_13w), xytext=((-8, -22) if r.site == "Kerch Strait" else (8, 6)), ha=("right" if r.site == "Kerch Strait" else "left"), textcoords="offset points", fontsize=12, color=INK)
ax.set_title("Flagged as disrupted (blue) only when both: drop of 10% or more and z of -2 or below", fontsize=14, color=INK, loc="left")
ax.set_xlabel("Change in weekly transits, 13 weeks after the event (%)"); ax.set_ylabel("z-score vs 52 placebo weeks")
save(fig, "z_vs_drop")
nums["flag13"] = int(ev.flag13.sum()); nums["scatter_n"] = int(len(ev))

# 4. GPR vs onsets on one weekly timeline
gpr = [float(x.split(":")[0]) for x in open("gpr_weekly_raw.txt").read().strip().split(",")]
alarm = json.load(open("alarm_results.json"))
sites = ["Suez Canal", "Bab el-Mandeb", "Hormuz", "Bosporus", "Panama Canal", "Taiwan Strait", "Malacca", "Cape of Good Hope", "Gibraltar"]
td = pd.to_datetime(dates)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
a1.plot(td, gpr, color=MUTED, lw=1.4)
a1.axhline(alarm["gpr_threshold_p90"], color=INK, ls="--", lw=1)
a1.text(td[-1], alarm["gpr_threshold_p90"] + 230, "dashed line: 90th percentile = alarm;  shaded: alarm episodes", fontsize=12, color=INK, ha="right")
for e in alarm["episodes"]:
    a1.axvspan(pd.Timestamp(e["start"]), pd.Timestamp(e["end"]) + pd.Timedelta(days=6), color=ACC, alpha=0.25, lw=0)
a1.set_ylabel("Geopolitical Risk Index\n(daily, weekly mean)")
for i, s in enumerate(sites):
    a = an[an.site == s]
    for r in a.itertuples():
        col = ACC if r.event else MUTED
        a2.plot([pd.Timestamp(r.week)], [i], marker="|", ms=22, mew=3, color=col)
a2.set_yticks(range(len(sites))); a2.set_yticklabels(sites, fontsize=12); a2.invert_yaxis()
a2.set_ylabel("Disruption onsets")
a2.set_title("Disruption onsets: blue = linked to a listed event, grey = not linked", fontsize=13, color=INK, loc="left")
a2.set_xlim(td[60], td[-1]); a2.set_ylim(len(sites) - 0.3, -0.7)
save(fig, "gpr_timeline")

# 5. Radar vs AIS at Hormuz
h = pd.read_csv("portwatch_hormuz_daily.csv", parse_dates=["date"])
ph = [("Pre-war", "2026-01-01", "2026-02-27"), ("Closure", "2026-02-28", "2026-06-17"),
      ("Two lanes", "2026-06-18", "2026-08-31"), ("Recent", "2026-09-01", "2026-10-11")]
pre = h[(h.date >= ph[0][1]) & (h.date <= ph[0][2])].n_total.mean()
ais = [h[(h.date >= a) & (h.date <= b)].n_total.mean() / pre * 100 for _, a, b in ph]
rad = json.load(open("radar_results.json"))["total_vs_prewar"]
radar = [rad["1_prewar"] * 100, rad["2_closure"] * 100, rad["3_two_lanes"] * 100, rad["4_recent"] * 100]
fig, ax = plt.subplots(figsize=(11, 5.5))
xx = np.arange(4); w = 0.36
b1 = ax.bar(xx - w / 2 - 0.01, ais, w, color=MUTED, label="AIS transits (PortWatch)")
b2 = ax.bar(xx + w / 2 + 0.01, radar, w, color=ACC, label="Sentinel-1 radar detections (own)")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.0f}%", ha="center", fontsize=13, color=INK)
ax.set_xticks(xx); ax.set_xticklabels([p[0] for p in ph]); ax.set_ylabel("% of pre-war level"); ax.set_ylim(0, 115)
ax.legend(frameon=False, loc="upper right")
save(fig, "radar_vs_ais")
nums["ais_phase"] = [round(v, 1) for v in ais]; nums["radar_phase"] = [round(v, 1) for v in radar]
json.dump(nums, open("charts/chart_numbers.json", "w"), indent=1)
print(nums)
