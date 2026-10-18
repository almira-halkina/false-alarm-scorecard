"""First-pass false-alarm scorecard on IMF PortWatch weekly chokepoint transits.

Input : weekly_raw.txt  (weekly mean daily transits, weeks start Monday 2019-01-07; pulled 2026-10-16)
Output: events_scored.csv, anomalies.csv, confusion.json
Run   : python3 scorecard.py        (link window 8 weeks, main result)
        python3 scorecard.py 13     (link window 13 weeks, robustness check)
"""
import json, datetime as dt
import numpy as np, pandas as pd

T0 = dt.date(2019, 1, 7)
NAMES = {"c1": "Suez Canal", "c2": "Panama Canal", "c3": "Bosporus", "c4": "Bab el-Mandeb",
         "c5": "Malacca", "c6": "Hormuz", "c7": "Cape of Good Hope", "c8": "Gibraltar",
         "c9": "Dover", "c11": "Taiwan Strait", "c28": "Kerch Strait"}
S = {}
for line in open("weekly_raw.txt"):
    k, v = line.strip().split("|")
    S[NAMES[k]] = np.array([float(x) for x in v.split(",")])
W = len(next(iter(S.values())))
dates = [T0 + dt.timedelta(7 * i) for i in range(W)]
# Known AIS artefact: week of 2020-01-27 spikes on most routes at once (e.g. Cape 177 vs ~45). Drop it.
ART = dates.index(dt.date(2020, 1, 27))
for k in S:
    S[k] = S[k].copy(); S[k][ART] = np.nan

SEASONAL = True
def deseason(x):
    """Divide by a week-of-year factor: median over years of x / centred 52-week mean (2019-2025)."""
    r = pd.Series(x).rolling(53, center=True, min_periods=40).mean().to_numpy()
    woy = np.array([min(d.isocalendar()[1], 52) for d in dates])
    f = np.ones(53)
    for k in range(1, 53):
        idx = [i for i in range(W) if woy[i] == k and dates[i].year <= 2025 and not np.isnan(x[i]) and r[i] > 1]
        if len(idx) >= 4:
            f[k] = np.median(x[idx] / r[idx])
    return x / f[woy]
RAW = {k: v.copy() for k, v in S.items()}
if SEASONAL:
    S = {k: deseason(v) for k, v in S.items()}

def week_of(d):
    return (d - T0).days // 7

BASE, MINBASE, PLACEBO = 26, 20, 52

def effect(x, t, h):
    """log change of mean over weeks [t, t+h) vs median of the 26 weeks before t."""
    pre = x[max(0, t - BASE):t]; pre = pre[~np.isnan(pre)]
    post = x[t:t + h]; post = post[~np.isnan(post)]
    if len(pre) < MINBASE or len(post) < max(1, h // 2) or np.median(pre) <= 0:
        return np.nan
    return np.log(max(post.mean(), 0.1)) - np.log(np.median(pre))

def zscore(x, t, h):
    """Effect divided by the SD of the same statistic computed at 52 earlier placebo dates."""
    e = effect(x, t, h)
    plac = [effect(x, s, h) for s in range(max(MINBASE, t - PLACEBO - h), t - h)]
    plac = np.array([p for p in plac if not np.isnan(p)])
    if len(plac) < 20 or np.isnan(e):
        return e, np.nan, np.nan
    return e, (e - plac.mean()) / plac.std(ddof=1), plac.std(ddof=1)

def recovery_weeks(x, t, band=0.10):
    """Weeks from the event until the 4-week mean is back within 10% of baseline, counted after the trough."""
    pre = x[max(0, t - BASE):t]; base = np.nanmedian(pre)
    roll = pd.Series(x).rolling(4, min_periods=3).mean().to_numpy()
    seg = roll[t + 3:min(t + 30, len(x))]
    tr = t + 3 + int(np.nanargmin(seg))
    for w in range(tr, len(x)):
        if roll[w] >= (1 - band) * base:
            return w - t, False
    return len(x) - 1 - t, True

# ---------------- event list (frame A: publicly flagged risks + frame B physical cases) ----------
# alarm: H = front-page global coverage with explicit shipping warnings; M = covered, shipping risk mentioned;
#        L = little advance alarm (physical or sudden). Coded by me from news coverage; to be replaced by GPR/GDELT.
E = [
 ("E01", "Gulf of Oman tanker attacks", "2019-06-13", "Interstate tension / attack on shipping", ["Hormuz"], "H", "no", "short"),
 ("E02", "Stena Impero seized by IRGC", "2019-07-19", "Interstate tension / attack on shipping", ["Hormuz"], "H", "no", "short"),
 ("E03", "Abqaiq-Khurais drone strikes", "2019-09-14", "Attack on energy infrastructure", ["Hormuz"], "H", "no", "short"),
 ("E04", "Soleimani killing and Iranian retaliation", "2020-01-03", "Interstate conflict", ["Hormuz"], "H", "no", "short"),
 ("E05", "COVID-19 declared a pandemic", "2020-03-11", "Pandemic (control case)", ["Suez Canal", "Panama Canal", "Malacca"], "M", "n/a", "long"),
 ("E06", "Ever Given grounding", "2021-03-23", "Infrastructure accident", ["Suez Canal"], "L", "yes", "short"),
 ("E07", "Russian invasion of Ukraine", "2022-02-24", "Interstate conflict", ["Bosporus", "Kerch Strait"], "H", "no", "long"),
 ("E08", "PLA drills after Pelosi visit", "2022-08-02", "Show of force", ["Taiwan Strait"], "H", "yes", "short"),
 ("E09", "Crimean bridge attack", "2022-10-08", "Interstate conflict", ["Kerch Strait"], "M", "no", "long"),
 ("E10", "Joint Sword drills", "2023-04-08", "Show of force", ["Taiwan Strait"], "M", "yes", "short"),
 ("E11", "Black Sea Grain Initiative collapse", "2023-07-17", "Interstate conflict", ["Bosporus"], "H", "no", "long"),
 ("E12", "Panama drought slot cuts", "2023-07-30", "Climatic restriction", ["Panama Canal"], "L", "yes", "medium"),
 ("E13", "Houthi seizure of Galaxy Leader", "2023-11-19", "Attack on shipping", ["Bab el-Mandeb", "Suez Canal"], "M", "yes", "long"),
 ("E13b", "US-UK strikes on the Houthis", "2024-01-12", "Interstate conflict", ["Bab el-Mandeb", "Suez Canal"], "H", "yes", "long"),
 ("E14", "Iran drone attack on Israel; MSC Aries seized", "2024-04-13", "Interstate conflict", ["Hormuz"], "H", "no", "short"),
 ("E15", "Joint Sword 2024A drills", "2024-05-23", "Show of force", ["Taiwan Strait"], "M", "yes", "short"),
 ("E16", "Iran missile strike on Israel", "2024-10-01", "Interstate conflict", ["Hormuz"], "H", "no", "short"),
 ("E17", "Joint Sword 2024B drills", "2024-10-14", "Show of force", ["Taiwan Strait"], "M", "yes", "short"),
 ("E18", "US threats to retake the Panama Canal", "2024-12-22", "Political threat", ["Panama Canal"], "M", "yes", "short"),
 ("E19", "Strait Thunder 2025A drills", "2025-04-01", "Show of force", ["Taiwan Strait"], "M", "yes", "short"),
 ("E20", "Israel-Iran war; parliament votes to close Hormuz", "2025-06-13", "Interstate conflict", ["Hormuz"], "H", "no", "short"),
 ("E21", "US-Israel strikes on Iran; Hormuz closed", "2026-02-28", "Interstate conflict", ["Hormuz"], "H", "no", "long"),
 ("E22", "Houthi blockade of Bab el-Mandeb declared", "2026-07-20", "Attack on shipping", ["Bab el-Mandeb"], "H", "yes", "long"),
 ("E23", "Panama draft cuts ahead of El Nino", "2026-07-24", "Climatic restriction", ["Panama Canal"], "M", "yes", "medium"),
]
DROP, ZCUT = -0.10, -2.0
rows = []
for eid, name, d, typ, sites, alarm, alt, persist in E:
    t = week_of(dt.date.fromisoformat(d))
    for s in sites:
        x = S[s]
        e4, z4, sd4 = zscore(x, t, 4)
        e13, z13, sd13 = zscore(x, t, 13)
        # use the worse of the two horizons for the flag
        flag = int(any((not np.isnan(e)) and (e <= np.log(1 + DROP)) and (z <= ZCUT)
                       for e, z in [(e4, z4), (e13, z13)]))
        rec, cens = recovery_weeks(x, t) if flag else (np.nan, False)
        rows.append(dict(id=eid, event=name, date=d, type=typ, site=s, alarm=alarm, alternative=alt,
                         hazard=persist, chg_4w=np.expm1(e4), z_4w=z4, chg_13w=np.expm1(e13), z_13w=z13,
                         noise_sd_4w=sd4, disrupted=flag, recovery_wk=rec, censored=cens,
                         base_pre=np.nanmedian(x[max(0, t - BASE):t])))
ev = pd.DataFrame(rows)
ev.to_csv("events_scored.csv", index=False)

# ---------------- frame B: automatic anomaly onsets ----------------
an = []
for s, x in S.items():
    last = -99
    for t in range(MINBASE + PLACEBO, W - 3):
        e, z, _ = zscore(x, t, 4)
        if not np.isnan(z) and e <= np.log(1 + DROP) and z <= ZCUT and t > last:
            rec, cens = recovery_weeks(x, t)
            an.append(dict(site=s, week=str(dates[t]), chg_4w=np.expm1(e), z_4w=z, episode_wk=rec, open=cens))
            last = t + min(max(rec, 13), 26)   # one onset per episode; an episode lasts until recovery, capped at 26 weeks so a later, deeper shock can register
an = pd.DataFrame(an)
# match each anomaly to an event at the same site within -2..+4 weeks
def match(r):
    t = week_of(dt.date.fromisoformat(r.week))
    for eid, name, d, *_rest in E:
        te = week_of(dt.date.fromisoformat(d))
        if r.site in _rest[1] and -2 <= t - te <= 4:
            return eid
    return ""
an["event"] = an.apply(match, axis=1)
an.to_csv("anomalies.csv", index=False)

# ---------------- confusion matrix: alarms scored as events, misses from frame B, quiet windows ----------
import sys
LINK = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # weeks after an alarm in which a disruption onset counts as linked
EXCLUDE_SITES = {"Kerch Strait"}       # single-digit daily counts after 2022 and likely AIS-dark traffic
EXCLUDE_TYPES = {"Pandemic (control case)"}
an["linked"] = an.event != ""
def linked(site, w):
    for eid, name, d, typ, sites, *_ in E:
        te = week_of(dt.date.fromisoformat(d))
        if site in sites and -2 <= w - te <= LINK:
            return eid
    return ""
an["event"] = [linked(r.site, week_of(dt.date.fromisoformat(r.week))) for r in an.itertuples()]
an.to_csv("anomalies.csv", index=False)
ok = ev[ev.z_13w.notna() & ~ev.site.isin(EXCLUDE_SITES) & ~ev.type.isin(EXCLUDE_TYPES)].copy()
lk = set(zip(an.event, an.site))
ok["hit"] = [int(r.disrupted == 1 or (r.id, r.site) in lk) for r in ok.itertuples()]
hits = ok[ok.hit == 1]; fas = ok[ok.hit == 0]
thin = EXCLUDE_SITES
ctrl = {eid for eid, n, d, typ, *_ in E if typ in EXCLUDE_TYPES}
miss = an[(an.event == "") & (~an.site.isin(thin))]
miss = miss[~miss.week.isin(["2020-05-25"])]   # first scorable week, still inside the COVID control window
quiet = 0
for s, x in S.items():
    if s in thin: continue
    aw = [week_of(dt.date.fromisoformat(d)) for eid, n, d, t_, sites, *_ in E if s in sites]
    nw = [week_of(dt.date.fromisoformat(w)) for w in an.loc[an.site == s, "week"]]
    for q0 in range(MINBASE + PLACEBO, W - 4, 13):
        if not any(q0 <= w < q0 + 13 for w in aw + nw):
            quiet += 1
h, f, m, q = len(hits), len(fas), len(miss), quiet
out = dict(cells=dict(hit=h, false_alarm=f, miss=m, quiet=q),
           hits=(hits.id + " " + hits.site).tolist(), false_alarms=(fas.id + " " + fas.site).tolist(),
           misses=(miss.site + " " + miss.week).tolist(), thin_sites=sorted(thin),
           unscored=(ev[~ev.index.isin(ok.index)].id + " " + ev[~ev.index.isin(ok.index)].site).tolist(),
           hit_rate=h / (h + m), false_alarm_rate=f / (f + q), precision=h / (h + f),
           noise_to_signal=(f / (f + q)) / (h / (h + m)), first_scored_week=str(dates[MINBASE + PLACEBO]))
json.dump(out, open("confusion.json", "w"), indent=1, default=str)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(ev[["id", "site", "alarm", "chg_4w", "z_4w", "chg_13w", "z_13w", "noise_sd_4w", "disrupted", "recovery_wk", "censored", "base_pre"]].round(2).to_string())
print(an.round(2).to_string())
print(json.dumps(out, indent=1, default=str))
