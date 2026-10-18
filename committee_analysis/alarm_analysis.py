"""Alarm side of the scorecard: Caldara-Iacoviello GPR against PortWatch chokepoint disruptions.

Inputs : weekly_raw.txt (PortWatch weekly transits), gpr_weekly_raw.txt (daily GPR averaged to the same
         Monday weeks: GPRD:THREAT:ACT), gpr_country_monthly.csv (monthly country GPR shares)
Needs  : scorecard.py in the same folder (reuses its cleaning, deseasonalising and onset search)
Output : alarm_results.json  (printed as well)
Run    : python3 alarm_analysis.py
"""
import json, runpy, io, contextlib, sys
import numpy as np, pandas as pd

sys.argv = ["scorecard.py", "8"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("scorecard.py")
S, dates, W, an, ev = g["S"], g["dates"], g["W"], g["an"], g["ev"]
week_of, dt = g["week_of"], g["dt"]
SITES = [s for s in S if s != "Kerch Strait"]
START = 72                       # first week the onset search can score (needs 20 + 52 weeks of history)

# ---------------- 1. GPR weekly ----------------
raw = open("gpr_weekly_raw.txt").read().strip().split(",")
gpr = pd.DataFrame([list(map(float, x.split(":"))) for x in raw], columns=["gpr", "threat", "act"])
gpr["date"] = dates
assert len(gpr) == W

# ---------------- 2. Disruption onsets per week (frame B, from scorecard.py) ----------------
onset = pd.DataFrame(0, index=range(W), columns=SITES)
for r in an.itertuples():
    if r.site in SITES:
        onset.loc[week_of(dt.date.fromisoformat(r.week)), r.site] = 1
# drop the first scorable week (COVID control window) as in scorecard.py
onset.loc[START] = 0
any_onset = onset.max(axis=1)

# ---------------- 3. Frame A from the index: GPR spike episodes ----------------
# spike = weekly GPR above its 90th percentile over 2019-2026; consecutive spike weeks form one episode
thr = gpr.gpr.quantile(0.90)
spike = gpr.gpr > thr
episodes, i = [], 0
while i < W:
    if spike[i]:
        j = i
        while j + 1 < W and spike[j + 1]:
            j += 1
        episodes.append((i, j)); i = j + 1
    else:
        i += 1
ep_rows = []
for a, b in episodes:
    win = range(a - 2, min(W, b + 9))          # onset from 2 weeks before to 8 weeks after the episode
    hit_sites = [s for s in SITES if onset.loc[[w for w in win if w >= 0], s].sum() > 0]
    ep_rows.append(dict(start=str(dates[a]), end=str(dates[b]), weeks=b - a + 1,
                        peak=round(gpr.gpr[a:b + 1].max()), scorable=a >= START, hit=int(bool(hit_sites)),
                        sites=";".join(hit_sites)))
ep = pd.DataFrame(ep_rows)
sc = ep[ep.scorable]
# permutation benchmark: chance that a random window of the same length contains an onset
rng = np.random.default_rng(7)
def random_hit_rate(n=5000):
    hits = 0
    for _ in range(n):
        L = int(rng.choice(sc.weeks))
        a = int(rng.integers(START + 2, W - L - 8))
        hits += any_onset[a - 2:a + L + 8].max()
    return hits / n
chance = random_hit_rate()

# ---------------- 4. Lead-lag: weekly GPR vs share of sites in an onset window ----------------
# disruption intensity = number of sites whose deseasonalised traffic is >=10% below its trailing 26-week median
def below(x):
    s = pd.Series(x)
    base = s.shift(1).rolling(26, min_periods=20).median()
    return (s / base < 0.9).astype(float)
intensity = pd.DataFrame({s: below(S[s]) for s in SITES}).sum(axis=1)
z = lambda s: (s - s.mean()) / s.std()
g_ = z(gpr.gpr[START:].reset_index(drop=True)); d_ = z(intensity[START:].reset_index(drop=True))
xc = {k: round(float(g_.corr(d_.shift(-k))), 3) for k in range(-12, 13)}   # k>0: GPR leads disruption by k weeks

# ---------------- 5. Site-specific alarm: country GPR vs monthly onsets ----------------
cg = pd.read_csv("gpr_country_monthly.csv")
MAP = {"Suez Canal": "EGY", "Bab el-Mandeb": "ISR", "Hormuz": "SAU", "Bosporus": "UKR",
       "Taiwan Strait": "TWN", "Malacca": "MYS", "Cape of Good Hope": "ZAF", "Panama Canal": "USA",
       "Gibraltar": "USA", "Dover": "USA"}
mon = pd.Series([d.strftime("%Y-%m") for d in dates])
rows = []
for s in ["Suez Canal", "Bab el-Mandeb", "Hormuz", "Bosporus", "Taiwan Strait"]:
    c = MAP[s]
    m_on = onset[s].groupby(mon).max()
    m_on = m_on[m_on.index >= str(dates[START])[:7]]
    x = cg.set_index("month")[c]
    thr_c = x.quantile(0.90)
    for m in m_on.index[:-2]:
        k = list(x.index).index(m) if m in x.index else None
        if k is None or k == 0:
            continue
        alarm_lvl = x.iloc[k - 1:k + 1].max()                   # alarm in the month or the month before
        fut = m_on.loc[m:].iloc[:3].max()                        # onset this month or in the next two
        rows.append(dict(site=s, month=m, country=c, alarm=alarm_lvl, alarm_flag=int(alarm_lvl > thr_c), onset=int(fut)))
cm = pd.DataFrame(rows)
def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
site_tab = []
for s, d in cm.groupby("site"):
    tp = int(((d.alarm_flag == 1) & (d.onset == 1)).sum()); fp = int(((d.alarm_flag == 1) & (d.onset == 0)).sum())
    fn = int(((d.alarm_flag == 0) & (d.onset == 1)).sum()); tn = int(((d.alarm_flag == 0) & (d.onset == 0)).sum())
    a = auc(d.onset.values, d.alarm.values)
    site_tab.append(dict(site=s, country=MAP[s], months=len(d), hits=tp, false_alarms=fp, misses=fn, quiet=tn,
                         auc=None if a is None else round(a, 2)))
pooled = cm.copy()
pooled["rank"] = pooled.groupby("site").alarm.rank(pct=True)     # within-site percentile so sites are comparable
pooled_auc = auc(pooled.onset.values, pooled["rank"].values)
glob = []
for r in cm.itertuples():
    k = list(cg.month).index(r.month)
    glob.append(cg.GPR.iloc[k - 1:k + 1].max())
global_auc = auc(cm.onset.values, np.array(glob))

# ---------------- 6. Data inventory (resolution table) ----------------
inventory = [
    dict(dataset="IMF PortWatch chokepoint transits", instrument="Satellite + terrestrial AIS (UN Global Platform)",
         space="Chokepoint (28 worldwide; 11 used)", time="Daily, from 1 Jan 2019, ~1 week lag", used_as="Outcome"),
    dict(dataset="Sentinel-1 GRD (own detections)", instrument="C-band SAR, 10 m, every ~12 days per orbit",
         space="0.025 deg grid over Hormuz narrows", time="Per pass (99 passes, 2025-26)", used_as="Outcome check where AIS goes dark"),
    dict(dataset="Geopolitical Risk Index (daily)", instrument="Newspaper text, 10 papers",
         space="Global", time="Daily, from 1985", used_as="Alarm"),
    dict(dataset="Country GPR", instrument="Newspaper text, country mentions",
         space="44 countries (no Iran or Yemen)", time="Monthly", used_as="Site-specific alarm"),
    dict(dataset="Event list (24 events)", instrument="News and official sources", space="Event-site pair",
         time="Day", used_as="Labels for validation"),
]

out = dict(
    gpr_threshold_p90=round(float(thr), 1),
    episodes=ep.to_dict(orient="records"),
    episodes_scorable=int(len(sc)), episodes_hit=int(sc.hit.sum()),
    episode_hit_rate=round(float(sc.hit.mean()), 3), chance_hit_rate=round(chance, 3),
    onset_weeks=int(any_onset.sum()),
    crosscorr=xc, best_lag=max(xc, key=lambda k: xc[k]),
    site_table=site_tab, pooled_country_auc=round(pooled_auc, 3), global_gpr_auc=round(global_auc, 3),
    n_site_months=int(len(cm)), n_onset_months=int(cm.onset.sum()),
    inventory=inventory,
)
json.dump(out, open("alarm_results.json", "w"), indent=1, default=str)
pd.set_option("display.width", 200)
print(ep.to_string())
print({k: out[k] for k in ["gpr_threshold_p90", "episodes_scorable", "episodes_hit", "episode_hit_rate", "chance_hit_rate", "onset_weeks", "best_lag", "pooled_country_auc", "global_gpr_auc", "n_site_months", "n_onset_months"]})
print(pd.DataFrame(site_tab).to_string())
print(xc)
