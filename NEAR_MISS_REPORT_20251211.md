# Near‑Miss Report — 2025‑12‑11 (09:30–14:30 ET)

This report lists **every observed time** the RL model produced a **TRIM/EXIT action (3/4/5) while the account was FLAT**.

When you are FLAT, TRIM/EXIT actions are **not actionable** (there is nothing to sell), so the engine maps them to **HOLD** → **no entry order is placed**.

## Quick summary

- **Time window**: 09:30 to 14:30 ET
- **Cycles in window (with RL/obs events)**: 498
- **Near‑miss cycles (TRIM/EXIT while FLAT)**: 383
- **First time the log captured raw actions** in this window: `10:36:35`

### Important limitation

- In early periods, your log sometimes records only the **final mapped action** (HOLD) without recording the **raw model output**.
  This report includes **only the moments where the raw TRIM/EXIT output was explicitly logged** (`Raw RL Output: action_raw=...`).

## Near‑miss cycles (exhaustive)

Legend:
- **raw** = what the model wanted to do
- **final** = what the trading engine actually used after safety rules

### #1 — 10:36:35–10:36:38 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=685.78, mean=273.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=620.96, mean=247.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.68, max=6870.33, mean=2742.59
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #2 — 10:37:11–10:37:14 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=685.78, mean=273.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=620.96, mean=247.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.62, max=6870.33, mean=2742.56
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #3 — 10:37:56–10:37:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=685.78, mean=273.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=620.96, mean=247.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.62, max=6870.33, mean=2742.53
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #4 — 10:38:32–10:38:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=685.78, mean=273.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=620.97, mean=247.90
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.66, max=6870.33, mean=2742.47
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #5 — 10:39:07–10:39:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=685.78, mean=273.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.85, max=620.97, mean=247.91
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.57, max=6869.58, mean=2742.43
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #6 — 10:39:42–10:39:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=685.78, mean=273.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.85, max=621.09, mean=247.92
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.52, max=6869.58, mean=2742.48
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #7 — 10:40:15–10:40:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=685.78, mean=273.91
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=621.12, mean=247.94
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.46, max=6868.59, mean=2742.35
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #8 — 10:40:46–10:40:49 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=685.96, mean=273.91
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=621.30, mean=247.94
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.69, max=6870.41, mean=2742.39
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #9 — 10:41:22–10:41:25 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=686.04, mean=273.92
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=621.47, mean=247.96
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.71, max=6871.43, mean=2742.52
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #10 — 10:41:57–10:42:00 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=686.47, mean=273.93
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=621.89, mean=247.97
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.71, max=6875.21, mean=2742.60
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #11 — 10:42:33–10:42:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=686.47, mean=273.95
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=621.97, mean=247.96
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.71, max=6875.55, mean=2742.78
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #12 — 10:43:09–10:43:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=686.68, mean=273.98
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.88, max=622.25, mean=248.03
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.78, max=6877.79, mean=2743.06
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #13 — 10:43:40–10:43:43 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.18, max=686.68, mean=273.98
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.88, max=622.22, mean=248.03
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.86, max=6877.64, mean=2743.03
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #14 — 10:44:16–10:44:19 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.18, max=686.73, mean=274.00
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.89, max=622.29, mean=248.06
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.97, max=6878.09, mean=2743.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.5
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #15 — 10:44:52–10:45:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.18, max=686.73, mean=274.00
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.88, max=622.29, mean=248.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.89, max=6878.09, mean=2743.42
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #16 — 10:45:39–10:45:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=686.73, mean=274.03
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=622.29, mean=248.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.70, max=6878.09, mean=2743.46
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #17 — 10:46:14–10:46:17 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=686.73, mean=274.06
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=622.30, mean=248.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.72, max=6878.09, mean=2743.46
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #18 — 10:46:50–10:47:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.18, max=686.97, mean=274.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.88, max=622.50, mean=248.23
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.85, max=6880.40, mean=2744.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #19 — 10:47:42–10:47:53 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=686.97, mean=274.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.88, max=622.50, mean=248.23
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.78, max=6880.40, mean=2744.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #20 — 10:48:26–10:48:29 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=686.97, mean=274.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=622.54, mean=248.27
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.75, max=6880.77, mean=2744.44
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #21 — 10:49:03–10:49:05 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=686.97, mean=274.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=622.54, mean=248.31
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.68, max=6880.77, mean=2744.78
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #22 — 10:49:35–10:49:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=686.97, mean=274.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=622.54, mean=248.31
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.65, max=6880.77, mean=2744.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #23 — 10:50:11–10:50:16 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=686.97, mean=274.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=622.54, mean=248.34
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.60, max=6880.77, mean=2745.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #24 — 10:50:46–10:50:52 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=686.97, mean=274.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=622.54, mean=248.34
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.60, max=6880.77, mean=2745.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #25 — 10:51:22–10:51:28 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=686.97, mean=274.22
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.85, max=622.54, mean=248.37
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.58, max=6880.77, mean=2745.33
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #26 — 10:51:58–10:52:03 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=686.97, mean=274.22
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.85, max=622.54, mean=248.37
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.54, max=6880.77, mean=2745.61
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #27 — 10:52:33–10:52:38 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=686.97, mean=274.24
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=622.54, mean=248.39
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.46, max=6880.77, mean=2745.58
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #28 — 10:53:09–10:53:14 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=686.97, mean=274.26
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=622.54, mean=248.41
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.45, max=6880.77, mean=2745.76
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #29 — 10:53:44–10:53:55 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=686.97, mean=274.25
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=622.54, mean=248.40
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.43, max=6880.77, mean=2745.69
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #30 — 10:54:28–10:54:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=686.97, mean=274.26
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=622.54, mean=248.41
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.40, max=6880.77, mean=2745.75
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #31 — 10:55:09–10:55:12 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=1.00, mean=-0.05
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.42, max=1.00, mean=-2.90
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #32 — 10:55:45–10:55:48 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.13, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.30, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #33 — 10:56:21–10:56:24 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.26, max=1.00, mean=-2.90
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #34 — 10:56:57–10:57:00 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.17, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #35 — 10:57:34–10:57:37 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.13, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #36 — 10:58:10–10:58:13 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.05, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #37 — 10:58:47–10:58:49 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.05, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #38 — 10:59:22–10:59:25 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.12, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #39 — 10:59:58–11:00:01 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.17, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #40 — 11:00:34–11:00:37 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.19, max=1.00, mean=-2.91
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #41 — 11:01:10–11:01:13 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.18, max=1.00, mean=-2.94
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #42 — 11:01:46–11:01:50 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.09, max=1.00, mean=-2.91
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #43 — 11:02:23–11:02:25 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.09, max=1.00, mean=-2.93
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #44 — 11:02:58–11:03:01 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.20, max=1.00, mean=-2.94
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #45 — 11:03:42–11:03:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.13, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.31, max=1.00, mean=-2.94
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #46 — 11:04:18–11:04:21 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.30, max=1.00, mean=-2.96
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #47 — 11:04:54–11:04:57 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.28, max=1.00, mean=-2.95
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #48 — 11:05:29–11:05:32 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.22
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.22
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.24, max=1.00, mean=-2.97
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #49 — 11:06:05–11:06:08 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.25
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.26, max=1.00, mean=-2.95
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #50 — 11:06:41–11:06:53 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.28
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.28, max=1.00, mean=-2.95
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #51 — 11:07:26–11:07:29 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.13, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.23
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.33, max=1.00, mean=-2.95
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #52 — 11:08:02–11:08:05 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.13, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.23
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.36, max=1.00, mean=-2.96
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #53 — 11:08:38–11:08:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.85, max=1.00, mean=-0.23
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.52, max=1.00, mean=-2.97
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #54 — 11:09:14–11:09:17 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.85, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.55, max=1.00, mean=-2.97
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #55 — 11:09:50–11:09:53 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.65, max=1.00, mean=-2.98
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #56 — 11:10:26–11:10:29 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.66, max=1.00, mean=-2.98
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #57 — 11:11:02–11:11:05 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.74, max=1.00, mean=-2.97
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #58 — 11:11:38–11:11:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.74, max=1.00, mean=-2.97
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #59 — 11:12:14–11:12:16 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.17, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.78, max=1.00, mean=-2.95
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #60 — 11:12:49–11:12:52 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.18, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.88, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.83, max=1.00, mean=-2.96
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #61 — 11:13:25–11:13:28 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.18, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.88, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.81, max=1.00, mean=-2.95
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #62 — 11:14:01–11:14:04 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.71, max=1.00, mean=-2.91
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #63 — 11:14:37–11:14:40 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.87, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.69, max=1.00, mean=-2.91
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #64 — 11:15:12–11:15:15 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.16, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.86, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.61, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #65 — 11:15:48–11:15:51 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.15, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.85, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.47, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #66 — 11:16:24–11:16:27 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.46, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #67 — 11:17:00–11:17:04 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.44, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #68 — 11:17:37–11:17:40 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.14, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.43, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #69 — 11:18:13–11:18:16 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.13, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.84, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.40, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #70 — 11:18:49–11:18:52 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.28, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #71 — 11:19:26–11:19:28 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.24, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #72 — 11:20:02–11:20:04 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.24, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #73 — 11:20:38–11:20:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.25, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #74 — 11:21:16–11:21:19 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.25, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #75 — 11:21:52–11:21:55 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.25, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #76 — 11:22:29–11:22:32 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.25, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #77 — 11:23:05–11:23:08 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.24, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #78 — 11:23:41–11:23:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.20, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #79 — 11:24:17–11:24:19 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.23, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #80 — 11:24:53–11:24:56 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.29, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #81 — 11:25:30–11:25:33 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.12, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.83, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.15, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #82 — 11:26:06–11:26:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.17, max=1.00, mean=-2.90
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #83 — 11:26:42–11:26:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.14, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #84 — 11:27:18–11:27:21 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.11, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.82, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.13, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #85 — 11:27:54–11:27:57 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.08, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #86 — 11:28:30–11:28:33 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.07
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.06, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #87 — 11:29:06–11:29:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.98, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #88 — 11:29:42–11:29:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.96, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #89 — 11:30:18–11:30:21 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.05
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.01, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #90 — 11:30:53–11:30:56 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.10, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #91 — 11:31:29–11:31:32 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.01, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #92 — 11:32:05–11:32:08 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.97, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #93 — 11:32:41–11:32:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.06
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.00, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #94 — 11:33:17–11:33:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.95, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #95 — 11:33:53–11:33:56 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.97, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #96 — 11:34:29–11:34:32 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.10, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.01, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #97 — 11:35:05–11:35:08 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.81, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-31.00, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #98 — 11:35:41–11:35:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.93, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #99 — 11:36:17–11:36:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.91, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #100 — 11:36:53–11:36:56 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.91, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #101 — 11:37:29–11:37:35 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.98, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #102 — 11:38:08–11:38:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.94, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #103 — 11:38:44–11:38:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.90, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #104 — 11:39:20–11:39:25 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #105 — 11:39:59–11:40:02 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #106 — 11:40:35–11:40:38 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.80, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #107 — 11:41:11–11:41:14 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #108 — 11:41:47–11:41:50 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.87, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #109 — 11:42:23–11:42:26 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.87, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #110 — 11:42:59–11:43:02 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.93, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #111 — 11:43:35–11:43:37 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.94, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #112 — 11:44:10–11:44:13 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.96, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #113 — 11:44:46–11:44:49 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.94, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #114 — 11:45:22–11:45:25 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.94, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #115 — 11:45:58–11:46:00 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.87, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #116 — 11:46:33–11:46:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.87, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #117 — 11:47:10–11:47:12 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #118 — 11:47:45–11:47:48 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #119 — 11:48:21–11:48:24 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.83, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #120 — 11:48:57–11:49:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #121 — 11:49:40–11:49:43 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.79, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #122 — 11:50:16–11:50:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.76, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #123 — 11:50:53–11:50:55 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #124 — 11:51:28–11:51:31 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.80, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #125 — 11:52:04–11:52:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.78, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #126 — 11:52:40–11:52:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.76, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #127 — 11:53:15–11:53:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #128 — 11:53:51–11:53:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.74, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #129 — 11:54:28–11:54:31 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.76, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #130 — 11:55:04–11:55:07 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #131 — 11:55:40–11:55:43 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.69, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #132 — 11:56:16–11:56:19 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #133 — 11:56:52–11:57:04 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.62, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #134 — 11:57:37–11:57:40 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.57, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #135 — 11:58:13–11:58:16 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.61, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #136 — 11:58:50–11:58:53 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #137 — 11:59:26–11:59:29 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #138 — 12:00:02–12:00:05 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.66, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #139 — 12:00:38–12:00:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.65, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #140 — 12:01:14–12:01:17 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.65, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #141 — 12:02:00–12:02:03 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #142 — 12:02:36–12:02:39 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #143 — 12:03:12–12:03:15 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.80, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #144 — 12:03:49–12:03:52 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.76, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #145 — 12:04:25–12:04:28 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #146 — 12:05:01–12:05:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #147 — 12:05:51–12:05:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.88, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #148 — 12:06:27–12:06:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.90
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #149 — 12:07:03–12:07:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.90
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #150 — 12:07:39–12:07:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.80, max=1.00, mean=-2.90
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #151 — 12:08:16–12:08:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #152 — 12:08:51–12:08:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #153 — 12:09:27–12:09:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.83, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #154 — 12:10:03–12:10:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #155 — 12:10:38–12:10:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #156 — 12:11:19–12:11:21 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.83, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #157 — 12:11:55–12:11:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.89, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #158 — 12:12:31–12:12:33 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.93, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #159 — 12:13:06–12:13:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.88, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #160 — 12:13:42–12:13:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.89
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #161 — 12:14:47–12:15:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #162 — 12:15:55–12:15:55 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #163 — 12:16:23–12:16:26 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.77, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #164 — 12:17:03–12:17:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.75, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #165 — 12:17:40–12:17:50 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #166 — 12:18:23–12:18:26 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.70, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #167 — 12:18:59–12:19:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.69, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #168 — 12:19:45–12:19:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.70, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #169 — 12:20:20–12:20:23 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.70, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #170 — 12:20:56–12:20:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #171 — 12:21:32–12:21:35 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #172 — 12:22:08–12:22:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #173 — 12:22:44–12:22:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.88
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #174 — 12:23:20–12:23:23 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.66, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #175 — 12:23:56–12:23:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.66, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #176 — 12:24:33–12:24:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.69, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #177 — 12:25:09–12:25:12 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.69, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #178 — 12:25:45–12:25:49 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #179 — 12:26:22–12:26:25 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #180 — 12:26:58–12:27:01 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #181 — 12:27:34–12:27:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #182 — 12:28:09–12:28:12 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #183 — 12:28:45–12:28:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.66, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #184 — 12:29:31–12:29:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #185 — 12:30:07–12:30:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.66, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #186 — 12:30:43–12:30:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.66, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #187 — 12:31:19–12:31:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.70, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #188 — 12:31:55–12:31:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #189 — 12:32:31–12:32:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #190 — 12:33:07–12:33:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #191 — 12:33:43–12:33:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #192 — 12:34:19–12:34:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #193 — 12:34:55–12:34:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #194 — 12:35:31–12:35:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.72, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #195 — 12:36:07–12:36:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #196 — 12:36:43–12:36:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #197 — 12:37:19–12:37:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #198 — 12:37:55–12:37:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.72, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #199 — 12:38:31–12:38:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.78, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #200 — 12:39:07–12:39:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.78, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #201 — 12:39:43–12:39:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #202 — 12:40:19–12:40:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.77, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #203 — 12:40:55–12:40:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.75, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #204 — 12:41:31–12:41:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.83, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #205 — 12:42:07–12:42:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.91, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #206 — 12:42:43–12:42:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.91, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #207 — 12:43:19–12:43:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.83, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #208 — 12:43:55–12:43:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #209 — 12:44:31–12:44:33 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #210 — 12:45:06–12:45:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #211 — 12:45:42–12:45:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.08
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #212 — 12:46:18–12:46:21 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #213 — 12:46:54–12:46:57 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #214 — 12:47:30–12:47:33 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.88, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #215 — 12:48:06–12:48:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.89, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #216 — 12:48:42–12:48:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #217 — 12:49:18–12:49:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.85, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #218 — 12:49:54–12:50:08 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.85, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #219 — 12:50:41–12:50:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.79, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #220 — 12:51:17–12:51:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.74, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #221 — 12:51:56–12:51:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.74, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #222 — 12:52:32–12:52:35 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.72, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #223 — 12:53:08–12:53:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #224 — 12:53:44–12:53:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.70, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #225 — 12:54:20–12:54:23 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #226 — 12:54:56–12:54:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #227 — 12:55:32–12:55:35 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #228 — 12:56:08–12:56:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.74, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #229 — 12:56:44–12:56:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.76, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #230 — 12:57:20–12:57:23 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.72, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #231 — 12:57:56–12:57:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #232 — 12:58:32–12:58:35 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.69, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #233 — 12:59:08–12:59:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.69, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #234 — 12:59:44–12:59:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #235 — 13:00:20–13:00:23 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #236 — 13:00:56–13:00:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.70, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #237 — 13:01:32–13:01:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #238 — 13:02:07–13:02:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.68, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #239 — 13:02:44–13:02:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #240 — 13:03:19–13:03:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.75, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #241 — 13:03:55–13:03:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.77, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #242 — 13:04:32–13:04:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.71, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #243 — 13:05:08–13:05:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.72, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #244 — 13:05:44–13:05:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.70, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #245 — 13:06:19–13:06:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.76, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #246 — 13:06:56–13:06:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.21
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.74, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #247 — 13:07:31–13:07:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.80, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #248 — 13:08:07–13:08:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.75, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #249 — 13:08:43–13:08:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.83, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #250 — 13:09:19–13:09:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.88, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #251 — 13:09:56–13:09:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.88, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #252 — 13:10:31–13:10:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.92, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #253 — 13:11:07–13:11:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.87, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #254 — 13:11:43–13:11:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #255 — 13:12:19–13:12:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.92, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #256 — 13:12:55–13:12:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.90, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #257 — 13:13:31–13:13:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.95, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #258 — 13:14:07–13:14:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.94, max=1.00, mean=-2.87
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #259 — 13:14:43–13:14:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.92, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #260 — 13:15:19–13:15:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.91, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #261 — 13:15:55–13:15:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.91, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #262 — 13:16:31–13:16:33 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.89, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #263 — 13:17:07–13:17:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.89, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #264 — 13:17:43–13:17:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #265 — 13:18:19–13:18:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #266 — 13:18:55–13:18:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.09
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.89, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #267 — 13:19:33–13:19:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #268 — 13:20:15–13:20:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.85, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #269 — 13:20:52–13:20:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.82, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #270 — 13:21:28–13:21:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #271 — 13:22:03–13:22:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.92, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #272 — 13:22:39–13:22:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.93, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #273 — 13:23:16–13:23:19 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.94, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #274 — 13:23:52–13:23:55 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.92, max=1.00, mean=-2.86
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #275 — 13:24:28–13:24:31 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.09, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.93, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #276 — 13:25:04–13:25:07 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.88, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #277 — 13:25:40–13:25:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #278 — 13:26:15–13:26:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.88, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #279 — 13:26:51–13:26:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.87, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #280 — 13:27:27–13:27:29 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.83, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #281 — 13:28:02–13:28:05 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.84, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #282 — 13:28:38–13:28:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.80, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.86, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #283 — 13:29:15–13:29:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.08, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.81, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #284 — 13:29:51–13:29:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.79, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #285 — 13:30:27–13:30:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.75, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #286 — 13:31:03–13:31:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.79, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.75, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #287 — 13:31:39–13:31:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.73, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #288 — 13:32:15–13:32:17 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.07, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.72, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #289 — 13:32:51–13:32:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.69, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #290 — 13:33:27–13:33:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #291 — 13:34:03–13:34:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.59, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #292 — 13:34:39–13:34:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.61, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #293 — 13:35:14–13:35:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.61, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #294 — 13:35:51–13:35:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.61, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #295 — 13:36:27–13:36:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.64, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #296 — 13:37:03–13:37:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.78, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.67, max=1.00, mean=-2.85
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #297 — 13:37:39–13:37:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.06, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.61, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #298 — 13:38:15–13:38:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.60, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #299 — 13:38:51–13:38:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.60, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #300 — 13:39:27–13:39:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.59, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #301 — 13:40:03–13:40:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.57, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #302 — 13:40:39–13:40:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.59, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #303 — 13:41:15–13:41:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.57, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #304 — 13:41:51–13:41:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.53, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #305 — 13:42:27–13:42:30 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.53, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #306 — 13:43:03–13:43:06 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.54, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #307 — 13:43:39–13:43:42 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.52, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #308 — 13:44:15–13:44:18 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.52, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #309 — 13:44:51–13:44:54 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.77, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.52, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #310 — 13:45:29–13:45:32 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.50, max=1.00, mean=-2.84
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #311 — 13:46:05–13:46:08 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.48, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #312 — 13:46:41–13:46:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.50, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #313 — 13:47:17–13:47:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.05, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.50, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #314 — 13:47:53–13:47:56 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.48, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #315 — 13:48:30–13:48:32 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.46, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #316 — 13:49:05–13:49:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.48, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #317 — 13:49:41–13:49:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.11
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.46, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #318 — 13:50:17–13:50:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.47, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #319 — 13:50:55–13:50:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.45, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #320 — 13:51:31–13:51:34 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #321 — 13:52:07–13:52:10 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #322 — 13:52:43–13:52:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #323 — 13:53:19–13:53:22 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.42, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #324 — 13:53:55–13:53:59 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.10
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.40, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #325 — 13:54:32–13:54:35 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.46, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #326 — 13:55:08–13:55:11 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.47, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #327 — 13:55:44–13:55:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.45, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #328 — 13:56:21–13:56:24 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.45, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #329 — 13:56:57–13:57:00 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.44, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #330 — 13:57:33–13:57:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.44, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #331 — 13:58:09–13:58:12 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.40, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #332 — 13:58:45–13:58:48 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.40, max=1.00, mean=-2.80
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #333 — 13:59:21–13:59:24 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.42, max=1.00, mean=-2.79
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #334 — 13:59:57–14:00:00 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.41, max=1.00, mean=-2.80
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #335 — 14:00:33–14:00:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.40, max=1.00, mean=-2.80
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #336 — 14:01:09–14:01:12 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.42, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #337 — 14:01:45–14:01:47 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #338 — 14:02:21–14:02:24 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #339 — 14:02:57–14:03:00 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.36, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #340 — 14:03:33–14:03:36 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.34, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #341 — 14:04:10–14:04:13 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.37, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #342 — 14:04:46–14:04:49 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.34, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #343 — 14:05:23–14:05:26 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.30, max=1.00, mean=-2.80
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #344 — 14:05:59–14:06:02 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.30, max=1.00, mean=-2.80
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #345 — 14:06:35–14:06:38 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.02, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.30, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #346 — 14:07:12–14:07:14 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.02, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.30, max=1.00, mean=-2.80
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #347 — 14:07:47–14:07:51 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.32, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #348 — 14:08:24–14:08:27 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #349 — 14:09:01–14:09:04 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.42, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #350 — 14:09:37–14:09:40 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #351 — 14:10:13–14:10:16 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #352 — 14:10:49–14:10:52 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.20
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #353 — 14:11:25–14:11:28 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #354 — 14:12:01–14:12:04 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.41, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #355 — 14:12:37–14:12:40 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.42, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #356 — 14:13:14–14:13:16 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.42, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #357 — 14:13:50–14:13:52 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.80
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #358 — 14:14:25–14:14:28 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #359 — 14:15:02–14:15:04 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #360 — 14:15:38–14:15:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.37, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #361 — 14:16:14–14:16:17 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.35, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #362 — 14:16:50–14:16:52 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.35, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #363 — 14:17:26–14:17:28 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.36, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #364 — 14:18:02–14:18:05 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.37, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #365 — 14:18:38–14:18:41 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #366 — 14:19:14–14:19:17 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.37, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #367 — 14:19:50–14:19:53 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.35, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #368 — 14:20:26–14:20:29 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.37, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #369 — 14:21:02–14:21:05 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.37, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #370 — 14:21:44–14:21:46 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.37, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #371 — 14:22:20–14:22:23 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.44, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #372 — 14:22:56–14:22:58 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #373 — 14:23:40–14:23:43 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.19
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #374 — 14:24:16–14:24:19 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #375 — 14:24:52–14:24:55 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.17
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.39, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #376 — 14:25:28–14:25:31 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.14
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #377 — 14:26:04–14:26:08 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.43, max=1.00, mean=-2.81
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #378 — 14:26:41–14:26:44 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.15
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.40, max=1.00, mean=-2.82
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #379 — 14:27:17–14:27:20 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.38, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #380 — 14:27:53–14:27:56 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.03, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.13
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.40, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #381 — 14:28:29–14:28:32 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.18
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.75, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.40, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #382 — 14:29:06–14:29:09 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.46, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

### #383 — 14:29:42–14:29:45 ET
- **Account state**: FLAT
- **Any BUY setup present?**: NO
- **SPY**:
  - obs stats: min=-3.04, max=1.00, mean=-0.16
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **QQQ**:
  - obs stats: min=-2.76, max=1.00, mean=-0.12
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.
- **SPX**:
  - obs stats: min=-30.46, max=1.00, mean=-2.83
  - raw: 4 (TRIM 70%)
  - final: 0 (HOLD) | source=RL | strength=0.3
  - **Why this prevented an entry**: model chose a sell/exit-type action while no position existed → engine converts it to HOLD, so no trade is opened.

## Appendix A — 09:30–10:36 “unknown‑raw HOLD cycles”

During this window your logs recorded the **final action** per symbol (almost always HOLD), but **did not yet include the debug line** `Raw RL Output: action_raw=...`.

So we can confirm **what the bot ended up doing** (HOLD), but we cannot confirm whether the model originally wanted TRIM/EXIT vs HOLD before the safety remap.

### Counts

- **Window**: 09:30:00–10:36:35 ET
- **Cycles found (from RL Inference lines)**: 115
- **All‑HOLD cycles**: 115

### Cycles (exhaustive)

#### A1 — 09:30:04–09:30:05 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A2 — 09:30:39–09:30:41 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A3 — 09:31:15–09:31:17 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A4 — 09:31:51–09:31:52 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A5 — 09:32:26–09:32:27 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A6 — 09:33:01–09:33:03 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A7 — 09:33:37–09:33:39 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A8 — 09:34:13–09:34:15 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A9 — 09:34:49–09:34:50 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A10 — 09:35:24–09:35:26 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A11 — 09:36:00–09:36:02 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A12 — 09:36:36–09:36:38 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A13 — 09:37:12–09:37:14 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A14 — 09:37:47–09:37:49 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A15 — 09:38:23–09:38:25 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A16 — 09:38:59–09:39:00 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A17 — 09:39:34–09:39:36 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A18 — 09:40:10–09:40:12 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A19 — 09:40:46–09:40:48 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A20 — 09:41:22–09:41:24 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A21 — 09:41:58–09:42:00 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A22 — 09:42:34–09:42:35 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A23 — 09:43:09–09:43:11 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A24 — 09:43:45–09:43:47 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A25 — 09:44:21–09:44:23 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A26 — 09:44:57–09:44:58 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A27 — 09:45:32–09:45:34 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A28 — 09:46:08–09:46:10 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A29 — 09:46:44–09:46:46 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A30 — 09:47:20–09:47:22 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A31 — 09:47:56–09:47:57 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A32 — 09:48:31–09:48:33 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A33 — 09:49:07–09:49:09 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A34 — 09:49:43–09:49:45 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A35 — 09:50:19–09:50:21 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A36 — 09:50:55–09:50:57 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A37 — 09:51:30–09:51:32 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A38 — 09:52:06–09:52:08 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A39 — 09:52:42–09:52:44 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A40 — 09:53:18–09:53:20 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A41 — 09:53:54–09:53:55 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A42 — 09:54:29–09:54:31 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A43 — 09:55:05–09:55:07 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A44 — 09:55:41–09:55:43 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A45 — 09:56:17–09:56:18 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A46 — 09:56:52–09:56:54 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A47 — 09:57:28–09:57:30 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A48 — 09:58:04–09:58:06 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A49 — 09:58:40–09:58:41 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A50 — 09:59:15–09:59:17 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A51 — 09:59:51–09:59:53 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A52 — 10:00:27–10:00:29 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A53 — 10:01:03–10:01:05 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A54 — 10:01:39–10:01:40 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A55 — 10:02:14–10:02:16 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A56 — 10:02:50–10:02:52 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A57 — 10:03:26–10:03:28 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A58 — 10:04:02–10:04:03 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A59 — 10:04:37–10:04:39 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A60 — 10:05:13–10:05:15 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A61 — 10:05:49–10:05:51 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A62 — 10:06:25–10:06:27 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A63 — 10:07:01–10:07:02 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A64 — 10:07:36–10:07:38 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A65 — 10:08:12–10:08:14 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A66 — 10:08:48–10:08:50 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A67 — 10:09:24–10:09:25 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A68 — 10:09:59–10:10:01 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A69 — 10:10:35–10:10:37 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A70 — 10:11:11–10:11:13 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A71 — 10:11:47–10:11:48 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A72 — 10:12:22–10:12:24 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A73 — 10:12:58–10:13:00 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A74 — 10:13:34–10:13:35 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A75 — 10:14:09–10:14:11 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A76 — 10:14:45–10:14:47 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A77 — 10:15:21–10:15:22 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A78 — 10:15:56–10:15:58 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A79 — 10:16:32–10:16:34 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A80 — 10:17:08–10:17:10 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A81 — 10:17:44–10:17:46 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A82 — 10:18:20–10:18:21 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A83 — 10:18:55–10:18:57 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A84 — 10:19:31–10:19:33 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A85 — 10:20:07–10:20:08 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A86 — 10:20:42–10:20:44 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A87 — 10:21:18–10:21:20 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A88 — 10:21:54–10:21:56 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A89 — 10:22:30–10:22:31 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A90 — 10:23:05–10:23:07 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A91 — 10:23:41–10:23:43 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A92 — 10:24:17–10:24:19 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A93 — 10:24:53–10:24:55 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A94 — 10:25:29–10:25:30 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A95 — 10:26:04–10:26:06 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A96 — 10:26:40–10:26:42 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A97 — 10:27:16–10:27:18 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A98 — 10:27:52–10:27:53 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A99 — 10:28:27–10:28:29 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A100 — 10:29:03–10:29:05 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A101 — 10:29:39–10:29:41 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A102 — 10:30:15–10:30:17 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A103 — 10:30:50–10:30:52 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A104 — 10:31:26–10:31:28 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A105 — 10:32:02–10:32:04 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A106 — 10:32:38–10:32:40 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A107 — 10:33:14–10:33:15 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A108 — 10:33:49–10:33:51 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A109 — 10:34:25–10:34:27 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A110 — 10:35:01–10:35:03 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A111 — 10:35:36–10:35:38 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A112 — 10:36:12–10:36:14 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A113 — 10:36:24–10:36:25 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A114 — 10:36:31–10:36:33 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.

#### A115 — 10:36:34–10:36:34 ET
- **SPY**: final action=0 (HOLD) | source=RL | strength=0.500
- **QQQ**: final action=0 (HOLD) | source=RL | strength=0.500
- **What it means (beginner)**: the bot decided to **do nothing** in this moment, so it did not place a trade.
