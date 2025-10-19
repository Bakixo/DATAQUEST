# viz.py
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

PROC = Path("data/processed")
FIGS = Path("reports/figures"); FIGS.mkdir(parents=True, exist_ok=True)
TABS = Path("reports/tables");  TABS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC/"skv_v0.csv", low_memory=False)

# --- Sanity checks
print("Yıl aralığı:", int(df["Year"].min()), "→", int(df["Year"].max()))
print("Ülke sayısı:", df["Country Code"].nunique())
print("Örnek sütunlar:", [c for c in df.columns if c not in ["Country Code","Country Name","Year","Region","Income Group"]][:10])

# Coverage (son 5 yıl)
cov = (df[df["Year"]>=2019]
       .groupby("Country Code")["SKV_v0"]
       .apply(lambda s: s.notna().mean())
       .reset_index(name="coverage_2019_2023"))
cov.to_csv(TABS/"coverage_last5.csv", index=False)

# --- Harita: SKV_v0 (2019–2023 ort)
last5 = (df[df["Year"]>=2019]
         .groupby(["Country Code","Country Name"], as_index=False)["SKV_v0"]
         .mean())
fig_map = px.choropleth(last5, locations="Country Code", color="SKV_v0",
                        hover_name="Country Name", title="SKV_v0 (2019–2023)")
fig_map.write_html(str(FIGS/"map_skv_v0_last5.html"))
print("saved:", FIGS/"map_skv_v0_last5.html")

# --- Scatter: CO2 vs Life Expectancy (2020)
yr = 2020
need = df[df["Year"]==yr][["Country Name","Country Code","LE","SKV_v0"]].copy()
# CO2 kolonu isimlendirmesi dosyaya göre değişebilir: üç olası isimden hangisi varsa onu al garanti olsun diye yaptık.
co2_col = next((c for c in ["CO2pc","CO2intensity","CO2_total"] if c in df.columns), None)
if co2_col:
    need[co2_col] = df[df["Year"]==yr][co2_col].values
    fig_sc = px.scatter(need, x=co2_col, y="LE", color="SKV_v0", hover_name="Country Name",
                        title=f"{co2_col} vs Life Expectancy ({yr})")
    fig_sc.write_html(str(FIGS/f"scatter_{co2_col}_LE_{yr}.html"))
    print("saved:", FIGS/f"scatter_{co2_col}_LE_{yr}.html")

# --- Lig tabloları (top/bottom 10)
league = last5.sort_values("SKV_v0", ascending=False)
league.head(10).to_csv(TABS/"top10_skv_v0_last5.csv", index=False)
league.tail(10).to_csv(TABS/"bottom10_skv_v0_last5.csv", index=False)
print("saved league tables:", TABS/"top10_skv_v0_last5.csv", TABS/"bottom10_skv_v0_last5.csv")

# --- Trend örneği: Türkiye + 3 referans (değiştirebilirsin)
watch = ["TUR","NOR","DEU","IND"]
trend = df[df["Country Code"].isin(watch)][["Country Code","Country Name","Year","SKV_v0"]]
trend.to_csv(TABS/"trend_watch.csv", index=False)
print("saved trend subset:", TABS/"trend_watch.csv")

# --- Pozitif sapkınlar: gelir grubuna göre z-skor (son 5 yıl)
tmp = (df[df["Year"]>=2019]
       .groupby(["Income Group","Country Code","Country Name"], as_index=False)["SKV_v0"]
       .mean())
tmp["z_in_group"] = tmp.groupby("Income Group")["SKV_v0"].transform(lambda s: (s - s.mean())/s.std(ddof=0))
pos_dev = tmp[tmp["z_in_group"]>=1.0].sort_values(["Income Group","z_in_group"], ascending=[True,False])
pos_dev.to_csv(TABS/"positive_deviants_last5.csv", index=False)
print("saved:", TABS/"positive_deviants_last5.csv")
