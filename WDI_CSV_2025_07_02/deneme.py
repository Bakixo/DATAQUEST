import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# --------------------
# Dosya yolları (aynı klasörden oku)
# --------------------
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# Bazı CSV'ler latin-1 ile gelebilir; oku -> gerekirse fallback dene yani kısaca güvenli okuma yapıyoruz.
def read_csv_safely(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")

wdi  = read_csv_safely("WDICSV.csv")
meta = read_csv_safely("WDICountry.csv")

# --------------------
# CO2 göstergesini veri setinden otomatik seç
# --------------------
available = set(wdi["Indicator Code"].dropna().unique())

preferred_co2 = [
    "EN.GHG.CO2.PC.CE.AR5",  # per-capita (AR5, consumption-based) - en iyisi
    "EN.ATM.CO2E.PC",        # eski per-capita
    "EN.GHG.CO2.RT.GDP.KD",  # CO2 intensity (düşük=iyi)
    "EN.GHG.CO2.MT.CE.AR5",  # total CO2 (milyon ton) - son çare
]
co2_code = next((c for c in preferred_co2 if c in available), None)
if co2_code is None:
    raise RuntimeError("Uygun CO2 göstergesi bulunamadı. Veri setinizde CO2 ile başlayan kodları kontrol edin.")

print("CO2 göstergesi olarak kullanılacak kod:", co2_code)

# --------------------
# 1) 6 göstergede filtrele
# --------------------
keep = [
    "SP.DYN.LE00.IN",      # Life expectancy
    "NY.GDP.PCAP.KD",      # GDP per capita (constant)
    "SE.SEC.ENRR",         # School enrollment, secondary (% net)
    "EG.USE.PCAP.KG.OE",   # Energy use (kg oil eq per capita)
    "EN.ATM.PM25.MC.M3",   # PM2.5 exposure (µg/m3)
    co2_code               # otomatik seçilen CO2
]
sub = wdi[wdi["Indicator Code"].isin(keep)].copy() # yanlızca bu 6 göstergenin satırları

# --------------------
# 2) geniş -> uzun form yani yılları tek sütuna alıyoruz çünkü böyle bir veride uzun form analiz için daha iyi.
# --------------------
id_cols = ["Country Name","Country Code","Indicator Name","Indicator Code"]
year_cols = [c for c in sub.columns if str(c).isdigit()]
long = sub.melt(id_vars=id_cols, value_vars=year_cols,
                var_name="Year", value_name="Value")
long["Year"] = long["Year"].astype(int)

# --------------------
# 3) pivot (ülke-yıl tek satır)
# --------------------
panel = (long
         .pivot_table(index=["Country Code","Country Name","Year"],
                      columns="Indicator Code", values="Value")
         .reset_index())

# Kodları kısa adlara çevir
rename_map = {
    "SP.DYN.LE00.IN": "LE",
    "NY.GDP.PCAP.KD": "GDPpc",
    "SE.SEC.ENRR": "School",
    "EG.USE.PCAP.KG.OE": "EnergyUse",
    "EN.ATM.PM25.MC.M3": "PM25",
}
if co2_code in ("EN.GHG.CO2.PC.CE.AR5", "EN.ATM.CO2E.PC"):
    rename_map[co2_code] = "CO2pc"
elif co2_code == "EN.GHG.CO2.RT.GDP.KD":
    rename_map[co2_code] = "CO2intensity"
elif co2_code == "EN.GHG.CO2.MT.CE.AR5":
    rename_map[co2_code] = "CO2_total"

panel = panel.rename(columns=rename_map)

# --------------------
# 4) meta ekle + yıl aralığı + temizlik
# --------------------
meta_keep = meta[["Country Code","Region","Income Group"]].drop_duplicates()
df = panel.merge(meta_keep, on="Country Code", how="left")

# Toplamlar/bölgeler (Region NaN olanlar) dışarı
df = df[df["Region"].notna()]

# Yıl filtresi
df = df[(df["Year"]>=2000) & (df["Year"]<=2023)].sort_values(["Country Code","Year"])

# Ülke bazında kısmi interpolasyon (≤3 yıl boşluk)
val_cols_base = ["LE","GDPpc","School","EnergyUse","PM25"]
if "CO2pc" in df.columns:
    val_cols = val_cols_base + ["CO2pc"]
elif "CO2intensity" in df.columns:
    val_cols = val_cols_base + ["CO2intensity"]
elif "CO2_total" in df.columns:
    val_cols = val_cols_base + ["CO2_total"]
else:
    raise RuntimeError("CO2 için beklenen sütun bulunamadı (rename aşaması).")

for c in val_cols:
    df[c] = df.groupby("Country Code")[c].transform(
        lambda s: s.interpolate(limit=3, limit_direction="both")
    )

# Uç değer kırpma (winsorize ~1%)
for c in val_cols:
    q1, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
    df[c] = df[c].clip(q1, q99)

# Temiz paneli kaydediyoruz.
PROC.mkdir(parents=True, exist_ok=True)
(df[val_cols + ["Country Code","Country Name","Year","Region","Income Group"]]
 .to_csv(PROC/"mini_panel_clean.csv", index=False))

# --------------------
# 5) SKV_v0 (hızlı ilk skor)
# --------------------
out_cols = ["LE","GDPpc","School"]         # iyi: yüksek olsun
# giriş setini CO2'ye göre kur
in_cols = ["EnergyUse","PM25"]             # kötü: düşük olsun
if "CO2pc" in df.columns:
    in_cols.append("CO2pc")
elif "CO2intensity" in df.columns:
    in_cols.append("CO2intensity")
elif "CO2_total" in df.columns:
    in_cols.append("CO2_total")

# MinMax + girdileri ters çevir yani 0 ile 1 arasına çek + SKV_v0 hesapla.
sc_out = MinMaxScaler()
sc_in  = MinMaxScaler()
df[[f"{c}_n" for c in out_cols]] = sc_out.fit_transform(df[out_cols])
df[[f"{c}_n" for c in in_cols ]] = sc_in.fit_transform(df[in_cols ])
for c in in_cols:
    df[f"{c}_n"] = 1 - df[f"{c}_n"]

df["SKV_v0"] = df[[f"{c}_n" for c in out_cols]].mean(axis=1) - \
               df[[f"{c}_n" for c in in_cols]].mean(axis=1)

df.to_csv(PROC/"skv_v0.csv", index=False)

print("✅ Çıktılar yazıldı:",
      PROC/"mini_panel_clean.csv",
      PROC/"skv_v0.csv", sep="\n - ")
