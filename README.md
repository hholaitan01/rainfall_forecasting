# Rainfall–Runoff Forecasting & Hydraulic Channel Design Decision-Support System

## Problem Statement

This tool provides an engineering decision-support system combining:

1. **Machine learning-based rainfall–runoff modelling** (XGBoost) for hydrological forecasting and flood risk assessment.
2. **Standards-based hydraulic channel design** (Rational Method + Manning's Equation) for drainage infrastructure sizing.

The system processes historical hydro-meteorological data (rainfall, discharge, temperature, pressure, wind speed, radiation) to train predictive models and generate design-grade channel sizing outputs.

---

## Engineering Philosophy

> **ML outputs are NEVER used to size channels.**

This is non-negotiable. ML models capture basin-scale hydrological behaviour across a large upstream catchment. They are appropriate for:

- Hydrological behaviour analysis
- Future discharge forecasting
- Flood risk assessment and early warning

Channel design must follow established engineering standards:

- **Rational Method**: Q = 0.278 × C × i × A
- **IDF Rainfall Intensity**: i = A / (Tc + B)  (Ikeja, Nigeria — 10-year return period)
- **Manning's Equation**: Q = (1/n) × A × R^(2/3) × S^(1/2)
- **Best Hydraulic Section** assumptions for both rectangular and trapezoidal channels

---

## ML Limitations

The XGBoost model:

- Is trained on historical monthly gauge data representing the entire river basin upstream of the gauge station.
- The gauge catchment is typically orders of magnitude larger than the local 0.5 km² design catchment.
- Peak ML discharge (e.g. 43 m³/s) vs rational method design discharge (e.g. 0.17 m³/s) reflects this scale difference — it is **not a design error**, it is **physically correct**.
- ML predictions carry uncertainty quantified by ±1 RMSE bands.
- Extrapolating beyond the observed historical range carries elevated uncertainty.

---

## Design Assumptions

| Parameter | Value | Basis |
|-----------|-------|-------|
| IDF constants (A, B) | 8266, 63.1 | Ikeja, Nigeria — 10-year return period |
| Manning's n | 0.015 | Concrete-lined channel |
| Runoff coefficient C | 0.75 | Mixed urban/semi-urban catchment |
| Catchment area | 0.5 km² | Local design catchment |
| Channel bed slope | 0.002 | Site survey |
| Side slope (trapezoidal) | z = 1.0 | 45° — best hydraulic section |
| Freeboard | F = 0.3 + 0.25y | Standard engineering practice |
| Velocity limits | 0.6 – 6.0 m/s | Self-cleansing to non-erosive |

---

## Proper Use

✅ Use the **Channel Design tab** outputs (Q_rational, dimensions) for construction drawings.  
✅ Use the **Flood Risk tab** to understand basin vulnerability and plan supplementary measures.  
✅ Use the **Forecast tab** for operational planning and early-warning system development.  
✅ Use the **Model Validation tab** to verify ML performance before trusting forecasts.

## Misuse Warnings

❌ Do NOT substitute Q_ML for Q_rational in any design calculation.  
❌ Do NOT use forecast values as guaranteed future conditions — they carry RMSE uncertainty.  
❌ Do NOT apply the 10-year IDF coefficients outside their geographical validity (Ikeja region).  
❌ Do NOT use Manning's n = 0.015 for unlined or grass channels — adjust accordingly.

---

## Project Structure

```
project/
├── engine/                    # Pure computation — no Streamlit
│   ├── data_loader.py         # CSV loading and cleaning
│   ├── feature_engineering.py # Lags, rolling stats, seasonality
│   ├── ml_model.py            # XGBoost training / loading
│   ├── forecast.py            # Recursive 12-month forecasting
│   ├── metrics.py             # RMSE, R², residual stats
│   ├── hydraulics.py          # Rational method + IDF
│   ├── channel_design.py      # Manning's equation solver
│   └── risk_assessment.py     # Flood risk comparison
│
├── visuals/
│   ├── plots.py               # Matplotlib time-series, scatter, residuals
│   └── channel_sections.py    # True-scale channel cross-sections
│
├── app.py                     # Streamlit UI only
├── requirements.txt
└── README.md
```

---

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload your processed CSV file (with a `date` index containing `rainfall` and `discharge` columns).

---

## References

- Manning, R. (1891). On the flow of water in open channels and pipes. *Trans. ICE Ireland*, 20, 161–207.
- ASCE. (1996). *Hydrology Handbook* (2nd ed.). American Society of Civil Engineers.
- Chow, V.T. (1959). *Open Channel Hydraulics*. McGraw-Hill.
- Chen & Liu (1987). IDF relationships for Lagos/Ikeja region.
