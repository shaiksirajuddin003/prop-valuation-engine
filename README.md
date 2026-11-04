# 🏢 VALUEX: Institutional AI Real Estate Valuation Engine (AVM)

An enterprise-grade Automated Valuation Model (AVM) and investment underwriting platform that calculates fair-market property valuations, unit pricing per square foot, and gross rental yields using gradient-boosted regression.

## 🚀 Key Features
- **Gradient-Boosted AVM Core:** Trained with `XGBRegressor` on multi-vector real estate telemetry (built-up area, micro-locality multipliers, transit proximity, floor-rise premiums, and age depreciation).
- **Portfolio Telemetry Ingestion Hub:** Ingests batch spreadsheets (`.xlsx`, `.csv`) to perform portfolio-wide valuations and identify undervalued/overvalued inventory.
- **Micro-Market Analytics:** Tracks locality price benchmarks, rate/sq.ft variations, and structural format yields.
- **Investment Underwriting Dossier:** Exports automated valuation memorandums with 95% confidence intervals and projected monthly cap rates (JSON format).

## 📁 Repository Structure
- `app.py`: Institutional dark-theme Streamlit dashboard.
- `train_model.py`: Automated regression training & feature alignment pipeline.
- `data/properties_1000.xlsx`: Verified baseline dataset containing 1,000 property transaction records.
- `requirements.txt`: Cloud environment dependencies.

## 💻 Local Setup
```bash
# 1. Clone repository
git clone [https://github.com/shaiksirajuddin003/prop-valuation-engine.git](https://github.com/shaiksirajuddin003/prop-valuation-engine.git)
cd prop-valuation-engine

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install packages
pip install -r requirements.txt

# 4. Launch terminal
streamlit run app.py 