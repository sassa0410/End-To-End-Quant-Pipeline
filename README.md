# End-to-End Quant Pipeline: ML Volatility Forecasting & MILP Portfolio Optimization

A complete quantitative infrastructure developed in Python for automating the rebalancing of a cryptocurrency portfolio. The system integrates Data Engineering, Machine Learning for risk forecasting (Dynamic Horse Race) and Operations Research (Gurobi) for portfolio allocation optimization with real market constraints.

## System Architecture

The pipeline is fully automated and orchestrated via the `main.py` entry point.
It is divided in 4 sequential components:

1. **Data Engineering (`build_features.py`)**
   * Processing of high-frequency historical data saved in `.parquet` format to maximize I/O efficiency.
   * Vectorized calculation of financial features: logarithmic returns, rolling volatility (1h, 4h) and momentum.
   
2. **Machine Learning & Model Selection (`train_model.py`)**
   * Implementation of an automated *Horse Race* (Model Selection) to predict future volatility (Target: 60m).
   * Models in competition: **Ridge Regression** (Baseline), **Random Forest** (Robustness) and **XGBoost** (Gradient Boosting to capture non-linear relationships).
   * Dynamic prediction of future volatility passed to the optimizer.

3. **Operations Research (`portfolio_optimization.py`)**
   * Mathematical model **MILP** (Mixed-Integer Linear Programming) solved via **Gurobi**.
   * **Objective:** Minimization of the expected risk of the portfolio penalizing the turnover.
   * **Constraints implemented:**
     * Maximum exposure limits for individual assets (max 30% to ensure diversification).
     * Real percentage transaction costs (0.1% fee).
     * Logical constraints (Big-M) and binary variables to impose a minimum turnover and avoid speculative over-trading.

4. **Automated Reporting (`generate_report.py`)**
   * Programmatic generation of an executive **Tear Sheet in PDF** (`fpdf2` e `matplotlib`).
   * Clear visualization of the Execution Plan, initial vs optimal weights and risk profile.


## Tech Stack
* **Programming Language:** Python 3
* **Data Processing:** Pandas, NumPy, PyArrow
* **Machine Learning:** Scikit-learn, XGBoost
* **Mathematical Optimization:** Gurobi (`gurobipy`)
* **Reporting:** Matplotlib, FPDF2


## How to Run the Project
The pipeline is orchestrated to run in sequence with a single command.

1. Clone the repository and install the dependencies:
   ```bash
   git clone https://github.com/sassa0410/End-To-End-Quant-Pipeline.git
   cd End-To-End-Quant-Pipeline
   pip install -r requirements.txt
   python main.py
   ```
