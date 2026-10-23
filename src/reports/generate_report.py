import pandas as pd
from fpdf import FPDF, XPos, YPos
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class QuantReport(FPDF):
    def header(self):
        # Header del Documento
        self.set_font("helvetica", style='B', size=14)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, text=" End-to-End Quant Pipeline: Strategy Tear Sheet ", new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_font("helvetica", style='I', size=10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, text="Machine Learning Volatility Forecasting & Gurobi MILP Portfolio Optimization", new_x="LMARGIN", new_y="NEXT", align='C')
        self.line(20, 25, 190, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", style='I', size=8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, text=f"Page {self.page_no()}", align='C')

def create_visualizations(df: pd.DataFrame, output_dir: str):
    #Genera e salva i grafici da inserire nel PDF
    os.makedirs(output_dir, exist_ok=True)
    
    df['Initial_Num'] = df['Initial_Weight'].str.rstrip('%').astype('float')
    df['Final_Num'] = df['Final_Weight'].str.rstrip('%').astype('float')
    
    # 1. Pie chart: Optimal Portfolio Allocation
    plt.figure(figsize=(6, 6))
    final_alloc = df[df['Final_Num'] > 0]
    plt.pie(final_alloc['Final_Num'], labels=final_alloc['Coin'], autopct='%1.1f%%', startangle=140,colors = plt.cm.Paired.colors)
    plt.title('Optimal Portfolio Allocation', fontweight='bold', fontsize=16, pad=6)
    plt.savefig(f'{output_dir}/pie_chart.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Bar chart: Initial vs Optimal Weights
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df['Coin']))
    width = 0.35
    plt.bar(x - width/2, df['Initial_Num'], width, label='Initial Weight', color='#d9534f')
    plt.bar(x + width/2, df['Final_Num'], width, label='Final Weight', color='#5cb85c')
    plt.ylabel('Weight (%)', fontweight='bold')
    plt.title('Portfolio Rebalancing: Initial vs Optimal (MILP)', fontweight='bold', fontsize=16, pad=10)
    plt.xticks(x, df['Coin'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bar_chart.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_pdf():
    logging.info("Processing visualizations and generating report...")
    
    results_path = 'reports/portfolio_results.csv'
    if not os.path.exists(results_path):
        logging.error(f"Portfolio results file not found at {results_path}. Run the optimization step first.")
        return

    df = pd.read_csv(results_path)
    create_visualizations(df, 'reports')
    
    pdf = QuantReport()
    pdf.set_margins(left=20, top=15, right=20)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ==========================================
    # PAGE 1: Executive Summary & Execution Plan
    # ==========================================
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, text="1. Executive Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("helvetica", "", 10)
    summary_text = (
        "This analysis is based on historical data collected over the one-week period from October 16 to October 23, 2026. It outlines the execution plan for an automated, volatility-targeting portfolio reallocation. "
        "The main quantitative objective is to minimize the portfolio's expected risk using ML "
        "forecasts, while respecting structural constraints such as maximum asset exposure limits "
        "and real-world transaction costs.\n\n"
        "Upon detecting a severe concentration of risk in highly volatile assets, the algorithmic pipeline triggered "
        "an immediate rebalancing process. The resulting strategy offloads toxic exposures and redistributes "
        "capital into lower-variance assets, ensuring optimal risk-adjusted stability without incurring excessive turnover fees."
    )
    pdf.multi_cell(0, 5, text=summary_text, align="L")
    pdf.ln(5)
    
    # Tabella Execution Plan
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, text="Trade Execution Plan", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("helvetica", "B", 8)
    
    pdf.set_left_margin(25)
    col_w = [35, 30, 30, 30, 30]
    headers = ["Asset", "Initial Weight", "Final Weight", "Action", "Trade Size"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, text=h, border=1, align="C")
    pdf.ln(8)
    
    pdf.set_font("helvetica", "", 9)
    for _, row in df.iterrows():
        pdf.cell(col_w[0], 8, text=str(row['Coin']), border=1, align="C")
        pdf.cell(col_w[1], 8, text=str(row['Initial_Weight']), border=1, align="C")
        pdf.cell(col_w[2], 8, text=str(row['Final_Weight']), border=1, align="C")
        pdf.cell(col_w[3], 8, text=str(row['Action']), border=1, align="C")
        pdf.cell(col_w[4], 8, text=str(row['Trade_Amount']), border=1, align="C")
        pdf.ln(8)
    pdf.set_left_margin(20)  
    pdf.ln(10)
    pdf.image("reports/pie_chart.png", w=80, x=(pdf.w - 80)/2)
    
    # ==========================================
    # PAGE 2: Machine Learning Analytics
    # ==========================================
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, text="2. Machine Learning Predictive Engine", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("helvetica", "", 10)
    ml_text = (
        "Risk assessment within the pipeline relies on a Dynamic Machine Learning Model Selection architecture. "
        "Historical high-frequency data is processed into vectorized features, including logarithmic returns, "
        "rolling volatility measures (1h, 4h), and momentum indicators. To model the non-linear relationships in the cryptocurrency "
        "market, multiple models (Ridge Regression, Random Forest, and XGBoost) are trained and evaluated through "
        "a continuous comparative process. The XGBoost Gradient Boosting algorithm consistently outperformed the baseline, "
        "identifying complex variance patterns. The forecasts generated by this model serve as the expected risk "
        "coefficients in the subsequent mathematical optimization stage."
    )
    pdf.multi_cell(0, 5, text=ml_text, align="L")
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, text="3. Operations Research & MILP Formulation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "", 10)
    or_text = (
        "Converting machine learning predictions into an effective financial strategy requires strict constraint management. "
        "To address this, Gurobi, a state-of-the-art mathematical solver, is used to frame the portfolio reallocation as a "
        "Mixed-Integer Linear Programming (MILP) problem. Mathematical Constraints Enforced:"
    )
    pdf.multi_cell(0, 5, text=or_text, align="L")
    pdf.ln(3)
    constraints = [
        ("**Diversification Cap:**", "A strict limit of 30% exposure per single asset ensures the portfolio is never overly reliant on a single point of failure."),
        ("**Transaction Fees:**", "The optimization model incorporates a 0.1% exchange fee, mathematically deducting costs from the final budget constraints. This prevents hyper-trading."),
        ("**Turnover Penalty:**", "A penalty term (1e-2) is introduced in the objective function to eliminate 'ghost trades', forcing the solver to ignore mathematically negligible optimizations that waste capital in fees."),
        ("**Big-M Constraints:**", "Binary variables dictate that if a trade occurs, it must meet a minimum size threshold, ensuring realistic execution on live exchanges.")
    ]
    pdf.set_left_margin(28)
    for title, desc in constraints:
        pdf.multi_cell(0, 4.5, text=f"{chr(149)} {title} {desc}", markdown=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        pdf.ln(1.5)
    pdf.set_left_margin(20)
    
    pdf.ln(6)
    pdf.image("reports/bar_chart.png", w=160, x=(pdf.w - 160) / 2)
    
    pdf.output("reports/TearSheet.pdf")
    logging.info("Report generated: reports/TearSheet.pdf")
    
    if os.path.exists("reports/pie_chart.png"): os.remove("reports/pie_chart.png")
    if os.path.exists("reports/bar_chart.png"): os.remove("reports/bar_chart.png")

if __name__ == "__main__":
    generate_pdf()