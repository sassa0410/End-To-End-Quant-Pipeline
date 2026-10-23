import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MILPOptimizer:
    """
    Mixed-Integer Linear Programming portfolio optimizer using Gurobi.
    Minimizes the expected risk forecasted via Machine Learning,
    enforcing maximum exposure limits and penalizing transaction costs.
    """
    
    def __init__(self, fee_rate: float = 0.001, minimum_trade: float = 0.02, max_trade: float = 0.30):
        self.fee_rate: float = fee_rate
        self.minimum_trade: float = minimum_trade
        self.max_trade: float = max_trade
        self.penalty: float = 1e-2  # Small penalty to encourage smaller trades
        self.env: gp.Env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()

    def _get_init_weights(self, coins: List[str]) -> Dict[str, float]:
        w_init: Dict[str, float] = {coin: 0.0 for coin in coins}
        if 'UNIUSDT' in w_init:
            w_init['UNIUSDT'] = 0.60  # Added to the coin most risky
        if 'AVAXUSDT' in w_init:
            w_init['AVAXUSDT'] = 0.40  # Added to an other risky coin
        return w_init

    def _build_model(self, coins: List[str], volatility: Dict[str, float], w_init: Dict[str, float]) -> Optional[gp.Model]:
        m= gp.Model("Crypto_Portfolio_Optimization", env=self.env)
    
        # Decision Variables
        w = m.addVars(coins, lb=0.0, ub=self.max_trade, name="weights")
        z = m.addVars(coins, lb=0.0, ub=1.0, name="trade_amount")
        y = m.addVars(coins, vtype=GRB.BINARY, name="trade_indicator")
    
    # Objective: Minimize Portfolio Volatility
        m.setObjective(
            gp.quicksum(volatility[coin]* w[coin] for coin in coins) + gp.quicksum(self.penalty * z[coin] for coin in coins),
            GRB.MINIMIZE
        )
    
    # Constraints
        for coin in coins:
            m.addConstr(z[coin] >= w[coin] - w_init[coin], name=f"abs_pos_{coin}")
            m.addConstr(z[coin] >= w_init[coin] - w[coin], name=f"abs_neg_{coin}")
            m.addConstr(z[coin] >= self.minimum_trade * y[coin], name=f"min_trade_{coin}")
            m.addConstr(z[coin] <= self.max_trade * y[coin], name=f"big_M_{coin}")
        
        m.addConstr(
            gp.quicksum(w[coin] for coin in coins) + gp.quicksum(z[coin] * self.fee_rate for coin in coins) == 1.0,
            name="budget"
        )
    
        m.optimize()
        return m if m.status == GRB.OPTIMAL else None
    
    def optimize_portfolio(self, prediction_path: str, output_dir: str = 'reports') -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Executes portfolio optimization by reading ML predictions.
        
        Args:
            prediction_path (str): Path to the CSV file containing forecasted volatilities.
            output_dir (str, optional): Directory to save the results. Defaults to 'reports'.

        Raises:
            ValueError: If no optimal solution is found.

        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: A DataFrame containing the Execution plan
            and a dictionary with risk metrics and transaction costs.
        """
        df = pd.read_csv(prediction_path)
        coins: List[str] = df['Coin'].tolist()
        volatility: Dict[str, float] = dict(zip(df['Coin'], df['Predicted_Vol_1h']))
        w_init = self._get_init_weights(coins)
        model = self._build_model(coins, volatility, w_init)
        
        if model is None:
            raise ValueError("No optimal solution found for the portfolio optimization problem.")
    
        results: List[Dict[str, str]] = []
        total_cost: float = 0.0
        for c in coins:
            weight: float = model.getVarByName(f"weights[{c}]").X
            trade_amount: float = model.getVarByName(f"trade_amount[{c}]").X
            is_traded: float = model.getVarByName(f"trade_indicator[{c}]").X
            if is_traded > 0.5 and trade_amount > 0.001:
                action: str = "Bought" if weight > w_init[c] else "Sold"
                total_cost += trade_amount * self.fee_rate
            else:
                action = "No Trade"
                trade_amount = 0.0
        
            results.append({
                'Coin': c,
                'Initial_Weight': f"{w_init[c]*100:.2f}%",
                'Final_Weight': f"{weight*100:.2f}%",            
                'Action': action,
                'Trade_Amount': f"{trade_amount*100:.2f}%"
            })
    
        df_results = pd.DataFrame(results)
        os.makedirs(output_dir, exist_ok=True)
        df_results.to_csv(f'{output_dir}/portfolio_results.csv', index=False)
        
        metrics: Dict[str, float] = {
            'expected_risk': model.objVal,
            'total_fees':total_cost
        }
        logging.info(f"Portfolio optimization completed. Expected Risk: {metrics['expected_risk']:.6f}, Total Fees: {metrics['total_fees']:.6f}\n")
        
        return df_results, metrics
        
if __name__ == "__main__":
    optimizer = MILPOptimizer()
    df_plan, metrics = optimizer.optimize_portfolio('data/processed/live_predictions.csv')
    print(df_plan.to_string(index=False))