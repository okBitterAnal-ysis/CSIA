# Experiment 2: Utility-Based Agent (Algorithmic Trading Bot)

price = float(input("Enter current stock price (e.g. 150): "))
prev_price = float(input("Enter previous stock price (e.g. 140): "))
volatility = float(input("Enter volatility between 0 and 1 (e.g. 0.25): "))
capital = float(input("Enter available capital (e.g. 100000): "))

# 1. Weights based on volatility
if volatility >= 0.40:
    w_profit, w_risk, w_cost = 0.4, 0.5, 0.1
    market = "High Volatility"
else:
    w_profit, w_risk, w_cost = 0.6, 0.3, 0.1
    market = "Normal"

# 2. Expected profits based on trend
change = price - prev_price
buy_p, sell_p = (80, 30) if change > 0 else ((25, 75) if change < 0 else (40, 40))
risk_penalty = volatility * 100
cost = 10

# 3. Compute utilities
u_buy = round(buy_p * w_profit - risk_penalty * w_risk - cost * w_cost)
u_sell = round(sell_p * w_profit - risk_penalty * w_risk - cost * w_cost)
u_hold = 5

utilities = {"BUY": u_buy, "SELL": u_sell, "HOLD": u_hold}
best_action = max(utilities, key=utilities.get)

# 4. Position sizing based on risk level
if volatility < 0.20:
    pos_pct, risk_lvl = 0.20, "Low"
elif volatility < 0.40:
    pos_pct, risk_lvl = 0.10, "Medium"
elif volatility < 0.60:
    pos_pct, risk_lvl = 0.05, "High"
else:
    pos_pct, risk_lvl = 0.0, "Very High"

position_size = round(capital * pos_pct) if best_action in ["BUY", "SELL"] else 0

# 5. Output Results
print("\n--- Trading Agent Assessment ---")
print(f"Market Condition : {market}")
print(f"Risk Level       : {risk_lvl}")
print(f"Utility Scores   : BUY = {u_buy}, SELL = {u_sell}, HOLD = {u_hold}")
print(f"Selected Action  : {best_action}")
print(f"Position Size    : Rs. {position_size}")
