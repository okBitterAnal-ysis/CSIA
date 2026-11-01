# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 2: PEAS, PAGE and Agent Design (Algorithmic Trading Bot)
# Source: EXP2_AISC_dhirensacherr (1).pdf
# =====================================================================

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Simple Reflex Agent - Market Trend Signal Generator
# ---------------------------------------------------------------------
def get_market_condition(volatility):
    """A simple rule to classify the market."""
    if volatility >= 0.40:
        return "High Volatility"
    else:
        return "Normal"

def get_utility_weights(market_condition):
    """In volatile markets, the agent becomes more careful about risk."""
    if market_condition == "High Volatility":
        return {
            "profit": 0.4,
            "risk": 0.5,
            "cost": 0.1,
        }
    return {
        "profit": 0.6,
        "risk": 0.3,
        "cost": 0.1,
    }

# ---------------------------------------------------------------------
# 2. Goal-Based Agent - Risk-Constrained Position Sizing
# ---------------------------------------------------------------------
def get_risk_level(volatility):
    """Risk is decided using simple if/elif rules."""
    if volatility < 0.20:
        return "Low"
    elif volatility < 0.40:
        return "Medium"
    elif volatility < 0.60:
        return "High"
    else:
        return "Very High"

def calculate_position_size(capital, risk_level):
    if risk_level == "Low":
        percentage = 0.20
    elif risk_level == "Medium":
        percentage = 0.10
    elif risk_level == "High":
        percentage = 0.05
    else:
        percentage = 0
    return round(capital * percentage)

# ---------------------------------------------------------------------
# 3. Utility-Based Agent - Dynamic Trade Optimizer
# ---------------------------------------------------------------------
def get_utility(expected_profit, risk_penalty, transaction_cost, weights):
    utility = (
        expected_profit * weights["profit"]
        - risk_penalty * weights["risk"]
        - transaction_cost * weights["cost"]
    )
    return round(utility)

def calculate_utilities(price, previous_price, volatility, weights):
    price_change = price - previous_price

    # Simple rule-based expected profit values
    if price_change > 0:
        buy_profit = 80
        sell_profit = 30
    elif price_change < 0:
        buy_profit = 25
        sell_profit = 75
    else:
        buy_profit = 40
        sell_profit = 40

    # Higher volatility means higher risk
    risk_penalty = volatility * 100
    transaction_cost = 10

    utilities = {
        "BUY": get_utility(buy_profit, risk_penalty, transaction_cost, weights),
        "SELL": get_utility(sell_profit, risk_penalty, transaction_cost, weights),
        "HOLD": 5,
    }
    return utilities

def select_best_action(utilities):
    return max(utilities, key=utilities.get)

# ---------------------------------------------------------------------
# 4. Main Driver with Input Validation & Execution Report
# ---------------------------------------------------------------------
def get_numeric_input(prompt, val_type=float, min_val=0.0, max_val=None):
    """Prompts the user for numeric input with validation and error handling."""
    while True:
        try:
            val = val_type(input(prompt))
            if min_val is not None and val < min_val:
                print(f"Error: Value must be at least {min_val}. Please try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"Error: Value must not exceed {max_val}. Please try again.")
                continue
            return val
        except ValueError:
            print(f"Invalid input. Please enter a valid numeric value.")

def main():
    print("=" * 45)
    print("        UTILITY-BASED TRADING AGENT")
    print("=" * 45)
    print("Please enter the market parameters below:\n")

    price = get_numeric_input("Enter the current price of the stock (Rs.): ", float, min_val=0.01)
    previous_price = get_numeric_input("Enter the previous price of the stock (Rs.): ", float, min_val=0.01)
    volatility = get_numeric_input("Enter the volatility of the stock (0.0 to 1.0, e.g., 0.25): ", float, min_val=0.0, max_val=1.0)
    capital = get_numeric_input("Enter the available capital (Rs.): ", float, min_val=0.0)

    market = get_market_condition(volatility)
    weights = get_utility_weights(market)

    utilities = calculate_utilities(price, previous_price, volatility, weights)
    action = select_best_action(utilities)

    risk_level = get_risk_level(volatility)

    if action in ["BUY", "SELL"]:
        position_size = calculate_position_size(capital, risk_level)
    else:
        position_size = 0

    print("\n" + "=" * 45)
    print("             TRADING AGENT REPORT")
    print("=" * 45)
    print(f"Current Price         : Rs. {price:,.2f}")
    print(f"Previous Price        : Rs. {previous_price:,.2f}")
    price_change = price - previous_price
    print(f"Price Change          : Rs. {price_change:+,.2f}")
    print(f"Volatility            : {volatility:.2f}")
    print(f"Market Condition      : {market}")
    print(f"Active Weights        : Profit: {weights['profit']}, Risk: {weights['risk']}, Cost: {weights['cost']}")
    print("-" * 45)
    print("Action Utilities:")
    print(f"   • BUY Utility       : {utilities['BUY']}")
    print(f"   • SELL Utility      : {utilities['SELL']}")
    print(f"   • HOLD Utility      : {utilities['HOLD']}")
    print("-" * 45)
    print(f"Recommended Action    : {action}")
    print(f"Risk Level            : {risk_level}")
    print(f"Available Capital     : Rs. {capital:,.2f}")
    print(f"Position Sizing       : Rs. {position_size:,.2f}")
    print("=" * 45)

    # Graph Generation & Printing
    print("\nGenerating Utility Agent Decision & Assessment Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Action Utilities Bar Chart
    actions = list(utilities.keys())
    values = list(utilities.values())
    colors = ['#2ecc71' if a == action else '#3498db' for a in actions]
    bars = ax1.bar(actions, values, color=colors, edgecolor='black', width=0.5)
    ax1.set_title(f'Action Utilities (Recommended: {action})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Utility Score', fontsize=11)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -12),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')

    # Weights Pie Chart
    weight_labels = [f"Profit ({weights['profit']})", f"Risk ({weights['risk']})", f"Cost ({weights['cost']})"]
    weight_vals = [weights['profit'], weights['risk'], weights['cost']]
    ax2.pie(weight_vals, labels=weight_labels, autopct='%1.1f%%', colors=['#48dbfb', '#ff6b6b', '#feca57'], startangle=140)
    ax2.set_title(f'Active Utility Weights ({market})', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plot_file = 'utility_trading_bot_plot.png'
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[Graph Printed] Plot successfully saved to '{plot_file}'")
    except Exception:
        pass

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
