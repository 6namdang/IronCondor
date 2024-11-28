
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

# # Black-Scholes Formula
# def black_scholes_price(S, K, T, r, sigma, option_type):
#     d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     if option_type == "call":
#         return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     elif option_type == "put":
#         return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# # Greeks (Delta, Gamma, Theta, Vega, Rho) Calculation
# def option_greeks(S, K, T, r, sigma, option_type):
#     d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
    
#     if option_type == "call":
#         delta = norm.cdf(d1)
#         gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
#         theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
#         vega = S * norm.pdf(d1) * np.sqrt(T)
#         rho = K * T * np.exp(-r * T) * norm.cdf(d2)
#     elif option_type == "put":
#         delta = norm.cdf(d1) - 1
#         gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
#         theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
#         vega = S * norm.pdf(d1) * np.sqrt(T)
#         rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
#     return delta, gamma, theta, vega, rho

# # Iron Condor Payoff Function
# def iron_condor_payoff(S_range, K1, K2, K3, K4, premium_call1, premium_put1, premium_call2, premium_put2):
#     payoff = []
#     for S in S_range:
#         # Payoff for each leg of the strategy
#         payoff_call1 = max(S - K1, 0) - premium_call1
#         payoff_put1 = max(K2 - S, 0) - premium_put1
#         payoff_call2 = premium_call2 - max(S - K3, 0)
#         payoff_put2 = premium_put2 - max(K4 - S, 0)
#         total_payoff = payoff_call1 + payoff_put1 + payoff_call2 + payoff_put2
#         payoff.append(total_payoff)
#     return payoff

# # Streamlit App
# st.title("Iron Condor Strategy Calculator with Real-Time Stock Data")

# st.sidebar.header("User Inputs")

# # User Inputs
# ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, MSFT)", value="AAPL").upper()

# # Fetch the current stock price using Yahoo Finance
# if ticker:
#     try:
#         stock_data = yf.Ticker(ticker)
#         current_price = stock_data.history(period="1d")['Close'].iloc[0]
#         st.sidebar.write(f"**Current Stock Price for {ticker}:** ${current_price:.2f}")
#     except:
#         st.sidebar.write("**Invalid ticker symbol. Please try again.**")
#         current_price = 0
# else:
#     current_price = 0

# if current_price > 0:
#     # Get input for strikes, expiration, etc. (if current stock price is fetched successfully)
#     K1 = st.sidebar.number_input("Short Call Strike (K1)", value=current_price * 1.05, step=1.0)
#     K2 = st.sidebar.number_input("Short Put Strike (K2)", value=current_price * 0.95, step=1.0)
#     K3 = st.sidebar.number_input("Long Call Strike (K3)", value=current_price * 1.1, step=1.0)
#     K4 = st.sidebar.number_input("Long Put Strike (K4)", value=current_price * 0.9, step=1.0)
#     T = st.sidebar.number_input("Time to Expiration (T) in Days", value=30, step=1) / 365.0
#     r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, step=0.01)
#     sigma = st.sidebar.number_input("Implied Volatility (\u03C3)", value=0.2, step=0.01)

#     premium_call1 = st.sidebar.number_input("Premium for Short Call (K1)", value=2.0, step=0.1)
#     premium_put1 = st.sidebar.number_input("Premium for Short Put (K2)", value=2.0, step=0.1)
#     premium_call2 = st.sidebar.number_input("Premium for Long Call (K3)", value=1.0, step=0.1)
#     premium_put2 = st.sidebar.number_input("Premium for Long Put (K4)", value=1.0, step=0.1)

#     # Compute Iron Condor Payoff
#     S_range = np.linspace(current_price - 20, current_price + 20, 500)
#     payoff = iron_condor_payoff(S_range, K1, K2, K3, K4, premium_call1, premium_put1, premium_call2, premium_put2)

#     # Plot Payoff Diagram
#     st.subheader("Payoff Diagram")
#     fig, ax = plt.subplots()
#     ax.plot(S_range, payoff, label="Iron Condor Payoff")
#     ax.axhline(0, color="black", linewidth=1, linestyle="--")
#     ax.set_xlabel("Stock Price at Expiration")
#     ax.set_ylabel("Profit / Loss")
#     ax.legend()
#     st.pyplot(fig)
def black_scholes_price(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Greeks (Delta, Gamma, Theta, Vega, Rho) Calculation
def option_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return delta, gamma, theta, vega, rho

# Iron Condor Payoff Function
def iron_condor_payoff(S_range, K1, K2, K3, K4, premium_call1, premium_put1, premium_call2, premium_put2):
    payoff = []
    for S in S_range:
        # Payoff for each leg of the strategy
        payoff_call1 = max(S - K1, 0) - premium_call1
        payoff_put1 = max(K2 - S, 0) - premium_put1
        payoff_call2 = premium_call2 - max(S - K3, 0)
        payoff_put2 = premium_put2 - max(K4 - S, 0)
        total_payoff = payoff_call1 + payoff_put1 + payoff_call2 + payoff_put2
        payoff.append(total_payoff)
    return payoff

# Input Validation Function
def validate_inputs(current_price, K1, K2, K3, K4, T, r, sigma):
    errors = []
    
    # Check strike price order
    if not (K4 < K2 < current_price < K1 < K3):
        errors.append("Strike prices must follow the order: K4 < K2 < Current Price < K1 < K3 !")
    
    # Check expiration time
    if T <= 0:
        errors.append("Expiration time must be positive ")
    
    # Check volatility range (typically 0-1)
    if sigma < 0 or sigma > 1:
        errors.append("Volatility should be between 0 and 1")
    
    # Check interest rate range (typically -1 to 1)
    if r < -1 or r > 1:
        errors.append("Interest rate should be between -1% and 100%")
    
    return errors

# Streamlit App
st.title("Iron Condor Strategy Calculator with Real-Time Stock Data")

st.sidebar.header("User Inputs")

# User Inputs
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, MSFT)", value="AAPL").upper()

# Fetch the current stock price using Yahoo Finance
if ticker:
    try:
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period="1d")['Close'].iloc[0]
        st.sidebar.write(f"**Current Stock Price for {ticker}:** ${current_price:.2f}")
    except:
        st.sidebar.write("**Invalid ticker symbol. Please try again.**")
        current_price = 0
else:
    current_price = 0

if current_price > 0:
    # Get input for strikes, expiration, etc. (if current stock price is fetched successfully)
    K1 = st.sidebar.number_input("Short Call Strike (K1)", value=current_price * 1.05, step=1.0)
    K2 = st.sidebar.number_input("Short Put Strike (K2)", value=current_price * 0.95, step=1.0)
    K3 = st.sidebar.number_input("Long Call Strike (K3)", value=current_price * 1.1, step=1.0)
    K4 = st.sidebar.number_input("Long Put Strike (K4)", value=current_price * 0.9, step=1.0)
    T = st.sidebar.number_input("Time to Expiration (T) in Days", value=30, step=1) / 365.0
    r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Implied Volatility (\u03C3)", value=0.2, step=0.01)

    premium_call1 = st.sidebar.number_input("Premium for Short Call (K1)", value=2.0, step=0.1)
    premium_put1 = st.sidebar.number_input("Premium for Short Put (K2)", value=2.0, step=0.1)
    premium_call2 = st.sidebar.number_input("Premium for Long Call (K3)", value=1.0, step=0.1)
    premium_put2 = st.sidebar.number_input("Premium for Long Put (K4)", value=1.0, step=0.1)

    # Validate Inputs
    validation_errors = validate_inputs(current_price, K1, K2, K3, K4, T, r, sigma)
    
    if validation_errors:
        st.error("Input Validation Errors:")
        for error in validation_errors:
            st.error(error)
    else:
        # Compute Iron Condor Payoff
        S_range = np.linspace(K4 * 0.8, K3 * 1.2, 500)
        payoff = iron_condor_payoff(S_range, K1, K2, K3, K4, premium_call1, premium_put1, premium_call2, premium_put2)

        # Plot Payoff Diagram with Enhanced Visualization
        st.subheader("Payoff Diagram")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(S_range, payoff, color='blue', linewidth=2, label="Iron Condor Payoff")
        
        # Highlight key regions
        ax.axhline(0, color="red", linewidth=1, linestyle="--", label="Break-Even")
        ax.fill_between(S_range, payoff, where=(S_range < K4) | (S_range > K3), 
                        color='red', alpha=0.2, label='Loss Regions')
        ax.fill_between(S_range, payoff, where=(S_range >= K2) & (S_range <= K1), 
                        color='green', alpha=0.2, label='Maximum Profit Region')
        
        ax.set_xlabel("Stock Price at Expiration", fontsize=12)
        ax.set_ylabel("Profit / Loss", fontsize=12)
        ax.set_title("Iron Condor Strategy Payoff", fontsize=14)
        
        # Annotate key points
        max_profit = max(payoff)
        max_profit_index = payoff.index(max_profit)
        max_profit_price = S_range[max_profit_index]
        
        ax.annotate(f'Max Profit: ${max_profit:.2f}', 
                    xy=(max_profit_price, max_profit), 
                    xytext=(10, 10), textcoords='offset points', 
                    ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)


    # Greeks Calculation
    delta1, gamma1, theta1, vega1, rho1 = option_greeks(current_price, K1, T, r, sigma, "call")
    delta2, gamma2, theta2, vega2, rho2 = option_greeks(current_price, K2, T, r, sigma, "put")
    delta3, gamma3, theta3, vega3, rho3 = option_greeks(current_price, K3, T, r, sigma, "call")
    delta4, gamma4, theta4, vega4, rho4 = option_greeks(current_price, K4, T, r, sigma, "put")

    # Plot Greeks
    st.subheader("Greeks Plot")

    # Plot Delta
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))

    ax[0, 0].plot(S_range, [option_greeks(S, K1, T, r, sigma, "call")[0] for S in S_range], label="Delta (Short Call)")
    ax[0, 0].plot(S_range, [option_greeks(S, K2, T, r, sigma, "put")[0] for S in S_range], label="Delta (Short Put)")
    ax[0, 0].plot(S_range, [option_greeks(S, K3, T, r, sigma, "call")[0] for S in S_range], label="Delta (Long Call)")
    ax[0, 0].plot(S_range, [option_greeks(S, K4, T, r, sigma, "put")[0] for S in S_range], label="Delta (Long Put)")
    ax[0, 0].set_title("Delta")
    ax[0, 0].set_xlabel("Stock Price at Expiration")
    ax[0, 0].set_ylabel("Delta")
    ax[0, 0].legend()

    # Plot Gamma
    ax[0, 1].plot(S_range, [option_greeks(S, K1, T, r, sigma, "call")[1] for S in S_range], label="Gamma (Short Call)")
    ax[0, 1].plot(S_range, [option_greeks(S, K2, T, r, sigma, "put")[1] for S in S_range], label="Gamma (Short Put)")
    ax[0, 1].plot(S_range, [option_greeks(S, K3, T, r, sigma, "call")[1] for S in S_range], label="Gamma (Long Call)")
    ax[0, 1].plot(S_range, [option_greeks(S, K4, T, r, sigma, "put")[1] for S in S_range], label="Gamma (Long Put)")
    ax[0, 1].set_title("Gamma")
    ax[0, 1].set_xlabel("Stock Price at Expiration")
    ax[0, 1].set_ylabel("Gamma")
    ax[0, 1].legend()

    # Plot Theta
    ax[1, 0].plot(S_range, [option_greeks(S, K1, T, r, sigma, "call")[2] for S in S_range], label="Theta (Short Call)")
    ax[1, 0].plot(S_range, [option_greeks(S, K2, T, r, sigma, "put")[2] for S in S_range], label="Theta (Short Put)")
    ax[1, 0].plot(S_range, [option_greeks(S, K3, T, r, sigma, "call")[2] for S in S_range], label="Theta (Long Call)")
    ax[1, 0].plot(S_range, [option_greeks(S, K4, T, r, sigma, "put")[2] for S in S_range], label="Theta (Long Put)")
    ax[1, 0].set_title("Theta")
    ax[1, 0].set_xlabel("Stock Price at Expiration")
    ax[1, 0].set_ylabel("Theta")
    ax[1, 0].legend()

    # Plot Vega
    ax[1, 1].plot(S_range, [option_greeks(S, K1, T, r, sigma, "call")[3] for S in S_range], label="Vega (Short Call)")
    ax[1, 1].plot(S_range, [option_greeks(S, K2, T, r, sigma, "put")[3] for S in S_range], label="Vega (Short Put)")
    ax[1, 1].plot(S_range, [option_greeks(S, K3, T, r, sigma, "call")[3] for S in S_range], label="Vega (Long Call)")
    ax[1, 1].plot(S_range, [option_greeks(S, K4, T, r, sigma, "put")[3] for S in S_range], label="Vega (Long Put)")
    ax[1, 1].set_title("Vega")
    ax[1, 1].set_xlabel("Stock Price at Expiration")
    ax[1, 1].set_ylabel("Vega")
    ax[1, 1].legend()

    # Plot Rho
    ax[2, 0].plot(S_range, [option_greeks(S, K1, T, r, sigma, "call")[4] for S in S_range], label="Rho (Short Call)")
    ax[2, 0].plot(S_range, [option_greeks(S, K2, T, r, sigma, "put")[4] for S in S_range], label="Rho (Short Put)")
    ax[2, 0].plot(S_range, [option_greeks(S, K3, T, r, sigma, "call")[4] for S in S_range], label="Rho (Long Call)")
    ax[2, 0].plot(S_range, [option_greeks(S, K4, T, r, sigma, "put")[4] for S in S_range], label="Rho (Long Put)")
    ax[2, 0].set_title("Rho")
    ax[2, 0].set_xlabel("Stock Price at Expiration")
    ax[2, 0].set_ylabel("Rho")
    ax[2, 0].legend()

    # Adjust layout for better presentation
    plt.tight_layout()
    st.pyplot(fig)
