# Iron Condor Strategy Calculator with Real-Time Stock Data

This project implements an Iron Condor options strategy calculator using real-time stock data fetched from Yahoo Finance. It leverages the Black-Scholes option pricing model and calculates option Greeks (Delta, Gamma, Theta, Vega, Rho) to provide a comprehensive analysis of the strategy. Additionally, it includes a Streamlit app to allow users to interactively visualize the payoff diagram and Greeks of their Iron Condor strategy.



https://github.com/user-attachments/assets/23eb96bc-5dfe-431d-9951-ba66efbfab95


## Features

- **Real-time Stock Data**: Fetches live stock prices using Yahoo Finance.
- **Black-Scholes Pricing**: Calculates the theoretical price of call and put options using the Black-Scholes formula.
- **Greeks Calculation**: Computes the option Greeks (Delta, Gamma, Theta, Vega, and Rho) for call and put options in the Iron Condor strategy.
- **Iron Condor Payoff**: Visualizes the payoff diagram of the Iron Condor strategy at expiration.
- **Interactive UI**: Built using Streamlit, allowing users to modify key parameters such as stock ticker, strike prices, time to expiration, and implied volatility.

## Input Parameters:

- **Stock Ticker**: Enter the ticker symbol of the stock you want to analyze (e.g., AAPL, MSFT).
- **Strike Prices**: Set the strike prices for the Iron Condor strategy:
  - **Short Call Strike (K1)**: Set the strike price for the short call.
  - **Short Put Strike (K2)**: Set the strike price for the short put.
  - **Long Call Strike (K3)**: Set the strike price for the long call.
  - **Long Put Strike (K4)**: Set the strike price for the long put.
- **Time to Expiration (T)**: Set the number of days until expiration (converted to years).
- **Risk-Free Interest Rate (r)**: Set the annual risk-free interest rate (e.g., 0.05 for 5%).
- **Implied Volatility (σ)**: Set the implied volatility of the underlying asset.
- **Premiums**: Set the premium prices for the call and put options in the Iron Condor strategy.

## Output:

- **Payoff Diagram**: Displays a graph showing the profit and loss for the Iron Condor strategy at different stock prices at expiration.
- **Greeks Plot**: Displays the Delta, Gamma, Theta, Vega, and Rho values for each leg of the Iron Condor strategy.

## Papers:

Black-Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637–654. https://doi.org/10.1086/260062

Merton, R. C. (1973). Theory of rational option pricing. The Bell Journal of Economics and Management Science, 4(1), 141–183. https://doi.org/10.2307/3003143

Hull, J. (2017). Options, futures, and other derivatives (10th ed.). Pearson Education.

Black, F., & Scholes, M. (1972). The valuation of option contracts and a test of market efficiency. The Journal of Finance, 27(2), 399–417. https://doi.org/10.1111/j.1540-6261.1972.tb00843.x

Breeden, D., & Litzenberger, R. (1978). Prices of state-contingent claims implicit in option prices. Journal of Business, 51(4), 621–651. https://doi.org/10.1086/296393




## Installation

To run this project locally, you need to install the following Python libraries:

```bash
pip install streamlit numpy matplotlib yfinance scipy
streamlit run app1.py
