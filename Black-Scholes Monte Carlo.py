"""
Monte Carlo Simulation for European call Option Pricing

1. Solve it analytically using the Black-Scholes formula.
2. Solve it using a Monte Carlo simulation, and model the distrubution with a gaussian curve.
3. Compute error scaling with the number of simulations.
"""


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def black_scholes(S0, K, T, r, sigma):
    """Calculate the Black-Scholes price of a European call option."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def gaussian_distribution(x, mu, sigma):
    """Calculate the Gaussian distribution value for given x, mean (mu), and standard deviation (sigma)."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def monte_carlo_call(S0, K, T, r, sigma, num_simulations):
    """Calculate the Monte Carlo price of a European call option."""
    # Simulate end-of-period stock prices
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Compute call payoffs
    payoffs = np.maximum(ST - K, 0)

    # Fit a Gaussian to the distribution of simulated stock prices
    mu = np.mean(ST)
    sigma_est = np.std(ST)
    x = np.linspace(min(ST), max(ST), 100)

    popt, _ = curve_fit(gaussian_distribution, 
                        np.sort(ST),                              # x-data: sorted prices
                        gaussian_distribution(np.sort(ST), mu, sigma_est),  # y-data: analytic Gaussian
                        p0=[mu, sigma_est])

    plt.figure(figsize=(10, 6))
    plt.hist(ST, bins=50, density=True, alpha=0.6, color='steelblue', label='Simulated prices')
    plt.plot(x, gaussian_distribution(x, *popt), 'r-', lw=2, label=f'Gaussian fit (μ={popt[0]:.2f}, σ={popt[1]:.2f})')
    plt.xlabel('Stock Price at Maturity')
    plt.ylabel('Probability Density')
    plt.title(f'Distribution of Simulated Stock Prices (n={num_simulations})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Discount the average payoff back to present value
    call_price = np.exp(-r * T) * np.mean(payoffs)
    print(f"Simulated end-of-period stock prices ({num_simulations}): {ST[:5]}")
    return call_price

def main():
    # Parameters
    S0 = 100  # Initial stock price
    K = 110   # Strike price
    T = 1     # Time to maturity (in years)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    # Analytical price using Black-Scholes formula
    bs_price = black_scholes(S0, K, T, r, sigma)
    print(f"Black-Scholes Price: {bs_price:.4f}")

    # Monte Carlo simulation
    num_simulations_list = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8]
    errors = []
    
    for num_simulations in num_simulations_list:
        mc_price = monte_carlo_call(S0, K, T, r, sigma, int(num_simulations))
        error = abs(mc_price - bs_price)
        errors.append(error)
        print(f"Monte Carlo Price with {num_simulations} simulations: {mc_price:.4f}, Error: {error:.4f}")

    # Plot error scaling
    plt.figure(figsize=(10, 6))
    plt.loglog(num_simulations_list, errors, marker='o')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Absolute Error')
    plt.title('Error Scaling of Monte Carlo Simulation')
    plt.grid(True)
    plt.show()

main()