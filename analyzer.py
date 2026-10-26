import numpy as np

def main():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate synthetic housing dataset (Square Footage vs Price in $1000s)
    n_samples = 100
    sqft = np.random.normal(1800, 400, n_samples)
    # True relationship: Price = 50 + 0.12 * sqft + random noise
    price = 50 + 0.12 * sqft + np.random.normal(0, 25, n_samples)
    
    print("--- Dataset Sample (First 5 rows) ---")
    for s, p in zip(sqft[:5], price[:5]):
        print(f"SqFt: {s:.1f} | Price: ${p:.1f}k")
        
    # 2. Data Manipulation for Regression
    # Construct feature matrix X with a column of ones for the intercept (bias term)
    X = np.vstack([np.ones(n_samples), sqft]).T
    y = price
    
    # 3. Implement Ordinary Least Squares (OLS) Closed-Form Solution
    # Formula: theta = (X^T * X)^(-1) * X^T * y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    intercept, slope = theta[0], theta[1]
    
    print("\n--- Fitted Regression Model Parameters ---")
    print(f"Intercept (b₀): ${intercept:.2f}k")
    print(f"Slope (b₁): ${slope:.4f}k per sq. ft.")
    print(f"Regression Equation: Price = {intercept:.2f} + {slope:.4f} * SqFt")
    
    # 4. Model Evaluation using NumPy Metrics
    y_pred = X @ theta
    mse = np.mean((y_pred - y) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R² Score): {r2_score:.4f}")
    
    # 5. Make a New Prediction
    target_sqft = 2100
    predicted_price = intercept + slope * target_sqft
    print(f"\n--- Prediction Example ---")
    print(f"Estimated price for a {target_sqft} sq. ft. home: ${predicted_price:.2f}k")

if __name__ == "__main__":
    main()
