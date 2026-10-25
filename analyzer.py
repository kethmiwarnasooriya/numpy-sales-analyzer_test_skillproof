import numpy as np

def main():
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    # Generate 30 days of synthetic daily sales data (ranging between $500 and $5,000)
    sales = np.random.uniform(500, 5000, size=30)
    
    print("--- Raw Daily Sales Data ---")
    print(np.round(sales, 2))
    
    # 1. Compute basic statistics using NumPy
    total_sales = np.sum(sales)
    mean_sales = np.mean(sales)
    max_sales = np.max(sales)
    min_sales = np.min(sales)
    peak_day = np.argmax(sales) + 1  # 1-indexed day
    
    print("\n--- Summary Statistics ---")
    print(f"Total Monthly Sales: ${total_sales:,.2f}")
    print(f"Average Daily Sales: ${mean_sales:,.2f}")
    print(f"Peak Sales Day: Day {peak_day} (${max_sales:,.2f})")
    print(f"Lowest Sales Day: ${min_sales:,.2f}")
    
    # 2. Normalize data (Min-Max Scaling to a 0-1 range)
    normalized_sales = (sales - min_sales) / (max_sales - min_sales)
    
    print("\n--- Normalized Sales (0 to 1) ---")
    print(np.round(normalized_sales, 4))

if __name__ == "__main__":
    main()
