import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate customer_metadata
n_customers = 100
customer_data = pd.DataFrame({
    "CustomerID": range(1001, 1001 + n_customers),
    "State": np.random.choice(["California", "Texas", "New York", "Florida", "Illinois"], n_customers),
    "City": np.random.choice(["Los Angeles", "Houston", "New York", "Miami", "Chicago"], n_customers),
    "Town": np.random.choice(["Santa Monica", "The Woodlands", "Brooklyn", "Sunny Isles Beach", "Naperville"], n_customers),
    "Recency": np.random.randint(1, 365, n_customers),  # Days since last purchase
    "Frequency": np.random.randint(1, 100, n_customers),  # Number of purchases
    "Monetary": np.random.uniform(100, 25000, n_customers).round(2)  # Total spend
})
customer_data.to_csv("data/raw/customer_metadata.csv", index=False)

# Generate sku_metadata
manufacturers = ["Nestlé", "PepsiCo", "Coca-Cola", "Unilever", "Procter & Gamble", 
                 "Mondelez", "Danone", "Kellogg's", "L'Oreal", "Heinz"]
segments = ["Beverages", "Food", "Household", "Personal Care"]
categories = {
    "Beverages": ["Soft Drinks", "Juices", "Energy Drinks", "Yogurt Drinks"],
    "Food": ["Snacks", "Breakfast Cereals", "Condiments", "Confectionery"],
    "Household": ["Cleaning Supplies", "Laundry Detergents"],
    "Personal Care": ["Beauty Products", "Skincare", "Oral Care"]
}

n_skus = 50
sku_data = pd.DataFrame({
    "SKUID": range(5001, 5001 + n_skus),
    "Manufacturer": np.random.choice(manufacturers, n_skus),
    "Segment": np.random.choice(segments, n_skus),
    "Category": [np.random.choice(categories[seg]) for seg in np.random.choice(segments, n_skus)]
})
sku_data.to_csv("data/raw/sku_metadata.csv", index=False)

# Generate transactions
weeks = pd.date_range("2022-01-01", "2023-12-31", freq="W")
transactions = []
for week in weeks:
    for customer_id in np.random.choice(customer_data["CustomerID"], size=20, replace=False):
        n_purchases = np.random.poisson(2)
        skus = np.random.choice(sku_data["SKUID"], n_purchases, replace=True)
        for sku in skus:
            quantity = np.random.randint(1, 5)
            unit_price = np.random.uniform(10, 100)
            transactions.append([
                week.strftime("%Y-%m-%d"),
                customer_id,
                sku,
                quantity,
                round(quantity * unit_price, 2)
            ])

transaction_df = pd.DataFrame(transactions, columns=["Week", "CustomerID", "SKUID", "Quantity", "OrderValue"])
transaction_df.to_csv("data/raw/transactions.csv", index=False)