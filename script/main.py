import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_json("ekasuwan_gwari_customers_data.json")

# 3.2 Data Cleaning
# 3.2.1 Handling Missing Data
num_imputer = SimpleImputer(strategy='mean')
df[['TimeSpent', 'PagesVisited', 'ItemsInCart']] = num_imputer.fit_transform(df[['TimeSpent', 'PagesVisited', 'ItemsInCart']])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[['Gender']] = cat_imputer.fit_transform(df[['Gender']])

df = df.dropna(thresh=len(df.columns) * 0.5)

# 3.2.2 Removing Duplicates
df = df.drop_duplicates(subset=['CustomerID'])

# 3.2.3 Data Normalization and Standardization
scaler = MinMaxScaler()
df[['TimeSpent', 'PagesVisited', 'ItemsInCart']] = scaler.fit_transform(df[['TimeSpent', 'PagesVisited', 'ItemsInCart']])

std_scaler = StandardScaler()
df[['TimeSpent', 'PagesVisited', 'ItemsInCart']] = std_scaler.fit_transform(df[['TimeSpent', 'PagesVisited', 'ItemsInCart']])

# 3.3 Data Preparation for KNN Algorithm
# Expand PurchasedItems into separate rows
def expand_purchased_items(df):
    # Create a new dataframe to store expanded items
    rows = []
    for index, row in df.iterrows():
        if isinstance(row['PurchasedItems'], list):
            for item in row['PurchasedItems']:
                row_copy = row.copy()
                row_copy['PurchasedItem'] = item['item']
                row_copy['ItemQuantity'] = item['quantity']
                rows.append(row_copy)
        else:
            rows.append(row)
    return pd.DataFrame(rows)

df_expanded = expand_purchased_items(df)

# Pivot the dataframe to have one row per customer and one column per item with quantities
df_pivot = df_expanded.pivot_table(index=['CustomerID'], columns='PurchasedItem', values='ItemQuantity', fill_value=0)

# Join with the original dataframe to retain other features
df_final = df.join(df_pivot, on='CustomerID')

# Prepare features for KNN
X = df_final[['CustomerID'] + list(df_pivot.columns)].copy()

# Drop CustomerID for the KNN model
X_knn = X.drop(columns=['CustomerID'])

# Apply KNN
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_knn)

# Predict neighbors for a specific customer
def find_neighbors(customer_id, X, knn):
    customer_index = X[X['CustomerID'] == customer_id].index[0]
    distances, indices = knn.kneighbors([X.drop(columns=['CustomerID']).iloc[customer_index]])
    return X.iloc[indices[0]], distances[0]

# Example: Find neighbors for a specific customer
customer_id = "C010"
neighbors, distances = find_neighbors(customer_id, X, knn)

# Display neighbors
print("Nearest Neighbors for Customer ID:", customer_id)
print(neighbors)
print("Distances:", distances)

# Visualization of Neighbors
def plot_neighbors(customer_id, X, neighbors, distances):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=neighbors.index, y=distances, hue=neighbors['CustomerID'], palette="viridis", s=100)
    plt.title(f'Nearest Neighbors for Customer ID {customer_id}')
    plt.xlabel('Neighbor Index')
    plt.ylabel('Distance')
    plt.show()

# Plot neighbors for a specific customer
plot_neighbors(customer_id, X, neighbors, distances)
