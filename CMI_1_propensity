import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Seed for reproducibility
np.random.seed(0)

# Create sample data
data = {
    'client_id': np.random.choice(['Client_A', 'Client_B', 'Client_C', 'Client_D', 'Client_E'], size=100),
    'product_id': np.random.choice(['Product_1', 'Product_2', 'Product_3', 'Product_4'], size=100),
    'purchase_date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
    'feature_1': np.random.rand(100),
    'feature_2': np.random.rand(100)
}

df = pd.DataFrame(data)

# Assume target is whether the product was purchased or not (binary 0/1)
df['purchased'] = np.random.randint(0, 2, size=100)

# Feature Engineering
df['purchase_count'] = df.groupby('client_id')['product_id'].transform('count')
df['days_since_last_purchase'] = (pd.to_datetime('2023-01-01') - df['purchase_date']).dt.days

# Create a one-hot encoded DataFrame of products purchased by each client
basket = df.groupby(['client_id', 'product_id']).size().unstack(fill_value=0)

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

# Calculate the association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Prepare feature matrix and target vector for propensity model
X = df[['feature_1', 'feature_2', 'purchase_count', 'days_since_last_purchase']]
y = df['purchased']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict propensity scores for the entire dataset
df['propensity_score'] = model.predict_proba(X)[:, 1]

# Function to get affinity-based recommendations
def get_affinity_recommendations(client_id, basket, rules):
    purchased_products = set(basket.loc[client_id][basket.loc[client_id] > 0].index)
    recommendations = set()
    
    for product in purchased_products:
        for _, row in rules.iterrows():
            if product in row['antecedents']:
                recommendations.update(row['consequents'])
    
    # Exclude products already purchased
    recommendations.difference_update(purchased_products)
    
    return list(recommendations)

# Function to combine affinity and propensity recommendations
def get_combined_recommendations(client_id, df, basket, rules):
    # Get affinity-based recommendations
    affinity_recommendations = get_affinity_recommendations(client_id, basket, rules)
    
    # Filter the original dataframe for the specific client and recommended products
    combined_df = df[(df['client_id'] == client_id) & (df['product_id'].isin(affinity_recommendations))]
    
    # Sort by propensity score
    combined_df = combined_df[['product_id', 'propensity_score']].drop_duplicates().sort_values(by='propensity_score', ascending=False)
    
    return combined_df

# Example: Get combined recommendations for Client_A
client_id = 'Client_A'
combined_recommendations = get_combined_recommendations(client_id, df, basket, rules)

print(f"Combined recommendations for {client_id} based on affinity and propensity scores:")
print(combined_recommendations)
