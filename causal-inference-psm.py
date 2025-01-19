#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min

# Sample data: Assume df contains features, 'treatment' column (1 = exposed, 0 = not), and 'outcome'
# Columns: 'feature1', 'feature2', ..., 'treatment', 'outcome'
df = pd.read_csv("marketing_data.csv")

# Step 1: Estimate Propensity Scores
X = df.drop(columns=['treatment', 'outcome'])  # Features
y = df['treatment']  # Treatment indicator
logit = LogisticRegression()
df['propensity_score'] = logit.fit(X, y).predict_proba(X)[:, 1]

# Step 2: Perform Nearest Neighbor Matching
treated = df[df['treatment'] == 1]
control = df[df['treatment'] == 0]
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity_score']])
control_indices, _ = pairwise_distances_argmin_min(treated[['propensity_score']], control[['propensity_score']])

matched_control = control.iloc[control_indices]
matched_treated = treated.reset_index(drop=True)

# Step 3: Calculate Average Treatment Effect (ATE)
treated_outcomes = matched_treated['outcome']
control_outcomes = matched_control['outcome']
ate = treated_outcomes.mean() - control_outcomes.mean()

print(f"Average Treatment Effect (ATE): {ate}")


# In[ ]:




