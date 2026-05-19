import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("Loading dataset...")
df = pd.read_csv('datasets/matches.csv')
df.columns = [c.strip() for c in df.columns]

features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away',
            'Form5Home', 'Form5Away']

df_clean = df[features + ['FTResult']].dropna()
print(f"Training on {len(df_clean)} matches...")

X = df_clean[features]
y = df_clean['FTResult']

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

os.makedirs('models', exist_ok=True)
with open('models/prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved — trained on {len(df_clean)} matches")
print(f"Classes: {model.classes_}")