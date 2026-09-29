import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('poker_data.csv')

features = ['avg_timing', 'bet_size_ratio', 'num_raises',
            'num_calls', 'hesitation_count', 'aggression']

X = df[features]
y = df['is_bluff']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    scale_pos_weight=len(y[y==0])/len(y[y==1]),
    eval_metric='logloss'
)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
joblib.dump(model, 'xgboost_poker.pkl')
print("XGBoost saved with 6 features!")