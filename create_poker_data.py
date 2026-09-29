import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

data = {
    'avg_timing': np.random.exponential(3, n),
    'bet_size_ratio': np.random.uniform(0, 3, n),
    'num_raises': np.random.randint(0, 5, n),
    'num_calls': np.random.randint(0, 10, n),
    'hesitation_count': np.random.randint(0, 5, n),
    'aggression': np.random.uniform(0, 2, n),
}

df = pd.DataFrame(data)

bluff_score = (
    (df['avg_timing'] < 1.5).astype(int) * 0.3 +
    (df['bet_size_ratio'] > 1.5).astype(int) * 0.3 +
    (df['num_raises'] > 2).astype(int) * 0.2 +
    (df['hesitation_count'] > 2).astype(int) * 0.2
)

df['is_bluff'] = (bluff_score + np.random.uniform(0, 0.3, n) > 0.5).astype(int)
df.to_csv('poker_data.csv', index=False)
print(f"Done! {len(df)} rows, bluff rate: {df['is_bluff'].mean():.2%}")