import pandas as pd

pairs = pd.read_csv("") \
           .sort_values("ml_score", ascending=False)

for N in [1485]:
    pairs.head(N).to_csv(
        f"",
        index=False
    )
