import pandas as pd

df = pd.read_csv("")

# gain 기준으로 정렬
df = df.sort_values("gain", ascending=False)

# 상위 300개만 선택
top300 = df.head(301)

top300[["leading_item_id", "following_item_id"]].to_csv(
    "",
    index=False
)

print("Top300 pairs saved:", len(top300))
