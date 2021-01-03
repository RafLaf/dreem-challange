import sys
import pandas as pd

arg = "xgb"
if len(sys.argv) > 1:
    arg = sys.argv[1]
filename = f"submission_{arg}.csv"


df = pd.read_csv(filename)
idx = pd.read_csv("sample_submission.csv")

new_df = pd.DataFrame(df.iloc[:, 1].values, index=idx.values[:, 0], columns=["sleep_stage"])

new_df.to_csv(f"{arg}.csv", index_label="index")
