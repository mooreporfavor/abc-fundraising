import pandas as pd
try:
    df = pd.read_csv("task_1_processed_v2.csv")
    with open("verification.txt", "w", encoding="utf-8") as f:
        f.write(str(df['Geography'].unique()))
    print("Done")
except Exception as e:
    with open("verification.txt", "w") as f:
        f.write(str(e))
