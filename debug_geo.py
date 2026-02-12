import pandas as pd

try:
    df = pd.read_csv("task_1_processed.csv", encoding="utf-8")
    unique_geos = df['Geography'].unique()
    
    with open("geography_debug_codepoints.txt", "w", encoding="utf-8") as f:
        for geo in unique_geos:
            if isinstance(geo, str):
                debug_str = "".join([f"{c}({ord(c)})" for c in geo])
                f.write(f"{geo}: {debug_str}\n")
            
    print("Debug file written.")

except Exception as e:
    print(f"Error: {e}")
