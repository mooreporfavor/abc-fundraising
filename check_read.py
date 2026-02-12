import pandas as pd
import traceback

try:
    print("Attempting to read task_1.csv...")
    df = pd.read_csv("task_1.csv", encoding="utf-8-sig")
    print("Success reading task_1.csv")
    
    print("Attempting to write test_output.csv...")
    df.to_csv("test_output.csv", index=False)
    print("Success writing test_output.csv")
    
    with open("success_check.txt", "w") as f:
        f.write("Success")

except Exception as e:
    with open("error_check.txt", "w") as f:
        f.write(traceback.format_exc())
    print(f"Error: {e}")
