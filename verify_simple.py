try:
    with open("task_1_processed.csv", "rb") as f: # Read bytes
        content = f.read()
        # Look for em dash (E2 80 94) or en dash (E2 80 93)
        if b'\xe2\x80\x94' in content:
            print("FOUND EM DASH")
        elif b'\xe2\x80\x93' in content:
            print("FOUND EN DASH")
        else:
            print("CLEAN")
    
    with open("simple_check.txt", "w") as f:
        f.write("Ran simple check")

except Exception as e:
    print(f"Error: {e}")
