import os
import re
import subprocess

stocks = [
    ("IDEA.NS", "Vodafone Idea"),
    ("YESBANK.NS", "Yes Bank"),
    ("SUZLON.NS", "Suzlon Energy"),
    ("IOB.NS", "Indian Overseas Bank"),
    ("UCOBANK.NS", "UCO Bank")
]

script_file = "advanced_stock_predictor.py"

with open(script_file, "r", encoding="utf-8") as f:
    original_code = f.read()

print("🚀 Starting Batch Prediction for 5 Penny Stocks...\n")
print(f"{'Stock Name':<25} | {'Ticker':<12} | {'Last Close':<12} | {'Predicted':<12} | {'Change (%)':<12} | {'Signal & Action'}")
print("-" * 125)

results = []

for ticker, name in stocks:
    # Update the script
    new_code = re.sub(r'TICKER\s*=\s*".*"', f'TICKER         = "{ticker}"', original_code)
    new_code = re.sub(r'TICKER_NAME\s*=\s*".*"', f'TICKER_NAME    = "{name}"', new_code)
    
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(new_code)
        
    # Run the script and capture output
    output_file = f"run_output_{ticker.replace('.', '_')}.txt"
    with open(output_file, "w", encoding="utf-8") as out_f:
        subprocess.run(["python", "-X", "utf8", script_file], stdout=out_f, stderr=subprocess.STDOUT)
    
    # Parse the output
    last_close = "N/A"
    predicted = "N/A"
    expected = "N/A"
    signal = "N/A"
    action = "N/A"
    
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[-50:]:
            line = line.strip()
            if "Last close price   :" in line:
                last_close = line.split(":")[-1].strip()
            elif "★ Ensemble prediction" in line:
                predicted = line.split(":")[-1].strip()
            elif "★ Expected change" in line:
                expected = line.split(":")[-1].strip()
            elif "★ SIGNAL:" in line:
                signal = line.split("★ SIGNAL:")[1].split("(Confidence")[0].strip()
            elif "★ ACTION:" in line:
                action = line.split("★ ACTION:")[-1].strip()
    
    print(f"{name:<25} | {ticker:<12} | {last_close:<12} | {predicted:<12} | {expected:<12} | {signal} | {action}")
    results.append({
        "name": name, "ticker": ticker, 
        "last_close": last_close, "predicted": predicted, 
        "expected": expected, "signal": signal
    })

# Restore original code
with open(script_file, "w", encoding="utf-8") as f:
    f.write(original_code)

print("\n✅ Batch execution completed. All outputs saved locally.")
