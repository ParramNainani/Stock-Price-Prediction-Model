import requests
import json
import re

names = [
    "One Point One", "Trigyn Techno.", "Durlax Top", "Pasupati Acrylon", "Aartech Solonics",
    "Imagica. Enter.", "Filatex India", "Suzlon Energy", "Tirupati Forge", "Sagility",
    "Globe Civil", "Radiant Cash", "Vertoz", "Ugar Sugar Works", "Motherson Wiring",
    "BMW Industries", "Paramount Comm.", "Welspun Special.", "HP Adhesives", "Narmada Agrobase",
    "Indbank Merchant", "Iris Clothings", "BCL Industries", "Nova Agritech", "Sudarshan Pharma",
    "XT Global Infot.", "Cool Caps", "Ambalal Sarabhai", "United Polyfab", "Tiger Logistics",
    "Ajanta Soya", "Magellanic Cloud", "Trident", "Atal Realtech", "Jindal Worldwide",
    "Kesar Petroprod.", "HMA Agro Inds.", "Balaxi Pharma", "Geekay Wires", "Surana Solar",
    "Hindcon Chemical", "Fineotex Chem", "Bhatia Communic.", "Vishal Fabrics", "Century Extrus.",
    "Hardwyn India", "Blue Cloud Soft.", "Polo Queen Ind.", "R&B Denims", "Pavna Industries"
]

results = []

print("Resolving tickers from Yahoo Finance API...")
for name in names:
    clean_name = name.replace(".", "").strip()
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_name}&quotesCount=5"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        quotes = data.get("quotes", [])
        
        # Look for National Stock Exchange of India (NSE) first
        ticker = None
        for q in quotes:
            exch = q.get('exchange', '')
            if exch == 'NSI':
                ticker = q['symbol']
                break
        
        # Fallback to BSE if NSE not found
        if not ticker:
            for q in quotes:
                exch = q.get('exchange', '')
                if exch == 'BSE':
                    ticker = q['symbol']
                    break

        if ticker:
            # ensure .NS for NSI
            if ticker.endswith(".BO"):
                pass
            elif not ticker.endswith(".NS"):
                ticker = ticker + ".NS"
            results.append((ticker, name))
            print(f"✓ {name:<25} -> {ticker}")
        else:
            # Fallback manual text processing
            tk = re.sub(r'[^a-zA-Z0-9]', '', clean_name).upper() + ".NS"
            results.append((tk, name))
            print(f"⚠ {name:<25} -> {tk} (Fallback)")
    except Exception as e:
        tk = re.sub(r'[^a-zA-Z0-9]', '', clean_name).upper() + ".NS"
        results.append((tk, name))
        print(f"❌ {name:<25} -> {tk} (Failed)")

# Generate batch list
with open("batch_runner_50.py", "w", encoding="utf-8") as f:
    f.write('''import os
import re
import subprocess
import time

stocks = [
''')
    for tk, nm in results:
        f.write(f'    ("{tk}", "{nm}"),\n')
    f.write(''']

script_file = "advanced_stock_predictor.py"

with open(script_file, "r", encoding="utf-8") as f:
    original_code = f.read()

print(f"🚀 Starting MASSIVE Batch Prediction for {len(stocks)} Stocks...\\n")
print(f"{'Stock Name':<25} | {'Ticker':<12} | {'Last Close':<12} | {'Predicted':<12} | {'Change (%)':<12} | {'Action / Advice'}")
print("-" * 135)

for ticker, name in stocks:
    new_code = re.sub(r'TICKER\\s*=\\s*".*"', f'TICKER         = "{ticker}"', original_code)
    new_code = re.sub(r'TICKER_NAME\\s*=\\s*".*"', f'TICKER_NAME    = "{name}"', new_code)
    
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(new_code)
        
    output_file = f"run_output_{ticker.replace('.', '_')}.txt"
    subprocess.run(["python", "-X", "utf8", script_file], stdout=open(output_file, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    
    # Parse the output
    last_close, predicted, expected, action = "N/A", "N/A", "N/A", "N/A"
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-100:]:
                line = line.strip()
                if "Last close price   :" in line:
                    last_close = line.split(":")[-1].strip()
                elif "★ Ensemble prediction" in line:
                    predicted = line.split(":")[-1].strip()
                elif "★ Expected change" in line:
                    expected = line.split(":")[-1].strip()
                elif "★ ACTION:" in line:
                    action = line.split("★ ACTION:")[-1].strip()
    except: pass
    
    if action == "N/A":
       action = "Model Failed/Insufficient Data"
    
    print(f"{name:<25} | {ticker:<12} | {last_close:<12} | {predicted:<12} | {expected:<12} | {action}")

with open(script_file, "w", encoding="utf-8") as f:
    f.write(original_code)

print("\\n✅ Full suite 50-stock execution completed.")
''')
print("\nGenerated batch_runner_50.py successfully!")
