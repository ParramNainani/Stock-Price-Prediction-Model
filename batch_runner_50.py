import os
import re
import subprocess
import time

stocks = [
    ("ONEPOINT.NS", "One Point One"),
    ("TRIGYN.NS", "Trigyn Techno."),
    ("DURLAX-SM.NS", "Durlax Top"),
    ("PASUPTAC.NS", "Pasupati Acrylon"),
    ("AARTECH.NS", "Aartech Solonics"),
    ("IMAGICAA.NS", "Imagica. Enter."),
    ("FILATEX.NS", "Filatex India"),
    ("SUZLON.NS", "Suzlon Energy"),
    ("TIRUPATIFL.NS", "Tirupati Forge"),
    ("SAGILITY.NS", "Sagility"),
    ("GLOBECIVIL.NS", "Globe Civil"),
    ("RADIANTCMS.NS", "Radiant Cash"),
    ("VERTOZ.NS", "Vertoz"),
    ("UGARSUGAR.NS", "Ugar Sugar Works"),
    ("MOTHERSONWIRING.NS", "Motherson Wiring"),
    ("BMW.BO", "BMW Industries"),
    ("PARACABLES.NS", "Paramount Comm."),
    ("WELSPLSOL.BO", "Welspun Special."),
    ("HPAL.NS", "HP Adhesives"),
    ("NARMADA.NS", "Narmada Agrobase"),
    ("INDBANK.NS", "Indbank Merchant"),
    ("IRISDOREME.NS", "Iris Clothings"),
    ("BCLIND.NS", "BCL Industries"),
    ("NOVAAGRI.NS", "Nova Agritech"),
    ("SUDARSHAN.BO", "Sudarshan Pharma"),
    ("XTGLOBALINFOT.NS", "XT Global Infot."),
    ("COOLCAPS-SM.NS", "Cool Caps"),
    ("AMBALALSA.BO", "Ambalal Sarabhai"),
    ("UNITEDPOLY.NS", "United Polyfab"),
    ("TIGERLOGS.NS", "Tiger Logistics"),
    ("AJANTSOY.BO", "Ajanta Soya"),
    ("MCLOUD.NS", "Magellanic Cloud"),
    ("TTFL.BO", "Trident"),
    ("ATALREAL.NS", "Atal Realtech"),
    ("JINDWORLD.NS", "Jindal Worldwide"),
    ("KESARPE.BO", "Kesar Petroprod."),
    ("HMAAGROINDS.NS", "HMA Agro Inds."),
    ("BALAXI.NS", "Balaxi Pharma"),
    ("GEEKAYWIRE.NS", "Geekay Wires"),
    ("SURANASOL.NS", "Surana Solar"),
    ("HINDCON.NS", "Hindcon Chemical"),
    ("FCL.NS", "Fineotex Chem"),
    ("BHATIA.BO", "Bhatia Communic."),
    ("VISHAL.BO", "Vishal Fabrics"),
    ("CENTEXT.NS", "Century Extrus."),
    ("HARDWYN.NS", "Hardwyn India"),
    ("BLUECLOUDS.BO", "Blue Cloud Soft."),
    ("PQIF.BO", "Polo Queen Ind."),
    ("RBDENIMS.NS", "R&B Denims"),
    ("PAVNAIND.NS", "Pavna Industries"),
]

script_file = r"C:\Users\parra\OneDrive\Desktop\AML\Stock_price_predictor\advanced_stock_predictor.py"

with open(script_file, "r", encoding="utf-8") as f:
    original_code = f.read()

summary_file = "batch_summary_results.txt"
header1 = f"🚀 Starting MASSIVE Batch Prediction for {len(stocks)} Stocks...\n"
header2 = f"{'Stock Name':<25} | {'Ticker':<12} | {'Last Close':<12} | {'Action':<15} | {'20-Day Target':<15} | {'Stop Loss':<15}"
header3 = "-" * 110

print(header1)
print(header2)
print(header3)

with open(summary_file, "w", encoding="utf-8") as sf:
    sf.write(header1 + "\n")
    sf.write(header2 + "\n")
    sf.write(header3 + "\n")

    for ticker, name in stocks:
        new_code = re.sub(r'TICKER\s*=\s*".*"', f'TICKER         = "{ticker}"', original_code)
        new_code = re.sub(r'TICKER_NAME\s*=\s*".*"', f'TICKER_NAME    = "{name}"', new_code)
        new_code = re.sub(r'plt\.show\(\)', '# plt.show()', new_code) # Ensure no graphs pop up
        
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(new_code)
            
        output_file = f"run_output_{ticker.replace('.', '_')}.txt"
        subprocess.run(["python", "-X", "utf8", script_file], stdout=open(output_file, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
        
        # Parse the output
        last_close, action, swing_setup = "N/A", "N/A", "N/A"
        target, stop_loss = "N/A", "N/A"
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-100:]:
                    line = line.strip()
                    if "Last close price   :" in line:
                        last_close = line.split(":")[-1].strip()
                    elif "★ SIGNAL:" in line:
                        action = line.split("★ SIGNAL:")[-1].strip()
                    elif "20-DAY" in line and ("SETUP:" in line or "NO 20-DAY" in line):
                        swing_setup = line
                        if "Target:" in line:
                            parts = line.split("|")
                            for p in parts:
                                if "Target:" in p:
                                    target = p.split("Target:")[-1].strip()
                                if "Stop Loss:" in p:
                                    stop_loss = p.split("Stop Loss:")[-1].strip()
        except: pass
        
        if action == "N/A":
           action = "Model Failed"
           target = "-"
           stop_loss = "-"
        else:
           # Clean up the action string slightly for the table
           action = action.split("(")[0].strip()
           if "NO 20-DAY SETUP" in swing_setup:
               target = "Wait"
               stop_loss = "Wait"

        result_str = f"{name:<25} | {ticker:<12} | {last_close:<12} | {action:<15} | {target:<15} | {stop_loss:<15}"
        print(result_str)
        sf.write(result_str + "\n")
        sf.flush()

with open(script_file, "w", encoding="utf-8") as f:
    f.write(original_code)

footer = "\n✅ Full suite 50-stock execution completed."
print(footer)
with open(summary_file, "a", encoding="utf-8") as sf:
    sf.write(footer + "\n")
