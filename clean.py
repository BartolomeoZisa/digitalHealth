import pandas as pd

# Load datasets
ucp = pd.read_csv("data/bbt_RAW_anon.csv")
td = pd.read_csv("data/bbt_RAW_sani_anon.csv")

# Remove duplicates
ucp = ucp.drop_duplicates()
td = td.drop_duplicates()

# ---- CLEAN TD ----
# Remove unwanted IDs
td_remove_ids = ["TD6", "TD20", "TD22", "TD19"]
td_clean = td[~td["id"].isin(td_remove_ids)]

# Load replacement TD data
td_replacements = pd.read_csv("data/TD6TD20TD22.csv")

# Append replacements
td_clean = pd.concat([td_clean, td_replacements], ignore_index=True)

# ---- CLEAN UCP ----
# Remove unwanted IDs
ucp_clean = ucp[ucp["id"] != "UCP7"]

# Load replacement UCP data
ucp_replacements = pd.read_csv("data/UCP7.csv")

# Append replacements
ucp_clean = pd.concat([ucp_clean, ucp_replacements], ignore_index=True)

# ---- FINAL DEDUP (optional but safe) ----
td_clean = td_clean.drop_duplicates()
ucp_clean = ucp_clean.drop_duplicates()

# Save cleaned files
td_clean.to_csv("data/bbt_RAW_TD_clean.csv", index=False)
ucp_clean.to_csv("data/bbt_RAW_UCP_clean.csv", index=False)

print("Cleaning completed successfully!")