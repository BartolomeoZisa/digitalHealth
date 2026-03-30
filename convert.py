import pandas as pd

files = [
    "data/TD6TD20TD22.parquet",
    "data/UCP7.parquet"
]

for file in files:
    df = pd.read_parquet(file)
    output = file.replace(".parquet", ".csv")
    df.to_csv(output, index=False)
    print(f"Converted {file} → {output}")