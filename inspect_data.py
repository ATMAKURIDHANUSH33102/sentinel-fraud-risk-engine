"""Quick data inspection script."""
import pandas as pd
import numpy as np

print("Loading data...")
trans = pd.read_csv("data/raw/train_transaction.csv")
ident = pd.read_csv("data/raw/train_identity.csv")

print(f"\n{'='*70}")
print(f"TRANSACTIONS: {trans.shape}")
print(f"{'='*70}")
print(f"Columns: {list(trans.columns[:15])}...")
print(f"Fraud rate: {trans['isFraud'].mean():.4f} ({(trans['isFraud'].sum())} fraud, {(~trans['isFraud']).sum()} legit)")
print(f"Missing values: {trans.isnull().sum().sum()}")
print(f"Data types: {dict(trans.dtypes.value_counts())}")
print(f"\nTransactionDT range: {trans['TransactionDT'].min()} to {trans['TransactionDT'].max()}")
print(f"TransactionAMT range: {trans['TransactionAMT'].min():.2f} to {trans['TransactionAMT'].max():.2f}")
print(f"\nFirst 3 rows:")
print(trans[["TransactionID", "TransactionDT", "TransactionAMT", "isFraud"]].head(3))

print(f"\n{'='*70}")
print(f"IDENTITY: {ident.shape}")
print(f"{'='*70}")
print(f"Missing values: {ident.isnull().sum().sum()}")

print(f"\n{'='*70}")
print("READY TO IMPLEMENT PIPELINE")
print(f"{'='*70}")
