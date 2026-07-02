import pandas as pd
import numpy as np

beans_df = pd.read_csv('Dry_Beans_Filtered_Assignment3.csv')
fashion_df = pd.read_csv('Fashion_MNIST_Filtered_Assignment3.csv')

print("Dry Beans Shape:", beans_df.shape)
print("Dry Beans Columns:", beans_df.columns.tolist())
print("Dry Beans Target Class Counts:\n", beans_df['Class'].value_counts())
print("\nFashion MNIST Shape:", fashion_df.shape)
print("Fashion MNIST label Counts:\n", fashion_df['label'].value_counts())
