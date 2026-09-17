import pandas as pd

df = pd.read_csv('raw_occupancy_dataset.csv')

ZONE_MAX = 4
MAX_STUDENTS = df['Expected_Students'].max()

df['Expected_Zone_Level'] = (df['Expected_Students'] / MAX_STUDENTS * ZONE_MAX).round().astype(int)
df['Actual_Zone_Level']   = (df['Actual_Students']   / MAX_STUDENTS * ZONE_MAX).round().astype(int)

df.to_csv('scaled_occupancy_dataset.csv', index=False)
print(df.head())