import pandas as pd

tr = pd.read_csv('data/manifests/train.csv')
va = pd.read_csv('data/manifests/val.csv')

print('All columns in train.csv:')
print(list(tr.columns))
print(f'\nTRAIN shape: {tr.shape}')
print(f'VAL shape: {va.shape}')

# Check behavior column distribution
if 'behavior' in tr.columns:
    print('\n=== BEHAVIOR DISTRIBUTION ===')
    print('\nTRAIN behaviors:')
    print(tr['behavior'].value_counts().sort_values(ascending=False))
    print('\nVAL behaviors:')
    print(va['behavior'].value_counts().sort_values(ascending=False))

# Check primary label distribution
if '_primary_label' in tr.columns:
    print('\n=== PRIMARY LABEL DISTRIBUTION ===')
    print('\nTRAIN primary labels:')
    print(tr['_primary_label'].value_counts().sort_values(ascending=False))
    print('\nVAL primary labels:')
    print(va['_primary_label'].value_counts().sort_values(ascending=False))
