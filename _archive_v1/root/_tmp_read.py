import pandas as pd
df = pd.read_excel(r'd:\cross_sucking\cross_sucking\data\manifests\final_manifest.xlsx')

print('---DATE VALUES---')
print(df['Date'].value_counts())
print()

print('---DURATION STATS---')
print(df['Duration'].describe())
print()

print('---LAST 5---')
print(df.tail(5).to_string())
print()

print('---TOTAL EVENTS:', len(df))
print()

print('---COHORT x DAY---')
print(pd.crosstab(df['Cohort'], df['Day']))
print()

print('---COHORT x BEHAVIOR---')
print(pd.crosstab(df['Cohort'], df['Behavior']))
print()

# Compare with existing MASTER_FINAL_CLEAN.csv
try:
    master = pd.read_csv(r'd:\cross_sucking\cross_sucking\data\manifests\MASTER_FINAL_CLEAN.csv')
    print('MASTER_FINAL_CLEAN shape:', master.shape)
    print('MASTER columns:', list(master.columns))
    print()
    if 'behavior' in master.columns:
        print('MASTER behavior dist:')
        print(master['behavior'].value_counts())
        print()
    if 'group' in master.columns:
        print('MASTER group dist:')
        print(master['group'].value_counts())
except Exception as e:
    print('Could not read MASTER:', e)

# Check Ended.by column unique values
print()
print('---ENDED_BY VALUES---')
print(df['Ended.by.initiator.or.receiver.'].value_counts())
print()
print('---PEN_LOCATION VALUES---')
print(df['Pen.location'].value_counts())
