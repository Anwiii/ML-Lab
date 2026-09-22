import comtradeapicall as ca
df = ca.getReference('reporter')
print(df[df['text'].str.contains('Russia|Ukraine|Belarus', case=False)])