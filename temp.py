import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# generate data
n_obs = 100
df = pd.DataFrame({'Community School?': np.random.choice(['Yes', 'No'], size=n_obs),
                   'Economic Need Index': np.random.uniform(size=n_obs),
                   'School Income Estimate': np.random.normal(loc=n_obs, size=n_obs)})

# your data pre-processing steps
df['color-code'] = np.where(df['Community School?']=='Yes', 'blue', 'red')
sc_income = df[~df['Economic Need Index'].isnull() & ~df['School Income Estimate'].isnull()]

# plot Economic Need Index vs School Income Estimate by group
groups = sc_income.groupby('Community School?')

fig, ax = plt.subplots(1, figsize=(40,20))

for label, group in groups:
    ax.scatter(group['Economic Need Index'], group['School Income Estimate'], 
               c=group['color-code'], label=label)

ax.set(xlabel='Economic Need', ylabel='School Income $', 
       title='Economic Need vs. School Income')
ax.legend(title='Community School?')
plt.show()