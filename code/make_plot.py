import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

regression = False
title = ''

if '--figure3a' in sys.argv:
    regression = True
    title = 'abalone'
elif '--figure3b' in sys.argv:
    regression = False
    title = 'adult'
else:
    raise Exception('No known figure specified.')

if len(sys.argv) < 4:
    raise Exception('Not enough input files specified (two are needed).')
else:
    data_ = pd.read_csv(sys.argv[2])
    data_np_ = pd.read_csv(sys.argv[3])
    if regression:
        data_['test_rmse_std_err'] = data_['test_rmse_std'] / np.sqrt(data_['runs'])
        data_np_['test_rmse_std_err'] = data_np_['test_rmse_std'] / np.sqrt(data_np_['runs'])
    else:
        data_['test_auc_std_err'] = data_['test_auc_std'] / np.sqrt(data_['runs'])
        data_np_['test_auc_std_err'] = data_np_['test_auc_std'] / np.sqrt(data_np_['runs'])

palette_ = iter(sns.color_palette())
palette = [next(palette_) for _ in range(5)]

if regression:
    data = data_.iloc[data_.groupby(['privacy_budget', 'Q', 'r1', 'init_ratio'], as_index=False).idxmin(axis=0)['test_rmse_mean'].values]
    data = data.rename(columns={'test_rmse_mean': 'test_score_mean', 'test_rmse_std_err': 'test_score_std_err'})
    maddock_row = data.query('Q==1.0 & r1==0.5 & init_ratio==0.0')
    s_bdt_row = data.copy()
    s_bdt_row = s_bdt_row.loc[s_bdt_row.groupby(['privacy_budget',], as_index=False).idxmin(axis=0)['test_score_mean'].values]
    data_np = data_np_.iloc[data_np_.groupby(['privacy_budget'], as_index=False).idxmin(axis=0)['test_rmse_mean'].values]
    data_np = data_np.rename(columns={'test_rmse_mean': 'test_score_mean', 'test_rmse_std_err': 'test_score_std_err'})
    data_np = data_np.append(data_np[data_np["privacy_budget"] == 0.0].copy())
    data_np.iloc[0, data_np.columns.get_loc("privacy_budget")] = 0.05
    data_np.iloc[-1, data_np.columns.get_loc("privacy_budget")] = 0.5
else:
    data = data_.iloc[data_.groupby(['privacy_budget', 'Q', 'r1'], as_index=False).idxmax(axis=0)['test_auc_mean'].values]
    data = data.rename(columns={'test_auc_mean': 'test_score_mean', 'test_auc_std_err': 'test_score_std_err'})
    maddock_row = data.query('Q==1.0 & r1==0.5')
    s_bdt_row = data.copy()
    s_bdt_row = s_bdt_row.loc[s_bdt_row.groupby(['privacy_budget',], as_index=False).idxmax(axis=0)['test_score_mean'].values]
    data_np = data_np_.iloc[data_np_.groupby(['privacy_budget'], as_index=False).idxmax(axis=0)['test_auc_mean'].values]
    data_np = data_np.rename(columns={'test_auc_mean': 'test_score_mean', 'test_auc_std_err': 'test_score_std_err'})
    data_np = data_np.append(data_np[data_np["privacy_budget"] == 0.0].copy())
    data_np.iloc[0, data_np.columns.get_loc("privacy_budget")] = 0.01
    data_np.iloc[-1, data_np.columns.get_loc("privacy_budget")] = 0.5

# Plotting
plt.figure(figsize=(10, 6))

if regression:
    rows = [maddock_row, s_bdt_row, data_np]
    labels = ['Maddock et al.', 'S-BDT', 'xgboost']
    colors = [0, 1, 2]
else:
    rows = [s_bdt_row, maddock_row, data_np]
    labels = ['S-BDT', 'Maddock et al.', 'xgboost']
    colors = [1, 0, 2]

for row, label, color in zip(rows, labels, colors):
    # Plot line
    ax = sns.lineplot(
        x='privacy_budget',
        y='test_score_mean',
        data=row,
        marker='o',
        label=label,
        c=palette[color]
    )
    plt.fill_between(
        x=row['privacy_budget'],
        y1=row['test_score_mean'] - row['test_score_std_err'],
        y2=row['test_score_mean'] + row['test_score_std_err'],
        alpha=0.2
    )

# Customize the plot
if not regression:
    ax.set_xscale('log')
    xticks=[0.01, 0.02, 0.03, 0.1, 0.5]
else:
    xticks=[0.05, 0.1, 0.2, 0.3, 0.5]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
plt.xlabel('Privacy Budget')
plt.ylabel('Test RMSE Mean' if regression else 'Test AUC Mean')
plt.title('Abalone' if regression else 'Adult')
plt.legend()
plt.savefig(f'{title}_plot_{datetime.now().strftime("%m.%d_%H:%M:%S")}.png')
