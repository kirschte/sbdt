import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

regression = False
learning_on_streams = False
title = ''

if '--figure3a' in sys.argv:
    regression = True
    learning_on_streams = False
    title = 'abalone'
elif '--figure3b' in sys.argv:
    regression = False
    learning_on_streams = False
    title = 'adult'
elif '--figure4a' in sys.argv:
    regression = True
    learning_on_streams = True
    title = 'abalone'
elif '--figure4b' in sys.argv:
    regression = False
    learning_on_streams = True
    title = 'adult'
else:
    raise Exception('No known figure specified.')

if not learning_on_streams and len(sys.argv) < 4:
    raise Exception('Not enough input files specified (two are needed).')
elif learning_on_streams and len(sys.argv) < 3:
    raise Exception('No input file specified.')
else:
    data_ = pd.read_csv(sys.argv[2])
    if not learning_on_streams:
        data_np_ = pd.read_csv(sys.argv[3])
    if regression:
        data_['test_rmse_std_err'] = data_['test_rmse_std'] / np.sqrt(data_['runs'])
        if not learning_on_streams:
            data_np_['test_rmse_std_err'] = data_np_['test_rmse_std'] / np.sqrt(data_np_['runs'])
    else:
        data_['test_auc_std_err'] = data_['test_auc_std'] / np.sqrt(data_['runs'])
        if not learning_on_streams:
            data_np_['test_auc_std_err'] = data_np_['test_auc_std'] / np.sqrt(data_np_['runs'])

palette_ = iter(sns.color_palette())
palette = [next(palette_) for _ in range(5)]

if regression:
    data = data_.iloc[data_.groupby(['privacy_budget', 'Q', 'r1', 'init_ratio', 'use_pf'], as_index=False).idxmin(axis=0)['test_rmse_mean'].values]
    data = data.rename(columns={'test_rmse_mean': 'test_score_mean', 'test_rmse_std_err': 'test_score_std_err'})
    maddock_row = data.query('Q==1.0 & r1==0.5 & init_ratio==0.0')
    if not learning_on_streams:
        s_bdt_row = data.copy()
        s_bdt_row = s_bdt_row.loc[s_bdt_row.groupby(['privacy_budget',], as_index=False).idxmin(axis=0)['test_score_mean'].values]
        data_np = data_np_.iloc[data_np_.groupby(['privacy_budget'], as_index=False).idxmin(axis=0)['test_rmse_mean'].values]
        data_np = data_np.rename(columns={'test_rmse_mean': 'test_score_mean', 'test_rmse_std_err': 'test_score_std_err'})
        data_np = data_np.append(data_np[data_np["privacy_budget"] == 0.0].copy())
        data_np.iloc[0, data_np.columns.get_loc("privacy_budget")] = 0.05
        data_np.iloc[-1, data_np.columns.get_loc("privacy_budget")] = 0.5
    else:
        s_bdt_data = data.copy()
        s_bdt_data = s_bdt_data.loc[s_bdt_data.groupby(['privacy_budget', 'use_pf'], as_index=False).idxmin(axis=0)['test_score_mean'].values]
        s_bdt_row = s_bdt_data.query('use_pf==True')
        s_bdt_row_noirf = s_bdt_data.query('use_pf==False')

else:
    data = data_.iloc[data_.groupby(['privacy_budget', 'Q', 'r1', 'use_pf'], as_index=False).idxmax(axis=0)['test_auc_mean'].values]
    data = data.rename(columns={'test_auc_mean': 'test_score_mean', 'test_auc_std_err': 'test_score_std_err'})
    maddock_row = data.query('Q==1.0 & r1==0.5')
    if not learning_on_streams:
        s_bdt_row = data.copy()
        s_bdt_row = s_bdt_row.loc[s_bdt_row.groupby(['privacy_budget',], as_index=False).idxmax(axis=0)['test_score_mean'].values]
        data_np = data_np_.iloc[data_np_.groupby(['privacy_budget'], as_index=False).idxmax(axis=0)['test_auc_mean'].values]
        data_np = data_np.rename(columns={'test_auc_mean': 'test_score_mean', 'test_auc_std_err': 'test_score_std_err'})
        data_np = data_np.append(data_np[data_np["privacy_budget"] == 0.0].copy())
        data_np.iloc[0, data_np.columns.get_loc("privacy_budget")] = 0.01
        data_np.iloc[-1, data_np.columns.get_loc("privacy_budget")] = 0.5
    else:
        s_bdt_data = data.copy()
        s_bdt_data = s_bdt_data.loc[s_bdt_data.groupby(['privacy_budget', 'use_pf'], as_index=False).idxmax(axis=0)['test_score_mean'].values]
        s_bdt_row = s_bdt_data.query('use_pf==True')
        s_bdt_row_noirf = s_bdt_data.query('use_pf==False')

# Plotting
plt.figure(figsize=(10, 6))

if regression:
    rows = [maddock_row, s_bdt_row, data_np] if not learning_on_streams else [maddock_row, s_bdt_row_noirf, s_bdt_row]
    labels = ['Maddock et al.', 'S-BDT', 'xgboost'] if not learning_on_streams else ['Maddock et al. (naive extra rounds)', 'S-BDT (naive extra rounds)', 'S-BDT (Rényi filter extra rounds)']
    colors = [0, 1, 2] if not learning_on_streams else [0, 1, 3]
else:
    rows = [maddock_row, s_bdt_row, data_np] if not learning_on_streams else [s_bdt_row_noirf, s_bdt_row, maddock_row]
    labels = ['Maddock et al.', 'S-BDT', 'xgboost'] if not learning_on_streams else  ['S-BDT (naive extra rounds)', 'S-BDT (Rényi filter extra rounds)', 'Maddock et al. (naive extra rounds)']
    colors = [0, 1, 2] if not learning_on_streams else [0, 1, 3]

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
