"""
Plot UNSW-NB15 Attack Category Distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
print('Loading UNSW-NB15 datasets...')
train_df = pd.read_csv('data/UNSW_NB15_training-set.csv')
test_df = pd.read_csv('data/UNSW_NB15_testing-set.csv')

print(f'Training set: {len(train_df)} samples')
print(f'Testing set: {len(test_df)} samples')

# Combine for full distribution
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f'Total samples: {len(combined_df)}')

# Get attack category distribution
attack_dist = combined_df['attack_cat'].fillna('Normal').value_counts().sort_values(ascending=True)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color palette
colors = plt.cm.Set3(np.linspace(0, 1, len(attack_dist)))

# Plot 1: Horizontal bar chart
ax1 = axes[0]
bars = ax1.barh(attack_dist.index, attack_dist.values, color=colors)
ax1.set_xlabel('Number of Samples', fontsize=12)
ax1.set_ylabel('Attack Category', fontsize=12)
ax1.set_title('UNSW-NB15 Attack Distribution', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, val in zip(bars, attack_dist.values):
    ax1.text(val + 500, bar.get_y() + bar.get_height()/2, 
             f'{val:,}', va='center', fontsize=9)

ax1.set_xlim(0, max(attack_dist.values) * 1.15)

# Plot 2: Pie chart
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    attack_dist.values, 
    labels=attack_dist.index, 
    autopct='%1.1f%%', 
    colors=colors,
    explode=[0.05 if cat == 'Normal' else 0 for cat in attack_dist.index]
)
ax2.set_title('Attack Category Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('attack_distribution_unswnb15.png', dpi=150, bbox_inches='tight')
plt.savefig('attack_distribution_unswnb15.pdf', bbox_inches='tight')
print('\nSaved: attack_distribution_unswnb15.png/pdf')

# Print distribution table
print('\n' + '='*50)
print('UNSW-NB15 Attack Distribution')
print('='*50)
total = len(combined_df)
for cat in attack_dist.sort_values(ascending=False).index:
    count = attack_dist[cat]
    pct = count / total * 100
    print(f'{cat:15s}: {count:>8,} ({pct:5.2f}%)')
print('='*50)
print(f'{"TOTAL":15s}: {total:>8,}')

plt.show()
