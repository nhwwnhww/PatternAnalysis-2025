import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('summary_all_metrics.csv')
df['base'] = df['image'].str.replace(r'(\.nii(\.gz)?)$', '', regex=True)
df['week'] = df['base'].str.extract(r'_Week(\d+)')[0].astype('Int64')

# 1) 直方图
plt.figure()
plt.hist(df['dice'], bins=30)
plt.title('Dice distribution (N={})'.format(len(df)))
plt.xlabel('Dice'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig('dice_hist.png')

# 2) 箱线图
plt.figure()
plt.boxplot(df['dice'].dropna())
plt.title('Dice boxplot')
plt.ylabel('Dice')
plt.tight_layout(); plt.savefig('dice_box.png')

# 3) 按周的均值±std（忽略缺失周）
byw = df.dropna(subset=['week']).groupby('week')['dice'].agg(['mean','std','count']).reset_index()
plt.figure()
plt.errorbar(byw['week'], byw['mean'], yerr=byw['std'], fmt='-o')
plt.title('Dice by Week (mean ± std)')
plt.xlabel('Week'); plt.ylabel('Dice')
for _, r in byw.iterrows():
    plt.text(r['week'], r['mean'], f"n={int(r['count'])}", fontsize=8, ha='center', va='bottom')
plt.tight_layout(); plt.savefig('dice_by_week.png')
print('Saved: dice_hist.png, dice_box.png, dice_by_week.png')
