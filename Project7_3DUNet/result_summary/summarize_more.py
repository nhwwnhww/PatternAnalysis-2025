import pandas as pd
import numpy as np

# 读取合并表
df = pd.read_csv('summary_all_metrics.csv')   # columns: image, mode, dice, source

# 去后缀，得到基础名（不带 .nii/.nii.gz）
base = df['image'].str.replace(r'(\.nii(\.gz)?)$', '', regex=True)

# 解析 subject 与 week（允许缺失）
df['subject_full'] = base                      # 例如 B006_Week0_LFOV 或 B006_Week0
df['subject_id']   = base.str.extract(r'^([A-Z]\d+)')[0]        # <- 有()的捕获分组
df['week']         = base.str.extract(r'_Week(\d+)')[0].astype('Int64')  # pandas 可空整型

print(f'Cases={len(df)}, mean={df.dice.mean():.4f}, std={df.dice.std(ddof=0):.4f}')
print('quartiles:', df.dice.quantile([0.1,0.25,0.5,0.75,0.9]).to_dict())

# Top/Bottom 10
topk = df.sort_values('dice', ascending=False).head(10)
lowk = df.sort_values('dice', ascending=True).head(10)
topk.to_csv('top10_cases.csv', index=False)
lowk.to_csv('bottom10_cases.csv', index=False)
print('Saved: top10_cases.csv, bottom10_cases.csv')

# 失败/优秀计数
print('<=0.20 count:', int((df.dice <= 0.20).sum()))
print('>=0.80 count:', int((df.dice >= 0.80).sum()))

# 每周统计（忽略缺失周）
by_week = df.dropna(subset=['week']).groupby('week', dropna=True)['dice'].agg(['count','mean','std'])
by_week.to_csv('by_week_stats.csv')
print('Saved: by_week_stats.csv')

# 每受试者统计（忽略无法解析 subject_id 的行）
by_subject = df.dropna(subset=['subject_id']).groupby('subject_id')['dice'].agg(['count','mean','std'])
by_subject.to_csv('by_subject_stats.csv')
print('Saved: by_subject_stats.csv')
