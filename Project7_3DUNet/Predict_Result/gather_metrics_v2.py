out ='.\\summary_metrics.csv' if '--out' not in sys.argv else sys.argv[sys.argv.index('--out')+1]
rows=[]
[rows.append(pd.read_csv(m)) for m in glob.glob(os.path.join(root,'**','metrics.csv'),recursive=True)]
df=pd.concat(rows,ignore_index=True); 
df=df.rename(columns={'dice_binary_prostate':'dice'}) if 'dice_binary_prostate' in df.columns else df
df.to_csv(out,index=False); print(f"[Summary] cases={len(df)}  mean={df.dice.mean():.4f}  std={df.dice.std(ddof=0):.4f}"); print(f"[Saved] {out}")
out ='.\\summary_metrics.csv' if '--out' not in sys.argv else sys.argv[sys.argv.index('--out')+1]
rows=[]
[rows.append(pd.read_csv(m)) for m in glob.glob(os.path.join(root,'**','metrics.csv'),recursive=True)]
df=pd.concat(rows,ignore_index=True); 
df=df.rename(columns={'dice_binary_prostate':'dice'}) if 'dice_binary_prostate' in df.columns else df
df.to_csv(out,index=False); print(f"[Summary] cases={len(df)}  mean={df.dice.mean():.4f}  std={df.dice.std(ddof=0):.4f}"); print(f"[Saved] {out}")
