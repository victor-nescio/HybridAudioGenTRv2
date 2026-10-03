import numpy as np, pandas as pd, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SINIF_SIRASI=list(pd.read_parquet("data/processed/veri_listesi.parquet").groupby("sinif_idx")["sinif"].first().values)
v=pd.read_parquet('data/processed/veri_listesi.parquet')
ds=v.groupby('dosya')['sinif_idx'].first().reset_index()
tv,te=train_test_split(ds['dosya'].values,test_size=0.15,stratify=ds['sinif_idx'].values,random_state=42)
d=pd.read_parquet('data/processed/ses_metin_eslestirme.parquet')
dc=d.groupby('dosya')['sinif_idx'].first().reset_index()
ctv,cte=train_test_split(dc,test_size=0.1,stratify=dc['sinif_idx'],random_state=42)
ctr,cvl=train_test_split(ctv,test_size=0.1/0.9,stratify=ctv['sinif_idx'],random_state=42)
ctr=set(ctr.dosya)
te_sorted=np.array(sorted(te))
z=np.load('results/preds_test_seg.npz')
gt=z['gt']; fid=z['dosya_id']
assert fid.max()+1==len(te_sorted), (fid.max()+1, len(te_sorted))
# sanity: class of file matches gt
cls_of=ds.set_index('dosya')['sinif_idx']
assert (cls_of.loc[te_sorted[fid]].values==gt).all(), 'eslesme bozuk'
gordu=np.array([f in ctr for f in te_sorted])[fid]
hedef=[SINIF_SIRASI.index(s) for s in ['kopek_sesleri','kus_sesleri','hava_olaylari']]
print('segment: CLAP-gordu', gordu.sum(), ' gormedi', (~gordu).sum())
for alt,m in [('CLAP gordu',gordu),('CLAP GORMEDI',~gordu)]:
    mt=m & np.isin(gt,hedef)
    print(f'\n== {alt}: hedef segment {mt.sum()}, hedef dosya {len(np.unique(fid[mt]))}')
    for k in ['H1','C2','C3','C4']:
        rb=(z['baseline'][mt]==gt[mt]).mean(); ra=(z[k][mt]==gt[mt]).mean()
        fb=f1_score(gt[m],z['baseline'][m],labels=hedef,average='macro',zero_division=0)
        fa=f1_score(gt[m],z[k][m],labels=hedef,average='macro',zero_division=0)
        print(f'  {k}: hedef recall {rb:.3f}->{ra:.3f} ({ra-rb:+.3f})   hedef macroF1 {fb:.3f}->{fa:.3f} ({fa-fb:+.3f})')
