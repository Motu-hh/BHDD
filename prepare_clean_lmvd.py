"""Clean fixed LMVD windows; freeze video-level splits; fit train-only scaling."""
from pathlib import Path
import json,csv,hashlib
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'dataset'/'LMVD_real_v1'
DST=ROOT/'dataset'/'LMVD_clean_v2'
MODES={'landmarks':136,'au_pose_gaze':29,'audio':128}


def dump(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def table(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def arrays(r):
    with np.load(SRC/'window_metadata'/f"{r['id']}.npz",allow_pickle=False) as m:
        visual=m['tracking_success_mask'].copy()
    audio=np.arange(186)<min(186,r['audio_rows'])
    for mode in MODES:
        a=np.load(SRC/mode/f"{r['id']}.npy",allow_pickle=False)
        mask=audio if mode=='audio' else visual
        yield mode,a,mask


def main():
    if DST.exists():raise RuntimeError('Output exists; refusing overwrite')
    records=sorted([json.loads(p.read_text(encoding='utf-8')) for p in (SRC/'records').glob('*.json')],key=lambda r:r['numeric_id'])
    eligible=[]; excluded=[];seen={}
    prefix_groups={}
    for r in records:
        if r['status']=='ready':
            signature=tuple(r['output_sha256'][f"{mode}/{r['id']}.npy"] for mode in MODES)
            prefix_groups.setdefault(signature,[]).append(r)
    conflicts={r['id'] for group in prefix_groups.values() if len({q['label'] for q in group})>1 for r in group}
    for r in records:
        reason=None
        if r['status']!='ready':reason='source_nonfinite_or_schema_review'
        elif r['id'] in conflicts:reason='identical_prefix_features_with_conflicting_labels'
        else:
            with np.load(SRC/'window_metadata'/f"{r['id']}.npz",allow_pickle=False) as m:
                valid=int(m['tracking_success_mask'].sum())
            if valid==0:reason='zero_successful_tracking_in_first_window'
            elif r['audio_rows']==0:reason='empty_audio'
            else:
                signature=tuple(r['output_sha256'][f"{mode}/{r['id']}.npy"] for mode in MODES)
                if signature in seen:
                    prior=seen[signature]
                    if prior['label']!=r['label']:raise RuntimeError('Identical features with conflicting labels')
                    reason='identical_prefix_features_same_label_duplicate_of_'+prior['id']
                else:seen[signature]=r
        if reason:excluded.append({'id':r['id'],'label':r['label'],'reason':reason})
        else:eligible.append(r)
    labels=[r['label'] for r in eligible]
    pool,test=train_test_split(np.arange(len(eligible)),test_size=0.1,stratify=labels,random_state=42)
    train,dev=train_test_split(pool,test_size=1/9,stratify=np.array(labels)[pool],random_state=43)
    splits={name:set(map(int,idx)) for name,idx in [('train',train),('dev',dev),('test',test)]}
    assert not(splits['train']&splits['dev'] or splits['train']&splits['test'] or splits['dev']&splits['test'])
    assert set.union(*splits.values())==set(range(len(eligible)))
    DST.mkdir(parents=True)
    for folder in (*MODES,'labels','masks','splits'): (DST/folder).mkdir()
    accum={mode:{'count':0,'sum':np.zeros(dim,dtype=np.float64),'sum_sq':np.zeros(dim,dtype=np.float64)} for mode,dim in MODES.items()}
    for i in sorted(splits['train']):
        for mode,a,mask in arrays(eligible[i]):
            x=a[mask].astype(np.float64)
            assert np.isfinite(x).all() and len(x)>0
            accum[mode]['count']+=len(x)
            accum[mode]['sum']+=x.sum(axis=0)
            accum[mode]['sum_sq']+=(x*x).sum(axis=0)
    stats={}
    for mode,a in accum.items():
        mean=a['sum']/a['count'];var=np.maximum(a['sum_sq']/a['count']-mean*mean,0)
        std=np.sqrt(var);std[std<1e-6]=1.0
        stats[mode+'_mean']=mean;stats[mode+'_std']=std
    np.savez(DST/'train_normalization.npz',**stats)
    manifest=[];output_stats={mode:{'max_abs':0.0,'sum':np.zeros(dim),'sum_sq':np.zeros(dim),'count':0} for mode,dim in MODES.items()}
    for i,r in enumerate(eligible):
        split=next(name for name,idx in splits.items() if i in idx)
        masks={}
        for mode,a,mask in arrays(r):
            b=np.zeros_like(a,dtype=np.float32)
            b[mask]=((a[mask].astype(np.float64)-stats[mode+'_mean'])/stats[mode+'_std']).astype(np.float32)
            assert np.isfinite(b).all() and (b[~mask]==0).all()
            np.save(DST/mode/f"{r['id']}.npy",b,allow_pickle=False)
            masks['audio_valid' if mode=='audio' else 'visual_valid']=mask
            output_stats[mode]['max_abs']=max(output_stats[mode]['max_abs'],float(np.abs(b).max()))
            if split=='train':
                x=b[mask].astype(np.float64);q=output_stats[mode]
                q['sum']+=x.sum(axis=0);q['sum_sq']+=(x*x).sum(axis=0);q['count']+=len(x)
        np.savez_compressed(DST/'masks'/f"{r['id']}.npz",**masks)
        (DST/'labels'/f"{r['id']}_Depression.csv").write_text(f"label\n{r['label']}\n",encoding='utf-8')
        manifest.append({'id':r['id'],'label':r['label'],'split':split,'visual_valid_rows':int(masks['visual_valid'].sum()),'audio_valid_rows':int(masks['audio_valid'].sum())})
        if (i+1)%250==0:print('Normalized',i+1,'/',len(eligible),flush=True)
    table(DST/'manifest.csv',manifest)
    table(DST/'excluded.csv',excluded)
    for name in splits:table(DST/'splits'/f'{name}.csv',[r for r in manifest if r['split']==name])
    # Future cross-validation stays inside development pool, away from test.
    pool=sorted(splits['train']|splits['dev'])
    cv=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(10,shuffle=True,random_state=42).split(pool,np.array(labels)[pool]),1):
        for role,indices in [('train',tr),('dev',va)]:
            for j in indices:cv.append({'fold':fold,'role':role,'id':eligible[pool[j]]['id'],'label':eligible[pool[j]]['label']})
    table(DST/'splits'/'development_cv10.csv',cv)
    scale_checks={}
    for mode,q in output_stats.items():
        mean=q['sum']/q['count'];std=np.sqrt(np.maximum(q['sum_sq']/q['count']-mean*mean,0))
        active=np.sqrt(np.maximum(accum[mode]['sum_sq']/accum[mode]['count']-(accum[mode]['sum']/accum[mode]['count'])**2,0))>=1e-6
        assert np.abs(mean).max()<1e-5
        assert np.max(np.abs(std[active]-1))<1e-5
        scale_checks[mode]={'train_max_abs_mean':float(np.abs(mean).max()),'train_max_std_error':float(np.abs(std[active]-1).max()),'all_splits_max_abs':q['max_abs'],'fitted_valid_train_rows':accum[mode]['count']}
    report={'source_samples':len(records),'included_samples':len(eligible),'excluded_samples':len(excluded),'exclusion_reasons':dict(Counter(r['reason'] for r in excluded)),
        'splits':{name:{'count':len(idx),'class_counts':dict(Counter(str(eligible[i]['label']) for i in idx))} for name,idx in splits.items()},
        'source_policy':'First-window prefix crops inherited from public loader, no claim of synchronized audio/video physical times',
        'clean_policy':'Tracker success==1 keeps visual frames; invalid frames and padding set to zero after train-only per-feature standardization. Audio padding excluded from normalization. No interpolation or frame deletion.',
        'split_policy':'Video-level stratified approx 80/10/10, seeds42/43. Subject grouping unavailable; subject leakage cannot be ruled out.',
        'test_policy':'Reserved. Not used for statistics fitting, model selection, or performance calculation.',
        'cv_policy':'10fold file is a future development-pool partition only. Normalization MUST be refit separately on each fold training portion; global train_normalization.npz is for fixed train/dev/test only.',
        'normalization':scale_checks,'formal_training_started':False}
    dump(DST/'preparation.json',report)
    (DST/'README.md').write_text('# LMVD clean v2\n\nFixed video-level train/dev/test splits, no subject-independence guarantee. Tracker failure frames remain in place as zeros; masks are included but the current model has no attention padding masks. Prefix crop audio/video time alignment remains unverified. Train-only per-feature scaling. Reserved test has no reported performance. Future cross-validation must refit scaling inside EACH training fold; do not reuse this holdout scaler. Local positional encoding correction changes original BHDD behavior. See preparation.json and excluded.csv.\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
