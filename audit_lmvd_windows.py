"""Retain metadata for the first visual window; do not shift/delete frames."""
import csv
import json
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'dataset'/'LMVD_real_v1'
ARCHIVE=Path(json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['archive'])


def inspect(name):
    stem=Path(name).stem
    columns=['timestamp','success','confidence','pose_Tz']
    with zipfile.ZipFile(ARCHIVE) as z:
        with z.open(name) as member:
            frame=pd.read_csv(member,skipinitialspace=True,nrows=915,usecols=columns,dtype=np.float32)
    n=len(frame)
    mask=np.zeros(915,dtype=np.bool_)
    mask[:n]=frame['success'].to_numpy()==1
    timestamp=np.full(915,np.nan,dtype=np.float32)
    confidence=np.zeros(915,dtype=np.float32)
    timestamp[:n]=frame['timestamp'].to_numpy()
    confidence[:n]=frame['confidence'].to_numpy()
    np.savez_compressed(DATA/'window_metadata'/f'{stem}.npz',tracking_success_mask=mask,timestamp=timestamp,confidence=confidence)
    pose=frame['pose_Tz'].to_numpy()
    finite=np.isfinite(pose)
    failed=frame['success'].to_numpy()!=1
    return {'id':stem,'visual_rows_retained':n,'tracking_success_rows':int(mask.sum()),
        'tracking_failed_rows':int(failed.sum()),'window_end_seconds':float(frame['timestamp'].iloc[-1]),
        'pose_tz_max_abs':float(np.abs(pose[finite]).max()) if finite.any() else None,
        'failed_tracking_pose_tz_max_abs':float(np.abs(pose[finite & failed]).max()) if (finite & failed).any() else None}


def main():
    folder=DATA/'window_metadata'
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        names=[n for n in z.namelist() if n.startswith('Video_feature/') and n.endswith('.csv')]
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(inspect,name) for name in names]):
            records.append(future.result())
            if len(records)%250==0: print('Window metadata',len(records),'/',len(names),flush=True)
    records.sort(key=lambda r:int(r['id']))
    with (DATA/'window_quality.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    summary={'samples':len(records),'samples_with_failed_tracking_in_retained_window':sum(r['tracking_failed_rows']>0 for r in records),
        'samples_with_zero_success_in_retained_window':sum(r['tracking_success_rows']==0 for r in records),
        'failed_tracking_rows':sum(r['tracking_failed_rows'] for r in records),
        'retained_rows':sum(r['visual_rows_retained'] for r in records),
        'warning':'Feature arrays remain raw. Masks describe tracker success only, not validated clinical signal or normalization. Audio time alignment is not inferred.'}
    (DATA/'window_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
