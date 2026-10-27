"""Stream and validate the supplied LMVD ZIP into fixed BHDD input tensors.
Raw archive is never modified. Labels come exclusively from its Readme.md.
"""
import argparse
import csv
import hashlib
import json
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LM = [f"{axis}_{i}" for i in range(68) for axis in ("x", "y")]
AU = [f"AU{i:02d}_r" for i in (1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45)]
POSE = [f"pose_{name}" for name in ("Tx","Ty","Tz","Rx","Ry","Rz")]
GAZE = [f"gaze_{eye}_{axis}" for eye in (0,1) for axis in ("x","y","z")]
APG = AU + POSE + GAZE
META = ["timestamp", "confidence", "success", "face_id"]
COLS = LM + APG + META


def label_for(sample):
    if 1 <= sample <= 601 or 1117 <= sample <= 1423:
        return 1
    if 602 <= sample <= 1116 or 1425 <= sample <= 1824:
        return 0
    raise ValueError(f"No documented label for {sample}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_one(archive, out, sample, names):
    stem, audio_name, video_name = names
    record = {"id": stem, "numeric_id": sample, "label": label_for(sample)}
    done = out / "records" / f"{stem}.json"
    if done.exists():
        existing = json.loads(done.read_text(encoding="utf-8"))
        if existing.get("status") == "ready" and all(
            (out / file).is_file() and sha(out / file) == digest
            for file, digest in existing["output_sha256"].items()
        ):
            return existing
    try:
        with zipfile.ZipFile(archive) as z:
            record["source_members"] = {name: {"crc32": z.getinfo(name).CRC, "bytes": z.getinfo(name).file_size} for name in (audio_name,video_name)}
            with z.open(audio_name) as member:
                audio = np.load(member, allow_pickle=False)
                while member.read(1024*1024):
                    pass
            if audio.ndim != 2 or audio.shape[1] != 128 or not len(audio):
                raise ValueError(f"Bad audio shape {audio.shape}")
            if not np.isfinite(audio).all():
                raise ValueError("Audio contains non-finite values")
            fixed_audio = np.zeros((186,128), dtype=np.float32)
            fixed_audio[:min(186,len(audio))] = audio[:186]
            first = []
            kept = 0
            rows = valid_success = nonfinite = multiple_face = 0
            tmin = tmax = None
            decreasing = 0
            previous_time = None
            missing = None
            with z.open(video_name) as member:
                reader = pd.read_csv(member, skipinitialspace=True,
                    usecols=lambda name: name.strip() in COLS,
                    dtype=np.float32, chunksize=8192)
                for chunk in reader:
                    chunk.columns = chunk.columns.str.strip()
                    missing = set(COLS)-set(chunk.columns)
                    if missing:
                        raise ValueError(f"Missing video columns {sorted(missing)}")
                    values = chunk[LM+APG].to_numpy(dtype=np.float32)
                    nonfinite += int((~np.isfinite(values)).sum())
                    valid_success += int((chunk['success']==1).sum())
                    multiple_face += int((chunk['face_id']!=0).sum())
                    times = chunk['timestamp'].to_numpy()
                    if not np.isfinite(times).all():
                        raise ValueError("Non-finite timestamps")
                    if len(times):
                        decreasing += int((np.diff(times)<0).sum())
                        if previous_time is not None:
                            decreasing += int(times[0]<previous_time)
                        previous_time = float(times[-1])
                        tmin = float(times.min()) if tmin is None else min(tmin,float(times.min()))
                        tmax = float(times.max()) if tmax is None else max(tmax,float(times.max()))
                    if kept < 915:
                        part = values[:915-kept].copy()
                        first.append(part)
                        kept += len(part)
                    rows += len(chunk)
                while member.read(1024*1024):
                    pass
            if not rows:
                raise ValueError("Empty video features")
            record.update({"video_rows":rows,"audio_rows":len(audio),
                "video_timestamp_min":tmin,"video_timestamp_max":tmax,
                "video_success_rows":valid_success,"nonzero_face_id_rows":multiple_face,
                "timestamp_decreases":decreasing,"selected_feature_nonfinite_values":nonfinite,
                "video_truncated_rows":max(0,rows-915),"audio_truncated_rows":max(0,len(audio)-186),
                "video_padded_rows":max(0,915-rows),"audio_padded_rows":max(0,186-len(audio))})
            if nonfinite or multiple_face or decreasing:
                raise ValueError("Video requires review: non-finite features, multiple face IDs or decreasing timestamps")
            fixed = np.zeros((915,165), dtype=np.float32)
            fixed[:kept] = np.concatenate(first)
            outputs = {
                f"landmarks/{stem}.npy":fixed[:,:136],
                f"au_pose_gaze/{stem}.npy":fixed[:,136:],
                f"audio/{stem}.npy":fixed_audio,
            }
            hashes = {}
            for file, array in outputs.items():
                dest = out / file
                with dest.with_suffix('.tmp').open('wb') as handle:
                    np.save(handle,array,allow_pickle=False)
                dest.with_suffix('.tmp').replace(dest)
                hashes[file] = sha(dest)
            label_path = out / "labels" / f"{stem}_Depression.csv"
            label_path.write_text(f"label\n{record['label']}\n",encoding='utf-8')
            hashes[str(label_path.relative_to(out))] = sha(label_path)
            record.update(status="ready", output_sha256=hashes, crc_read_passed=True)
    except Exception as error:
        record.update(status="needs_review",error=f"{type(error).__name__}: {error}")
    done.write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,default=ROOT.parent.parent/'Dataset'/'LMVD_Feature.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'dataset'/'LMVD_real_v1')
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    for folder in ('landmarks','au_pose_gaze','audio','labels','records'):
        (out/folder).mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate archive member names')
        audio = {int(Path(n).stem):n for n in names if n.startswith('Audio_feature/') and n.endswith('.npy')}
        video = {int(Path(n).stem):n for n in names if n.startswith('Video_feature/') and n.endswith('.csv')}
        if audio.keys()!=video.keys():
            raise ValueError('Audio and video sample IDs differ')
        readme = z.read('Readme.md').decode('utf-8-sig')
        for token in ('001-601','1117-1423','0602-1116','1425-1824'):
            if token not in readme:
                raise ValueError('Unexpected label README; manual review needed')
        schema = {'archive':str(args.archive.resolve()),'archive_bytes':args.archive.stat().st_size,
            'archive_mtime_ns':args.archive.stat().st_mtime_ns,'landmark_columns':LM,'auxiliary_columns':APG,
            'shapes':{'landmarks':[915,136],'au_pose_gaze':[915,29],'audio':[186,128]},
            'policy':'First 915 visual rows and first 186 audio rows; zero-pad shorter sequences, matching existing loader. No frame deletion, resampling or normalization. Landmark xy interleaved to match Net.view(...,68,2). AU intensity only, six pose and six gaze vector columns. This is documented reconstruction, not author-confirmed original preprocessing.',
            'label_source':'Archive Readme.md; 1=001-601 and 1117-1423; 0=602-1116 and 1425-1824',
            'split_status':'Not split; participant/group identifiers unavailable in supplied archive',
            'source_readme':readme,'sample_count':len(audio)}
        schema_path = out/'schema.json'
        if schema_path.exists() and json.loads(schema_path.read_text(encoding='utf-8'))!=schema:
            raise RuntimeError('Output schema/source changed; use a new output directory')
        schema_path.write_text(json.dumps(schema,indent=2),encoding='utf-8')
        (out/'SOURCE_Readme.md').write_text(readme,encoding='utf-8')
        work = {i:(Path(audio[i]).stem,audio[i],video[i]) for i in sorted(audio)}
    start = time.monotonic()
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_one,args.archive,out,i,names) for i,names in work.items()]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if len(records)%25==0 or record['status']!='ready' or len(records)==len(work):
                progress={'processed':len(records),'total':len(work),'ready':sum(r['status']=='ready' for r in records),
                    'needs_review':sum(r['status']!='ready' for r in records),'elapsed_seconds':round(time.monotonic()-start,1)}
                (out/'progress.json').write_text(json.dumps(progress),encoding='utf-8')
                print(json.dumps(progress),flush=True)
                if record['status']!='ready': print(json.dumps(record),flush=True)
    records.sort(key=lambda r:r['numeric_id'])
    fields=['id','numeric_id','label','status','video_rows','audio_rows','video_success_rows','nonzero_face_id_rows','timestamp_decreases','selected_feature_nonfinite_values','video_truncated_rows','audio_truncated_rows','video_padded_rows','audio_padded_rows','error']
    with (out/'quality.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    summary={'archive_members_crc_verified':sum(2 for r in records if r.get('crc_read_passed'))+1,
        'sample_count':len(records),'status_counts':dict(Counter(r['status'] for r in records)),
        'ready_label_counts':dict(Counter(str(r['label']) for r in records if r['status']=='ready')),
        'review_ids':[r['id'] for r in records if r['status']!='ready'],
        'failed_tracking_rows_in_ready_samples':sum(r['video_rows']-r['video_success_rows'] for r in records if r['status']=='ready'),
        'missing_number_1424':'Not present in either modality and not included in documented label ranges',
        'elapsed_seconds':round(time.monotonic()-start,1),
        'training_status':'Not started; source grouping/split and preprocessing protocol need review'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
