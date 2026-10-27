"""Full fixed-split LMVD training for the local cleaned/corrected variant.
Uses all fixed training and development samples. Never evaluates reserved test.
"""
import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'/'BHDD'))
from dataloader import BHDDDataset
from train import build_adjacency_matrix,reconstruction_loss
from bhdd_lmvd_corrected import Net


def parse_args():
    p=argparse.ArgumentParser(description='Full fixed-split LMVD local baseline; no test evaluation')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--run-name',default='lmvd_full_seed0')
    p.add_argument('--resume',action='store_true',help='Continue from last completed epoch in the same run directory')
    p.add_argument('--check-only',action='store_true',help='Validate data, GPU and model forward; do not train or create a run directory')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.patience)<=0 or a.learning_rate<=0:p.error('Training numeric arguments must be positive')
    if not a.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.run_name):p.error('Run name must contain only letters, digits, underscore or hyphen')
    return a


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def main():
    args=parse_args();data=ROOT/'dataset'/'LMVD_clean_v2';out=ROOT/'outputs'/args.run_name
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: use the bhdd Python environment')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    frame=pd.read_csv(data/'manifest.csv',dtype={'id':str})
    train=frame[frame.split=='train'].copy();dev=frame[frame.split=='dev'].copy();test=frame[frame.split=='test']
    assert len(train)>0 and len(dev)>0
    assert not(set(train.id)&set(dev.id) or set(train.id)&set(test.id) or set(dev.id)&set(test.id))
    def dataset(rows):return BHDDDataset(data/'landmarks',data/'au_pose_gaze',data/'audio',data/'labels',[i+'.npy' for i in rows.id])
    train_ds=dataset(train);dev_ds=dataset(dev)
    for stem in pd.concat([train,dev]).id:
        for sub,suffix in [('landmarks','.npy'),('au_pose_gaze','.npy'),('audio','.npy'),('labels','_Depression.csv')]:
            if not(data/sub/(stem+suffix)).is_file():raise FileNotFoundError(stem+' '+sub)
    device=torch.device('cuda');adj=build_adjacency_matrix().to(device)
    net=Net(adj).to(device)
    print(f'GPU: {torch.cuda.get_device_name(0)} | train={len(train)}, dev={len(dev)}, reserved test={len(test)}',flush=True)
    if args.check_only:
        net.eval();lm,au,audio,_=train_ds[0]
        with torch.no_grad():logits=net(lm[None].to(device),au[None].to(device),audio[None].to(device))[0]
        assert logits.shape==(1,2) and torch.isfinite(logits).all().item()
        print('CHECK PASSED: all training/dev files found; GPU forward finite; no training started',flush=True)
        return
    config={**vars(args),'train_count':len(train),'dev_count':len(dev),'test_count':len(test),
        'model':'bhdd_lmvd_corrected.Net','loss':'CE + 0.7*clamp(mean MSE,1e-10,1e10)/1e10, original scale',
        'optimizer':'Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4','scheduler':'none',
        'selection':'highest dev macro F1; earliest epoch on ties; argmax classification',
        'data_manifest_sha256':digest(data/'manifest.csv'),'normalization_sha256':digest(data/'train_normalization.npz'),
        'source_model_sha256':digest(ROOT/'model'/'BHDD'/'model.py'),'corrected_model_sha256':digest(ROOT/'bhdd_lmvd_corrected.py'),
        'training_script_sha256':digest(Path(__file__)),'reserved_test_evaluated':False,
        'limitations':'Local cleaned/corrected version; video-level split only, subject overlap cannot be excluded; independent prefix crops have unverified physical time alignment; zero inputs have no attention padding mask; auxiliary loss is very small after scaling; single seed exploratory baseline, not exact paper reproduction'}
    optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    first=1;best=-1.;best_epoch=None;stale=0;history=[]
    if args.resume:
        last=out/'last.pth'
        if not last.is_file():raise FileNotFoundError(last)
        checkpoint=torch.load(last,map_location=device,weights_only=True)
        old=checkpoint['config']
        for key in ('seed','batch_size','learning_rate','patience','data_manifest_sha256','normalization_sha256','source_model_sha256','corrected_model_sha256','training_script_sha256'):
            if old[key]!=config[key]:raise RuntimeError('Resume configuration/source differs: '+key)
        net.load_state_dict(checkpoint['net'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        first=checkpoint['epoch']+1;best=checkpoint['best_dev_macro_f1'];best_epoch=checkpoint['best_epoch'];stale=checkpoint['stale_epochs'];history=checkpoint['history']
        torch.set_rng_state(checkpoint['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint['cuda_rng']])
    else:
        if out.exists():raise RuntimeError('Run directory exists; choose a new --run-name or use --resume')
        out.mkdir(parents=True)
        train.to_csv(out/'train_ids.csv',index=False);dev.to_csv(out/'dev_ids.csv',index=False)
    if first>args.epochs:raise RuntimeError('Requested epochs already completed; increase --epochs to continue')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()],force=True)
    logging.info('Starting at epoch %d; max epochs %d; best dev F1 %.6f; original test not evaluated',first,args.epochs,best)
    dev_loader=DataLoader(dev_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    started=time.monotonic()
    try:
        for epoch in range(first,args.epochs+1):
            train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed+epoch))
            net.train();total=0.;ce_total=0.;rec_total=0.;count=0;max_norm=0.;saturated=0
            progress=tqdm(train_loader,desc=f'Epoch {epoch}/{args.epochs} train')
            for lm,au,audio,label in progress:
                lm,au,audio,label=[x.to(device) for x in (lm,au,audio,label)]
                optimizer.zero_grad(set_to_none=True)
                logits,_,_,visual,rebuilt=net(lm,au,audio)
                ce=torch.nn.functional.cross_entropy(logits,label);rec=reconstruction_loss(visual,rebuilt);loss=ce+0.7*rec
                if not torch.isfinite(loss).item():raise RuntimeError('Non-finite training loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                n=len(label);count+=n;total+=loss.item()*n;ce_total+=ce.item()*n;rec_total+=rec.item()*n
                max_norm=max(max_norm,norm.item());saturated+=int(rec.item()>=1.)
                progress.set_postfix(loss=f'{total/count:.4f}')
            net.eval();truth=[];pred=[];dev_loss=0.
            with torch.no_grad():
                for lm,au,audio,label in tqdm(dev_loader,desc='Development'):
                    lm,au,audio,label=[x.to(device) for x in (lm,au,audio,label)]
                    logits,_,_,visual,rebuilt=net(lm,au,audio)
                    loss=torch.nn.functional.cross_entropy(logits,label)+0.7*reconstruction_loss(visual,rebuilt)
                    if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():raise RuntimeError('Non-finite dev output')
                    dev_loss+=loss.item()*len(label);truth+=label.cpu().tolist();pred+=logits.argmax(1).cpu().tolist()
            score=float(f1_score(truth,pred,labels=[0,1],average='macro',zero_division=0))
            row={'epoch':epoch,'train_loss':total/count,'train_ce':ce_total/count,'train_reconstruction':rec_total/count,'dev_loss':dev_loss/len(truth),
                'dev_macro_f1':score,'dev_accuracy':float(accuracy_score(truth,pred)),'dev_balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
                'max_gradient_norm_before_clipping':max_norm,'reconstruction_saturated_steps':saturated}
            history.append(row)
            improved=score>best
            if improved:best=score;best_epoch=epoch;stale=0
            else:stale+=1
            checkpoint={'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best_dev_macro_f1':best,'best_epoch':best_epoch,
                'stale_epochs':stale,'history':history,'config':config,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            if improved:
                atomic_save(checkpoint,out/'best_dev.pth')
                pd.DataFrame({'id':dev.id.tolist(),'label':truth,'prediction':pred}).to_csv(out/'best_dev_predictions.csv',index=False)
            atomic_save(checkpoint,out/'last.pth')
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            logging.info('Epoch %d: train_loss=%.6f dev_loss=%.6f dev_macro_f1=%.6f best_epoch=%s',epoch,row['train_loss'],row['dev_loss'],score,best_epoch)
            if stale>=args.patience:
                logging.info('Early stopping: no dev macro F1 improvement for %d epochs',stale);break
    except KeyboardInterrupt:
        logging.info('Interrupted. Resume replays the incomplete epoch from last completed epoch checkpoint.');return
    summary={'status':'completed','epochs_completed':history[-1]['epoch'],'best_epoch':best_epoch,'best_dev_macro_f1':best,'test_evaluated':False,
        'seed':args.seed,'train_samples':len(train),'dev_samples':len(dev),'elapsed_this_invocation_seconds':time.monotonic()-started,
        'limitations':config['limitations']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    logging.info('Completed. Best development macro F1 %.6f at epoch %s. Results: %s',best,best_epoch,out)


if __name__=='__main__':main()
