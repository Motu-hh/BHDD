"""Three-epoch real-data smoke run on fixed training/development subsets.
Not a formal experiment; reserved test is never used for inference or metrics.
"""
import sys,json,csv,random,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'/'BHDD'))
from dataloader import BHDDDataset
from train import build_adjacency_matrix,reconstruction_loss
from bhdd_lmvd_corrected import Net
DATA=ROOT/'dataset'/'LMVD_clean_v2'
OUT=ROOT/'outputs'/'lmvd_clean_short_seed0'


def subset(frame,split,per_class):
    f=frame[frame.split==split]
    return pd.concat([f[f.label==c].sample(n=per_class,random_state=0) for c in (0,1)]).sort_values('id')

def main():
    if OUT.exists():raise RuntimeError('Output exists; refusing overwrite')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    manifest=pd.read_csv(DATA/'manifest.csv',dtype={'id':str})
    train=subset(manifest,'train',32);dev=subset(manifest,'dev',16)
    assert set(train.id).isdisjoint(set(dev.id))
    test_ids=set(manifest.loc[manifest.split=='test','id'])
    assert not ((set(train.id)|set(dev.id))&test_ids)
    OUT.mkdir(parents=True)
    config={'purpose':'3-epoch real-data engineering smoke test, not paper reproduction or confirmatory comparison',
        'seed':0,'train_samples':64,'dev_samples':32,'batch_size':4,'epochs':3,'learning_rate':1e-5,
        'model':'local sequence-first positional encoding correction',
        'loss':'Original cross entropy + 0.7*clamp(mean reconstruction MSE,1e-10,1e10)/1e10',
        'schedule':'none for this short numerical check','normalizer':'fixed full-training-split statistics; never fitted on dev/test',
        'invalid_frames':'zero after normalization; current architecture has no attention padding masks',
        'selection':'highest development macro F1, first epoch on ties; reserved test never evaluated',
        'limits':'video-level split only; prefix crop modality alignment unverified; subset metrics not full-dataset results',
        'data_manifest_sha256':hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest(),
        'normalization_sha256':hashlib.sha256((DATA/'train_normalization.npz').read_bytes()).hexdigest()}
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    train.to_csv(OUT/'train_subset.csv',index=False);dev.to_csv(OUT/'dev_subset.csv',index=False)
    def make_loader(rows,shuffle):
        ds=BHDDDataset(DATA/'landmarks',DATA/'au_pose_gaze',DATA/'audio',DATA/'labels',[i+'.npy' for i in rows.id])
        return DataLoader(ds,batch_size=4,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(0))
    train_loader=make_loader(train,True);dev_loader=make_loader(dev,False)
    device=torch.device('cuda');adj=build_adjacency_matrix().to(device)
    net=Net(adj).to(device);optim=torch.optim.Adam(net.parameters(),lr=1e-5,weight_decay=1e-4)
    best=-1.;history=[];best_epoch=None;started=time.monotonic()
    for epoch in range(1,4):
        net.train();total_loss=0.;total_ce=0.;total_rec=0.;max_grad=0.;saturated=0;count=0
        for lm,au,audio,label in train_loader:
            lm,au,audio,label=[v.to(device) for v in (lm,au,audio,label)]
            optim.zero_grad(set_to_none=True)
            logits,_,_,visual,rebuilt=net(lm,au,audio)
            ce=torch.nn.functional.cross_entropy(logits,label);rec=reconstruction_loss(visual,rebuilt);loss=ce+0.7*rec
            if not torch.isfinite(loss).item():raise RuntimeError('Non-finite loss')
            loss.backward()
            for name,p in net.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all().item():raise RuntimeError('Non-finite gradient '+name)
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
            optim.step()
            n=len(label);count+=n;total_loss+=loss.item()*n;total_ce+=ce.item()*n;total_rec+=rec.item()*n
            max_grad=max(max_grad,norm.item());saturated+=rec.item()>=1.0
        net.eval();truth=[];pred=[];dev_loss=0.
        with torch.no_grad():
            for lm,au,audio,label in dev_loader:
                lm,au,audio,label=[v.to(device) for v in (lm,au,audio,label)]
                logits,_,_,visual,rebuilt=net(lm,au,audio)
                loss=torch.nn.functional.cross_entropy(logits,label)+0.7*reconstruction_loss(visual,rebuilt)
                if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():raise RuntimeError('Non-finite development output')
                dev_loss+=loss.item()*len(label);truth.extend(label.cpu().tolist());pred.extend(logits.argmax(1).cpu().tolist())
        score=float(f1_score(truth,pred,average='macro',zero_division=0))
        row={'epoch':epoch,'train_loss':total_loss/count,'train_ce':total_ce/count,'train_rec':total_rec/count,
            'dev_loss':dev_loss/len(truth),'dev_macro_f1':score,'dev_balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
            'max_gradient_norm_before_clipping':max_grad,'reconstruction_saturated_steps':saturated}
        history.append(row);print(json.dumps(row),flush=True)
        if score>best:
            best=score;best_epoch=epoch
            torch.save({'net':net.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch,'config':config,'dev_macro_f1':score},OUT/'best_dev.pth')
            pd.DataFrame({'id':dev.id.tolist(),'label':truth,'prediction':pred}).to_csv(OUT/'best_dev_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(OUT/'history.csv',index=False)
    ckpt=torch.load(OUT/'best_dev.pth',map_location=device,weights_only=True)
    fresh=Net(adj).to(device);fresh.load_state_dict(ckpt['net'],strict=True);fresh.eval()
    replay=[]
    with torch.no_grad():
        for lm,au,audio,_ in dev_loader:replay.extend(fresh(lm.to(device),au.to(device),audio.to(device))[0].argmax(1).cpu().tolist())
    saved=pd.read_csv(OUT/'best_dev_predictions.csv')
    assert replay==saved.prediction.tolist()
    with torch.no_grad():
        lm,au,audio,_=next(iter(dev_loader))
        pair=fresh(lm[:1].repeat(2,1,1).to(device),au[:1].repeat(2,1,1).to(device),audio[:1].repeat(2,1,1).to(device))[0]
        torch.testing.assert_close(pair[0],pair[1],atol=1e-5,rtol=1e-5)
    result={'status':'PASS','epochs':3,'training_updates':48,'train_samples':64,'dev_samples':32,'reserved_test_evaluated':False,
        'best_epoch':best_epoch,'best_dev_macro_f1':best,'checkpoint_replay_matches':True,
        'any_reconstruction_saturation':any(r['reconstruction_saturated_steps'] for r in history),
        'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'elapsed_seconds':time.monotonic()-started,
        'interpretation':'Numerical and pipeline check only; subset development score is not generalization or paper reproduction evidence'}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
