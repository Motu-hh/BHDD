"""Real LMVD data smoke check, 3 updates only; no performance estimate."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'model' / 'BHDD'))
from train import build_adjacency_matrix, reconstruction_loss
from dataloader import BHDDDataset
from bhdd_lmvd_corrected import Net, SequenceFirstPositionalEncoding


def main():
    torch.manual_seed(0)
    data = ROOT/'dataset'/'LMVD_real_v1'
    out = ROOT/'outputs'/'lmvd_real_smoke_seed0'
    if out.exists():
        raise RuntimeError('Output already exists')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    records = [json.loads(p.read_text(encoding='utf-8')) for p in (data/'records').glob('*.json')]
    ids = {label:sorted([r['id'] for r in records if r['status']=='ready' and r['label']==label],key=int)[:2] for label in (0,1)}
    assert all(len(v)==2 for v in ids.values())
    files = [f'{stem}.npy' for label in (0,1) for stem in ids[label]]
    dataset = BHDDDataset(data/'landmarks',data/'au_pose_gaze',data/'audio',data/'labels',files)
    inputs = [torch.stack([dataset[i][j] for i in range(4)]) for j in range(4)]
    lm, au, audio, labels = [x.cuda() for x in inputs]
    assert list(lm.shape)==[4,915,136] and list(au.shape)==[4,915,29] and list(audio.shape)==[4,186,128]
    for tensor in (lm,au,audio): assert torch.isfinite(tensor).all().item()
    device=torch.device('cuda')
    net = Net(build_adjacency_matrix().to(device)).to(device)
    net.eval()
    pe=SequenceFirstPositionalEncoding(512).to(device)
    with torch.no_grad():
        encoded=pe(torch.zeros(915,2,512,device=device))
        time_delta=(encoded[1,0]-encoded[0,0]).abs().max().item()
        slot_delta=(encoded[0,1]-encoded[0,0]).abs().max().item()
        assert time_delta>0 and slot_delta==0
        outputs=net(lm[:1].repeat(2,1,1),au[:1].repeat(2,1,1),audio[:1].repeat(2,1,1))[0]
        identical_delta=(outputs[0]-outputs[1]).abs().max().item()
        torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-5,atol=1e-5)
    print('PASS: corrected positional encoding and identical-input consistency',flush=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-5,weight_decay=1e-4)
    steps=[]
    for step in range(3):
        net.train()
        index=torch.tensor([step%2,2+step%2],device=device)
        optimizer.zero_grad(set_to_none=True)
        logits,_,_,visual,rebuilt=net(lm[index],au[index],audio[index])
        ce=torch.nn.functional.cross_entropy(logits,labels[index])
        rec=reconstruction_loss(visual,rebuilt)
        loss=ce+0.7*rec
        assert torch.isfinite(loss).item()
        loss.backward()
        for name,parameter in net.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f'Non-finite gradients: {name}')
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        record={'step':step+1,'cross_entropy':ce.item(),'reconstruction_loss':rec.item(),'combined_loss':loss.item(),'gradient_norm_before_clipping':norm.item()}
        steps.append(record)
        print(json.dumps(record),flush=True)
    net.eval()
    with torch.no_grad():
        result=net(lm[:2],au[:2],audio[:2])[0]
        assert result.shape==(2,2) and torch.isfinite(result).all().item()
    out.mkdir(parents=True)
    pd.DataFrame(steps).to_csv(out/'steps.csv',index=False)
    evidence={'status':'PASS','real_data':True,'purpose':'Data and numerical smoke check only; no validation/test metrics',
        'seed':0,'sample_ids':files,'sample_count':4,'optimizer_updates':3,
        'model_variant':'Local sequence-first positional encoding correction; original source retained',
        'positional_time_delta':time_delta,'positional_batch_slot_delta':slot_delta,
        'identical_input_logit_delta':identical_delta,'steps':steps,
        'gpu':torch.cuda.get_device_name(0),'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,
        'original_model_sha256':hashlib.sha256((ROOT/'model'/'BHDD'/'model.py').read_bytes()).hexdigest(),
        'remaining':'No held-out performance claim; participant IDs and split protocol unresolved'}
    (out/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('PASS: 3 real-data optimizer updates and inference',flush=True)


if __name__=='__main__':
    main()
