"""Write a factual handoff for LMVD preparation from saved run records."""
from pathlib import Path
import json
import csv
import hashlib

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'dataset'/'LMVD_real_v1'
OUT=ROOT/'outputs'/'lmvd_real_smoke_seed0'


def main():
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    windows=json.loads((DATA/'window_summary.json').read_text(encoding='utf-8'))
    smoke=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    schema=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))
    assert summary['sample_count']==windows['samples']==schema['sample_count']==1823
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (DATA/'records').glob('*.json')]
    ready=[r for r in records if r['status']=='ready']
    assert len(ready)==summary['status_counts']['ready']
    ready_files=[]
    for record in ready:
        for relative,digest in record['output_sha256'].items():
            path=DATA/relative
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, str(path)
        ready_files.append({'id':record['id'],'label':record['label'],'landmarks':f"landmarks/{record['id']}.npy",'au_pose_gaze':f"au_pose_gaze/{record['id']}.npy",'audio':f"audio/{record['id']}.npy",'label_file':f"labels/{record['id']}_Depression.csv",'tracking_metadata':f"window_metadata/{record['id']}.npz",'split':'UNASSIGNED'})
    ready_files.sort(key=lambda r:int(r['id']))
    with (DATA/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=ready_files[0].keys());writer.writeheader();writer.writerows(ready_files)
    verdict={
        'preparation_status':'COMPLETE_WITH_QUALITY_FINDINGS',
        'total_samples':1823,'converted_finite_samples':len(ready),'review_ids':summary['review_ids'],
        'converted_outputs_sha256_rechecked':True,
        'window_quality':windows,'real_data_smoke_status':smoke['status'],
        'formal_training_started':False,
        'formal_training_readiness':'NOT_READY: tracking failures and extreme finite pose values need cleaning; modality time alignment and participant grouping require protocol decisions',
        'original_model_unchanged':smoke['original_model_sha256']==hashlib.sha256((ROOT/'model'/'BHDD'/'model.py').read_bytes()).hexdigest(),
    }
    (OUT/'preparation_handoff.json').write_text(json.dumps(verdict,indent=2),encoding='utf-8')
    notes=f"""# LMVD 真实数据准备记录

## 当前结论

已读取你提供的 LMVD_Feature.zip，并完成全部 1823 组数据的流式读取和质量检查。原压缩包未修改，也未将约58GiB原始CSV完整展开。
{len(ready)} 组成功转换为有限值固定输入；{len(summary['review_ids'])} 组需要复核，编号：{', '.join(summary['review_ids'])}。未自动填补异常值。
ready 仅表示输入格式和有限值检查通过，不表示已清理完成或可直接正式训练。

## 文件与标签

- landmarks：915×136，按 x0,y0,x1,y1,... 排列以匹配模型的68×2展开。
- au_pose_gaze：915×29，17个AU强度、6个姿态、6个视线向量值。
- audio：186×128，原包音频特征。
- 标签依据包内 Readme.md：001–601、1117–1423 为1；602–1116、1425–1824 为0。
- 全量类别计数为908个1和915个0；1424不在原包和README标签区间中。
- schema.json 保留准确列名和转换策略，records/*.json 保留逐样本记录，manifest.csv 列出成功输出和标签来源。

## 预处理边界

当前 v1 是工程检查输入：沿用原加载器的前915个视频帧、前186个音频帧截断及短序列补零；未归一化、未删除跟踪失败帧、未推断音视频时间对齐。
原压缩包保留完整序列，后续可依据确定的时间规则重新处理。
window_metadata/*.npz 保存前915帧的 tracking_success_mask、timestamp、confidence；补齐位置时间戳为NaN是显式缺帧标记，不是输入特征的NaN。

## 数据质量发现

- 首个保留窗口含跟踪失败帧的样本：{windows['samples_with_failed_tracking_in_retained_window']} / 1823。
- 首窗口没有任何成功跟踪帧的样本：{windows['samples_with_zero_success_in_retained_window']}。
- 首窗口真实行共{windows['retained_rows']}，其中{windows['failed_tracking_rows']}行被源跟踪器标为失败。
- 例如001号前915帧有528帧跟踪失败，头部位置最大绝对值28613630，最大值发生在失败帧；0602号最大值34329676，同样发生在失败帧。
- 这些值即便有限也不能当成正常行为特征；不能直接把当前转换版用于论文级训练。
- 例如001号前915帧只覆盖约30.467秒；音频序列没有在本包中提供逐帧时间戳，不能假定前186个音频帧与其覆盖时间相同。
- 本包只提供样本编号，未提供被试/同一人多个视频的对应表，尚未建立被试独立划分。

## 已完成的真实数据计算检查

使用0602、0603、001、002号样本，seed=0，batch size=2，完成3次优化器更新和后续eval推理。没有计算或报告验证/测试准确率，没有保存这次试跑的训练权重。
模型使用 bhdd_lmvd_corrected.py，仅修正序列优先布局的位置编码，官方 model/BHDD/model.py 保持不变。
修正后相同输入位于不同批次位置的logit最大差={smoke['identical_input_logit_delta']}。
峰值显存约{smoke['peak_gpu_mib']:.1f} MiB，仅代表这次小规模检查。

重要：三次更新的第1和第3次重建损失达到原函数clamp上限1.0；梯度裁剪前范数分别约3818.56和1434.11。数值有限不代表优化稳定，异常姿态值和特征量纲问题需先处理。
三项未使用参数仍保持原代码行为，详见先前实现核查记录。

## 下一阶段

1. 确定无效跟踪帧的掩码/插值策略，保留时间轴并处理零有效帧样本。
2. 确认音频特征的时间采样规则，制定统一时间窗口；不能仅凭数组长度推断对齐。
3. 获取被试分组信息，或明确只能采用较弱的视频级探索性划分；归一化统计仅由训练集计算。
4. 固定原版与位置编码修正版的角色、预处理和评估协议，再设计正式训练。

当前阶段是数据准备与计算检查，不是论文结果复现。详细数值见quality.csv、window_quality.csv、summary.json和本目录verification.json。
"""
    (OUT/'LMVD真实数据准备说明.md').write_text(notes,encoding='utf-8')
    (DATA/'README.md').write_text('# LMVD real-data engineering preparation v1\n\nNOT READY FOR FORMAL TRAINING. Fixed prefix crops preserve the legacy loader behavior but modality time alignment is unresolved. Raw tracker-failure values are retained and masks are provided. Use manifest.csv for convertible inputs, review quality records, and consult outputs/lmvd_real_smoke_seed0/LMVD真实数据准备说明.md. Original archive remains unchanged.\n',encoding='utf-8')
    print(json.dumps(verdict,indent=2),flush=True)


if __name__=='__main__':main()
