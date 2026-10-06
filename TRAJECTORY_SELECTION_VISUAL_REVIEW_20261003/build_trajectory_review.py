from pathlib import Path
import argparse, hashlib, json, csv, shutil
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def metrics(c,g):
    d=np.linalg.norm(c-g[:,None],axis=-1)
    return d.mean(-1),d[:,:,-1]

def category(a,f,s,A=.3,F=.5,R=.2):
    good=(a<=A)&(f<=F); ia=int(a.argmin()); jf=int(f.argmin())
    if good[s]: return '准确选中：双oracle' if s==ia==jf else '准确选中：可接受非双oracle'
    if good.any(): return '好候选未选中'
    if a.min()<=A and f.min()<=F: return 'ADE/FDE分属不同候选'
    if a.min()<=A: return '路径接近、终点不足'
    if f.min()<=F: return '终点接近、路径不足'
    q=a+.5*f
    return '候选不足、选择接近最优' if q[s]-q.min()<=R else '候选不足且选择有额外损失'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.project); out=Path(args.output)
    source=root/'F16_RANKABILITY_PROBE_SERIES/results/cache/val_seed1.npz'
    reference=root/'F16_RANKABILITY_PROBE_SERIES/results/p0_identity/CACHE_IDENTITY_AUDIT.md'
    before={str(p):sha(p) for p in (source,reference)}
    z=np.load(source,allow_pickle=False)
    ids=z['sample_id']; c=z['candidate_final'];gt=z['future_relative'];past=z['past_relative'];coarse=z['candidate_coarse'];logits=z['native_logits'];sel=z['native_selected_index']
    assert c.shape==(1516,20,12,2) and len(np.unique(ids))==len(ids)
    assert all(np.isfinite(v).all() for v in (c,gt,past,coarse,logits))
    a,f=metrics(c,gt); ac,fc=metrics(coarse,gt)
    assert np.max(np.abs(a-z['ADE_final']))<1e-5 and np.max(np.abs(f-z['FDE_final']))<1e-5
    assert np.array_equal(sel,logits.argmax(1)) and np.array_equal(a.argmin(1),z['oracle_final_index'])
    prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    speed=np.linalg.norm(np.diff(np.concatenate((np.zeros((len(c),20,1,2)),c),2),axis=2)/.4,axis=-1)
    vel=np.diff(np.concatenate((np.zeros((len(c),20,1,2)),c),2),axis=2)/.4
    acc=np.linalg.norm(np.diff(vel,axis=2)/.4,axis=-1); jerk=np.linalg.norm(np.diff(vel,n=2,axis=2)/(.4**2),axis=-1)
    rows=[]; payload=[]; counts={}
    for i,sid in enumerate(ids):
        s=int(sel[i]); ia=int(a[i].argmin()); jf=int(f[i].argmin()); ij=int((a[i]+.5*f[i]).argmin());typ=category(a[i],f[i],s);counts[typ]=counts.get(typ,0)+1
        rank=np.argsort(-logits[i],kind='stable'); ranks=np.empty(20,int);ranks[rank]=np.arange(1,21)
        row=dict(sample_id=int(sid),category=typ,selected_slot=s,minade_slot=ia,minfde_slot=jf,joint_slot=ij,selected_ADE=float(a[i,s]),selected_FDE=float(f[i,s]),minADE=float(a[i,ia]),minFDE=float(f[i,jf]),FDE_at_minADE=float(f[i,ia]),ADE_at_minFDE=float(a[i,jf]),ADE_regret=float(a[i,s]-a[i,ia]),FDE_regret=float(f[i,s]-f[i,jf]),joint_regret=float(a[i,s]+.5*f[i,s]-(a[i]+.5*f[i]).min()),minADE_score_rank=int(ranks[ia]),minFDE_score_rank=int(ranks[jf]),same_good_count=int(((a[i]<=.3)&(f[i]<=.5)).sum()),selected_speed_max=float(speed[i,s].max()),selected_accel_max=float(acc[i,s].max()),selected_jerk_max=float(jerk[i,s].max()))
        rows.append(row)
        def rnd(x): return np.asarray(x).round(5).tolist()
        payload.append(dict(id=int(sid),p=rnd(past[i]),g=rnd(gt[i]),c=rnd(c[i]),co=rnd(coarse[i]),a=rnd(a[i]),f=rnd(f[i]),ac=rnd(ac[i]),fc=rnd(fc[i]),l=rnd(logits[i]),prob=rnd(prob[i]),s=s,speed=rnd(speed[i].max(1)),acc=rnd(acc[i].max(1)),jerk=rnd(jerk[i].max(1))))
    # Synthetic boundary/tie tests: partitions reflect SAME candidate, never separate minima.
    assert category(np.array([.1,.8]),np.array([.8,.1]),0)=='ADE/FDE分属不同候选'
    assert category(np.array([.1,.8]),np.array([.1,.8]),1)=='好候选未选中'
    assert category(np.array([.1,.8]),np.array([.1,.8]),0)=='准确选中：双oracle'
    assert sum(counts.values())==1516
    out.mkdir(parents=True,exist_ok=False)
    template=Path(__file__).with_name('trajectory_review_template.html')
    text=template.read_text(encoding='utf-8').replace('/*DATA_PLACEHOLDER*/',json.dumps(payload,separators=(',',':'),ensure_ascii=False,allow_nan=False))
    (out/'index.html').write_text(text,encoding='utf-8')
    with open(out/'classification.csv','w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    after={p:sha(Path(p)) for p in before};assert before==after
    report=dict(status='PASS',samples=1516,candidates=20,dt_seconds=.4,scope='ETH existing development cache; original F16 V2 seed1 native scoring/final refiner output. Not latest generated-head results; not held-out test.',thresholds=dict(ADE_m=.3,FDE_m=.5,joint_regret_m=.2,joint_fde_weight=.5),threshold_role='Descriptive adjustable thresholds only; no change to formal major gates.',counts=counts,selected_ADE_mean=float(a[np.arange(1516),sel].mean()),selected_FDE_mean=float(f[np.arange(1516),sel].mean()),minADE_mean=float(a.min(1).mean()),minFDE_mean=float(f.min(1).mean()),source_sha_before=before,source_sha_after=after,checks=['explicit unique sample IDs','finite N1516 K20 T12 xy2','cached ADE/FDE independently recomputed within1e-5','selected index equals actual native logits argmax','minADE index matches cached oracle','same-candidate classification synthetic tests','all samples partitioned','input cache/reference SHA unchanged'])
    (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'README.md').write_text('''# 候选生成—模型筛选人工复核\n\n双击 index.html，无服务器、网络或GPU依赖。全部1516个既有ETH开发案例，不是独立测试。来源是已审计的原生F16 V2 seed1：20条final候选与原模型真实logits/argmax。不是最新生成修正实验，也没有用旧分数假称重新评分。\n\n默认分类：同一条候选ADE<=0.30m且FDE<=0.50m才算接近；joint=A​​DE+0.5FDE仅用于本诊断选择损失排序，不冒充生产训练目标。joint regret<=0.20m为选择接近最优。阈值可改，分类立即重算，可导出当前分类CSV。是描述性阈值，无外部公认标准、无DEV调参/修改正式门槛。\n\n八个互斥分类，依次判断：准确选中双oracle、准确选中可接受非双oracle、好候选未选中、ADE/FDE分属不同候选、仅路径接近、仅终点接近、候选不足选择接近最优、候选不足且选择额外损失。oracle是事后使用GT的评估参照，不是推理输出；exact oracle命中率不等于预测质量。\n\n界面显示相对最后观测位置的物理坐标(米)，等比例坐标，8步历史/12步未来dt0.4s。20条候选全部绘制；可比较粗轨迹与refiner后最终轨迹，模型选中、GT、ADE/FDE/joint oracle独立高亮；表格点击可看任意候选，时间滑块查看逐时刻。softmax只是原生评分的归一化，不是经校准的真实概率。速度/加速度/jerk为有限差分描述，不凭单案例推断机制或安全性。\n\n分类CSV保留所有样本，不只挑好看的图。排序最大regret用于定位失败，默认样本ID顺序用于避免只看极端案例。可导出人工标注JSON，与固定样本ID对应：支持人类形成假设，不作为独立泛化证明、不训练任何模型。没有图像场景/地图/邻居背景，不据图判定碰撞或场景可行性。\n\n输入只读，audit.json保存输入SHA前后和全部数值核验。没有旧结果/模型/科学脚本改写。人工评审后若选新实验方向应预注册并继续原有对照/审计标准。\n''',encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(template,out/template.name)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
