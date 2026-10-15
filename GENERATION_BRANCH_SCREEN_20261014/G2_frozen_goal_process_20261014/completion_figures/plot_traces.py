"""Saved-data fixed-hash trace illustration; not additional model evaluation or sample selection by outcome."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c, numpy as np,hashlib,csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.8,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42})
OUT=Path(__file__).resolve().parent
def main():
 lock=c.read(c.H/"source_manifest.json");result=c.read(c.H/"RESULT.json");c.verify(result["output_sha"]);source=dict(result["output_sha"]);source[str(c.H/"source_manifest.json")]=c.sha(c.H/"source_manifest.json")
 data=c.load();ix=data["split"]["outer"];chosen=[]
 for group in range(3):
  eligible=np.flatnonzero(data["groups"][ix]==group);chosen.append(min(eligible,key=lambda j:hashlib.sha256(("20261014-g2trace-v1:"+str(int(data["ids"][ix[j]]))).encode()).hexdigest()))
 csvrows=[];selection=[];outputs={}
 for seed in (42,43,44):
  z=c.npz(c.H/f"runs/seed_{seed}/outer.npz");fig,axes=plt.subplots(3,2,figsize=(7.204724409448819,6.102362204724409));fig.subplots_adjust(left=.09,right=.98,bottom=.08,top=.92,wspace=.34,hspace=.58)
  for k,j in enumerate(chosen):
   group=("stable","transition","change")[k];gt=z["gt"][j].astype(float);B=z["paths"][j];A=z["base_paths"][j];E=np.linalg.norm(B-gt[None],axis=-1);J=np.maximum(E.mean(-1)/.2,E[:,-1]/.4);slot=int(J.argmin());assert np.array_equal(A[:,-1],B[:,-1]);i=ix[j];past=data["past"][i];time=np.arange(1,13)*.4
   ax,speedax=axes[k];ax.plot(past[:,0],past[:,1],color="#9A9A9A",marker=".",lw=.9,label="observed")
   for port,path,col,style in (("A",A[slot],"#638BA6","--"),("B",B[slot],"#CB806D","-"),("GT",gt,"#222222","-")):
    full=np.concatenate((np.zeros((1,2)),path));ax.plot(full[:,0],full[:,1],color=col,ls=style,lw=1.3,label=port);vel=np.linalg.norm(np.diff(full,axis=0)/.4,axis=-1);speedax.plot(time,vel,color=col,ls=style,lw=1.3,label=port)
    for t in range(12):csvrows.append(dict(seed=seed,group=group,sample_id=int(z["sample_id"][j]),slot=slot,port=port,time_s=float(time[t]),x_m=float(path[t,0]),y_m=float(path[t,1]),speed_mps=float(vel[t])))
   obs_speed=float(np.linalg.norm(data["vel"][i]));speedax.scatter([0],[obs_speed],color="#777777",s=8,zorder=3);ax.set_aspect("equal",adjustable="datalim");ax.set_xlabel("x (m)");ax.set_ylabel("y (m)");speedax.set_xlabel("Future time (s)");speedax.set_ylabel("Speed (m/s)");speedax.set_ylim(bottom=0)
   ax.set_title(f"{group}; ID {int(z['sample_id'][j])}; slot {slot}",fontsize=7)
   for a,label in ((ax,chr(97+k*2)),(speedax,chr(98+k*2))):a.text(-.15,1.09,label,transform=a.transAxes,fontweight="bold",fontsize=8)
   if k==0:ax.legend(fontsize=6,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.43),handlelength=1.4,columnspacing=.7)
   selection.append(dict(seed=seed,group=group,sample_id=int(z["sample_id"][j]),candidate=slot,candidate_selection="posthoc_B_samepath_Jclosest; A uses identical slot and endpoint",window_selection="fixed_SHA256_tag_and_ID_noGT_outcome",B_J=float(J[slot]),observed_speed=obs_speed))
  fig.suptitle(f"Same endpoints, different temporal process | seed {seed}",fontsize=8)
  for ext in ("svg","pdf","png"):assert not (OUT/f"paired_traces_seed{seed}.{ext}").exists()
  fig.savefig(OUT/f"paired_traces_seed{seed}.svg")
  fig.savefig(OUT/f"paired_traces_seed{seed}.pdf")
  fig.savefig(OUT/f"paired_traces_seed{seed}.png",dpi=600)
  for ext in ("svg","pdf","png"):
   path=OUT/f"paired_traces_seed{seed}.{ext}";outputs[str(path)]=c.sha(path)
  plt.close(fig);c.guard()
 path=OUT/"SOURCE_DATA.csv"
 with path.open("x",newline="",encoding="utf8") as f:
  writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
 outputs[str(path)]=c.sha(path);c.verify(source);c.write(OUT/"TRACE_MANIFEST.json",dict(status="PASS",fixed_three_outer_windows_across_all3seeds=True,source=source,outputs=outputs,selection=selection,source_rows=len(csvrows),not_representative_population_or_significance_test=True,GT_only_posthoc_candidate_display_no_training_or_policy_use=True,no_network_forward_optimizer_or_quality_audit_repeat=True,backend="python"))
 print("TRACE_PASS",len(csvrows),flush=True)
if __name__=="__main__":main()
