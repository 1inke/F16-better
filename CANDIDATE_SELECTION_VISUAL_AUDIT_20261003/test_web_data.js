"use strict";
// Pure file-based checks: no browser control, no training, no source mutation.
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict"),path=require("node:path");
const root=__dirname,sandbox={window:{AUDIT_MODELS:{}}};vm.createContext(sandbox);
for(const name of ["common","native","pairwise","readout","messages"])
  vm.runInContext(fs.readFileSync(path.join(root,"data",name+".js"),"utf8"),sandbox);
const C=sandbox.window.AUDIT_COMMON,M=sandbox.window.AUDIT_MODELS;
function csv(name){const rows=fs.readFileSync(path.join(root,name),"utf8").trim().split(/\r?\n/).map(x=>x.split(","));const head=rows.shift();return rows.map(row=>Object.fromEntries(head.map((k,i)=>[k,row[i]])));}
function classify(a,f,s,al,fl){const good=a.map((v,k)=>v<=al&&f[k]<=fl);const ag=Math.min(...a)<=al,fg=Math.min(...f)<=fl;return good[s]?0:good.some(Boolean)?1:ag&&fg?2:ag?3:fg?4:5;}
const rows=csv("per_sample_classification.csv"),sensitivity=csv("threshold_sensitivity.csv");
assert.equal(C.ids.length,1516);assert.equal(new Set(C.ids).size,1516);assert.equal(rows.length,6064);
let coords=0,maxRoundedADEError=0,maxRoundedFDEError=0;
for(const row of rows){
  const key=row.model,i=Number(row.row_index),d=M[key],s=d.d[i].selected;
  assert.equal(C.ids[i],Number(row.sample_id));assert.equal(s,Number(row.selected));
  assert.equal(classify(d.a[i],d.f[i],s,.3,.5),Number(row.category));
  assert.equal(d.p[i].length,20);assert.equal(d.a[i].length,20);
  const nativeArgmax=C.scores[i].reduce((best,v,k)=>v>C.scores[i][best]?k:best,0);assert.equal(s,nativeArgmax);
  for(let k=0;k<20;k++){
    assert.equal(d.p[i][k].length,12);const error=d.p[i][k].map((p,t)=>{assert.equal(p.length,2);assert.ok(p.every(Number.isFinite));coords+=2;return Math.hypot(p[0]-C.gt[i][t][0],p[1]-C.gt[i][t][1]);});
    const ade=error.reduce((a,b)=>a+b,0)/12,fde=error[11];
    maxRoundedADEError=Math.max(maxRoundedADEError,Math.abs(ade-d.a[i][k]));maxRoundedFDEError=Math.max(maxRoundedFDEError,Math.abs(fde-d.f[i][k]));
  }
}
for(const row of sensitivity){
  const d=M[row.model],group=row.group,cat=C.labels.indexOf(row.category),al=Number(row.ade_limit),fl=Number(row.fde_limit);
  let n=0,denominator=0;
  for(let i=0;i<C.ids.length;i++){if(group!=="overall"&&["stable","transition","change"][C.groups[i]]!==group)continue;denominator++;if(classify(d.a[i],d.f[i],d.d[i].selected,al,fl)===cat)n++;}
  assert.equal(n,Number(row.n));assert.equal(denominator,Number(row.denominator));
}
// Both GT and candidate web coordinates are rounded per component by <=5e-6m.
assert.ok(maxRoundedADEError<1.5e-5);assert.ok(maxRoundedFDEError<1.5e-5);
console.log(JSON.stringify({status:"PASS",sample_rows:rows.length,threshold_group_rows:sensitivity.length,finite_coordinates_checked:coords,maxRoundedADEError,maxRoundedFDEError}));
