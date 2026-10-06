"use strict";
// Independent file-based UI-data checks; no inference or training.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const root=__dirname,sandbox={window:{PULSE_CASES:{}}};vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root,'data/common.js'),'utf8'),sandbox);
const C=sandbox.window.PULSE_REVIEW;assert.equal(C.cases.length,1516);assert.equal(new Set(C.cases.map(r=>r.id)).size,1516);
function good(r,k,al=.2,fl=.4){return r.ade[k].some((a,j)=>a<=al&&r.fde[k][j]<=fl);}
function flag(old,now){return !old&&now?'rescued':old&&!now?'lost':old&&now?'retained':'missing';}
assert.equal(good({ade:{x:[.1,.9]},fde:{x:[.8,.1]}},'x'),false,'Separate minima are NOT a good same path');
assert.equal(good({ade:{x:[.2]},fde:{x:[.4]}},'x'),true,'Inclusive boundary');
for(const entry of C.counts){const pool=C.cases.filter(r=>entry.group==='overall'||r.clear),seed=entry.seed,newkey='learned'+seed;assert.equal(pool.length,entry.n);
 for(const name of ['rescued','lost','retained','missing'])assert.equal(pool.filter(r=>flag(good(r,entry.reference,entry.ade_limit,entry.fde_limit),good(r,newkey,entry.ade_limit,entry.fde_limit))===name).length,entry[name]);}
let coords=0,maxError=0;
for(const r of C.cases){vm.runInContext(fs.readFileSync(path.join(root,'data/cases',r.id+'.js'),'utf8'),sandbox);const d=sandbox.window.PULSE_CASES[r.id];assert.equal(d.id,r.id);assert.equal(d.gt.length,12);assert.equal(d.past.length,8);const actual=d.scores.reduce((best,v,k)=>v>d.scores[best]?k:best,0);assert.equal(actual,r.selected);
 for(const key of C.keys){assert.equal(d.paths[key].length,20);assert.equal(r.ade[key].length,20);for(let k=0;k<20;k++){const pp=d.paths[key][k];assert.equal(pp.length,12);const e=pp.map((p,t)=>{assert.equal(p.length,2);assert.ok(p.every(Number.isFinite));coords+=2;return Math.hypot(p[0]-d.gt[t][0],p[1]-d.gt[t][1]);});const a=e.reduce((a,b)=>a+b,0)/12;maxError=Math.max(maxError,Math.abs(a-r.ade[key][k]),Math.abs(e[11]-r.fde[key][k]));}
 const closest=r.ade[key].map((a,k)=>Math.max(a/.2,r.fde[key][k]/.4)).reduce((best,x,k,arr)=>x<arr[best]?k:best,0);assert.equal(good(r,key),r.ade[key][closest]<=.2&&r.fde[key][closest]<=.4);}
 delete sandbox.window.PULSE_CASES[r.id];
}
assert.ok(maxError<4e-6);const result={status:'PASS',IDs:1516,methods:7,classification_count_rows:C.counts.length,finite_coordinates:coords,rounded_metric_max_error:maxError,native_argmax_exact:true};fs.writeFileSync(path.join(root,'DATA_QA.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
