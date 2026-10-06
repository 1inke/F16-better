"use strict";
(() => {
const C=window.PULSE_REVIEW,$=id=>document.getElementById(id),ns="http://www.w3.org/2000/svg";
if(!C){$("heading").textContent="数据未加载，请完整保留data目录。";return;}
const labels={all:"全部",rescued:"救回 · 原没有 / 新有",lost:"退化 · 原有 / 新没有",retained:"仍有合格候选",missing:"仍没有合格候选"};
const colors={gt:"#007f5f",past:"#35404a",candidate:"#b4c2cb",closest:"#00858c",selected:"#c45028",ade:"#2874ac",fde:"#8956a3",active:"#182f44"};
const q=new URLSearchParams(location.search),state={seed:Number(q.get("seed"))||C.default_seed,row:0,filter:q.get("filter")||C.default_filter,highlight:null,list:[],ticket:0};
if(![1,2,3].includes(state.seed))state.seed=1;
$("seed").value=state.seed;
const fmt=(v,n=3)=>Number(v).toFixed(n),limit=()=>$("threshold").value.split(",").map(Number),keys=()=>["native","fixed"+state.seed,"learned"+state.seed];
const modelName=k=>k==="native"?"原F16 / V2":k.startsWith("fixed")?"固定脉冲响应对照":"可学习脉冲响应";
const storage="pulse_response_human_review_20261004_"+C.source_sha.slice(0,12);
let notes={};try{notes=JSON.parse(localStorage.getItem(storage)||"{}");}catch(_){}
const drafts={};let noteKey="",data=null;
function node(tag,attrs={},text){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function add(parent,tag,attrs,text){const e=node(tag,attrs,text);parent.appendChild(e);return e;}
function html(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function stats(r,k){
 const [al,fl]=limit(),a=r.ade[k],f=r.fde[k],metric=a.map((x,j)=>Math.max(x/al,f[j]/fl)),slot=metric.reduce((best,x,j)=>x<metric[best]?j:best,0);
 const ia=a.reduce((best,x,j)=>x<a[best]?j:best,0),iff=f.reduce((best,x,j)=>x<f[best]?j:best,0);
 const good=a.map((x,j)=>x<=al&&f[j]<=fl),any=good.some(Boolean),ag=a[ia]<=al,fg=f[iff]<=fl;
 return {slot,ia,iff,good,any,count:good.filter(Boolean).length,margin:metric[slot],category:good[r.selected]?"所选 / 继承slot合格":any?"集合有合格候选但未选":ag&&fg?"两个min分别合格但不同路径":ag?"仅ADE接近":fg?"仅FDE接近":"双指标均不足"};
}
function flag(r,seed=state.seed){const old=stats(r,$("reference").value==="native"?"native":"fixed"+seed).any,now=stats(r,"learned"+seed).any;return !old&&now?"rescued":old&&!now?"lost":old&&now?"retained":"missing";}
function inGroup(r){const g=$("group").value;return g==="all"||g==="clear"&&r.clear||["stable","transition","change"][r.group]===g;}
function stable(r){const flags=[1,2,3].map(s=>flag(r,s));switch($("consistency").value){case"rescued3":return flags.every(x=>x==="rescued");case"lost3":return flags.every(x=>x==="lost");case"variable":return new Set(flags).size>1;default:return true;}}
function magnitude(r){return Math.abs(stats(r,"learned"+state.seed).margin-stats(r,$("reference").value==="native"?"native":"fixed"+state.seed).margin);}
function refresh(reset=true){
 const group=C.cases.filter(inGroup),eligible=group.filter(stable),counts={all:eligible.length};
 for(const f of ["rescued","lost","retained","missing"])counts[f]=eligible.filter(r=>flag(r)===f).length;
 $("counts").replaceChildren();for(const [f,label]of Object.entries(labels)){const b=html("button",undefined,"count"+(state.filter===f?" active":""));b.setAttribute("aria-pressed",String(state.filter===f));b.append(html("span",label),html("strong",counts[f]),html("small",`当前范围 · seed${state.seed}`));b.onclick=()=>{state.filter=f;refresh();};$("counts").appendChild(b);}
 state.list=eligible.filter(r=>state.filter==="all"||flag(r)===state.filter).map(r=>r.row);
 state.list.sort((a,b)=>$("sort").value==="magnitude"?magnitude(C.cases[b])-magnitude(C.cases[a])||C.cases[a].id-C.cases[b].id:C.cases[a].id-C.cases[b].id);
 if(reset||!state.list.includes(state.row))state.row=state.list[0];
 $("sample").replaceChildren();for(const i of state.list){const r=C.cases[i],option=html("option",`ID ${r.id} · ${labels[flag(r)]}`);option.value=i;$("sample").appendChild(option);}$("sample").value=state.row;
 state.highlight=null;loadCase();
}
function loadCase(){
 const ticket=++state.ticket,i=state.row;data=null;
 if(i===undefined){$("heading").textContent="当前条件没有样本，可切类别 / 种子 / 范围。";$("position").textContent="0例";$("panels").replaceChildren();$("candidate-table").replaceChildren();$("seed-flags").textContent="";for(const id of ["error","speed","accel"])$(id).replaceChildren();for(const id of ["previous","next","save-note","export-svg"])$(id).disabled=true;return;}
 const id=C.cases[i].id;$("heading").textContent=`正在加载 ID ${id}…`;
 const draw=()=>{if(ticket!==state.ticket)return;data=window.PULSE_CASES[id];render();};
 if(window.PULSE_CASES[id]){draw();return;}
 const script=document.createElement("script");script.src=`data/cases/${id}.js`;script.onload=draw;script.onerror=()=>{if(ticket===state.ticket)$("heading").textContent="案例文件加载失败，请保持data/cases目录完整。";};document.body.appendChild(script);
}
function tip(e,text){const p=$("tooltip");p.textContent=text;p.hidden=false;p.style.left=Math.max(6,Math.min(innerWidth-300,e.clientX+12))+"px";p.style.top=Math.max(6,Math.min(innerHeight-90,e.clientY+12))+"px";}
function bind(e,k,key){e.style.cursor="pointer";e.onclick=()=>{state.highlight=k;render();};e.onmouseenter=event=>tip(event,`${modelName(key)} · Slot${k+1} · ADE ${fmt(C.cases[state.row].ade[key][k],4)} / FDE ${fmt(C.cases[state.row].fde[key][k],4)}m`);e.onmouseleave=()=>$("tooltip").hidden=true;}
function dimensions(svg){const w=Math.max(240,svg.getBoundingClientRect().width),h=parseFloat(getComputedStyle(svg).height)||240;svg.setAttribute("viewBox",`0 0 ${w} ${h}`);return[w,h];}
function axes(svg,xd,yd,xlabel,ylabel){svg.replaceChildren();const[w,h]=dimensions(svg),L=52,R=14,T=22,B=40,bw=w-L-R,bh=h-T-B;const x=v=>L+(v-xd[0])/(xd[1]-xd[0])*bw,y=v=>h-B-(v-yd[0])/(yd[1]-yd[0])*bh;const g=add(svg,"g",{});
 for(let j=0;j<5;j++){const xx=xd[0]+j*(xd[1]-xd[0])/4,yy=yd[0]+j*(yd[1]-yd[0])/4;add(g,"line",{x1:x(xx),x2:x(xx),y1:T,y2:h-B,stroke:"#e9eff3"});add(g,"line",{x1:L,x2:w-R,y1:y(yy),y2:y(yy),stroke:"#e9eff3"});add(g,"text",{x:x(xx),y:h-B+17,"text-anchor":"middle"},fmt(xx,1));add(g,"text",{x:L-6,y:y(yy)+4,"text-anchor":"end"},fmt(yy,2));}
 add(g,"rect",{x:L,y:T,width:bw,height:bh,fill:"none",stroke:"#c3cfd7"});add(svg,"text",{x:L+bw/2,y:h-4,"text-anchor":"middle"},xlabel);add(svg,"text",{x:L,y:13},ylabel);
 const cid="clip_"+svg.id;const clip=add(add(svg,"defs",{}),"clipPath",{id:cid});add(clip,"rect",{x:L,y:T,width:bw,height:bh});const plot=add(g,"g",{"clip-path":`url(#${cid})`});return {g:plot,x,y,w,h,L,R,T,B,bw,bh};
}
function line(sc,pts,color,width=2,dash="",opacity=1){return add(sc.g,"path",{d:pts.map((p,j)=>(j?"L":"M")+sc.x(p[0])+","+sc.y(p[1])).join(" "),fill:"none",stroke:color,"stroke-width":width,"stroke-dasharray":dash,opacity});}
function sharedDomain(){const r=C.cases[state.row],pts=[...data.past,...data.gt,[0,0]];for(const k of keys()){const s=stats(r,k);if($("show-closest").checked)pts.push(...data.paths[k][s.slot]);if($("show-selected").checked)pts.push(...data.paths[k][r.selected]);if($("show-ade").checked)pts.push(...data.paths[k][s.ia]);if($("show-fde").checked)pts.push(...data.paths[k][s.iff]);if(state.highlight!==null)pts.push(...data.paths[k][state.highlight]);if(!$("focus").checked)pts.push(...data.paths[k].flat());}if($("neighbors").checked&&!$("focus").checked)pts.push(...data.neighbors.flat());
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 return {cx:(xmin+xmax)/2,cy:(ymin+ymax)/2,sx:Math.max(xmax-xmin,.4)*1.13,sy:Math.max(ymax-ymin,.4)*1.13};
}
function spatial(svg,key,domain){const r=C.cases[state.row],s=stats(r,key),[w,h]=dimensions(svg),scale=Math.max(domain.sx/(w-66),domain.sy/(h-62));const xd=[domain.cx-(w-66)*scale/2,domain.cx+(w-66)*scale/2],yd=[domain.cy-(h-62)*scale/2,domain.cy+(h-62)*scale/2];const sc=axes(svg,xd,yd,"沿历史朝向 x (m)","横向 y (m)");const pp=k=>[[0,0],...data.paths[key][k]];
 if($("neighbors").checked)data.neighbors.forEach(p=>line(sc,p,"#ae947a",1,"3 3"));
 if($("show-all").checked)data.paths[key].forEach((p,k)=>{line(sc,pp(k),colors.candidate,1.15,"",.7);bind(line(sc,pp(k),"transparent",9),k,key);});
 const refs=[["show-closest",s.slot,colors.closest,"",3],["show-ade",s.ia,colors.ade,"7 4",2.1],["show-fde",s.iff,colors.fde,"2 4",2.1],["show-selected",r.selected,colors.selected,"",2.2]];
 refs.forEach(([on,k,c,dash,width])=>{if($(on).checked){const p=line(sc,pp(k),c,width,dash);bind(p,k,key);const end=pp(k).at(-1);add(sc.g,"circle",{cx:sc.x(end[0]),cy:sc.y(end[1]),r:3.8,fill:"white",stroke:c,"stroke-width":1.8});}});
 line(sc,data.past,colors.past,2);data.past.forEach(p=>add(sc.g,"circle",{cx:sc.x(p[0]),cy:sc.y(p[1]),r:2,fill:colors.past}));line(sc,[[0,0],...data.gt],colors.gt,2.8);
 data.gt.forEach((p,t)=>{const dot=add(sc.g,"circle",{cx:sc.x(p[0]),cy:sc.y(p[1]),r:2,fill:colors.gt});dot.onmouseenter=e=>tip(e,`GT · ${(t+1)*.4}s`);dot.onmouseleave=()=>$("tooltip").hidden=true;});
 if(state.highlight!==null)line(sc,pp(state.highlight),colors.active,2.8,"9 3");
 add(sc.g,"circle",{cx:sc.x(0),cy:sc.y(0),r:4,fill:colors.past});const gt=data.gt.at(-1);add(sc.g,"path",{d:`M${sc.x(gt[0])-5},${sc.y(gt[1])-5}l10 10M${sc.x(gt[0])-5},${sc.y(gt[1])+5}l10 -10`,stroke:colors.gt,"stroke-width":2});
 add(svg,"text",{x:sc.L+4,y:sc.T+14},$("focus").checked?"聚焦 · 远候选可裁剪":"完整20候选 · 等比例米制");
}
function quality(svg,key,domain){const r=C.cases[state.row],s=stats(r,key),[al,fl]=limit(),sc=axes(svg,[0,domain[0]],[0,domain[1]],"ADE · 整条路径 (m)","FDE · 终点 (m)");
 add(sc.g,"rect",{x:sc.x(0),y:sc.y(fl),width:sc.x(al)-sc.x(0),height:sc.y(0)-sc.y(fl),fill:"#dfeee5"});for(const[x1,y1,x2,y2]of [[al,0,al,domain[1]],[0,fl,domain[0],fl]])add(sc.g,"line",{x1:sc.x(x1),y1:sc.y(y1),x2:sc.x(x2),y2:sc.y(y2),stroke:colors.gt,"stroke-dasharray":"4 4"});
 r.ade[key].forEach((a,k)=>{add(sc.g,"circle",{cx:sc.x(a),cy:sc.y(r.fde[key][k]),r:4.4,fill:s.good[k]?colors.gt:colors.candidate});bind(add(sc.g,"circle",{cx:sc.x(a),cy:sc.y(r.fde[key][k]),r:12,fill:"transparent"}),k,key);});
 const references=[[s.slot,colors.closest],...($("show-ade").checked?[[s.ia,colors.ade]]:[]),...($("show-fde").checked?[[s.iff,colors.fde]]:[]),...(state.highlight!==null?[[state.highlight,colors.active]]:[])];for(const[k,c]of references)add(sc.g,"circle",{cx:sc.x(r.ade[key][k]),cy:sc.y(r.fde[key][k]),r:7,fill:"none",stroke:c,"stroke-width":2});
 add(sc.g,"rect",{x:sc.x(r.ade[key][r.selected])-6,y:sc.y(r.fde[key][r.selected])-6,width:12,height:12,fill:"none",stroke:colors.selected,"stroke-width":2});
 if($("quality-focus").checked)add(svg,"text",{x:sc.L+3,y:sc.T+13},"达标框聚焦 · 远差散点已裁");
}
function pathSlot(key){const r=C.cases[state.row],mode=$("path-mode").value;return mode==="closest"?stats(r,key).slot:mode==="selected"?r.selected:state.highlight??r.selected;}
function errors(path){return path.map((p,t)=>Math.hypot(p[0]-data.gt[t][0],p[1]-data.gt[t][1]));}
function velocity(path){return path.map((p,t)=>p.map((v,c)=>(v-(t?path[t-1][c]:0))/.4));}
function speed(path){return velocity(path).map(p=>Math.hypot(...p));}
function accel(path){const v=velocity(path);return v.slice(1).map((p,t)=>Math.hypot(...p.map((x,c)=>(x-v[t][c])/.4)));}
function curves(){const cs=["#2874ac","#8a819b","#ce543d"],pick=keys().map((key,j)=>({key,c:cs[j],p:data.paths[key][pathSlot(key)]}));
 for(const [id,fn,start]of [["error",errors,.4],["speed",speed,.4],["accel",accel,.8]]){const series=pick.map(x=>({...x,v:fn(x.p)}));if(id!=="error")series.unshift({c:"#273c4a",v:fn(data.gt),key:"GT"});const max=Math.max(.02,...series.flatMap(x=>x.v))*1.1;const sc=axes($(id),[0,4.8],[0,max],"预测时间 (s)",id==="error"?"到GT距离 (m)":id==="speed"?"速度 (m/s)":"加速度 (m/s²)");series.forEach(x=>line(sc,x.v.map((v,t)=>[start+t*.4,v]),x.c,2.1,x.key==="GT"?"4 3":""));}
}
function table(){const r=C.cases[state.row],order=data.scores.map((_,k)=>k).sort((a,b)=>data.scores[b]-data.scores[a]||a-b);$("candidate-table").replaceChildren();order.forEach((k,rank)=>{const tr=html("tr",undefined,k===state.highlight?"active":"");tr.append(html("td",`Slot${k+1} / #${rank+1}${k===r.selected?" · 所选":""}`),html("td",fmt(data.scores[k])));for(const key of keys())tr.append(html("td",`${fmt(r.ade[key][k],4)} / ${fmt(r.fde[key][k],4)}${stats(r,key).good[k]?" ✓":""}`));tr.onclick=()=>{state.highlight=k;$("path-mode").value="linked";render();};$("candidate-table").appendChild(tr);});}
function render(){
 if(!data)return;const r=C.cases[state.row],ks=keys(),[al,fl]=limit();$("position").textContent=`${state.list.indexOf(state.row)+1} / ${state.list.length}例（全量1516）`;$("sample").value=state.row;
 $("previous").disabled=state.list.indexOf(state.row)<=0;$("next").disabled=state.list.indexOf(state.row)>=state.list.length-1;$("save-note").disabled=$("export-svg").disabled=false;
 $("heading").textContent=`ID ${r.id} · ${r.clear?"明确运动变化":r.uncertain?"事件边界不确定":"无明确事件"} · ${C.event_labels[r.event_kind]} · 正式${["stable","transition","change"][r.group]}组`;
 const flags=[1,2,3].map(s=>`seed${s}: ${labels[flag(r,s)]}`).join("　|　");$("seed-flags").textContent=flags+`。当前比较${$("reference").value==="native"?"原F16":"固定响应"}，同路径阈值${al}/${fl}m；${r.onset<99?"GT事后事件起始步 "+r.onset+"（约"+fmt(r.onset*.4,1)+"s）":"GT无满足持久规则的事件，99不是实际时差"}。`;
 $("panels").replaceChildren();ks.forEach(key=>{const s=stats(r,key),panel=html("section",undefined,"panel");panel.append(html("h3",modelName(key)+(key!=="native"?` · seed${state.seed}`:"")));panel.append(html("span",s.any?`集合双达标 · ${s.count}/20条`:"集合无双达标候选","status"+(s.any?" good":"")));
 const m=html("div",undefined,"metric");m.append(html("div",`独立min ADE / FDE：${fmt(r.ade[key][s.ia],4)} / ${fmt(r.fde[key][s.iff],4)}m · ${s.ia===s.iff?"同slot":"不同slot"}`),html("strong",`最近双阈值 Slot${s.slot+1}：${fmt(r.ade[key][s.slot],4)} / ${fmt(r.fde[key][s.slot],4)}m`),html("div",`双阈值距离 ${fmt(s.margin,4)}（≤1合格）；${s.category}`),html("div",`${key==="native"?"真实所选":"继承slot参考"} Slot${r.selected+1}：${fmt(r.ade[key][r.selected],4)} / ${fmt(r.fde[key][r.selected],4)}m`));panel.append(m);
 const space=node("svg",{id:"space_"+key,role:"img","aria-label":modelName(key)+"候选与GT等比例图"}),qual=node("svg",{id:"quality_"+key,class:"quality",role:"img","aria-label":modelName(key)+" ADE FDE散点"});panel.append(space,qual);
 const pulse=html("div",undefined,"pulse-detail");if(key!=="native"){const slot=pathSlot(key),p=data.pulse[key];pulse.textContent=`曲线Slot${slot+1} · 脉冲center ${fmt(p.center[slot])} / span ${fmt(p.span[slot])}（归一化预测时间参数，不是GT事件时刻）。`;}
 else pulse.textContent=`曲线Slot${pathSlot(key)+1} · 原评分最高slot不随GT改变。`;
 panel.append(pulse);$("panels").appendChild(panel);});
 const domain=sharedDomain(),qualityDomain=$("quality-focus").checked?[al*2,fl*2]:[Math.max(al,...ks.flatMap(k=>r.ade[k]))*1.08,Math.max(fl,...ks.flatMap(k=>r.fde[k]))*1.08];ks.forEach(key=>{spatial($("space_"+key),key,domain);quality($("quality_"+key),key,qualityDomain);});curves();table();
 $("slot-detail").textContent=state.highlight===null?"点击候选 / 散点，三个方法联动高亮同slot；同slot不意味着最终轨迹或终点相同。深青轨迹由GT事后选定，不是部署选择。":`当前高亮 Slot${state.highlight+1}：`+ks.map(k=>`${modelName(k)} ${fmt(r.ade[k][state.highlight],4)} / ${fmt(r.fde[k][state.highlight],4)}m`).join(" → ");
 const key=r.id+":"+state.seed;if(noteKey!==key){const v=drafts[key]||notes[key]||{};$("human-tag").value=v.tag||"未标注";$("human-note").value=v.note||"";noteKey=key;}$("note-status").textContent=drafts[key]?"有未保存修改":notes[key]?"本例已保存；请导出长期保留":"未标注";
}
function download(name,text,type){$("export-name").textContent=name;$("export-text").value=text;$("export-fallback").open=true;const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement("a");a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),2000);}
for(const x of C.representatives){const o=html("option",`Seed${x.seed} · ${x.transition==="rescued"?"救回":"退化"} · ${x.rule==="median_change"?"中位变化":"最大变化"} · ID${x.id}`);o.value=JSON.stringify(x);$("representatives").appendChild(o);}
for(const id of ["reference","threshold","group","consistency","sort"])$(id).onchange=()=>refresh();
$("seed").onchange=()=>{state.seed=Number($("seed").value);refresh();};
$("sample").onchange=()=>{state.row=Number($("sample").value);state.highlight=null;loadCase();};
for(const [id,step]of [["previous",-1],["next",1]])$(id).onclick=()=>{state.row=state.list[state.list.indexOf(state.row)+step];state.highlight=null;loadCase();};
function go(id){const row=C.cases.findIndex(x=>x.id===Number(id));if(row<0){$("position").textContent="该ID不在1516样本内";return;}state.filter="all";$("group").value="all";$("consistency").value="all";state.row=row;refresh(false);}
$("go").onclick=()=>go($("id").value);$("id").onkeydown=e=>{if(e.key==="Enter")go($("id").value);};
$("representatives").onchange=()=>{if(!$("representatives").value)return;const x=JSON.parse($("representatives").value);state.seed=x.seed;$("seed").value=x.seed;state.filter=x.transition;$("reference").value="native";$("threshold").value="0.2,0.4";$("group").value="clear";$("consistency").value="all";state.row=x.row;refresh(false);};
for(const id of ["show-all","show-ade","show-fde","show-closest","show-selected","neighbors","focus","quality-focus","path-mode"])$(id).onchange=render;
$("focus-best").onclick=()=>{$("focus").checked=$("quality-focus").checked=$("show-closest").checked=true;$("show-selected").checked=false;render();};
$("full-view").onclick=()=>{$("focus").checked=$("quality-focus").checked=false;$("show-selected").checked=$("show-all").checked=true;render();};
$("clear-slot").onclick=()=>{state.highlight=null;render();};
for(const id of ["human-tag","human-note"])$(id).oninput=()=>{if(!noteKey)return;drafts[noteKey]={tag:$("human-tag").value,note:$("human-note").value};$("note-status").textContent="有未保存修改；关闭前保存";};
$("save-note").onclick=()=>{if(!data)return;const r=C.cases[state.row];notes[noteKey]={sample_id:r.id,seed:state.seed,reference:$("reference").value,thresholds:limit(),classification:flag(r),tag:$("human-tag").value,note:$("human-note").value,highlighted_slot_1based:state.highlight===null?null:state.highlight+1,path_mode:$("path-mode").value,source_sha:C.source_sha,saved_at:new Date().toISOString()};delete drafts[noteKey];try{localStorage.setItem(storage,JSON.stringify(notes));}catch(_){}$("note-status").textContent="已保存，请导出JSON长期保留";};
$("export-notes").onclick=()=>download("human_pulse_response_observations.json",JSON.stringify({study:"pulse_response_20261004",source_sha:C.source_sha,annotations:Object.values(notes)},null,2),"application/json;charset=utf-8");
$("select-export").onclick=()=>{$("export-text").focus();$("export-text").select();};
$("export-svg").onclick=()=>{if(!data)return;const out=node("svg",{xmlns:ns,width:1500,height:850,viewBox:"0 0 1500 850"});out.append(node("rect",{width:1500,height:850,fill:"white"}),node("style",{},"text{font:13px Segoe UI,Microsoft YaHei,sans-serif;fill:#203344}"),node("text",{x:20,y:25},`ID${data.id} · Seed${state.seed} · 同路径 ${limit().join(" / ")}m · 新路径继承原slot，未重新评分`));
 keys().forEach((key,j)=>{out.append(node("text",{x:20+j*500,y:56},modelName(key)));const a=$("space_"+key).cloneNode(true);a.setAttribute("x",j*500);a.setAttribute("y",65);a.setAttribute("width",500);a.setAttribute("height",450);out.append(a);const b=$("quality_"+key).cloneNode(true);b.setAttribute("x",j*500);b.setAttribute("y",535);b.setAttribute("width",500);b.setAttribute("height",290);out.append(b);});download(`ID${data.id}_seed${state.seed}_pulse_comparison.svg`,new XMLSerializer().serializeToString(out),"image/svg+xml");};
let timer;new ResizeObserver(()=>{clearTimeout(timer);timer=setTimeout(render,100);}).observe($("panels"));
state.row=C.cases.findIndex(x=>x.id===(Number(q.get("id"))||C.default_id));if(state.row<0)state.row=0;if(q.has("id")){go(q.get("id"));}else refresh(false);
})();
