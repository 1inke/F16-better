"use strict";
(() => {
 const C=window.AUDIT_COMMON, M=window.AUDIT_MODELS, $=id=>document.getElementById(id);
 if(!C||!M.native){document.body.textContent="数据未加载。请完整保留data文件夹，并重新打开index.html。";return;}
 const ns="http://www.w3.org/2000/svg", colors={gt:"#007f5f",selected:"#bd4824",ade:"#2367a7",fde:"#8455a5",joint:"#00878d",past:"#35404a",candidate:"#aebbc5",active:"#142e40"};
 const params=new URLSearchParams(location.search);
 const state={model:params.get("model")||"native",category:-1,row:0,highlight:null,list:[],cats:[],goodCount:[]};
 if(!C.models.some(x=>x.key===state.model))state.model="native";
 const storageKey="trajectory_visual_review_v1_"+C.data_sha.slice(0,12);
 let notes={};try{notes=JSON.parse(localStorage.getItem(storageKey)||"{}");}catch(_){notes={};}
 const drafts={};let annotationKey=null;
 const fmt=(v,n=3)=>Number(v).toFixed(n), esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
 const meta=()=>C.models.find(x=>x.key===state.model), D=()=>M[state.model], stats=i=>D().d[i];
 const limits=()=>[Number($("ade-limit").value),Number($("fde-limit").value)];
 const node=(tag,attrs={},text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
 const add=(parent,tag,attrs,text)=>{const n=node(tag,attrs,text);parent.appendChild(n);return n;};
 function tooltip(event,text){const p=$("tooltip");p.textContent=text;p.hidden=false;p.style.left=Math.max(6,Math.min(innerWidth-300,event.clientX+14))+"px";p.style.top=Math.max(6,Math.min(innerHeight-85,event.clientY+14))+"px";}
 function hideTip(){$("tooltip").hidden=true;}
 function choose(k){state.highlight=k;renderCase();}
 function classify(){
  const [al,fl]=limits();if(!(al>0&&fl>0)){return false;}
  state.cats=[];state.goodCount=[];
  D().a.forEach((a,i)=>{const f=D().f[i],s=stats(i).selected;const good=a.map((v,k)=>v<=al&&f[k]<=fl);
   const any=good.some(Boolean),ag=Math.min(...a)<=al,fg=Math.min(...f)<=fl;
   state.cats.push(good[s]?0:any?1:ag&&fg?2:ag?3:fg?4:5);state.goodCount.push(good.filter(Boolean).length);});
  return true;
 }
 function categoryPanel(){
  const p=$("categories");p.replaceChildren();
  C.labels.forEach((name,cat)=>{const count=state.cats.filter(x=>x===cat).length;
   const b=document.createElement("button");b.className="category"+(state.category===cat?" active":"");b.setAttribute("aria-pressed",state.category===cat);
   b.innerHTML=`${esc(name)}<strong>${count}</strong><small>${fmt(count/C.ids.length*100,1)}% · 全1516例</small>`;
   b.onclick=()=>{state.category=state.category===cat?-1:cat;refreshList(true);};p.appendChild(b);});
 }
 function passesFlag(i){const d=stats(i);switch($("flag").value){case"ade":return d.selected_exact_ade_slot;case"fde":return d.selected_exact_fde_slot;case"joint":return d.selected_exact_joint_slot;case"different":return !d.oracle_same_slot;case"dominated":return d.selected_pareto_dominated;default:return true;}}
 function refreshList(reset=false){
  if(!D())return;
  if(!classify()){ $("position").textContent="阈值须大于0";return; }
  categoryPanel();const group=$("group").value;
  state.list=C.ids.map((_,i)=>i).filter(i=>(state.category<0||state.cats[i]===state.category)&&(group==="all"||C.groups[i]===Number(group))&&passesFlag(i));
  const sort=$("sort").value;
  const metric={regret:"joint_regret","ade-regret":"ade_regret",oracle:"joint_oracle",selected:"joint_selected"}[sort];
  state.list.sort((a,b)=>metric?(stats(b)[metric]-stats(a)[metric]||C.ids[a]-C.ids[b]):C.ids[a]-C.ids[b]);
  if(reset||!state.list.includes(state.row))state.row=state.list[0];
  const select=$("sample");select.replaceChildren();
  state.list.forEach(i=>{const opt=document.createElement("option");opt.value=i;opt.textContent=`ID ${C.ids[i]} · ${C.labels[state.cats[i]]} · regret ${fmt(stats(i).joint_regret)}m`;select.appendChild(opt);});
  if(state.row!==undefined)select.value=state.row;
  state.highlight=null;renderCase();
 }
 function dimensions(svg){const w=Math.max(260,svg.getBoundingClientRect().width),h=Number.parseFloat(getComputedStyle(svg).height)||360;svg.setAttribute("viewBox",`0 0 ${w} ${h}`);return[w,h];}
 function axes(svg,xDomain,yDomain,xLabel,yLabel){
  svg.replaceChildren();const [w,h]=dimensions(svg),L=58,R=18,T=20,B=46,bw=w-L-R,bh=h-T-B;
  const x=v=>L+(v-xDomain[0])/(xDomain[1]-xDomain[0])*bw;
  const y=v=>h-B-(v-yDomain[0])/(yDomain[1]-yDomain[0])*bh;
  const defs=add(svg,"defs"),clip=add(defs,"clipPath",{id:svg.id+"-clip"});add(clip,"rect",{x:L,y:T,width:bw,height:bh});
  const g=add(svg,"g",{"clip-path":`url(#${svg.id}-clip)`});
  for(let j=0;j<=4;j++){
   const xv=xDomain[0]+j*(xDomain[1]-xDomain[0])/4,yv=yDomain[0]+j*(yDomain[1]-yDomain[0])/4;
   add(svg,"line",{x1:x(xv),y1:T,x2:x(xv),y2:h-B,stroke:"#edf1f4"});
   add(svg,"line",{x1:L,y1:y(yv),x2:w-R,y2:y(yv),stroke:"#edf1f4"});
   add(svg,"text",{x:x(xv),y:h-B+19,"text-anchor":"middle"},fmt(xv,1));
   add(svg,"text",{x:L-9,y:y(yv)+4,"text-anchor":"end"},fmt(yv,1));
  }
  add(svg,"rect",{x:L,y:T,width:bw,height:bh,fill:"none",stroke:"#b9c7d0"});
  add(svg,"text",{x:L+bw/2,y:h-8,"text-anchor":"middle"},xLabel);
  add(svg,"text",{x:16,y:T+bh/2,transform:`rotate(-90 16 ${T+bh/2})`,"text-anchor":"middle"},yLabel);
  return {g,x,y,w,h,L,R,T,B,bw,bh};
 }
 function line(g,points,sc,color,width=2,dash="",opacity=1){
  return add(g,"path",{d:points.map((p,i)=>(i?"L":"M")+sc.x(p[0])+","+sc.y(p[1])).join(" "),fill:"none",stroke:color,"stroke-width":width,"stroke-dasharray":dash,opacity,"stroke-linejoin":"round","stroke-linecap":"round"});
 }
 function domainForSpatial(svg,points){
  const [w,h]=dimensions(svg),bw=w-76,bh=h-66,xs=points.map(x=>x[0]),ys=points.map(x=>x[1]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const perPixel=Math.max((xmax-xmin)/bw,(ymax-ymin)/bh,.3/Math.min(bw,bh))*1.15;
  return [[(xmin+xmax)/2-bw*perPixel/2,(xmin+xmax)/2+bw*perPixel/2],[(ymin+ymax)/2-bh*perPixel/2,(ymin+ymax)/2+bh*perPixel/2]];
 }
 function domainBoth(){
  const i=state.row,d=stats(i),data=D(),focus=$("focus").checked;
  let points=[...C.past[i],...C.gt[i],...data.p[i][d.selected],[0,0]];
  if(!focus)points.push(...data.p[i].flat());
  if($("show-neighbors").checked&&!focus)points.push(...C.neighbors[i].flat());
  // The paired space plots have identical measured width and height at wide layouts.
  // At stacked/tablet layouts use a shared square-expanded domain for both actual aspect ratios.
  const x=domainForSpatial($("candidates-plot"),points),y=domainForSpatial($("selected-plot"),points);
  const center=[(x[0][0]+x[0][1])/2,(x[1][0]+x[1][1])/2];
  const spanX=Math.max(x[0][1]-x[0][0],y[0][1]-y[0][0]),spanY=Math.max(x[1][1]-x[1][0],y[1][1]-y[1][0]);
  return {center,spanX,spanY};
 }
 function spatial(svg,all,commonDomain){
  const [w,h]=dimensions(svg),bw=w-76,bh=h-66;
  const scale=Math.max(commonDomain.spanX/bw,commonDomain.spanY/bh);
  const xd=[commonDomain.center[0]-bw*scale/2,commonDomain.center[0]+bw*scale/2],yd=[commonDomain.center[1]-bh*scale/2,commonDomain.center[1]+bh*scale/2];
  const sc=axes(svg,xd,yd,"沿历史朝向 x (m)","横向 y (m)");
  const i=state.row,d=stats(i),data=D(),points=k=>[[0,0],...data.p[i][k]];
  add(sc.g,"line",{x1:sc.x(xd[0]),y1:sc.y(0),x2:sc.x(xd[1]),y2:sc.y(0),stroke:"#d0dbe3","stroke-dasharray":"2 4"});
  if($("show-neighbors").checked)C.neighbors[i].forEach((p,k)=>{line(sc.g,p,sc,"#a89075",1.1,"3 3",.7);add(sc.g,"circle",{cx:sc.x(p.at(-1)[0]),cy:sc.y(p.at(-1)[1]),r:2,fill:"#a89075"});});
  if(all&&$("show-all").checked)data.p[i].forEach((p,k)=>{
   const path=line(sc.g,points(k),sc,colors.candidate,1.3,"",.72);
   path.style.cursor="pointer";path.onmouseenter=e=>tooltip(e,`Slot ${k+1} · ADE ${fmt(data.a[i][k])}m / FDE ${fmt(data.f[i][k])}m`);path.onmouseleave=hideTip;path.onclick=()=>choose(k);
   const hit=line(sc.g,points(k),sc,"transparent",8);hit.style.cursor="pointer";hit.onmouseenter=path.onmouseenter;hit.onmouseleave=hideTip;hit.onclick=path.onclick;
  });
  const refs=[[$("show-joint").checked,d.oracle_joint,colors.joint,"8 3 2 3"],[$("show-fde").checked,d.oracle_fde,colors.fde,"2 5"],[$("show-ade").checked,d.oracle_ade,colors.ade,"7 4"]];
  refs.forEach(([on,k,color,dash])=>{if(on){line(sc.g,points(k),sc,color,2.2,dash);const end=points(k).at(-1);add(sc.g,"circle",{cx:sc.x(end[0]),cy:sc.y(end[1]),r:4,fill:"white",stroke:color,"stroke-width":1.8});}});
  line(sc.g,C.past[i],sc,colors.past,2.2);C.past[i].forEach(p=>add(sc.g,"circle",{cx:sc.x(p[0]),cy:sc.y(p[1]),r:2,fill:colors.past}));
  line(sc.g,[[0,0],...C.gt[i]],sc,colors.gt,3);const gt=C.gt[i].at(-1);
  add(sc.g,"path",{d:`M${sc.x(gt[0])-5},${sc.y(gt[1])-5}L${sc.x(gt[0])+5},${sc.y(gt[1])+5}M${sc.x(gt[0])-5},${sc.y(gt[1])+5}L${sc.x(gt[0])+5},${sc.y(gt[1])-5}`,stroke:colors.gt,"stroke-width":2});
  line(sc.g,points(d.selected),sc,colors.selected,3);const se=points(d.selected).at(-1);add(sc.g,"rect",{x:sc.x(se[0])-4,y:sc.y(se[1])-4,width:8,height:8,fill:colors.selected});
  if(state.highlight!==null){line(sc.g,points(state.highlight),sc,colors.active,2.3,"10 3");const e=points(state.highlight).at(-1);add(sc.g,"circle",{cx:sc.x(e[0]),cy:sc.y(e[1]),r:6,fill:"none",stroke:colors.active,"stroke-width":2});}
  add(sc.g,"circle",{cx:sc.x(0),cy:sc.y(0),r:4,fill:colors.past});
  add(svg,"text",{x:sc.L+5,y:sc.T+15},$("focus").checked?"局部聚焦 · 远候选可能被裁":"完整范围 · 1:1米制比例");
 }
 function quality(){
  const i=state.row,d=stats(i),a=D().a[i],f=D().f[i],[al,fl]=limits();
  const sc=axes($("quality-plot"),[0,Math.max(...a,al)*1.1+.015],[0,Math.max(...f,fl)*1.1+.025],"ADE · 整条路径 (m)","FDE · 终点 (m)");
  add(sc.g,"rect",{x:sc.x(0),y:sc.y(fl),width:sc.x(al)-sc.x(0),height:sc.y(0)-sc.y(fl),fill:"#dff0e8",opacity:.75});
  add(sc.g,"line",{x1:sc.x(al),y1:sc.T,x2:sc.x(al),y2:sc.h-sc.B,stroke:colors.gt,"stroke-dasharray":"4 4"});
  add(sc.g,"line",{x1:sc.L,y1:sc.y(fl),x2:sc.w-sc.R,y2:sc.y(fl),stroke:colors.gt,"stroke-dasharray":"4 4"});
  a.forEach((v,k)=>{const p=add(sc.g,"circle",{cx:sc.x(v),cy:sc.y(f[k]),r:state.highlight===k?7:4.5,fill:v<=al&&f[k]<=fl?colors.gt:colors.candidate,stroke:state.highlight===k?colors.active:"white","stroke-width":state.highlight===k?2:1});
   const hit=add(sc.g,"circle",{cx:sc.x(v),cy:sc.y(f[k]),r:13,fill:"transparent"});hit.style.cursor="pointer";hit.onclick=()=>choose(k);hit.onmouseenter=e=>tooltip(e,`Slot ${k+1} · ADE ${fmt(v)}m · FDE ${fmt(f[k])}m`);hit.onmouseleave=hideTip;
  });
  [[$("show-ade").checked,d.oracle_ade,colors.ade],[$("show-fde").checked,d.oracle_fde,colors.fde],[$("show-joint").checked,d.oracle_joint,colors.joint]].forEach(([on,k,color])=>{if(on)add(sc.g,"circle",{cx:sc.x(a[k]),cy:sc.y(f[k]),r:7,fill:"none",stroke:color,"stroke-width":2});});
  add(sc.g,"rect",{x:sc.x(a[d.selected])-6,y:sc.y(f[d.selected])-6,width:12,height:12,fill:"none",stroke:colors.selected,"stroke-width":2.2});
 }
 function errors(){
  const i=state.row,d=stats(i);const values=k=>D().p[i][k].map((p,t)=>Math.hypot(p[0]-C.gt[i][t][0],p[1]-C.gt[i][t][1]));
  const curves=[[d.selected,colors.selected,""],...($('show-ade').checked?[[d.oracle_ade,colors.ade,"7 4"]]:[]),...($('show-fde').checked?[[d.oracle_fde,colors.fde,"2 5"]]:[]),...($('show-joint').checked?[[d.oracle_joint,colors.joint,"8 3 2 3"]]:[])];
  const max=Math.max(...curves.flatMap(([k])=>values(k)),.1)*1.12;
  const sc=axes($("error-plot"),[0,4.8],[0,max],"预测时间 (s)","到GT距离 (m)");
  curves.forEach(([k,color,dash])=>line(sc.g,values(k).map((e,t)=>[(t+1)*.4,e]),sc,color,2,dash));
 }
 function badge(text,cls=""){return`<span class="badge ${cls}">${esc(text)}</span>`;}
 function table(){
  const i=state.row,d=stats(i),a=D().a[i],f=D().f[i],score=C.scores[i];
  const order=score.map((_,k)=>k).sort((x,y)=>score[y]-score[x]||x-y);const p=$("candidate-table");p.replaceChildren();
  order.forEach((k,rank)=>{const tr=document.createElement("tr");tr.className=(k===d.selected?"selected-row ":"")+(k===state.highlight?"active":"");
   let marks="";if(k===d.selected)marks+=badge("所选","selected");if(k===d.oracle_ade)marks+=badge("minA","ade");if(k===d.oracle_fde)marks+=badge("minF","fde");if(k===d.oracle_joint)marks+=badge("联合","joint");
   tr.innerHTML=`<td>${rank+1}</td><td><button aria-label="高亮slot ${k+1}">Slot ${k+1}</button>${marks}</td><td>${fmt(score[k])}</td><td>${fmt(a[k])}</td><td>${fmt(f[k])}</td><td>${fmt(a[k]+.5*f[k])}</td>`;
   tr.querySelector("button").onclick=()=>choose(k);tr.onclick=e=>{if(e.target.tagName!=="BUTTON")choose(k);};p.appendChild(tr);});
 }
 function renderCase(){
  if(!D())return;
  const i=state.row;const empty=i===undefined;
  ["candidates-plot","quality-plot","selected-plot","error-plot"].forEach(id=>{if(empty)$(id).replaceChildren();});
  $("position").textContent=empty?"没有符合筛选条件的样本":`${state.list.indexOf(i)+1} / ${state.list.length}（全量1516）`;
  $("previous").disabled=empty||state.list.indexOf(i)<=0;$("next").disabled=empty||state.list.indexOf(i)>=state.list.length-1;
  if(empty){$("case-heading").textContent="当前类别或筛选没有样本；可调整口径或显示全部。";$("metrics").replaceChildren();$("candidate-table").replaceChildren();$("highlight-detail").textContent="无样本";$("secondary-metrics").replaceChildren();$("save-note").disabled=true;$("export-svg").disabled=true;return;}
  $("save-note").disabled=false;$("export-svg").disabled=false;
  const d=stats(i),cat=state.cats[i],a=D().a[i],f=D().f[i];$("sample").value=i;
  $("case-heading").textContent=`ID ${C.ids[i]} · ${["stable","transition","change"][C.groups[i]]} · ${C.labels[cat]} · 所选Slot ${d.selected+1}`;
  $("metrics").innerHTML=[
   [state.model==="native"?"模型所选 ADE / FDE":"继承slot参照 ADE / FDE",`${fmt(d.ade_selected)} / ${fmt(d.fde_selected)} m`,state.model==="native"?"同一条实际选择轨迹":"非修正后重新评分选择"],
   ["集合 minADE / minFDE",`${fmt(d.minade)} / ${fmt(d.minfde)} m`,d.oracle_same_slot?"来自同一slot":"来自不同slot，不能拼成一条轨迹"],
   ["选择遗憾 ΔADE / ΔFDE",`${fmt(d.ade_regret)} / ${fmt(d.fde_regret)} m`,"所选误差减相应集合最小误差"],
   ["同路径合格候选 / 联合遗憾",`${state.goodCount[i]} 条 / ${fmt(d.joint_regret)} m`,"联合Q=ADE+0.5FDE，不是独立min之和"]
  ].map(([name,v,detail])=>`<div class="metric"><span>${name}</span><strong>${v}</strong><small>${detail}</small></div>`).join("");
  const com=domainBoth();spatial($("candidates-plot"),true,com);spatial($("selected-plot"),false,com);quality();errors();table();
  $("highlight-detail").textContent=state.highlight===null?`精确最优命中：minADE ${d.selected_exact_ade_slot?"是":"否"} · minFDE ${d.selected_exact_fde_slot?"是":"否"} · 联合 ${d.selected_exact_joint_slot?"是":"否"}。点击轨迹或表格可查看任意候选。`:`高亮Slot ${state.highlight+1} · ADE ${fmt(a[state.highlight])}m · FDE ${fmt(f[state.highlight])}m · Q ${fmt(a[state.highlight]+.5*f[state.highlight])}m`;
  $("secondary-metrics").innerHTML=`minADE候选：Slot ${d.oracle_ade+1}，对应FDE ${fmt(d.fde_at_minade)}m，评分排名 ${d.oracle_ade_score_rank}<br>minFDE候选：Slot ${d.oracle_fde+1}，对应ADE ${fmt(d.ade_at_minfde)}m，评分排名 ${d.oracle_fde_score_rank}<br>联合最优：Slot ${d.oracle_joint+1}，Q ${fmt(d.joint_oracle)}m，评分排名 ${d.oracle_joint_score_rank}<br>所选被其他候选在ADE/FDE上支配：${d.selected_pareto_dominated?"是":"否"}<br>集合终点分散度 ${fmt(d.endpoint_spread)}m · 10cm重复对率 ${fmt(d.duplicate_fraction_10cm*100,1)}%<br>所选speed p95 ${fmt(d.selected_speed_p95)}m/s · acceleration mean ${fmt(d.selected_acceleration_mean)}m/s² · jerk mean ${fmt(d.selected_jerk_mean)}m/s³（dt=0.4s；不自动判安全）`;
  const key=state.model+":"+C.ids[i];
  if(annotationKey!==key){const record=drafts[key]||notes[key]||{};$("human-tag").value=record.tag||"";$("human-note").value=record.note||"";annotationKey=key;}
  const saved=notes[key];$("note-status").textContent=drafts[key]?"有未保存的修改；切换图层不会丢失，关闭页面前请保存。":saved?.saved_at?"此例备注已保存 · "+saved.saved_at:"尚未标注；请保存后导出JSON长期保留。";
 }
 function loadModel(key){
  state.model=key;$("selection-semantics").textContent=meta().selection+(key==="native"?"：native logits argmax。": "：沿用原生slot ID，不对修正后路径重新评分，不代表新筛选性能。");
  $("selection-semantics").classList.toggle("warning",key!=="native");
  if(M[key]){refreshList();return Promise.resolve();}
  $("position").textContent="加载完整1516例…";const script=document.createElement("script");script.src=`data/${key}.js`;
  return new Promise((resolve,reject)=>{script.onload=()=>{refreshList();resolve();};script.onerror=()=>{$("position").textContent="数据读取失败，请保持文件夹完整。";reject(new Error("Data loading failed"));};document.body.appendChild(script);});
 }
 function download(name,text,type){$("export-name").textContent="文件名："+name;$("export-text").value=text;$("export-fallback").open=true;const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement("a");a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),2000);}
 C.models.forEach(m=>{const o=document.createElement("option");o.value=m.key;o.textContent=m.label;$("model").appendChild(o);});$("model").value=state.model;
 $("model").onchange=()=>loadModel($("model").value);
 $("preset").onchange=()=>{if($("preset").value!=="custom"){const [a,f]=$("preset").value.split(",");$("ade-limit").value=a;$("fde-limit").value=f;refreshList();}};
 ["ade-limit","fde-limit"].forEach(id=>$(id).onchange=()=>{$("preset").value="custom";refreshList();});
 ["group","flag","sort"].forEach(id=>$(id).onchange=()=>refreshList(true));
 $("sample").onchange=()=>{state.row=Number($("sample").value);state.highlight=null;renderCase();};
 $("previous").onclick=()=>{state.row=state.list[state.list.indexOf(state.row)-1];state.highlight=null;renderCase();};
 $("next").onclick=()=>{state.row=state.list[state.list.indexOf(state.row)+1];state.highlight=null;renderCase();};
 $("reset").onclick=()=>{state.category=-1;$("group").value="all";$("flag").value="all";refreshList();};
 function go(){const value=$("id-search").value;if(value==="")return;const i=C.ids.indexOf(Number(value));if(i<0){$("position").textContent="该ID不在既有development1516例中";return;}state.category=-1;$("group").value="all";$("flag").value="all";state.row=i;refreshList();}
 $("go").onclick=go;$("id-search").onkeydown=e=>{if(e.key==="Enter")go();};
 ["show-all","show-ade","show-fde","show-joint","show-neighbors","focus"].forEach(id=>$(id).onchange=renderCase);
 ["human-tag","human-note"].forEach(id=>$(id).oninput=()=>{if(!annotationKey)return;drafts[annotationKey]={tag:$("human-tag").value,note:$("human-note").value};$("note-status").textContent="有未保存的修改；关闭页面前请保存。";});
 $("save-note").onclick=()=>{if(state.row===undefined)return;const i=state.row,[al,fl]=limits();
  notes[state.model+":"+C.ids[i]]={model:state.model,sample_id:C.ids[i],tag:$("human-tag").value,note:$("human-note").value,
   diagnostic_class:C.labels[state.cats[i]],ade_limit:al,fde_limit:fl,selected_slot_1based:stats(i).selected+1,source_cache_SHA256:C.data_sha,saved_at:new Date().toISOString()};
  delete drafts[state.model+":"+C.ids[i]];
  try{localStorage.setItem(storageKey,JSON.stringify(notes));$("note-status").textContent="备注已在此浏览器保存。请导出JSON以长期保留。";}catch(_){$("note-status").textContent="浏览器未允许本地存储；备注仍可立即导出JSON。";}};
 $("export-notes").onclick=()=>download("human_trajectory_observations.json",JSON.stringify({source_cache_SHA256:C.data_sha,annotations:Object.values(notes)},null,2),"application/json;charset=utf-8");
 $("select-export").onclick=()=>{$("export-text").focus();$("export-text").select();};
 $("export-svg").onclick=()=>{const s=$("selected-plot").cloneNode(true);s.setAttribute("xmlns",ns);const title=node("title",{},`ID${C.ids[state.row]} ${meta().label} ${C.labels[state.cats[state.row]]}`);s.prepend(title);s.prepend(node("style",{},'text{font:12px "Segoe UI","Microsoft YaHei",sans-serif;fill:#203344}'));download(`ID${C.ids[state.row]}_${state.model}_selection.svg`,new XMLSerializer().serializeToString(s),"image/svg+xml");};
 let resizeTimer;new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(D()&&state.row!==undefined)renderCase();},80);}).observe($("candidates-plot").parentElement);
 loadModel(state.model).then(()=>{const i=C.ids.indexOf(Number(params.get("id")));if(params.has("id")&&i>=0){state.row=i;refreshList();}});
})();
