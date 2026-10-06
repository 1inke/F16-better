"use strict";
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/linke/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=__dirname;let activeBrowser;
async function main(){
 const browser=activeBrowser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({viewport:{width:1580,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('file:///'+path.join(root,'index.html').replaceAll('\\','/'));
 await page.waitForFunction(()=>document.querySelector('#heading').textContent.startsWith('ID 29410'));
 const defaultCounts=await page.locator('#counts .count strong').allTextContents();assert.deepEqual(defaultCounts,['469','8','12','80','369']);
 assert.equal(await page.locator('#sample option').count(),8);assert.equal(await page.locator('#panels section').count(),3);
 await page.screenshot({path:path.join(root,'preview_default.png'),fullPage:true});
 const plotBoxes=await page.locator('.panel > svg:not(.quality)').evaluateAll(els=>els.map(e=>({width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height,viewBox:e.getAttribute('viewBox')})));
 assert.ok(plotBoxes.every(b=>Math.abs(b.width-plotBoxes[0].width)<1&&b.height===plotBoxes[0].height));
 await page.locator('#counts .count').nth(2).click();await page.waitForFunction(()=>document.querySelector('#heading').textContent.startsWith('ID 7109'));assert.equal(await page.locator('#sample option').count(),12);
 await page.locator('#consistency').selectOption('lost3');await page.waitForFunction(()=>document.querySelector('#sample').options.length===6);assert.equal(await page.locator('#sample option').count(),6);
 // Robust lost sample and robust rescued sample across all seeds.
 await page.setViewportSize({width:1580,height:2000});
 for(const [id,name]of [[29334,'preview_degraded.png'],[29719,'preview_rescued.png']]){
   await page.locator('#id').fill(String(id));await page.locator('#go').click();await page.waitForFunction(id=>document.querySelector('#heading').textContent.startsWith('ID '+id),id);
   const top=await page.locator('#heading').evaluate(e=>e.getBoundingClientRect().top+window.scrollY);
   await page.screenshot({path:path.join(root,name),clip:{x:0,y:Math.max(0,top-8),width:1580,height:1040}});
   await page.locator('#focus-best').click();assert.ok(await page.locator('#quality-focus').isChecked());
   await page.screenshot({path:path.join(root,name.replace('.png','_focused.png')),clip:{x:0,y:Math.max(0,top-8),width:1580,height:1040}});
   await page.locator('#full-view').click();assert.ok(!await page.locator('#quality-focus').isChecked());
 }
 await page.locator('details > summary').first().click();await page.locator('#candidate-table tr').nth(4).click();assert.match(await page.locator('#slot-detail').textContent(),/当前高亮 Slot/);assert.equal(await page.locator('#path-mode').inputValue(),'linked');
 await page.locator('#human-note').fill('QA test draft: NOT a human observation');await page.locator('#show-fde').check();assert.equal(await page.locator('#human-note').inputValue(),'QA test draft: NOT a human observation');await page.locator('#human-note').fill('');
 await page.locator('#reference').selectOption('fixed');await page.locator('#seed').selectOption('2');
 await page.waitForFunction(()=>document.querySelector('#heading').textContent.startsWith('ID'));
 assert.deepEqual(await page.locator('#counts .count strong').allTextContents(),['1516','21','22','707','766']);
 await page.locator('#threshold').selectOption('0.3,0.5');assert.equal((await page.locator('#counts .count strong').allTextContents())[0],'1516');
 await page.locator('#threshold').selectOption('0.2,0.4');await page.locator('#reference').selectOption('native');await page.locator('#consistency').selectOption('rescued3');await page.locator('#counts .count').nth(2).click();
 await page.waitForFunction(()=>document.querySelector('#heading').textContent.startsWith('当前条件没有样本'));assert.equal(await page.locator('#sample option').count(),0);assert.ok(await page.locator('#save-note').isDisabled());
 await page.locator('#consistency').selectOption('all');await page.locator('#id').fill('29334');await page.locator('#go').click();await page.waitForFunction(()=>document.querySelector('#heading').textContent.startsWith('ID 29334'));
 await page.locator('#export-svg').click();assert.match(await page.locator('#export-text').inputValue(),/^<svg/);fs.writeFileSync(path.join(root,'example_ID29334_comparison.svg'),await page.locator('#export-text').inputValue());
 await page.locator('#export-notes').click();const exported=JSON.parse(await page.locator('#export-text').inputValue());assert.deepEqual(exported.annotations,[]);
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(250);const mobile=await page.evaluate(()=>({inner:innerWidth,width:document.documentElement.scrollWidth,panels:[...document.querySelectorAll('.panel')].map(e=>e.getBoundingClientRect().width)}));assert.ok(mobile.width<=mobile.inner+1);await page.screenshot({path:path.join(root,'preview_mobile.png'),fullPage:true});
 assert.deepEqual(errors,[]);await browser.close();
 const report={status:'PASS_OFFLINE_FILE_BROWSER_QA',browser:'Playwright existing Microsoft Edge headless, file:// offline',default_counts:defaultCounts,robust_lost_filter_n:6,all_methods_aligned_dimensions:plotBoxes,mobile,no_console_errors:errors.length===0,candidate_click_linked:true,unsaved_draft_survives_layer_change:true,empty_filter_safe:true,svg_export_tested:true,json_text_export_tested:true,human_annotation_saved:false,download_event_tested:false,localStorage_roundtrip_tested:false};fs.writeFileSync(path.join(root,'UI_QA.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}
main().catch(async e=>{console.error(e);if(activeBrowser)await activeBrowser.close();process.exitCode=1;});
