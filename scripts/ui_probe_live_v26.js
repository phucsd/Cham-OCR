const {chromium}=require('C:/Users/admin/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const out="E:/Phuc's Data/Github/Cham-OCR/REPORTS/live_v26_audit_2026_11_22/ui";
 fs.mkdirSync(out,{recursive:true});
 const errors=[],networkErrors=[],dialogs=[];
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('requestfailed',r=>networkErrors.push({url:r.url(),reason:r.failure()}));
 page.on('dialog',async d=>{dialogs.push(d.message());await d.dismiss()});
 try{
  await page.goto('https://ocr.cham.asia',{waitUntil:'domcontentloaded',timeout:45000});
  await page.waitForTimeout(1800);
  const state=await page.evaluate(()=>({
    title:document.title,url:location.href,heading:[...document.querySelectorAll('h1,h2,h3')].map(x=>x.textContent.trim()).slice(0,30),
    selects:[...document.querySelectorAll('select')].map(x=>({id:x.id,name:x.name,options:[...x.options].map(o=>({value:o.value,label:o.textContent.trim()}))})),
    fileInputs:[...document.querySelectorAll('input[type=file]')].map(x=>({id:x.id,name:x.name,accept:x.accept,multiple:x.multiple})),
    buttons:[...document.querySelectorAll('button')].filter(x=>x.offsetWidth||x.offsetHeight).map(x=>({id:x.id,txt:x.textContent.trim().slice(0,80),title:x.title,disabled:x.disabled})).slice(0,120),
    testIds:[...document.querySelectorAll('[data-testid]')].map(x=>x.getAttribute('data-testid')).slice(0,30),
    canvasCount:document.querySelectorAll('canvas').length,
    samples:[...document.querySelectorAll('img')].slice(0,15).map(x=>({alt:x.alt,src:x.getAttribute('src')})),
  }));
  fs.writeFileSync(path.join(out,'dom_inventory.json'),JSON.stringify({state,errors,networkErrors,dialogs,utc:new Date().toISOString()},null,2));
  await page.screenshot({path:path.join(out,'initial_page.png'),fullPage:true,timeout:15000});
  console.log('UI_PROBE_TITLE',state.title,'SELECTS',state.selects.map(s=>s.id).join(','),'INPUTS',JSON.stringify(state.fileInputs),'BUTTONS',JSON.stringify(state.buttons.slice(0,30)));
  console.log('UI_PROBE_ERRORS',JSON.stringify(errors),'NET',JSON.stringify(networkErrors));
 }catch(e){console.log('UI_PROBE_FAILURE',String(e));}
 finally{await browser.close();}
})();
