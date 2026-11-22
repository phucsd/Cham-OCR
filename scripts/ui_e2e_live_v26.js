const {chromium}=require('C:/Users/admin/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const root="E:/Phuc's Data/Github/Cham-OCR";
 const out=root+'/REPORTS/live_v26_audit_2026_11_22/ui';fs.mkdirSync(out,{recursive:true});
 const cases=[],errors=[],net=[],responses=[],downloads=[];
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox','--disable-gpu']});
 const page=await browser.newPage({viewport:{width:1360,height:960},acceptDownloads:true});
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('requestfailed',r=>net.push({url:r.url(),msg:r.failure()?.errorText}));
 page.on('response',r=>{if(/\/ocr($|\?)/.test(r.url()))responses.push({url:r.url(),status:r.status()})});
 page.on('download',async d=>{downloads.push({name:d.suggestedFilename(),url:d.url()});try{await d.saveAs(path.join(out,'download_'+d.suggestedFilename().replace(/[<>:"/\\|?*]/g,'_')))}catch(e){downloads.push({error:String(e)})}});
 try{
  await page.goto('https://ocr.cham.asia',{waitUntil:'domcontentloaded',timeout:35000});
  await page.waitForTimeout(1300);
  const file=path.join(root,'ocr-benchmark/data/images/test_006_lvl1.png');
  const model=page.locator('#model-select');
  await model.selectOption('v24_1_r4');
  await page.locator('#file-input').setInputFiles(file);
  await page.waitForTimeout(2100);
  const button=page.locator('#btn-run');
  cases.push({name:'upload_image',enabled:await button.isEnabled(),
      visibleImages:await page.locator('img[src^="data:"]').count(),
      queuedText:(await page.locator('body').innerText()).slice(-350)});
  if(await button.isEnabled()){
    const responseWait=page.waitForResponse(r=>r.url().endsWith('/ocr'),{timeout:120000}).catch(e=>null);
    await button.click({timeout:20000});
    const res=await responseWait;
    await page.waitForTimeout(1300);
    const txt=await page.locator('body').innerText();
    cases.push({name:'transcribe_from_UI',server_response:res?.status()||null,
      statusText:txt.slice(-900),runEnabledAfter:await button.isEnabled(),visibleChars:txt.length});
  } else cases.push({name:'transcribe_from_UI',error:'Upload did not enable Transcribe button'});
  const exportHeader=page.locator('button[title*="Export"]').first();
  if(await exportHeader.count()){
    await exportHeader.click();
    await page.waitForTimeout(450);
    const modal=await page.evaluate(()=>[...document.querySelectorAll('[role=dialog],.modal,dialog')].filter(x=>x.offsetWidth||x.offsetHeight).map(x=>({id:x.id,text:x.textContent.trim().slice(0,850)})));
    const btns=await page.locator('button:visible').allTextContents();
    cases.push({name:'open_export',modal,buttons:btns.map(x=>x.trim()).filter(Boolean).slice(-28)});
  }
  await page.keyboard.press('Escape');
  await model.selectOption('v25');
  cases.push({name:'model_selector_v25',selected:await model.inputValue()});
  await model.selectOption('auto');
  cases.push({name:'model_selector_auto',selected:await model.inputValue()});
 }catch(e){cases.push({name:'exception',error:String(e)});console.log('UI_E2E_EXCEPTION',String(e))}
 finally{
  fs.writeFileSync(path.join(out,'e2e_results.json'),JSON.stringify({cases,errors,net,responses,downloads,utc:new Date().toISOString()},null,2));
  console.log('UI_E2E',JSON.stringify({cases,errors,netCount:net.length,responses,downloads}).slice(0,8000));
  await browser.close();
 }
})();
