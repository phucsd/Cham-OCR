const {chromium}=require('C:/Users/admin/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const root="E:/Phuc's Data/Github/Cham-OCR";
 const out=root+'/REPORTS/live_v26_audit_2026_11_22/ui';fs.mkdirSync(out,{recursive:true});
 const results=[],pageErrors=[],failedRequests=[];
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox','--disable-gpu']});
 const page=await browser.newPage({viewport:{width:1280,height:950},acceptDownloads:true});
 page.on('pageerror',e=>pageErrors.push(String(e)));
 page.on('requestfailed',r=>failedRequests.push({url:r.url(),message:r.failure()?.errorText}));
 async function step(name,fn){try{const result=await fn();results.push({name,ok:true,...result});console.log('OK',name,JSON.stringify(result).slice(0,300))}catch(e){results.push({name,ok:false,error:String(e)});console.log('FAIL',name,String(e).slice(0,200))}}
 try{
  await page.goto('https://ocr.cham.asia',{waitUntil:'domcontentloaded',timeout:35000});
  await page.locator('#model-select').selectOption('v24_1_r4');
  await page.locator('#file-input').setInputFiles(path.join(root,'ocr-benchmark/data/images/test_006_lvl1.png'));
  await page.waitForTimeout(900);
  await step('transcribe',async()=>{
    await page.locator('#btn-run').waitFor({state:'visible',timeout:12000});
    const [resp]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/ocr'),{timeout:90000}),page.locator('#btn-run').click()]);
    await page.waitForTimeout(600);
    return {status:resp.status(),lines:(await page.locator('body').innerText()).match(/\b\d+ textlines\b/)?.[0]||null}
  });
  const targets=[['txt','Unicode Plaintext'],['md','Scholarly Markdown Report'],['json','Structured JSON'],['xml','Archival PAGE-XML'],['zip','Active Learning Dataset']];
  for(const [ext,txt] of targets){
    await step('export_'+ext,async()=>{
      const open=page.locator('button[title*="Export"]').first();
      await open.click();
      const btn=page.getByRole('button',{name:new RegExp(txt)});
      await btn.waitFor({state:'visible',timeout:7000});
      const [dl]=await Promise.all([page.waitForEvent('download',{timeout:18000}),btn.click()]);
      const name=dl.suggestedFilename();const target=path.join(out,'e2e_'+ext+'_'+name.replace(/[<>:"/\\|?*]/g,'_'));
      await dl.saveAs(target);
      return {filename:name,bytes:fs.statSync(target).size,path:target}
    });
  }
  await page.keyboard.press('Escape');
  await step('pdf_import',async()=>{
    const fixture=path.join(out,'two_page_sample.pdf');
    await page.locator('#file-input').setInputFiles(fixture);
    await page.locator('#pdf-modal').waitFor({state:'visible',timeout:25000});
    const msg=await page.locator('#pdf-modal-info').innerText();
    await page.locator('#pdf-page-range').fill('1-2');
    await page.locator('#btn-confirm-pdf').click();
    await page.locator('#pdf-modal').waitFor({state:'hidden',timeout:40000});
    await page.waitForTimeout(800);
    const txt=await page.locator('body').innerText();
    return {info:msg,queueVisible:txt.slice(0,350),hasPage1:txt.includes('_p001.png'),hasPage2:txt.includes('_p002.png')}
  });
 }catch(e){results.push({name:'uncaught',ok:false,error:String(e)})}
 finally{
  fs.writeFileSync(path.join(out,'exports_pdf_results.json'),JSON.stringify({results,pageErrors,failedRequests,utc:new Date().toISOString()},null,2));
  console.log('FINAL_UI',JSON.stringify({resultCount:results.length,failed:results.filter(x=>!x.ok).length,pageErrors,failedRequests}).slice(0,2500));
  await browser.close();
 }
})();
