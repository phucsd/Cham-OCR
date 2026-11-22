const fs=require('fs'),path=require('path');
const {chromium}=require('C:/Users/admin/node_modules/playwright');
const root="E:/Phuc's Data/Github/Cham-OCR/REPORTS/live_v26_audit_2026_11_22";
const md=fs.readFileSync(path.join(root,'FINAL_REPORT_VI.md'),'utf8').replace(/\r/g,'');
const out=path.join(root,'Bao_cao_kiem_thu_OCR_Cham_V26_2026-11-22.pdf');
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function fmt(s){
 let src=esc(s);
 src=src.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\*([^*]+)\*/g,'<em>$1</em>');
 src=src.replace(/!\[([^\]]*)\]\(([^)]*)\)/g,(_all,alt,rel)=>{
   const img=path.resolve(root,rel);
   if(!img.startsWith(root)||!fs.existsSync(img))return '(Chart unavailable: '+esc(rel)+')';
   return '<img class="chart" alt="'+alt+'" src="data:image/png;base64,'+fs.readFileSync(img).toString('base64')+'"/>';
 });
 return src;
}
const lines=md.split('\n'),html=[];let i=0;
while(i<lines.length){
 const ln=lines[i].trim();
 if(!ln){i++;continue}
 const head=/^(#{1,4}) (.*)$/.exec(ln);
 if(head){html.push('<h'+head[1].length+'>'+fmt(head[2])+'</h'+head[1].length+'>');i++;continue}
 if(ln.startsWith('|')&&i+1<lines.length&&/^\|[\s\-:|]+\|?$/.test(lines[i+1].trim())){
  const split=l=>l.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(s=>s.trim());
  html.push('<table><thead><tr>'+split(ln).map(s=>'<th>'+fmt(s)+'</th>').join('')+'</tr></thead><tbody>');
  i+=2;
  while(i<lines.length&&lines[i].trim().startsWith('|')){
    html.push('<tr>'+split(lines[i]).map(s=>'<td>'+fmt(s)+'</td>').join('')+'</tr>');i++;
  }
  html.push('</tbody></table>');continue;
 }
 if(/^[-*] /.test(ln)||/^\d+\.\s+/.test(ln)){
  const ordered=/^\d+\.\s+/.test(ln);html.push(ordered?'<ol>':'<ul>');
  while(i<lines.length&&(ordered?/^\d+\.\s+/:/^[-*] /).test(lines[i].trim())){
   html.push('<li>'+fmt(lines[i].trim().replace(/^(\d+\.\s+|[-*] )/,''))+'</li>');i++;
  }
  html.push(ordered?'</ol>':'</ul>');continue;
 }
 if(/^!\[/.test(ln)){html.push('<figure>'+fmt(ln)+'</figure>');i++;continue}
 if(/^---+$/.test(ln)){html.push('<hr/>');i++;continue}
 const arr=[];
 while(i<lines.length&&lines[i].trim()&&!/^(#{1,4} |[-*] |\d+\. |\|.*\||!\[)/.test(lines[i].trim())){arr.push(lines[i].trim());i++}
 if(!arr.length){html.push('<p>'+fmt(ln)+'</p>');i++;continue}
 html.push('<p>'+fmt(arr.join(' '))+'</p>');
}
const css=[
 '@page {size:A4; margin:16mm 12mm 16mm 12mm}',
 'html {font-family:"Segoe UI","Arial","V1_EFEO Cam AiaTrang",sans-serif;color:#182434;font-size:9px;line-height:1.4}',
 'body{margin:0 auto;max-width:100%}',
 'h1{font-size:21px;color:#0a4963;border-bottom:2px solid #16758e;padding-bottom:12px;margin-bottom:22px}',
 'h2{font-size:15px;margin:24px 0 10px;color:#0c536e;break-after:avoid;border-bottom:1px solid #cadde5;padding-bottom:5px}',
 'h3{font-size:12px;margin:14px 0 7px;color:#235773;break-after:avoid}',
 'p{margin:7px 0;orphans:3;widows:3}strong{color:#0d4158}',
 'table{width:100%;border-collapse:collapse;font-size:7px;margin:7px 0 12px;page-break-inside:auto;table-layout:auto}',
 'th,td{border:1px solid #d8e3e9;padding:4px 5px;vertical-align:top;overflow-wrap:anywhere}',
 'th{background:#e8f0f4;color:#123c55;text-align:left}tr:nth-child(even) td{background:#f7fafb}',
 'tr{page-break-inside:avoid;break-inside:avoid}li{margin:4px 0}ol,ul{padding-left:25px}',
 'figure{margin:18px 0;text-align:center;page-break-inside:avoid}img.chart{max-width:99%;max-height:225mm;object-fit:contain}'
].join('\n');
(async()=>{
 const b=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--no-sandbox']});
 const pg=await b.newPage();
 await pg.setContent('<!DOCTYPE html><html lang="vi"><meta charset="utf-8"/><style>'+css+'</style><body>'+html.join('\n')+'</body></html>',{waitUntil:'load'});
 await pg.pdf({path:out,format:'A4',printBackground:true,preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<div></div>',footerTemplate:'<div style="width:100%;font-size:8px;color:#888;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'});
 await b.close();
 const fd=fs.openSync(out,'r'),h=Buffer.alloc(5);fs.readSync(fd,h,0,5,0);fs.closeSync(fd);
 console.log('PDF_CREATED',out,'bytes',fs.statSync(out).size,'signature',h.toString());
})().catch(e=>{console.error('PDF_FAILED',String(e));process.exit(1)});
