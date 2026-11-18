// Demo-question eval: runs window.__demoProbe(q) in the real page for each row of ml/data/demo_questions.csv.
// Usage: node demo_eval.cjs ml/data/demo_questions.csv <url> [mini]   (mini: wait for the MiniLM second opinion; needs http)
const {chromium}=require('playwright');const fs=require('fs');
const rows=fs.readFileSync(process.argv[2],'utf8').trim().split('\n').slice(1).map(l=>{const [q,gold,seen]=l.split(',');return{q,gold:gold.split(';'),seen}});
(async()=>{const b=await chromium.launch();const p=await b.newPage();await p.goto(process.argv[3]);if(process.argv[4]==='mini'){await p.waitForSelector('body[data-mini="ready"]',{timeout:120000,state:'attached'});console.log('MiniLM second opinion ready')}
const res=[];for(const r of rows){const o=await p.evaluate(q=>window.__demoProbe(q),r.q);
 const esc=r.gold[0]==='ESC';const ok=esc?o.stamp==='ESCALATED':(o.stamp==='SOURCED'&&r.gold.includes(o.top));
 const at3=esc?o.stamp==='ESCALATED':o.shown.some(x=>r.gold.includes(x));res.push({...r,...o,ok,at3})}
const S=f=>res.filter(f);const pr=(n,a)=>n+': answered right '+a.filter(x=>x.ok).length+'/'+a.length+', right passage shown '+a.filter(x=>x.at3).length+'/'+a.length;
console.log(pr('all',res));console.log(pr('unseen',S(x=>x.seen==='new')));
const right=res.filter(x=>x.ok).length,min=+process.env.MIN_RIGHT||0;
if(min)console.log(right>=min?'GREEN: '+right+' >= MIN_RIGHT '+min:'RED: '+right+' < MIN_RIGHT '+min);if(min&&right<min)process.exitCode=1;
for(const x of res)console.log((x.ok?'ok ':x.at3?'~  ':'X  ')+x.stamp.padEnd(10)+x.q+'  -> '+x.top+(x.ok?'':'  [gold '+x.gold.join('|')+']'));await b.close()})();