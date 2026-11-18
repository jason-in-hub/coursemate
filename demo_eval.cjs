// Demo-question eval: runs window.__demoProbe(q) in the real page for each row of ml/data/demo_questions.csv.
// Usage: node demo_eval.cjs ml/data/demo_questions.csv <url> [mini]   (mini: wait for the MiniLM second opinion; needs http)
const {chromium}=require('playwright');const fs=require('fs');
const rows=fs.readFileSync(process.argv[2],'utf8').trim().split('\n').slice(1).map(l=>{const [q,gold,seen]=l.split(',');return{q,gold:gold.split(';'),seen}});
(async()=>{const b=await chromium.launch();const p=await b.newPage();await p.goto(process.argv[3]);if(process.argv[4]==='mini'){await p.waitForSelector('body[data-mini="ready"]',{timeout:120000,state:'attached'});console.log('MiniLM second opinion ready')}
const res=[];for(const r of rows){const o=await p.evaluate(q=>window.__demoProbe(q),r.q);
 // gold may list ESC: escalating is then also acceptable. first = the first passage the student reads (the answer, or the top best match)
 const escOK=r.gold.includes('ESC'),onlyEsc=r.gold.length===1&&escOK;
 const ok=(o.stamp==='ESCALATED'&&escOK)||(o.stamp==='SOURCED'&&r.gold.includes(o.top));
 const at3=(o.stamp==='ESCALATED'&&escOK)||o.shown.some(x=>r.gold.includes(x));
 const first=(o.stamp==='ESCALATED'&&escOK)||(!onlyEsc&&o.shown.length>0&&r.gold.includes(o.shown[0]));res.push({...r,...o,ok,at3,first})}
const S=f=>res.filter(f);const pr=(n,a)=>n+': answered right '+a.filter(x=>x.ok).length+'/'+a.length+', first passage shown is right '+a.filter(x=>x.first).length+'/'+a.length+', right passage among those shown '+a.filter(x=>x.at3).length+'/'+a.length;
console.log(pr('all',res));console.log(pr('unseen',S(x=>x.seen==='new')));
const right=res.filter(x=>x.ok).length,min=+process.env.MIN_RIGHT||0;
const wrong=res.filter(x=>x.stamp==='SOURCED'&&!x.gold.includes(x.top)).length,maxw=process.env.MAX_WRONG===undefined?Infinity:+process.env.MAX_WRONG;
console.log('wrong answers (SOURCED from a passage not in gold): '+wrong);
const bad=(min&&right<min)||wrong>maxw;if(min||maxw<Infinity)console.log(bad?'RED: right '+right+' (min '+min+'), wrong '+wrong+' (max '+maxw+')':'GREEN: right '+right+' >= '+min+', wrong '+wrong+' <= '+maxw);if(bad)process.exitCode=1;
for(const x of res)console.log((x.ok?'ok ':x.first?'~1 ':x.at3?'~  ':'X  ')+x.stamp.padEnd(10)+x.q+'  -> '+x.top+(x.ok?'':'  [gold '+x.gold.join('|')+']'));await b.close()})();