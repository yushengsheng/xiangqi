"""浅色双平台棋盘看板，附带下方 AI 对战控件。"""

WEB_DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>象棋盘面同步</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;gap:14px;overflow-x:hidden;overflow-y:auto;background:linear-gradient(145deg,#eef3ef,#dce7df);font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;padding:12px 12px 28px}
.board-shell{width:min(92vw,calc(78vh * .9));aspect-ratio:9/10;flex:0 0 auto;filter:drop-shadow(0 20px 25px rgba(48,76,56,.20));cursor:pointer}
svg{display:block;width:100%;height:100%}
.grid-line{stroke:#6d7767;stroke-width:4;fill:none;stroke-linecap:round}
.river-label{fill:#78816e;font-family:"STKaiti","KaiTi",serif;font-size:48px;font-weight:700;letter-spacing:13px;opacity:.72}
.piece-char{font-family:"STKaiti","KaiTi","Songti SC",serif;font-weight:700;font-size:55px;text-anchor:middle;dominant-baseline:central}
.dock{width:min(92vw,calc(78vh * .9));background:rgba(255,252,244,.92);border:1px solid #c9d2c0;border-radius:16px;padding:12px 14px;display:grid;gap:8px;box-shadow:0 10px 24px rgba(48,76,56,.12)}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
.badge{font-size:13px;font-weight:700;color:#35503a;background:#e7efe3;border-radius:999px;padding:4px 10px}
.status{font-size:14px;color:#3d4a3b;flex:1;min-width:180px}
.suggest{font-family:"STKaiti","KaiTi",serif;font-size:22px;font-weight:700;color:#ba382d}
.meta{font-size:12px;color:#6d7767}
.btn{border:0;border-radius:10px;padding:7px 12px;font-weight:700;cursor:pointer;background:#35503a;color:#f7f3e6}
.btn.alt{background:#ece7d6;color:#35503a}
.btn:disabled{opacity:.45;cursor:default}
.setting{display:flex;align-items:center;gap:6px;font-size:13px;color:#52604f}.setting input,.setting select{width:78px;border:1px solid #b8c4b1;border-radius:8px;padding:6px 8px;background:#fffdf7;color:#283526;font-weight:700}.setting select{width:112px}
.think{width:8px;height:8px;border-radius:50%;background:#ba382d;display:inline-block;margin-right:6px;box-shadow:0 0 0 0 rgba(186,56,45,.6);animation:pulse 1.2s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 8px rgba(186,56,45,0)}}
</style>
</head>
<body>
<main class="board-shell" aria-label="中国象棋实时盘面"><div id="board"></div></main>
<section class="dock">
  <div class="row">
    <span class="badge" id="side-badge">AI 执下方</span>
    <span class="status" id="status">正在连接盘面…</span>
    <span class="meta" id="score"></span>
  </div>
  <div class="row">
    <span class="suggest" id="suggest">等待建议</span>
    <div>
      <button class="btn" id="btn-play" type="button" title="仅在网页棋盘中模拟对弈，不用于正在进行的实盘">棋盘模拟</button>
      <button class="btn alt" id="btn-follow" type="button">跟随实盘</button>
      <button class="btn alt" id="btn-undo" type="button" disabled>回退一步</button>
      <button class="btn alt" id="btn-turn" type="button">轮到上方</button>
    </div>
  </div>
  <div class="row">
    <div class="setting"><label for="ai-level">AI 强度</label><select id="ai-level"><option value="normal">普通</option><option value="advanced">进阶</option><option value="expert">高级</option></select></div>
    <span class="meta">选择后立即生效</span>
  </div>
</section>
<script>
(()=>{
const text={r_k:"帥",r_a:"仕",r_b:"相",r_n:"傌",r_r:"俥",r_c:"炮",r_p:"兵",b_k:"将",b_a:"士",b_b:"象",b_n:"馬",b_r:"車",b_c:"砲",b_p:"卒"};
const root=document.querySelector('#board');
const x=c=>70+c*95,y=r=>70+r*95;
let board=[], ai=null, selected=null, lastMove=null, gameName='象棋';

function grid(){
  let s='<rect x="24" y="24" width="852" height="952" rx="28" fill="url(#wood)" stroke="#a4ab94" stroke-width="10"/><rect x="52" y="52" width="796" height="896" rx="10" fill="#b9c1aa" stroke="#8e9882" stroke-width="5"/>';
  for(let r=0;r<10;r++)s+=`<path class="grid-line" d="M70 ${y(r)} H830"/>`;
  for(let c=0;c<9;c++){
    const p=x(c);
    s+=c===0||c===8?`<path class="grid-line" d="M${p} 70 V925"/>`:`<path class="grid-line" d="M${p} 70 V450 M${p} 545 V925"/>`;
  }
  return s+'<path class="grid-line" d="M355 70 L545 260 M545 70 L355 260 M355 735 L545 925 M545 735 L355 925"/><text class="river-label" x="143" y="512">楚 河</text><text class="river-label" x="592" y="512">汉 界</text>';
}

function arrow(from,to,color){
  if(!from||!to)return '';
  return `<line x1="${x(from.col)}" y1="${y(from.row)}" x2="${x(to.col)}" y2="${y(to.row)}" stroke="${color}" stroke-width="10" stroke-linecap="round" marker-end="url(#arr)" opacity=".9"/>`;
}

function render(){
  let s=`<svg viewBox="0 0 900 1000" role="img" aria-label="实时象棋盘面"><defs>
    <linearGradient id="wood" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#d8dfc8"/><stop offset="1" stop-color="#aab59f"/></linearGradient>
    <radialGradient id="face"><stop stop-color="#fff9dd"/><stop offset=".72" stop-color="#f7ddb1"/><stop offset="1" stop-color="#c99757"/></radialGradient>
    <filter id="shadow" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="2" dy="5" stdDeviation="4" flood-color="#42503d" flood-opacity=".38"/></filter>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#c45c38"/></marker>
  </defs>${grid()}`;
  const sug=ai&&ai.suggestion;
  if(lastMove&&lastMove.from&&lastMove.to)s+=arrow(lastMove.from,lastMove.to,'#5b7c9a');
  if(sug&&sug.from&&sug.to)s+=arrow(sug.from,sug.to,'#c45c38');
  const targets=new Set();
  if(selected&&ai&&ai.legal_moves){
    ai.legal_moves.forEach(m=>{
      if(m.from&&m.from.row===selected.row&&m.from.col===selected.col){
        targets.add(m.to.row+','+m.to.col);
        s+=`<circle cx="${x(m.to.col)}" cy="${y(m.to.row)}" r="${m.captured?16:10}" fill="${m.captured?'none':'#35503a'}" stroke="#35503a" stroke-width="5" opacity=".55"/>`;
      }
    });
  }
  (Array.isArray(board)?board:[]).forEach((row,r)=>(row||[]).forEach((p,c)=>{
    if(!p||!text[p])return;
    const red=p.startsWith('r_'),ink=red?'#ba382d':'#242922',rim=red?'#d39a62':'#b98952';
    const isSel=selected&&selected.row===r&&selected.col===c;
    s+=`<g transform="translate(${x(c)} ${y(r)})" filter="url(#shadow)">
      <circle r="43" fill="${rim}" stroke="${isSel?'#c45c38':'#8b603d'}" stroke-width="${isSel?6:3}"/>
      <circle r="35" fill="url(#face)" stroke="#f9edc5" stroke-width="3"/>
      <text class="piece-char" fill="${ink}">${text[p]}</text></g>`;
  }));
  root.innerHTML=s+'</svg>';
}

function renderDock(){
  const badge=document.getElementById('side-badge');
  const status=document.getElementById('status');
  const suggest=document.getElementById('suggest');
  const score=document.getElementById('score');
  const playBtn=document.getElementById('btn-play');
  const undoBtn=document.getElementById('btn-undo');
  const levelInput=document.getElementById('ai-level');
  if(!ai){status.textContent='正在连接盘面…';return;}
  const engineLabel=ai.active_engine==='pikafish'?'Pikafish':(ai.active_engine==='builtin-fallback'?'内置回退':'内置');
  badge.textContent=`${gameName} · AI 执下方 · ${ai.bottom_label||''} · ${engineLabel}`;
  status.textContent=ai.error||ai.status||'';
  if(ai.thinking){const dot=document.createElement('span');dot.className='think';status.prepend(dot);}
  status.title=ai.engine_error||'';
  suggest.textContent=(ai.suggestion&& (ai.suggestion.zh||ai.suggestion.uci)) || (ai.to_move==='unknown'?'等待确认回合':(ai.play_mode?'请走上方棋子':'等待下方建议'));
  if(ai.suggestion&&typeof ai.suggestion.score==='number'){
    const v=(ai.suggestion.score/100).toFixed(2);
    score.textContent=`${engineLabel} · 深度 ${ai.suggestion.depth||0} · ${(ai.suggestion.nodes||0).toLocaleString()}节点 · 评估 ${v}`;
  }else score.textContent=ai.play_mode?'本地对弈：你走上方':'实盘助手';
  playBtn.textContent=ai.play_mode?'重开模拟':'棋盘模拟';
  undoBtn.disabled=!ai.play_mode||!ai.can_undo;
  undoBtn.title=ai.variation_count?`已保存 ${ai.variation_count} 条变招分支`:'';
  if(document.activeElement!==levelInput){
    levelInput.value=(ai.time_ms>=8000&&ai.engine_threads>=6)?'expert':((ai.time_ms>=3000&&ai.engine_threads>=4)?'advanced':'normal');
  }
}

function cellFromEvent(ev){
  const svg=root.querySelector('svg');
  if(!svg)return null;
  const pt=svg.createSVGPoint();
  pt.x=ev.clientX; pt.y=ev.clientY;
  const p=pt.matrixTransform(svg.getScreenCTM().inverse());
  const c=Math.round((p.x-70)/95), r=Math.round((p.y-70)/95);
  if(r<0||r>9||c<0||c>8)return null;
  return {row:r,col:c};
}

async function post(path, body){
  const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  const data=await res.json();
  if(!res.ok) throw new Error(data.error||`HTTP ${res.status}`);
  return data;
}

function applyAi(data){
  if(data.game_name) gameName=data.game_name;
  if(data.ai) ai=data.ai;
  if(ai&&ai.play_mode&&ai.board) board=ai.board;
  else if(data.board) board=data.board;
  lastMove=data.last_move||null;
  if(ai&&ai.play_mode&&ai.history&&ai.history.length){
    const mv=data.ai.history[data.ai.history.length-1];
    if(mv&&mv.from) lastMove=mv;
  }
  render(); renderDock();
}

root.addEventListener('click', async ev=>{
  if(!ai||!ai.play_mode) return;
  const cell=cellFromEvent(ev);
  if(!cell) return;
  const piece=board[cell.row]&&board[cell.row][cell.col];
  if(selected){
    if(selected.row===cell.row&&selected.col===cell.col){selected=null;render();return;}
    if(piece&&piece[0]===ai.top_side){selected=cell;render();return;}
    try{
      const data=await post('/api/ai/play',{from:selected,to:cell});
      selected=null;
      applyAi(data);
    }catch(error){showError(error);}
    return;
  }
  if(piece&&piece[0]===ai.top_side){selected=cell;render();}
});

function showError(error){
  console.error(error);
  document.getElementById('status').textContent=`操作失败：${error.message}`;
}
async function action(path,body){
  try{applyAi(await post(path,body));}catch(error){showError(error);}
}

document.getElementById('btn-play').onclick=async()=>{
  if(!ai||!ai.play_mode){
    const ok=window.confirm('棋盘模拟会暂时脱离实盘画面，仅用于网页内对弈。真实对局请使用“跟随实盘”。继续吗？');
    if(!ok)return;
  }
  selected=null;
  await action('/api/ai/start',{});
};
document.getElementById('btn-follow').onclick=async()=>{
  selected=null;
  await action('/api/ai/follow',{});
};
document.getElementById('btn-undo').onclick=async()=>{
  selected=null;
  await action('/api/ai/undo',{});
};
document.getElementById('ai-level').onchange=async ev=>{
  const presets={
    normal:{time_ms:1000,max_depth:60,engine_threads:2,engine_hash_mb:64},
    advanced:{time_ms:3000,max_depth:80,engine_threads:4,engine_hash_mb:128},
    expert:{time_ms:8000,max_depth:100,engine_threads:6,engine_hash_mb:256}
  };
  const select=ev.currentTarget;
  const chosen=select.value;
  select.disabled=true;
  try{applyAi(await post('/api/ai/config',{engine_kind:'pikafish',...presets[chosen]}));}
  catch(error){
    renderDock();
    document.getElementById('status').textContent=`AI 强度切换失败：${error.message}`;
  }finally{select.disabled=false;}
};
document.getElementById('btn-turn').onclick=async()=>{
  if(!ai) return;
  const next=ai.to_move===ai.top_side?ai.bottom_side:ai.top_side;
  await action('/api/ai/turn',{side:next});
};

function connect(){
  const ws=new WebSocket(`ws://${location.hostname}:__WS_PORT__`);
  ws.onmessage=e=>{try{applyAi(JSON.parse(e.data))}catch(error){showError(error);}};
  ws.onclose=()=>setTimeout(connect,1000);
}
render(); connect();
})();
</script>
</body>
</html>
"""
