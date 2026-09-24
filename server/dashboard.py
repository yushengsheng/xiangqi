"""浅色双平台棋盘看板，附带下方 AI 对战控件。"""

WEB_DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>象棋盘面同步</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;gap:14px;overflow-x:hidden;overflow-y:auto;background:linear-gradient(145deg,#eef3ef,#dce7df);font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;padding:12px 12px 28px}
.board-shell{width:min(92vw,calc(78vh * .9));aspect-ratio:9/10;flex:0 0 auto;filter:drop-shadow(0 20px 25px rgba(48,76,56,.20))}
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
.btn.reset{background:#8f3f37;color:#fff7ee}
.btn:disabled{opacity:.45;cursor:default}
.setting{display:flex;align-items:center;gap:6px;font-size:13px;color:#52604f}.setting input,.setting select{width:78px;border:1px solid #b8c4b1;border-radius:8px;padding:6px 8px;background:#fffdf7;color:#283526;font-weight:700}.setting select{width:112px}
.think{width:8px;height:8px;border-radius:50%;background:#ba382d;display:inline-block;margin-right:6px;box-shadow:0 0 0 0 rgba(186,56,45,.6);animation:pulse 1.2s infinite}
.think[hidden]{display:none}
@keyframes pulse{70%{box-shadow:0 0 0 8px rgba(186,56,45,0)}}
</style>
</head>
<body>
<main class="board-shell" aria-label="中国象棋实时盘面"><div id="board"></div></main>
<section class="dock">
  <div class="row">
    <span class="badge" id="side-badge">AI 执下方</span>
    <span class="status" id="status"><span class="think" id="thinking" hidden></span><span id="status-message">正在连接盘面…</span></span>
    <span class="meta" id="score"></span>
  </div>
  <div class="row">
    <span class="suggest" id="suggest">等待建议</span>
    <span class="meta">实时只读 · 合法走子跟踪</span>
  </div>
  <div class="row">
    <div class="setting"><label for="ai-level">AI 强度</label><select id="ai-level"><option value="eco">节能</option><option value="normal">普通</option><option value="advanced">进阶</option><option value="expert">高级</option></select></div>
    <button class="btn reset" id="btn-reset" type="button">清理缓存并重新读取对局</button>
  </div>
</section>
<script>
(()=>{
const text={r_k:"帥",r_a:"仕",r_b:"相",r_n:"傌",r_r:"俥",r_c:"炮",r_p:"兵",b_k:"将",b_a:"士",b_b:"象",b_n:"馬",b_r:"車",b_c:"砲",b_p:"卒"};
const root=document.querySelector('#board');
const x=c=>70+c*95,y=r=>70+r*95;
let board=[], ai=null, lastMove=null, gameName='象棋';
let latestSessionRevision=-1,latestStateTimestamp=0,latestServerId='',ws=null,reconnectTimer=null;
let syncDescription='',recognitionPending=false,captureStatus='waiting';
let resetRequestedAt=0,manualResetPending=false;
let resetWatchdog=null;
const pieceNodes=new Map();
let lastArrowSignature='',suggestionArrowSignature='';

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

function initBoard(){
  root.innerHTML=`<svg viewBox="0 0 900 1000" role="img" aria-label="实时象棋盘面"><defs>
    <linearGradient id="wood" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#d8dfc8"/><stop offset="1" stop-color="#aab59f"/></linearGradient>
    <radialGradient id="face"><stop stop-color="#fff9dd"/><stop offset=".72" stop-color="#f7ddb1"/><stop offset="1" stop-color="#c99757"/></radialGradient>
    <filter id="shadow" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="2" dy="5" stdDeviation="4" flood-color="#42503d" flood-opacity=".38"/></filter>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#c45c38"/></marker>
  </defs>${grid()}<g id="last-arrow"></g><g id="suggestion-arrow"></g><g id="pieces"></g></svg>`;
}

function makePiece(p,r,c){
  const red=p.startsWith('r_'),ink=red?'#ba382d':'#242922',rim=red?'#d39a62':'#b98952';
  const node=document.createElementNS('http://www.w3.org/2000/svg','g');
  node.setAttribute('transform',`translate(${x(c)} ${y(r)})`);
  node.setAttribute('filter','url(#shadow)');
  node.setAttribute('data-piece',p);
  node.innerHTML=`<circle r="43" fill="${rim}" stroke="#8b603d" stroke-width="3"/>
    <circle r="35" fill="url(#face)" stroke="#f9edc5" stroke-width="3"/>
    <text class="piece-char" fill="${ink}">${text[p]}</text>`;
  return node;
}

function render(){
  const pieces=root.querySelector('#pieces');
  if(!pieces)return;
  for(let r=0;r<10;r++){
    for(let c=0;c<9;c++){
      const p=Array.isArray(board)&&Array.isArray(board[r])?board[r][c]:null;
      const key=`${r}-${c}`,previous=pieceNodes.get(key);
      if(previous&&previous.getAttribute('data-piece')===p)continue;
      if(previous){previous.remove();pieceNodes.delete(key);}
      if(p&&text[p]){
        const node=makePiece(p,r,c);
        pieces.appendChild(node);
        pieceNodes.set(key,node);
      }
    }
  }
  const sug=ai&&ai.suggestion;
  const lastArrow=lastMove&&lastMove.from&&lastMove.to
    ?arrow(lastMove.from,lastMove.to,'#5b7c9a'):'';
  const suggestionArrow=sug&&sug.from&&sug.to
    ?arrow(sug.from,sug.to,'#c45c38'):'';
  if(lastArrow!==lastArrowSignature){
    root.querySelector('#last-arrow').innerHTML=lastArrow;
    lastArrowSignature=lastArrow;
  }
  if(suggestionArrow!==suggestionArrowSignature){
    root.querySelector('#suggestion-arrow').innerHTML=suggestionArrow;
    suggestionArrowSignature=suggestionArrow;
  }
}

function renderDock(){
  const badge=document.getElementById('side-badge');
  const status=document.getElementById('status');
  const statusMessage=document.getElementById('status-message');
  const thinking=document.getElementById('thinking');
  const suggest=document.getElementById('suggest');
  const score=document.getElementById('score');
  const levelInput=document.getElementById('ai-level');
  const resetButton=document.getElementById('btn-reset');
  const setText=(node,value)=>{if(node.textContent!==value)node.textContent=value;};
  if(!ai){setText(statusMessage,'正在连接盘面…');thinking.hidden=true;return;}
  const engineLabel='Pikafish';
  setText(badge,`${gameName} · 跟随实盘 · AI 执下方 · ${ai.bottom_label||''} · ${engineLabel}`);
  const rebuilding=manualResetPending;
  setText(statusMessage,rebuilding?'正在后台重新读取当前对局，当前盘面会暂时保留…':((recognitionPending&&syncDescription)||ai.error||ai.status||syncDescription||''));
  thinking.hidden=!ai.thinking;
  status.title=ai.engine_error||'';
  setText(suggest,rebuilding?'正在重新读取盘面':((ai.suggestion&& (ai.suggestion.zh||ai.suggestion.uci)) || (ai.to_move==='unknown'?'等待确认回合':(ai.play_mode?'请走上方棋子':'等待下方建议'))));
  if(ai.suggestion&&typeof ai.suggestion.score==='number'){
    const v=(ai.suggestion.score/100).toFixed(2);
    setText(score,`${engineLabel} · 深度 ${ai.suggestion.depth||0} · ${(ai.suggestion.nodes||0).toLocaleString()}节点 · 评估 ${v}`);
  }else setText(score,'实盘助手');
  if(document.activeElement!==levelInput){
    levelInput.value=(ai.time_ms>=8000&&ai.engine_threads>=6)?'expert':((ai.time_ms>=3000&&ai.engine_threads>=4)?'advanced':((ai.time_ms>=1000&&ai.engine_threads>=2)?'normal':'eco'));
  }
  if(resetButton){
    resetButton.disabled=rebuilding;
    setText(resetButton,rebuilding?'正在重新读取…':'清理缓存并重新读取对局');
  }
}

async function post(path, body){
  const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  const data=await res.json();
  if(!res.ok) throw new Error(data.error||`HTTP ${res.status}`);
  return data;
}

function applyAi(data){
  const revision=Number(data.session_revision||0);
  const timestamp=Number(data.timestamp||0);
  const serverId=String(data.server_instance_id||'');
  if(serverId&&serverId!==latestServerId){
    if(latestServerId&&timestamp&&timestamp<latestStateTimestamp)return;
    latestServerId=serverId;
    latestSessionRevision=-1;
    latestStateTimestamp=0;
  }
  if(revision<latestSessionRevision)return;
  if(revision===latestSessionRevision&&timestamp&&timestamp<latestStateTimestamp)return;
  if(revision>latestSessionRevision){
    latestSessionRevision=revision;
    latestStateTimestamp=0;
  }
  if(timestamp)latestStateTimestamp=Math.max(latestStateTimestamp,timestamp);
  if(data.game_name) gameName=data.game_name;
  syncDescription=(data.recognition_pending&&data.recognition_rejection)||data.description||'';
  recognitionPending=Boolean(data.recognition_pending);
  captureStatus=data.capture_status||captureStatus;
  manualResetPending=Boolean(data.manual_reset_pending);
  if(!manualResetPending){
    resetRequestedAt=0;
    if(resetWatchdog){clearTimeout(resetWatchdog);resetWatchdog=null;}
  }
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

function showError(error){
  console.error(error);
  document.getElementById('thinking').hidden=true;
  document.getElementById('status-message').textContent=`操作失败：${error.message}`;
}
document.getElementById('ai-level').onchange=async ev=>{
  const presets={
    eco:{time_ms:700,max_depth:60,engine_threads:1,engine_hash_mb:32},
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
    document.getElementById('status-message').textContent=`AI 强度切换失败：${error.message}`;
  }finally{select.disabled=false;}
};
document.getElementById('btn-reset').onclick=async ev=>{
  const button=ev.currentTarget;
  resetRequestedAt=Date.now();
  button.disabled=true;
  button.textContent='正在重新读取…';
  if(resetWatchdog)clearTimeout(resetWatchdog);
  resetWatchdog=setTimeout(async()=>{
    if(!manualResetPending)return;
    await refreshSnapshot();
    if(manualResetPending){
      manualResetPending=false;
      resetRequestedAt=0;
      renderDock();
      document.getElementById('status-message').textContent='重读等待超时，请检查棋盘画面后重试';
    }
    resetWatchdog=null;
  },8500);
  try{
    latestStateTimestamp=0;
    applyAi(await post('/api/live/reset',{}));
  }catch(error){
    resetRequestedAt=0;
    if(resetWatchdog){clearTimeout(resetWatchdog);resetWatchdog=null;}
    manualResetPending=false;
    renderDock();showError(error);
  }
};
function connect(){
  if(ws&&(ws.readyState===WebSocket.OPEN||ws.readyState===WebSocket.CONNECTING))return;
  ws=new WebSocket(`ws://${location.hostname}:__WS_PORT__`);
  ws.onmessage=e=>{try{applyAi(JSON.parse(e.data))}catch(error){showError(error);}};
  ws.onerror=()=>{try{ws.close()}catch(_error){}};
  ws.onclose=()=>{
    ws=null;
    if(reconnectTimer)clearTimeout(reconnectTimer);
    reconnectTimer=setTimeout(connect,600);
  };
}
async function refreshSnapshot(){
  try{
    const res=await fetch('/api/status',{cache:'no-store'});
    if(res.ok)applyAi(await res.json());
  }catch(_error){/* WebSocket reconnect and the next snapshot will recover. */}
}
async function initialize(){
  // 每次打开/刷新看板都回到真实盘面，避免上一次模拟对弈残留。
  try{applyAi(await post('/api/ai/follow',{}));}catch(error){showError(error);}
  connect();
  await refreshSnapshot();
  setInterval(refreshSnapshot,600);
}
initBoard(); render(); initialize();
})();
</script>
</body>
</html>
"""
