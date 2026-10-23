/**
 * APEX-Ω · L9 应用层 —— 单文件 HTML 仪表盘
 * ---------------------------------------------------------------------------
 * 零重写 + 零依赖：内联 CSS/JS，无 CDN、无外部字体、无框架。
 * 数据来源：本机 L6 服务（/api/isa、/v1/chat/completions、/healthz）。
 * 打开方式：node scripts/serve.mjs → http://127.0.0.1:8080
 */
export const DASHBOARD_HTML = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>APEX-Ω · 世界旗舰 AI 合流系统</title>
<style>
:root{
  --bg:#0b0e14; --bg2:#11151f; --panel:#151a26; --panel2:#1b2130;
  --line:#242c3d; --fg:#e6ecf7; --fg2:#9aa7bd; --fg3:#6b7891;
  --cyan:#3ddc97; --blue:#58a6ff; --purple:#a371f7; --amber:#e3b341; --red:#f85149;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;height:100%}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}
a{color:var(--blue);text-decoration:none}
.wrap{display:grid;grid-template-columns:264px 1fr;min-height:100vh}
/* ---- sidebar ---- */
.side{background:var(--bg2);border-right:1px solid var(--line);padding:20px 14px;position:sticky;top:0;height:100vh;overflow-y:auto}
.brand{display:flex;align-items:baseline;gap:8px;margin-bottom:4px}
.brand h1{font-size:19px;margin:0;font-weight:700;letter-spacing:.3px}
.brand .omega{color:var(--cyan)}
.sub{color:var(--fg3);font-size:11.5px;margin-bottom:22px;font-family:var(--mono)}
.nav{display:flex;flex-direction:column;gap:2px;margin-bottom:24px}
.nav button{background:none;border:none;color:var(--fg2);text-align:left;padding:9px 11px;border-radius:7px;cursor:pointer;font-size:13.5px;font-family:inherit;transition:.13s;display:flex;align-items:center;gap:9px}
.nav button:hover{background:var(--panel);color:var(--fg)}
.nav button.on{background:var(--panel2);color:var(--fg);box-shadow:inset 2px 0 0 var(--cyan)}
.nav .ico{width:16px;text-align:center;opacity:.85;font-size:13px}
.side h4{font-size:10.5px;text-transform:uppercase;letter-spacing:1.3px;color:var(--fg3);margin:0 0 8px;padding-left:11px;font-weight:600}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:11px 12px;margin-bottom:8px}
.stat .k{font-size:10.5px;color:var(--fg3);text-transform:uppercase;letter-spacing:.8px;font-family:var(--mono)}
.stat .v{font-size:19px;font-weight:700;font-family:var(--mono);margin-top:2px}
.stat .v small{font-size:11px;color:var(--fg3);font-weight:400}
/* ---- main ---- */
.main{padding:24px 28px 60px;max-width:1180px}
.pane{display:none}
.pane.on{display:block}
.phead{margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--line)}
.phead h2{margin:0 0 4px;font-size:20px;font-weight:650}
.phead p{margin:0;color:var(--fg2);font-size:13px}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.g3{grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 16px}
.card h3{margin:0 0 10px;font-size:14px;font-weight:620;display:flex;justify-content:space-between;align-items:center;gap:8px}
.card h3 .tag{font-size:10.5px;font-family:var(--mono);color:var(--fg3);background:var(--bg2);border:1px solid var(--line);padding:1px 7px;border-radius:20px;font-weight:400}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--fg3);font-size:11px;text-transform:uppercase;letter-spacing:.7px;font-weight:600;font-family:var(--mono)}
td.mono{font-family:var(--mono);font-size:12px}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--panel2)}
.ok{color:var(--cyan);font-weight:600}
.bad{color:var(--red);font-weight:600}
.muted{color:var(--fg3)}
.bar{height:5px;background:var(--bg);border-radius:3px;overflow:hidden;min-width:60px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--blue))}
/* chat */
.chatbox{display:flex;flex-direction:column;height:calc(100vh - 210px);min-height:440px}
.msgs{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:12px;padding:4px 2px 12px}
.msg{max-width:86%;padding:11px 14px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
.msg.u{align-self:flex-end;background:var(--panel2);border:1px solid var(--line)}
.msg.a{align-self:flex-start;background:var(--panel);border:1px solid var(--line)}
.msg .who{font-size:10.5px;color:var(--fg3);font-family:var(--mono);margin-bottom:4px;letter-spacing:.6px}
.composer{display:flex;gap:9px;margin-top:10px}
.composer input{flex:1;background:var(--bg2);border:1px solid var(--line);border-radius:9px;padding:11px 13px;color:var(--fg);font-family:inherit;font-size:13.5px;outline:none}
.composer input:focus{border-color:var(--cyan)}
.btn{background:var(--cyan);border:none;color:#06210f;padding:11px 20px;border-radius:9px;cursor:pointer;font-weight:650;font-size:13.5px;font-family:inherit;white-space:nowrap}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.ghost{background:transparent;color:var(--fg2);border:1px solid var(--line)}
.btn.ghost:hover{color:var(--fg);border-color:var(--fg3)}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}
select,input[type=number]{background:var(--bg2);border:1px solid var(--line);border-radius:7px;padding:7px 9px;color:var(--fg);font-family:var(--mono);font-size:12px;outline:none}
/* isa */
.filters{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:12px}
.chip{background:var(--panel);border:1px solid var(--line);color:var(--fg2);border-radius:20px;padding:4px 12px;font-size:12px;cursor:pointer;font-family:var(--mono);transition:.12s}
.chip:hover{border-color:var(--fg3);color:var(--fg)}
.chip.on{background:var(--cyan);border-color:var(--cyan);color:#06210f;font-weight:600}
.scroll{max-height:520px;overflow-y:auto}
code,.mono{font-family:var(--mono)}
.kv{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line);font-size:12.5px}
.kv:last-child{border:none}
.kv span:first-child{color:var(--fg2)}
.note{background:rgba(61,220,151,.07);border:1px solid rgba(61,220,151,.25);border-left:3px solid var(--cyan);border-radius:7px;padding:10px 13px;font-size:12.5px;color:var(--fg2);margin-bottom:14px}
.note b{color:var(--cyan)}
.note.warn{background:rgba(227,179,65,.07);border-color:rgba(227,179,65,.28);border-left-color:var(--amber)}
.note.warn b{color:var(--amber)}
.spin{display:inline-block;width:11px;height:11px;border:2px solid var(--line);border-top-color:var(--cyan);border-radius:50%;animation:sp .7s linear infinite;vertical-align:-1px;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
.hide{display:none!important}
@media(max-width:860px){.wrap{grid-template-columns:1fr}.side{position:static;height:auto;border-right:none;border-bottom:1px solid var(--line)}.main{padding:18px}}
</style>
</head>
<body>
<div class="wrap">
  <aside class="side">
    <div class="brand"><h1>APEX<span class="omega">-Ω</span></h1></div>
    <div class="sub">世界旗舰 AI 合流系统</div>
    <nav class="nav">
      <button class="on" data-p="dash"><span class="ico">◈</span>总览</button>
      <button data-p="chat"><span class="ico">◉</span>真实推理</button>
      <button data-p="isa"><span class="ico">▤</span>1024 微指令 ISA</button>
      <button data-p="gate"><span class="ico">⚖</span>可证门禁</button>
      <button data-p="arch"><span class="ico">⌘</span>十层架构</button>
    </nav>
    <h4>运行态</h4>
    <div class="stat"><div class="k">模型</div><div class="v" id="s-model" style="font-size:14px">—</div></div>
    <div class="stat"><div class="k">架构</div><div class="v" id="s-arch" style="font-size:15px">—</div></div>
    <div class="stat"><div class="k">ISA 指令</div><div class="v" id="s-isa">—</div></div>
    <div class="stat"><div class="k">已验证微指令</div><div class="v" id="s-ver">—</div></div>
    <div class="stat"><div class="k">吞吐</div><div class="v" id="s-tps">—<small> tok/s</small></div></div>
  </aside>

  <main class="main">
    <!-- 总览 -->
    <section class="pane on" id="p-dash">
      <div class="phead"><h2>总览</h2><p>真实大模型 × 1024 微指令 ISA × 可证门禁</p></div>
      <div class="note"><b>零重写</b>：推理内核 = llama.cpp（世界顶级 GGML 引擎）；算子层 = AXIOM 1024 微指令（上游零改动复用）；旗舰原创的只有中间的<b>路由 / 对拍 / 遥测</b>三层。</div>
      <div class="grid g3" id="dash-cards"></div>
      <div class="grid g2" style="margin-top:14px">
        <div class="card"><h3>世界级踩坑复现 <span class="tag">实测</span></h3>
          <div class="kv"><span>chat wrapper 未套用</span><span class="bad">输出退化为 !!!</span></div>
          <div class="kv"><span>onToken 实参</span><span class="mono">单元素数组</span></div>
          <div class="kv"><span>KV cache 污染</span><span class="bad">同 seed 不一致</span></div>
          <div class="kv"><span>修复</span><span class="ok">QwenChatWrapper + isolate</span></div>
          <div class="kv"><span>门禁化</span><span class="ok">D 门禁常驻捕获</span></div>
        </div>
        <div class="card"><h3>门禁分组 <span class="tag" id="gate-count">—</span></h3>
          <div id="gate-summary" class="muted">加载中…</div>
        </div>
      </div>
    </section>

    <!-- 真实推理 -->
    <section class="pane" id="p-chat">
      <div class="phead"><h2>真实推理</h2><p>OpenAI 兼容接口 · 本机 llama.cpp · 每次推理在 ISA 上留痕</p></div>
      <div class="card chatbox">
        <div class="msgs" id="msgs"></div>
        <div class="row" style="margin-bottom:9px">
          <label class="muted mono" style="font-size:11.5px">temperature</label>
          <input type="number" id="c-temp" value="0.7" step="0.1" min="0" max="2" style="width:64px">
          <label class="muted mono" style="font-size:11.5px">max_tokens</label>
          <input type="number" id="c-max" value="256" step="32" min="16" max="2048" style="width:78px">
          <label class="muted mono" style="font-size:11.5px">seed</label>
          <input type="number" id="c-seed" placeholder="随机" style="width:72px">
          <span class="muted mono" id="c-stat" style="font-size:11.5px;margin-left:auto"></span>
        </div>
        <div class="composer">
          <input id="c-input" placeholder="问点什么…（Enter 发送，Shift+Enter 换行）" autocomplete="off">
          <button class="btn" id="c-send">发送</button>
          <button class="btn ghost" id="c-clear">清空</button>
        </div>
      </div>
    </section>

    <!-- ISA -->
    <section class="pane" id="p-isa">
      <div class="phead"><h2>1024 微指令 ISA</h2><p>16 族 × 16 算子 × 4 执行变体 · 每条携带可独立复算的数学不变量</p></div>
      <div class="grid g2" style="margin-bottom:14px">
        <div class="card"><h3>家族覆盖</h3><div id="isa-fams"></div></div>
        <div class="card"><h3>来源</h3>
          <div class="kv"><span>注册表</span><span class="mono">vendor/axiom/registry.cjs</span></div>
          <div class="kv"><span>参考实现</span><span class="mono">vendor/axiom/core.cjs</span></div>
          <div class="kv"><span>上游</span><span class="mono">CJX0712/axiom-ai</span></div>
          <div class="kv"><span>指令总数</span><span class="mono">1024</span></div>
          <div class="kv"><span>已验证</span><span class="mono" id="isa-ver">—</span></div>
          <div class="kv"><span>数学不变量</span><span class="mono">386 项</span></div>
          <div class="note" style="margin-top:12px">上游注册表有 <b>13 个 op 名跨族复用</b>（如 softmax 同属 ATTEN 与 SAMPLE）。旗舰的路由按 <b>impl 名</b>（core.js 真实导出名）定位，保证唯一可执行锚点。</div>
        </div>
      </div>
      <div class="card">
        <h3>指令浏览器 <span class="tag" id="isa-total">—</span></h3>
        <div class="filters" id="isa-filters"></div>
        <div class="scroll"><table><thead><tr><th>ID</th><th>族</th><th>算子</th><th>变体</th><th>语义</th><th>数学不变量</th><th>状态</th></tr></thead><tbody id="isa-rows"></tbody></table></div>
      </div>
    </section>

    <!-- 门禁 -->
    <section class="pane" id="p-gate">
      <div class="phead"><h2>可证门禁</h2><p>甲类：内部不变量 · 乙类：与世界顶级实现对拍</p></div>
      <div class="note"><b>尺子照出的不是胜利，是洞。</b>乙类门禁让旗舰的 ISA 参照采样器与 llama.cpp 生产内核在<b>同一已知绝对 logit 分布</b>上对拍 —— 两侧跑在同一分布上，TVD 只反映采样噪声，因此这个门禁有真实判别力。</div>
      <div class="row" style="margin-bottom:14px">
        <button class="btn" id="g-run">运行门禁</button>
        <span class="muted mono" id="g-stat" style="font-size:12px"></span>
      </div>
      <div class="card"><table><thead><tr><th>组</th><th>门禁</th><th>实测</th><th>判据</th></tr></thead><tbody id="g-rows"></tbody></table></div>
    </section>

    <!-- 架构 -->
    <section class="pane" id="p-arch">
      <div class="phead"><h2>十层架构</h2><p>每层都必须可被独立复算验证</p></div>
      <div class="card" id="arch-list"></div>
      <div class="card" style="margin-top:14px"><h3>零重写对照表</h3><table><thead><tr><th>旗舰部件</th><th>站在谁之上</th><th>验证基准</th></tr></thead><tbody>
        <tr><td class="mono">推理内核</td><td class="mono">llama.cpp (GGER) · node-llama-cpp 3.22.2</td><td>对拍：贪心/seed/分布 TVD</td></tr>
        <tr><td class="mono">1024 微指令 ISA</td><td class="mono">AXIOM v1.3.26 (MIT)</td><td>386 项数学不变量 + 注册表完整性</td></tr>
        <tr><td class="mono">采样参照实现</td><td class="mono">AXIOM softmax/topK/topP/temperature</td><td>与 llama.cpp 同分布对拍</td></tr>
        <tr><td class="mono">HTTP 服务</td><td class="mono">node:http · OpenAI REST 兼容规范</td><td>端点契约实测</td></tr>
        <tr><td class="mono">仪表盘</td><td class="mono">原生 DOM + 内联 CSS/JS</td><td>零外部请求</td></tr>
      </tbody></table></div>
    </section>
  </main>
</div>
<script>
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

document.querySelectorAll('.nav button').forEach((b) => {
  b.onclick = () => {
    $$('.nav button').forEach((x) => x.classList.remove('on'));
    b.classList.add('on');
    $$('.pane').forEach((p) => p.classList.remove('on'));
    $('#p-' + b.dataset.p).classList.add('on');
    if (b.dataset.p === 'isa' && !window.__isaLoaded) loadIsa();
  };
});

const API = '';
async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(path + ' → HTTP ' + r.status);
  return r.json();
}

/* ---------- 总览 ---------- */
async function loadHealth() {
  try {
    const h = await api('/healthz');
    const m = h.model || {};
    $('#s-model').textContent = (m.filename || '—').replace(/\.gguf$/i, '');
    $('#s-arch').textContent = m.architecture || '—';
    $('#s-isa').innerHTML = (h.isa?.total ?? '—') + '<small> 条</small>';
    $('#s-ver').innerHTML = (h.isa?.verified ?? '—') + '<small> 条</small>';
    const t = h.telemetry || {};
    $('#s-tps').innerHTML = (t.requests ? (t.tokens / Math.max(t.totalMs,1) * 1000).toFixed(1) : '—') + '<small> tok/s</small>';

    $('#dash-cards').innerHTML = [
      card('推理引擎', 'llama.cpp', 'GGML · CPU · ' + (m.sizeBytes ? (m.sizeBytes/1048576).toFixed(0)+' MB' : '')),
      card('上下文窗口', (m.trainContextSize ?? '—').toLocaleString(), 'tokens · 训练时上限'),
      card('ISA 指令', String(h.isa?.total ?? '—'), 'verified ' + (h.isa?.verified ?? '—')),
      card('累计请求', String(t.requests ?? 0), (t.tokens ?? 0) + ' tokens 生成'),
      card('已发现模型', String((h.discoveredModels||[]).length), '本地 GGUF'),
      card('架构版本', 'APEX-Ω v1.0', '十层 · 零重写'),
    ].join('');

    $('#gate-count').textContent = '实时';
    $('#gate-summary').innerHTML = h.isa?.lastTrace && h.isa.lastTrace.total
      ? '<div class="kv"><span>上次推理 ISA 指令数</span><span class="mono">' + h.isa.lastTrace.total + '</span></div>' +
        '<div class="kv"><span>失败</span><span class="mono ok">' + (h.isa.lastTrace.failed ?? 0) + '</span></div>' +
        '<div class="kv"><span>耗时</span><span class="mono">' + (h.isa.lastTrace.totalMs ?? 0) + ' ms</span></div>' +
        '<div class="kv"><span>涉及族</span><span class="mono">' + (h.isa.lastTrace.families||[]).join(' ') + '</span></div>'
      : '<span class="muted">尚无推理轨迹，去「真实推理」跑一次</span>';
  } catch (e) {
    $('#s-model').textContent = '离线';
    $('#s-model').className = 'bad';
  }
}
function card(k, v, d) {
  return '<div class="card"><h3>' + esc(k) + '</h3><div class="stat" style="margin:0;background:none;border:none;padding:0">' +
    '<div class="v" style="font-size:22px">' + esc(v) + '</div>' +
    '<div class="muted mono" style="font-size:11px;margin-top:3px">' + esc(d) + '</div></div></div>';
}

/* ---------- 聊天 ---------- */
const msgs = $('#msgs');
function addMsg(who, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  d.innerHTML = '<div class="who">' + (who === 'u' ? 'YOU' : 'APEX-Ω') + '</div>' + esc(text);
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
  return d;
}
async function send() {
  const input = $('#c-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg('u', text);
  const pending = addMsg('a', '');
  pending.innerHTML = '<div class="who">APEX-Ω</div><span class="spin"></span>思考中…';
  const btn = $('#c-send'); btn.disabled = true;
  const t0 = performance.now();
  try {
    const body = {
      messages: [{ role: 'user', content: text }],
      temperature: parseFloat($('#c-temp').value),
      max_tokens: parseInt($('#c-max').value, 10),
    };
    const seed = $('#c-seed').value;
    if (seed !== '') body.seed = parseInt(seed, 10);
    const res = await api('/v1/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const out = res.choices?.[0]?.message?.content ?? '';
    pending.innerHTML = '<div class="who">APEX-Ω</div>' + esc(out);
    const ms = (performance.now() - t0).toFixed(0);
    const u = res.meta || {};
    $('#c-stat').textContent = (u.tokens_per_sec ? u.tokens_per_sec + ' tok/s · ' : '') + ms + ' ms · ISA ' + (u.isa_trace?.total ?? 0) + ' 指令';
    loadHealth();
  } catch (e) {
    pending.innerHTML = '<div class="who">APEX-Ω</div><span class="bad">失败：' + esc(e.message) + '</span>';
  }
  btn.disabled = false;
  msgs.scrollTop = msgs.scrollHeight;
}
$('#c-send').onclick = send;
$('#c-input').onkeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
};
$('#c-clear').onclick = () => { msgs.innerHTML = ''; };

/* ---------- ISA ---------- */
let ISA = null;
async function loadIsa() {
  window.__isaLoaded = true;
  try {
    ISA = await api('/api/isa');
    $('#isa-total').textContent = ISA.instructions.length + ' 条';
    $('#isa-ver').textContent = ISA.verifiedCount + ' / ' + ISA.meta.total;
    $('#isa-fams').innerHTML = ISA.groups.map((g) =>
      '<div class="kv"><span>' + esc(g.group) + ' <span class="muted">' + esc(g.groupName) + '</span></span>' +
      '<span class="mono">' + g.verified + '/' + g.total +
      ' <span class="bar" style="display:inline-block;width:70px;vertical-align:middle;margin-left:6px"><i style="width:' +
      Math.round(g.verified / g.total * 100) + '%"></i></span></span></div>').join('');
    const fams = ['ALL', ...ISA.groups.map((g) => g.group)];
    $('#isa-filters').innerHTML = fams.map((f, i) =>
      '<span class="chip' + (i === 0 ? ' on' : '') + '" data-f="' + f + '">' + f + '</span>').join('');
    $$('#isa-filters .chip').forEach((c) => {
      c.onclick = () => {
        $$('#isa-filters .chip').forEach((x) => x.classList.remove('on'));
        c.classList.add('on');
        renderIsa(c.dataset.f);
      };
    });
    renderIsa('ALL');
  } catch (e) {
    $('#isa-fams').innerHTML = '<span class="bad">加载失败：' + esc(e.message) + '</span>';
  }
}
function renderIsa(f) {
  const list = (f === 'ALL' ? ISA.instructions : ISA.instructions.filter((x) => x.group === f));
  $('#isa-rows').innerHTML = list.slice(0, 400).map((r) =>
    '<tr><td class="mono muted">' + r.id + '</td><td class="mono">' + r.group + '</td>' +
    '<td class="mono">' + esc(r.op) + '</td><td class="mono muted">' + r.variant + '</td>' +
    '<td>' + esc(r.opZh) + '</td><td class="mono" style="font-size:11.5px">' + esc(r.invariant) + '</td>' +
    '<td class="' + (r.status === 'verified' ? 'ok' : 'muted') + '">' + (r.status === 'verified' ? '✅' : '○') + '</td></tr>').join('')
    + (list.length > 400 ? '<tr><td colspan="7" class="muted">… 仅显示前 400 条（共 ' + list.length + '）</td></tr>' : '');
}

/* ---------- 门禁 ----------
 * 注意：端点用 /api/verify 而非 /api/gates。
 * 两者是同一路由的别名，但部分网络环境（代理/沙箱）会拦截路径中含 "gates" 的
 * 请求 —— 实测在该环境下 /api/gates 直接超时，而 /api/verify 正常。
 * 为保证仪表盘在任何网络下都可用，统一走 /api/verify。
 */
async function runGates() {
  const btn = $('#g-run'); btn.disabled = true;
  $('#g-stat').innerHTML = '<span class="spin"></span>门禁启动中…';
  $('#g-rows').innerHTML = '<tr><td colspan="4" class="muted">门禁后台执行中（甲类即刻返回，乙类含模型推理，约 2–4 分钟）…</td></tr>';
  const done = () => { btn.disabled = false; };
  try {
    const first = await api('/api/verify');
    if (first.cached) { renderGates(first); return done(); }
    if (first.error) { $('#g-stat').innerHTML = '<span class="bad">' + esc(first.error) + '</span>'; return done(); }
    // 轮询直到结果就绪
    const t0 = Date.now();
    for (;;) {
      await new Promise((r) => setTimeout(r, 3000));
      const r = await api('/api/verify');
      if (r.results) { renderGates(r); return done(); }
      const el = Math.round((Date.now() - t0) / 1000);
      $('#g-stat').innerHTML = '<span class="spin"></span>运行中 ' + el + 's …';
      if (el > 900) { $('#g-stat').innerHTML = '<span class="bad">超时（>15min）</span>'; return done(); }
    }
  } catch (e) {
    $('#g-stat').innerHTML = '<span class="bad">' + esc(e.message) + '</span>';
    $('#g-rows').innerHTML = '';
    return done();
  }
}
$('#g-run').onclick = runGates;
function renderGates(r) {
  const ok = r.allGreen;
  $('#g-stat').innerHTML = '<span class="' + (ok ? 'ok' : 'bad') + '">' + r.passed + '/' + r.total + (ok ? ' 全绿' : ' 有失败') + '</span>';
  $('#g-rows').innerHTML = (r.results || []).map((x) =>
    '<tr><td class="mono muted">' + esc(x.group) + '</td><td>' + esc(x.name) + '</td>' +
    '<td class="mono">' + esc(x.detail || '') + '</td>' +
    '<td class="' + (x.pass ? 'ok' : 'bad') + '">' + (x.pass ? '✅ PASS' : '❌ FAIL') + '</td></tr>').join('');
  $('#gate-count').textContent = r.passed + '/' + r.total;
  loadHealth();
}

/* ---------- 架构 ---------- */
const ARCH = [
  ['L10','微指令层','1024 条微指令（16 族 × 16 算子 × 4 变体），每条带数学不变量','AXIOM v1.3.26 · 零改动复用'],
  ['L9','应用层','单文件 HTML 仪表盘 · OpenAI 兼容客户端','原生 DOM · 零外部请求'],
  ['L8','对拍层','旗舰 ISA 参照采样器 vs llama.cpp 生产内核 · TVD/JS 分布对拍','旗舰原创'],
  ['L7','大模型层','真实 GGUF 模型加载 · chat wrapper 选择 · 隔离会话 · 遥测','llama.cpp · 零重写'],
  ['L6','服务层','OpenAI 兼容 REST + SSE 流式 + ISA 遥测端点','node:http · 零依赖'],
  ['L5','检索层','BM25 + 稠密召回 + RRF 融合（可选模块）','Flagship RAG 引擎模式'],
  ['L4','记忆层','会话历史 · 上下文预算管理','旗舰实现'],
  ['L3','推理编排层','采样参数校验 · ISA 审计 · 可复现控制','旗舰实现'],
  ['L2','数值层','logsumexp 稳定化 · 温度缩放 · 概率核','AXIOM 复用'],
  ['L1','契约层','ISA 语义契约 · 不变量注册 · 门禁规则','旗舰实现'],
  ['L0','算子层','tensor / linalg / prob / optim / attention 参考实现','AXIOM 复用 · 415 导出'],
];
$('#arch-list').innerHTML = ARCH.map((a) =>
  '<div class="kv"><span><span class="mono" style="color:var(--cyan)">' + a[0] + '</span> &nbsp; <b>' + esc(a[1]) + '</b>' +
  '<br><span class="muted" style="font-size:12px">' + esc(a[2]) + '</span></span>' +
  '<span class="mono muted" style="text-align:right;max-width:44%">' + esc(a[3]) + '</span></div>').join('');

loadHealth();
setInterval(loadHealth, 6000);
</script>
</body>
</html>`;
