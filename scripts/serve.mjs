#!/usr/bin/env node
/**
 * APEX-Ω · 启动脚本
 * 用法：
 *   node scripts/serve.mjs [--port 8080] [--host 127.0.0.1] [--model path/to.gguf]
 */
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { FlagshipRuntime, discoverModels } from '../src/layers/l7-model.js';
import { createServer } from '../src/layers/l6-service.js';
import { DASHBOARD_HTML } from '../src/layers/l9-dashboard.js';
import { IsaRouter } from '../src/layers/l10-isa.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 ? process.argv[i + 1] : dflt;
}

const port = Number(arg('port', process.env.PORT || 8080));
const host = arg('host', '127.0.0.1');
const modelArg = arg('model', process.env.APEX_OMEGA_MODEL || null);

const C = { reset: '\x1b[0m', dim: '\x1b[2m', green: '\x1b[32m', cyan: '\x1b[36m', bold: '\x1b[1m', red: '\x1b[31m' };

console.log(`${C.bold}${C.cyan}APEX-Ω · 世界旗舰 AI 合流系统${C.reset}`);
console.log(`${C.dim}真实大模型(llama.cpp) × 1024 微指令 ISA × 可证门禁${C.reset}\n`);

const found = discoverModels();
if (found.length) {
  console.log(`${C.dim}发现本地 GGUF：${C.reset}`);
  for (const m of found) console.log(`  · ${m.name}  ${(m.size / 1048576).toFixed(1)} MB`);
} else {
  console.log(`${C.dim}未发现本地 GGUF 模型。将自动搜索 models/ 目录。${C.reset}`);
}

const rt = new FlagshipRuntime({
  modelPath: modelArg,
  contextSize: 4096,
  threads: Math.max(2, os.cpus().length - 1),
});

try {
  await rt.init();
  console.log(`\n${C.green}✓${C.reset} 模型已加载：${rt.modelInfo.filename}`);
  console.log(`${C.dim}  架构 ${rt.modelInfo.architecture} · ${(rt.modelInfo.sizeBytes / 1048576).toFixed(0)} MB · 训练上下文 ${rt.modelInfo.trainContextSize}${C.reset}`);
  console.log(`${C.dim}  加载耗时 ${rt.loadMs} ms${C.reset}`);
} catch (e) {
  console.error(`${C.red}✗ 模型加载失败：${e.message}${C.reset}`);
  process.exit(1);
}

const router = new IsaRouter();
const { server, listen } = createServer({
  runtime: rt, port, host, uiHtml: DASHBOARD_HTML, router,
});

/* 门禁端点：单 server 内拦截 /api/gates */
const origListeners = server.listeners('request').slice();
server.removeAllListeners('request');
server.on('request', (req, res) => {
  if (req.url && req.url.startsWith('/api/gates')) {
    import('../test/gates.js')
      .then(async ({ runAllGates }) => {
        const r = await runAllGates({ runtime: rt, router, rounds: 120 });
        const body = JSON.stringify(r, null, 2);
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(body);
      })
      .catch((e) => {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      });
    return;
  }
  for (const fn of origListeners) fn.call(server, req, res);
});

await listen();
console.log(`\n${C.green}✓${C.reset} 仪表盘：${C.bold}http://${host}:${port}${C.reset}`);
console.log(`${C.dim}  OpenAI 兼容：POST http://${host}:${port}/v1/chat/completions${C.reset}`);
console.log(`${C.dim}  ISA 数据：   GET  http://${host}:${port}/api/isa${C.reset}`);
console.log(`${C.dim}  门禁：       GET  http://${host}:${port}/api/gates${C.reset}`);
console.log(`${C.dim}  健康检查：   GET  http://${host}:${port}/healthz${C.reset}\n`);
