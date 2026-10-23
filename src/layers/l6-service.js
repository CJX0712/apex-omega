/**
 * APEX-Ω · L6 服务层 —— OpenAI 兼容 HTTP 服务
 * ---------------------------------------------------------------------------
 * 零重写：HTTP 语义复用 Node.js 内置 `node:http`（世界标准运行时）；
 *         OpenAI 兼容协议**规格**直接对齐 OpenAI REST API（业界事实标准）；
 *         不引入 express/fastify 等框架（零额外依赖）。
 *
 * 端点：
 *   GET  /v1/models              模型清单
 *   POST /v1/chat/completions    对话（流式 SSE + 非流式）
 *   POST /v1/completions         文本补全
 *   GET  /healthz                健康检查（含 ISA 遥测）
 *   GET  /api/isa                ISA 指令浏览器数据
 *   GET  /api/trace              最近一次推理的 ISA 轨迹
 *   GET  /                        单文件仪表盘
 */
import http from 'node:http';
import { IsaRouter, REGISTRY, GROUP_STATS, ISA_META, VERIFIED } from '../layers/l10-isa.js';
import { discoverModels } from '../layers/l7-model.js';
import { runAllGates } from '../gates.js';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
};

/**
 * @param {{runtime:import('./l7-model.js').FlagshipRuntime, port?:number, host?:string,
 *          uiHtml?:string, router?:IsaRouter}} opts
 */
export function createServer(opts) {
  const { runtime, port = 8080, host = '127.0.0.1' } = opts;
  const router = opts.router ?? new IsaRouter();
  const gateRounds = opts.gateRounds ?? 60;
  let lastTrace = { trace: [], summary: {} };
  let gateCache = null;      // 最近一次门禁结果（null = 未跑过）
  let gateRunning = false;   // 防止并发重入
  let gateStartedAt = 0;

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host ?? 'localhost'}`);
    const p = url.pathname;

    try {
      /* ---------------- 静态 / 仪表盘 ---------------- */
      if (req.method === 'GET' && (p === '/' || p === '/index.html')) {
        res.writeHead(200, { 'Content-Type': MIME['.html'] });
        res.end(opts.uiHtml ?? '<!doctype html><meta charset=utf-8><h1>APEX-Ω</h1><p>仪表盘未构建</p>');
        return;
      }

      /* ---------------- OpenAI 兼容：模型清单 ---------------- */
      if (req.method === 'GET' && p === '/v1/models') {
        const info = runtime.modelInfo ?? {};
        const id = info.filename?.replace(/\.gguf$/i, '') ?? 'apex-omega';
        return json(res, 200, {
          object: 'list',
          data: [{
            id,
            object: 'model',
            created: Math.floor(Date.now() / 1000),
            owned_by: 'apex-omega',
            meta: {
              architecture: info.architecture,
              size_bytes: info.sizeBytes,
              train_context_size: info.trainContextSize,
              engine: 'llama.cpp (node-llama-cpp 3.22.2)',
              isa: ISA_META.version,
            },
          }],
        });
      }

      /* ---------------- 健康检查 ---------------- */
      if (req.method === 'GET' && p === '/healthz') {
        return json(res, 200, {
          ok: true,
          model: runtime.modelInfo ?? null,
          telemetry: runtime.telemetry,
          isa: {
            version: ISA_META.version,
            total: ISA_META.total,
            verified: VERIFIED.length,
            lastTrace: lastTrace.summary,
          },
          discoveredModels: discoverModels().map((m) => ({ name: m.name, size: m.size })),
        });
      }

      /* ---------------- ISA 浏览器数据 ---------------- */
      if (req.method === 'GET' && p === '/api/isa') {
        return json(res, 200, {
          meta: ISA_META,
          groups: GROUP_STATS,
          verifiedCount: VERIFIED.length,
          instructions: REGISTRY.map((r) => ({
            id: r.id, group: r.group, groupName: r.groupName, op: r.op, opZh: r.opZh,
            variant: r.variant, name: r.name, semantics: r.semantics,
            invariant: r.invariant, status: r.status, impl: r.impl,
          })),
        });
      }

      if (req.method === 'GET' && p === '/api/trace') {
        return json(res, 200, lastTrace);
      }

      /* ---------------- 门禁（异步执行 + 轮询） ----------------
       * 完整门禁含模型推理，耗时数分钟；不能挂在 HTTP 连接上等。
       * 首次 GET 触发后台运行并返回 {started:true}，客户端轮询至出现结果。
       */
      if (req.method === 'GET' && (p === '/api/gates' || p === '/api/verify')) {
        if (gateCache) return json(res, 200, { ...gateCache, cached: true });
        if (gateRunning) return json(res, 200, { running: true, startedAt: gateStartedAt });
        gateRunning = true;
        gateStartedAt = Date.now();
        // 后台执行，不阻塞响应。
        // 必须用 setImmediate 让出事件循环：否则 json() 的响应还没来得及 flush，
        // 就被 llama.cpp 的同步原生推理占满事件循环（实测表现为请求超时）。
        setImmediate(() => {
          runAllGates({ runtime, router, rounds: gateRounds })
            .then((r) => { gateCache = r; })
            .catch((e) => { gateCache = { allGreen: false, error: e.message, total: 0, passed: 0, results: [] }; })
            .finally(() => { gateRunning = false; });
        });
        return json(res, 200, { started: true, startedAt: gateStartedAt });
      }
      if (req.method === 'DELETE' && (p === '/api/gates' || p === '/api/verify')) {
        gateCache = null;
        return json(res, 200, { cleared: true });
      }

      /* ---------------- OpenAI 兼容：对话 ---------------- */
      if (req.method === 'POST' && (p === '/v1/chat/completions' || p === '/v1/completions')) {
        const body = await readJson(req);
        const isChat = p.endsWith('/chat/completions');

        let prompt;
        if (isChat) {
          const msgs = body.messages ?? [];
          // 取最后一条 user 消息作为提示；保留 system 作为系统提示
          const userMsg = [...msgs].reverse().find((m) => m.role === 'user');
          const sysMsg = msgs.find((m) => m.role === 'system');
          prompt = userMsg?.content ?? '';
          body.__system = sysMsg?.content;
        } else {
          prompt = body.prompt ?? '';
        }
        if (typeof prompt !== 'string' || !prompt.trim()) {
          return json(res, 400, { error: { message: 'prompt/messages 不能为空', type: 'invalid_request_error' } });
        }

        const genOpts = {
          maxTokens: body.max_tokens ?? body.max_completion_tokens ?? 256,
          temperature: body.temperature ?? 0.7,
          topP: body.top_p ?? 0.95,
          topK: body.top_k ?? 40,
          seed: body.seed,
          repeatPenalty: body.repeat_penalty,
          systemPrompt: body.__system,
          sessionKey: body.session_id ?? 'default',
          // 每一次真实推理都必须经过 ISA 审计层，留下可追溯的微指令轨迹
          router,
        };

        const modelId = runtime.modelInfo?.filename?.replace(/\.gguf$/i, '') ?? 'apex-omega';
        const created = Math.floor(Date.now() / 1000);

        if (body.stream) {
          res.writeHead(200, {
            'Content-Type': 'text/event-stream; charset=utf-8',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
          });
          let n = 0;
          const traceFrom = router.trace.length;
          try {
            const out = await runtime.chat(prompt, {
              ...genOpts,
              onToken: (_t, partial) => {
                const delta = partial.slice(n);
                n = partial.length;
                if (!delta) return;
                res.write(`data: ${JSON.stringify({
                  id: `chatcmpl-${created}`, object: 'chat.completion.chunk', created,
                  model: modelId,
                  choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
                })}\n\n`);
              },
            });
            lastTrace = {
              trace: router.trace.slice(-200),
              summary: router.summary(traceFrom),
            };
            res.write(`data: ${JSON.stringify({
              id: `chatcmpl-${created}`, object: 'chat.completion.chunk', created, model: modelId,
              choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
              usage: { prompt_tokens: 0, completion_tokens: out.tokenCount, total_tokens: out.tokenCount },
            })}\n\n`);
          } catch (e) {
            res.write(`data: ${JSON.stringify({ error: { message: e.message } })}\n\n`);
          }
          res.write('data: [DONE]\n\n');
          res.end();
          return;
        }

        try {
          const traceFrom = router.trace.length;   // 本次推理的 ISA 轨迹起点
          const out = await runtime.chat(prompt, genOpts);
          lastTrace = {
            trace: router.trace.slice(-200),
            summary: router.summary(traceFrom),     // 只含本次推理产生的微指令
          };
          return json(res, 200, {
            id: `chatcmpl-${created}`,
            object: isChat ? 'chat.completion' : 'text_completion',
            created,
            model: modelId,
            choices: isChat
              ? [{
                  index: 0,
                  message: { role: 'assistant', content: out.text },
                  finish_reason: 'stop',
                }]
              : [{
                  index: 0,
                  text: out.text,
                  finish_reason: 'stop',
                }],
            usage: {
              prompt_tokens: 0,
              completion_tokens: out.tokenCount,
              total_tokens: out.tokenCount,
            },
            meta: {
              ms: out.ms,
              tokens_per_sec: out.tokensPerSec,
              isa_trace: router.summary(traceFrom),
            },
          });
        } catch (e) {
          return json(res, 500, { error: { message: e.message, type: 'server_error' } });
        }
      }

      /* ---------------- 404 ---------------- */
      json(res, 404, { error: { message: `no route for ${req.method} ${p}`, type: 'not_found' } });
    } catch (e) {
      json(res, 500, { error: { message: e.message, stack: e.stack } });
    }
  });

  return { server, router, listen: () => new Promise((res) => server.listen(port, host, () => res(port))) };
}

function json(res, code, obj) {
  const body = JSON.stringify(obj, null, 2);
  res.writeHead(code, { 'Content-Type': MIME['.json'], 'Content-Length': Buffer.byteLength(body) });
  res.end(body);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (c) => {
      raw += c;
      if (raw.length > 8e6) { reject(new Error('payload too large')); req.destroy(); }
    });
    req.on('end', () => {
      try { resolve(raw ? JSON.parse(raw) : {}); } catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}
