/**
 * APEX-Ω · 旗舰统一入口
 * ---------------------------------------------------------------------------
 * 一站式使用：加载模型 → 提问 → 拿到带 ISA 轨迹的结果。
 *
 *   import { Flagship } from 'apex-omega';
 *   const f = new Flagship();
 *   await f.ready();
 *   const r = await f.ask('什么是零重写？');
 *   console.log(r.text, r.isaTrace);
 */
export { FlagshipRuntime, discoverModels, tokenId } from './layers/l7-model.js';
export { IsaRouter, isaIntegrityGate, ISA_META, REGISTRY, VERIFIED, GROUP_STATS } from './layers/l10-isa.js';
export { ReferenceSampler, DuelGate, Distances } from './layers/l8-duel.js';
export { createServer } from './layers/l6-service.js';
export { DASHBOARD_HTML } from './layers/l9-dashboard.js';
export { runIsaGates, runDuelGates, runAllGates } from './gates.js';

import { FlagshipRuntime } from './layers/l7-model.js';
import { IsaRouter } from './layers/l10-isa.js';
import { createServer } from './layers/l6-service.js';
import { DASHBOARD_HTML } from './layers/l9-dashboard.js';

/**
 * 旗舰门面：把十层收敛成一个对象。
 */
export class Flagship {
  constructor(opts = {}) {
    this.runtime = new FlagshipRuntime(opts);
    this.router = opts.router ?? new IsaRouter();
  }

  /** 加载模型并就绪 */
  async ready() {
    await this.runtime.init();
    return this;
  }

  /** 提问（自动留下 ISA 轨迹） */
  async ask(prompt, opts = {}) {
    const res = await this.runtime.chat(prompt, { ...opts, router: this.router });
    return {
      text: res.text,
      tokens: res.tokens,
      tokenCount: res.tokenCount,
      ms: res.ms,
      tokensPerSec: res.tokensPerSec,
      isaTrace: this.router.trace.slice(),
      isaSummary: this.router.summary(),
    };
  }

  /** 启动服务（含单文件仪表盘） */
  async serve(opts = {}) {
    const { server, listen } = createServer({
      runtime: this.runtime,
      router: this.router,
      uiHtml: DASHBOARD_HTML,
      port: opts.port ?? 8080,
      host: opts.host ?? '127.0.0.1',
    });
    await listen();
    return server;
  }

  /** 运行全部门禁 */
  async gates(opts = {}) {
    const { runAllGates } = await import('./gates.js');
    return runAllGates({ runtime: this.runtime, router: this.router, ...opts });
  }

  async dispose() {
    await this.runtime.dispose();
  }
}

export const VERSION = '1.0.0';
