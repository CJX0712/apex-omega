/**
 * APEX-Ω · 门禁总装（src/gates.js）
 * 注：test/gates.js 仅为本文件的再导出（保持旧导入路径可用）
 * ---------------------------------------------------------------------------
 * 旗舰的门禁分两类，缺一不可：
 *
 *   甲类 · 内部不变量门禁（ISA 自证，**无需模型**）
 *     复用 AXIOM 的 386 项数学不变量（零重写），外加旗舰自己的 ISA 完整性门禁
 *     与参照采样器分布自检。这些证明「旗舰的算子层是对的」。
 *     → CI / 离线 / 无 GPU 环境可完整复现。
 *
 *   乙类 · 对外对拍门禁（与世界顶级实现互证，**需要模型**）
 *     旗舰的 ISA 参照采样器 vs llama.cpp 的生产级采样内核。
 *     这些证明「旗舰与世界顶级引擎接得对」。
 *     模型缺失时优雅降级为 skipped，不冒充通过。
 *
 * 乙类才是旗舰的立身之本：甲类能证明自己没写错，但证明不了自己跟世界接得上。
 */
import { IsaRouter, isaIntegrityGate, ISA_META, VERIFIED, AXIOM } from './layers/l10-isa.js';
import { FlagshipRuntime, discoverModels } from './layers/l7-model.js';
import { DuelGate } from './layers/l8-duel.js';

/**
 * 甲类门禁：ISA 完整性 + AXIOM 386 项数学不变量 + 参照采样器自检
 * （全部无需模型，可离线复现）
 */
export function runIsaGates() {
  const results = [];
  const t0 = Date.now();

  // A1 ISA 注册表完整性
  const integ = isaIntegrityGate();
  for (const r of integ.results) {
    results.push({ group: 'ISA', name: r.name, pass: r.ok, detail: r.detail });
  }

  // A2 AXIOM 386 项数学不变量（零重写，直接跑上游权威自检）
  const checks = AXIOM.runChecks();
  const mathFailed = checks.results.filter((c) => !c.pass).length;
  for (const c of checks.results) {
    if (!c.pass) {
      results.push({
        group: 'ISA',
        name: c.name,
        pass: false,
        detail: `residual=${c.residual} tol=${c.tol}${c.error ? ` error=${c.error}` : ''}`,
      });
    }
  }
  results.push({
    group: 'ISA',
    name: `AXIOM 数学不变量全量（${checks.total} 项）`,
    pass: mathFailed === 0,
    detail: mathFailed === 0
      ? `${checks.total}/${checks.total} 全绿`
      : `${checks.total - mathFailed}/${checks.total}，失败 ${mathFailed}`,
  });

  // A3 参照采样器分布自检（模型无关）
  const duel = new DuelGate(null, { router: new IsaRouter() });
  const self = duel.samplerSelfCheck();
  results.push({ group: 'ISA', name: self.name, pass: self.pass, detail: self.detail });

  return { results, ms: Date.now() - t0, total: results.length };
}

/**
 * 乙类门禁：与世界顶级实现对拍（需要 GGUF 模型）
 */
export async function runDuelGates(opts = {}) {
  const results = [];
  const router = opts.router ?? new IsaRouter();
  const hasModel = opts.runtime?.model || discoverModels().length > 0;

  if (!hasModel) {
    return {
      results: [{
        group: '对拍',
        name: '乙类对拍门禁',
        pass: true,
        skipped: true,
        detail: '未发现 GGUF 模型，乙类门禁跳过（不冒充通过）。下载模型后重跑即可激活。',
      }],
      router,
      runtime: null,
      skipped: true,
    };
  }

  const rt = opts.runtime ?? new FlagshipRuntime({
    contextSize: opts.contextSize ?? 512,
    threads: opts.threads ?? 4,
  });
  await rt.init();
  const duel = new DuelGate(rt, { router });

  const gates = [
    () => duel.wrapperHealth('用一句话介绍你自己。'),
    () => duel.greedyConsistency('The capital of France is'),
    () => duel.seedReproducibility('用一句话解释什么是零重写原则。', { seed: 20261024 }),
    () => duel.samplingDistributionMatch('介绍一下你自己', { rounds: opts.rounds ?? 60, maxTokens: 1 }),
  ];

  for (const fn of gates) {
    const t0 = Date.now();
    try {
      const r = await fn();
      results.push({
        group: '对拍',
        name: r.name,
        pass: r.pass,
        detail: r.detail,
        ms: Date.now() - t0,
        sample: r.sample ?? r.a,
        // 分布对拍的原始数据一并留档，便于人工复核
        theory: r.theory,
        observed: r.observed,
        gaps: r.gaps,
      });
    } catch (e) {
      results.push({ group: '对拍', name: '门禁异常', pass: false, detail: e.message });
    }
  }

  return { results, router, runtime: rt };
}

/**
 * 完整门禁运行
 */
export async function runAllGates(opts = {}) {
  const isa = runIsaGates();
  const duel = await runDuelGates(opts);
  const results = [...isa.results, ...duel.results];
  const effective = results.filter((r) => !r.skipped);
  const passed = effective.filter((r) => r.pass).length;
  return {
    total: effective.length,
    passed,
    failed: effective.length - passed,
    allGreen: passed === effective.length,
    results,
    meta: {
      isa: ISA_META,
      verified: VERIFIED.length,
      ms: isa.ms,
      duelSkipped: !!duel.skipped,
    },
  };
}
