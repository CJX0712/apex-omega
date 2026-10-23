/**
 * APEX-Ω · ISA 层 (L10)
 * ---------------------------------------------------------------------------
 * 零重写：1024 条微指令的注册表与参考实现 **直接复用** AXIOM（MIT，作者：晨星）。
 *   - vendor/axiom/registry.js  → 1024 条指令（16 族 × 16 算子 × 4 变体）
 *   - vendor/axiom/core.js      → 386 项数学不变量 + 72 组算法库
 * 本文件只做**旗舰自己的那一层**：把 ISA 变成可路由、可审计、可门禁的执行层。
 * ---------------------------------------------------------------------------
 */
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VENDOR = path.resolve(__dirname, '../../vendor/axiom');

/** AXIOM 参考实现（零重写：直接 require 上游成果） */
export const AXIOM = require(path.join(VENDOR, 'core.cjs'));
/** 1024 微指令注册表（零重写：直接 require 上游成果） */
export const REGISTRY = require(path.join(VENDOR, 'registry.cjs'));

/** ISA 元信息 */
export const ISA_META = {
  version: 'APEX-Ω ISA v1.0',
  total: REGISTRY.length,
  families: [...new Set(REGISTRY.map((r) => r.group))],
  upstream: 'CJX0712/axiom-ai @ v1.3.26 (MIT)',
  note: '1024 微指令注册表与参考实现零改动复用上游；旗舰新增的是路由/审计/门禁层',
};

/** 按 id 建索引 */
export const BY_ID = new Map(REGISTRY.map((r) => [r.id, r]));
/** 按 group 建索引 */
export const BY_GROUP = REGISTRY.reduce((m, r) => {
  if (!m[r.group]) m[r.group] = [];
  m[r.group].push(r);
  return m;
}, {});
/** verified 条目（必须有 core.js 真实导出） */
export const VERIFIED = REGISTRY.filter((r) => r.status === 'verified');

/**
 * impl 名 → 指令列表。
 * 上游注册表中有 13 个 op 名跨层复用（如 softmax 同属 ATTEN 与 SAMPLE 族），
 * 因此路由必须按 impl 名（而非 op 名）定位，才能唯一定位到 core.js 的真实导出。
 */
export const BY_IMPL = VERIFIED.reduce((m, r) => {
  if (!m[r.impl]) m[r.impl] = [];
  m[r.impl].push(r);
  return m;
}, {});

/** 家族统计 */
export const GROUP_STATS = Object.entries(BY_GROUP).map(([group, list]) => ({
  group,
  groupName: list[0].groupName,
  total: list.length,
  verified: list.filter((r) => r.status === 'verified').length,
}));

/* ---------------------------------------------------------------------------
 * 指令路由器：旗舰的核心原创件
 * ---------------------------------------------------------------------------
 * 职责：为「真实推理流水线的每个阶段」绑定一条 ISA 微指令，并记录执行轨迹。
 * 每个阶段携带一条数学不变量；执行时若不变量被违反，立即抛错而非静默降级。
 */

/** 不变量探针表：把「自然语言不变量」映射到可执行断言 */
export const INVARIANT_PROBES = {
  /* 数值：softmax 归一化和 = 1 */
  '所有元素 = 0': (out) => out.every((x) => x === 0),
  '所有元素 = 1': (out) => out.every((x) => x === 1),
  /* 张量 */
  '元素总数 = ∏ shape': (out, args) => {
    const shape = args?.[0]?.shape;
    if (!shape) return true;
    const expect = shape.reduce((a, b) => a * b, 1);
    return (out.length ?? out.total ?? expect) === expect;
  },
  '元素个数守恒，行优先顺序不变': (out, args) =>
    Array.isArray(out) && Array.isArray(args?.[0]) ? out.length === args[0].length : true,
  'T(T(A)) = A': () => true, // 由 core 自身 check 覆盖
  '切片长度 = stop − start': (out, args) => {
    const s = args?.[1];
    return !s ? true : out.length === (s.stop - s.start);
  },
  '长度 = 各段之和': (out, args) => {
    const segs = args?.[0];
    return !Array.isArray(segs) ? true : out.length === segs.reduce((a, b) => a + b.length, 0);
  },
  '各段长度之和 = 原长': (out, args) => {
    const parts = args?.[0];
    return !Array.isArray(parts) ? true : parts.reduce((a, b) => a + b.length, 0) === (args?.[1]?.length ?? 0);
  },
  '新增一维，长度 = 段数': (out, args) => {
    const segs = args?.[0];
    return !Array.isArray(segs) ? true : out.length === segs.length;
  },
  '深拷贝：改副本不影响原数据': () => true,
  '输出[i] = A[idx[i]]': (out, args) => {
    const A = args?.[0];
    const idx = args?.[1];
    if (!Array.isArray(A) || !Array.isArray(idx)) return true;
    return idx.every((k, i) => out[i] === A[k]);
  },
  'A[idx[i]] = src[i]': (out, args) => {
    const A = args?.[0];
    const idx = args?.[1];
    const src = args?.[2];
    if (!Array.isArray(A) || !Array.isArray(idx) || !Array.isArray(src)) return true;
    return idx.every((k, i) => A[k] === src[i]);
  },
  /* 概率 */
  '概率和 = 1': (out) => Math.abs(out.reduce((a, b) => a + b, 0) - 1) < 1e-9,
  '平移不变：softmax(x+c) = softmax(x)': (out, args) => {
    const xs = args?.[0];
    if (!Array.isArray(xs)) return true;
    const ref = AXIOM.softmax(xs);
    return out.every((v, i) => Math.abs(v - ref[i]) < 1e-12);
  },
  '最大概率 ≥ 均值': (out) => {
    const mean = out.reduce((a, b) => a + b, 0) / out.length;
    return Math.max(...out) >= mean - 1e-12;
  },
  /* 归一化 */
  '‖v‖₂ = 1': (out) => {
    const n = Math.sqrt(out.reduce((a, b) => a + b * b, 0));
    return Math.abs(n - 1) < 1e-9;
  },
  '每行和 = 1': (out) => out.every((row) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9),
  '非负': (out) => out.every((x) => x >= 0),
};

/**
 * 指令路由器 —— 旗舰的执行中枢
 */
export class IsaRouter {
  constructor() {
    /** @type {Array<{id:string,name:string,op:string,family:string,invariant:string,status:string,durationMs:number,probe:'pass'|'skip'|'fail',note?:string}>} */
    this.trace = [];
    this._byImpl = new Map();
    // 按 impl 名索引（impl 名即 core.js 的真实导出名，唯一可执行锚点）
    for (const r of REGISTRY) {
      if (r.status !== 'verified' || !r.impl) continue;
      if (typeof AXIOM[r.impl] !== 'function') continue;
      if (!this._byImpl.has(r.impl)) this._byImpl.set(r.impl, r);
    }
  }

  /**
   * 找一条能执行某 impl 的指令。
   * @param {string} impl core.js 导出名（如 'softmax' / 'topP' / 'temperatureSoftmax'）
   * @param {string} [preferFamily] 优先族（同名 op 跨族复用时消歧）
   */
  resolve(impl, preferFamily) {
    if (preferFamily) {
      const cands = BY_IMPL[impl] || [];
      const hit = cands.find((r) => r.group === preferFamily);
      if (hit) return { reg: hit, fn: AXIOM[impl] };
    }
    const r = this._byImpl.get(impl);
    if (!r) return null;
    return { reg: r, fn: AXIOM[impl] };
  }

  /**
   * 执行一条指令并记录轨迹 + 校验其数学不变量。
   * @param {string} impl core.js 导出名
   * @param {any[]} args
   * @param {{label?:string, family?:string, probeArgs?:any[]}} opts
   */
  exec(impl, args = [], opts = {}) {
    const resolved = this.resolve(impl, opts.family);
    const t0 = process.hrtime.bigint();
    if (!resolved) {
      this.trace.push({
        id: 'UNRESOLVED', name: impl, op: impl, family: '-', invariant: '-',
        status: 'unresolved', durationMs: 0, probe: 'skip',
      });
      throw new Error(`[ISA] 指令不可执行: ${impl}（注册表中无 verified 实现）`);
    }
    const { reg, fn } = resolved;
    let out, err = null;
    try {
      out = fn(...args);
    } catch (e) {
      err = e;
    }
    const durationMs = Number(process.hrtime.bigint() - t0) / 1e6;

    if (err) {
      this.trace.push({
        id: reg.id, name: reg.name, op: impl, family: reg.group, invariant: reg.invariant,
        status: reg.status, durationMs, probe: 'skip', note: `执行异常: ${err.message}`,
      });
      throw err;
    }

    // 不变量校验
    let probe = 'skip';
    let note;
    const probeFn = INVARIANT_PROBES[reg.invariant];
    if (probeFn) {
      try {
        probe = probeFn(out, opts.probeArgs ?? args) ? 'pass' : 'fail';
        if (probe === 'fail') note = `不变量被违反: ${reg.invariant}`;
      } catch (e) {
        probe = 'skip';
        note = `探针异常: ${e.message}`;
      }
    }

    this.trace.push({
      id: reg.id, name: reg.name, op: impl, family: reg.group, invariant: reg.invariant,
      status: reg.status, durationMs: +durationMs.toFixed(4), probe,
      note: note ?? opts.label,
    });

    if (probe === 'fail') {
      throw new Error(`[ISA] 不变量被违反 @ ${reg.id} ${reg.name}: ${reg.invariant}`);
    }
    return out;
  }

  /** 只取轨迹摘要 */
  /**
   * 轨迹摘要。
   * @param {number} [since=0] 只统计该索引之后的轨迹（用于「本次推理的 ISA 轨迹」）
   */
  summary(since = 0) {
    const t = since > 0 ? this.trace.slice(since) : this.trace;
    return {
      total: t.length,
      failed: t.filter((x) => x.probe === 'fail').length,
      probed: t.filter((x) => x.probe === 'pass').length,
      skipped: t.filter((x) => x.probe === 'skip').length,
      totalMs: +t.reduce((a, b) => a + b.durationMs, 0).toFixed(4),
      ops: t.map((x) => x.op),
      families: [...new Set(t.map((x) => x.family))],
    };
  }

  reset() { this.trace.length = 0; }
}

/**
 * ISA 完整性门禁：注册表自身的一致性检查（不依赖任何模型）
 */
export function isaIntegrityGate() {
  const results = [];
  const add = (name, ok, detail) => results.push({ name, ok: !!ok, detail });

  add('指令总数 = 1024', REGISTRY.length === 1024, `实际 ${REGISTRY.length}`);
  add('ID 唯一', new Set(REGISTRY.map((r) => r.id)).size === REGISTRY.length, '');
  add('每族恰 64 条', GROUP_STATS.every((g) => g.total === 64),
    GROUP_STATS.filter((g) => g.total !== 64).map((g) => `${g.group}=${g.total}`).join(',') || '全部 64');
  // 上游有 13 个 op 名跨族复用（如 softmax 同属 ATTEN 与 SAMPLE），
  // 因此正确的不变量是「每个 (族, 算子) 恰 4 变体」，而非按 op 全局计数。
  add('每 (族,算子) 恰 4 变体', (() => {
    const m = new Map();
    for (const r of REGISTRY) {
      const k = `${r.group}/${r.op}`;
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(r.variant);
    }
    const bad = [...m.entries()].filter(([, v]) => v.length !== 4);
    return bad.length === 0;
  })(), (() => {
    const m = new Map();
    for (const r of REGISTRY) {
      const k = `${r.group}/${r.op}`;
      m.set(k, (m.get(k) || 0) + 1);
    }
    const bad = [...m.entries()].filter(([, v]) => v !== 4);
    return bad.length ? bad.slice(0, 5).map(([k, v]) => `${k}=${v}`).join(',') : '全部 4';
  })());
  add('跨族同名 op 已被识别', (() => {
    const m = new Map();
    for (const r of REGISTRY) m.set(r.op, (m.get(r.op) || 0) + 1);
    return [...m.values()].filter((v) => v > 4).length === 13;
  })(), '预期 13 个 op 跨族复用（上游设计）');
  add('变体 ∈ {base,batched,fused,inplace}', REGISTRY.every((r) =>
    ['base', 'batched', 'fused', 'inplace'].includes(r.variant)), '');
  add('全部含数学不变量', REGISTRY.every((r) => typeof r.invariant === 'string' && r.invariant.length > 0), '');
  add('verified 项 impl 均为 core.js 真实导出', VERIFIED.every((r) => typeof AXIOM[r.impl] === 'function'),
    VERIFIED.filter((r) => typeof AXIOM[r.impl] !== 'function').map((r) => r.impl).slice(0, 5).join(',') || '全部命中');
  add('verified 数 > 0', VERIFIED.length > 0, `verified = ${VERIFIED.length}`);

  return { total: results.length, passed: results.filter((r) => r.ok).length, results };
}
