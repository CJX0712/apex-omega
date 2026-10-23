#!/usr/bin/env node
/**
 * APEX-Ω · 门禁总跑
 * 用法：node test/run.js [--rounds N] [--json]
 */
import { runAllGates } from './gates.js';

const args = process.argv.slice(2);
const roundsIdx = args.indexOf('--rounds');
const rounds = roundsIdx >= 0 ? Number(args[roundsIdx + 1]) : 12;
const asJson = args.includes('--json');

const C = {
  reset: '\x1b[0m', dim: '\x1b[2m', red: '\x1b[31m', green: '\x1b[32m',
  yellow: '\x1b[33m', cyan: '\x1b[36m', bold: '\x1b[1m',
};

// --json 模式下 stdout 必须只有纯 JSON（供 CI 的 require() 直接解析），横幅一律不打印
if (!asJson) {
  console.log(`${C.bold}${C.cyan}APEX-Ω · 门禁总跑${C.reset}`);
  console.log(`${C.dim}甲类：ISA 内部不变量（AXIOM 386 项数学不变量 + ISA 完整性）
乙类：对外对拍（旗舰 ISA 参照采样器 vs llama.cpp 生产内核）${C.reset}\n`);
}

const t0 = Date.now();
const report = await runAllGates({ rounds });

if (asJson) {
  console.log(JSON.stringify(report, null, 2));
} else {
  let lastGroup = '';
  for (const r of report.results) {
    if (r.group !== lastGroup) {
      lastGroup = r.group;
      console.log(`\n${C.bold}── ${lastGroup} ──${C.reset}`);
    }
    const icon = r.pass ? `${C.green}✅${C.reset}` : `${C.red}❌${C.reset}`;
    const detail = r.detail ? ` ${C.dim}${r.detail}${C.reset}` : '';
    console.log(`  ${icon} ${r.name}${detail}`);
  }
  console.log('');
  const all = report.allGreen;
  console.log(`${C.bold}总计: ${report.passed}/${report.total} ${all ? C.green + 'PASS' : C.red + 'FAIL'}${C.reset}` +
    ` ${C.dim}· ${((Date.now() - t0) / 1000).toFixed(1)}s · ISA ${report.meta.isa.total} 指令 (verified ${report.meta.verified})${C.reset}`);
}

process.exit(report.allGreen ? 0 : 1);
