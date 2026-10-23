#!/usr/bin/env node
/**
 * APEX-Ω · 构建单文件 HTML
 * 把仪表盘导出为独立可双击打开的单文件（零外部请求）。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DASHBOARD_HTML } from '../src/layers/l9-dashboard.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'dist');
const outFile = path.join(outDir, 'apex-omega.html');

fs.mkdirSync(outDir, { recursive: true });

const html = DASHBOARD_HTML;
fs.writeFileSync(outFile, html, 'utf8');

// 构建期断言：单文件不得含外部资源引用
const externals = [
  /<script[^>]+src=/i,
  /<link[^>]+href=["']https?:/i,
  /@import\s+url\(/i,
  /fetch\(['"]https?:/i,
];
const violations = externals.filter((re) => re.test(html));

const stats = {
  file: path.relative(root, outFile).replace(/\\/g, '/'),
  bytes: Buffer.byteLength(html, 'utf8'),
  lines: html.split('\n').length,
  externals: violations.length,
};

console.log('APEX-Ω · 单文件构建');
console.log(`  输出   ${stats.file}`);
console.log(`  大小   ${(stats.bytes / 1024).toFixed(1)} KB`);
console.log(`  行数   ${stats.lines}`);
console.log(`  外部资源引用 ${stats.externals === 0 ? '0 ✅（零依赖）' : stats.externals + ' ❌'}`);

if (violations.length) {
  console.error('\n❌ 构建失败：单文件不得含外部资源引用');
  process.exit(1);
}
console.log('\n✅ 单文件构建完成 —— 可双击离线打开');
