# vendor/axiom — 零改动 vendored 副本

本目录包含 [CJX0712/axiom-ai](https://github.com/CJX0712/axiom-ai) v1.3.26 的
**零改动**副本，仅做两处适配：

| 上游文件 | 本地文件 | 改动 | 原因 |
|---|---|---|---|
| `src/core.js` | `core.cjs` | 仅改扩展名 | 本项目为 ESM（`"type": "module"`），`.js` 会被当 ESM 解析导致 `module.exports` 失效 |
| `src/registry.js` | `registry.cjs` | 仅改扩展名 | 同上 |

**代码内容零改动**，原始 MIT 许可证保留于 `LICENSE`。

- `core.cjs` — 415 个导出：数学/统计/优化/概率图/RL/注意力/梯度等参考实现，
  每项带可被独立实现交叉验证的数学不变量；`runChecks()` 为 386 项权威自检。
- `registry.cjs` — 1024 条微指令注册表（16 族 × 16 算子 × 4 执行变体），
  每条含语义契约、数学不变量、实现状态。

上游仓库：https://github.com/CJX0712/axiom-ai · 作者 晨星 · MIT
