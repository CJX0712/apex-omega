/**
 * APEX-Ω · 门禁总装（再导出）
 * ---------------------------------------------------------------------------
 * 门禁的权威实现位于 src/gates.js（服务层同样需要它）。
 * 本文件仅为兼容既有 `node test/run.js` 调用路径而保留。
 */
export {
  runIsaGates,
  runDuelGates,
  runAllGates,
} from '../src/gates.js';
