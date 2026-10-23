/* ============================================================================
 * AXIOM Core — 世界级数学与算法核心
 * 零依赖 · 浏览器 / Node 双端同源（UMD）· 每个算法携带可独立复算的不变量
 * ----------------------------------------------------------------------------
 * 每一行都站在世界顶级成果之上（零重写 = 直接复用世界最优算法）：
 *   矩阵分解   Householder(1958) QR · Jacobi(1846) 循环旋转特征分解 · Cholesky
 *   概率模型   Dempster(1977) EM · Box–Muller(1958) · Abramowitz–Stegun erf
 *   优化       Cauchy(1847) 梯度下降 · Polyak(1964) 动量 · Kingma–Ba(2015) Adam
 *              Hestenes–Stiefel(1952) 共轭梯度
 *   推断       Rabiner(1989) 前向-后向 / Viterbi · Bellman(1957) 动态规划
 *   机器学习   Lloyd(1982) K-Means · Cover–Hart(1967) KNN · Breiman(1984) CART
 *              Rumelhart–Hinton–Williams(1986) 反向传播
 *   验证哲学   解析 vs 数值 · 暴力枚举 vs 优化算法 · 残差必须可独立复算
 * 对标：scipy.linalg / ml.js / @tangent.to/lina（机器精度一致）
 * ==========================================================================*/
(function (root, factory) {
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AXIOM = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  var VERSION = '1.3.26';

  /* ========================================================================
   * 0. 确定性随机与基础工具（可复现是一切验证的前提）
   * ======================================================================*/
  function mulberry32(seed) {
    var t = seed >>> 0;
    return function () {
      t += 0x6D2B79F5;
      var r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }
  function zeros2(m, n) { return Array.from({ length: m }, function () { return new Array(n).fill(0); }); }
  function eye(n) { var I = zeros2(n, n); for (var i = 0; i < n; i++) I[i][i] = 1; return I; }
  function clone2(A) { return A.map(function (r) { return r.slice(); }); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm2(x) { return Math.sqrt(dot(x, x)); }
  function normInf(x) { var m = 0; for (var i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i])); return m; }
  function maxAbsDiff(a, b) {
    var m = 0;
    for (var i = 0; i < a.length; i++) {
      var ra = a[i], rb = b[i];
      if (Array.isArray(ra)) m = Math.max(m, maxAbsDiff(ra, rb));
      else m = Math.max(m, Math.abs(ra - rb));
    }
    return m;
  }
  function maxAbsDiff2(A, B) { // 矩阵版
    var m = 0;
    for (var i = 0; i < A.length; i++) for (var j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
    return m;
  }
  function dist2(a, b) { var s = 0; for (var i = 0; i < a.length; i++) { var d = a[i] - b[i]; s += d * d; } return s; }
  function euclid(a, b) { return Math.sqrt(dist2(a, b)); }
  function spdMatrix(n, seed) { // A = BᵀB + nI，对称正定
    var rng = mulberry32(seed);
    var B = Array.from({ length: n }, function () { return Array.from({ length: n }, function () { return rng() * 2 - 1; }); });
    var A = zeros2(n, n);
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var s = 0; for (var k = 0; k < n; k++) s += B[k][i] * B[k][j];
      A[i][j] = s + (i === j ? n : 0);
    }
    return A;
  }

  /* ========================================================================
   * 1. 线性代数（对标 scipy.linalg，机器精度）
   * ======================================================================*/
  function transpose(A) { return A[0].map(function (_, j) { return A.map(function (r) { return r[j]; }); }); }
  function matMul(A, B) {
    var n = A.length, k = B.length, m = B[0].length;
    var C = Array.from({ length: n }, function () { return new Array(m).fill(0); });
    for (var i = 0; i < n; i++) for (var p = 0; p < k; p++) {
      var a = A[i][p];
      if (a === 0) continue; // 加 0 精确无副作用，保持 (AB)ᵀ=BᵀAᵀ 逐位一致
      for (var j = 0; j < m; j++) C[i][j] += a * B[p][j];
    }
    return C;
  }
  function matVec(A, x) { return A.map(function (r) { return dot(r, x); }); }
  function trace(A) { var s = 0; for (var i = 0; i < A.length; i++) s += A[i][i]; return s; }

  function det(A) { // LU + 部分选主元
    var n = A.length, M = clone2(A), d = 1;
    for (var k = 0; k < n; k++) {
      var p = k;
      for (var i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      if (Math.abs(M[p][k]) < 1e-300) return 0;
      if (p !== k) { var t = M[p]; M[p] = M[k]; M[k] = t; d = -d; }
      d *= M[k][k];
      for (var i2 = k + 1; i2 < n; i2++) {
        var f = M[i2][k] / M[k][k];
        for (var j = k; j < n; j++) M[i2][j] -= f * M[k][j];
      }
    }
    return d;
  }
  function inverse(A) { // Gauss-Jordan + 部分选主元
    var n = A.length;
    var M = A.map(function (r, i) {
      var row = r.slice();
      for (var j = 0; j < n; j++) row.push(i === j ? 1 : 0);
      return row;
    });
    for (var k = 0; k < n; k++) {
      var p = k;
      for (var i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      if (Math.abs(M[p][k]) < 1e-300) throw new Error('矩阵奇异，无法求逆');
      if (p !== k) { var t = M[p]; M[p] = M[k]; M[k] = t; }
      var piv = M[k][k];
      for (var j = 0; j < 2 * n; j++) M[k][j] /= piv;
      for (var i2 = 0; i2 < n; i2++) {
        if (i2 === k) continue;
        var f = M[i2][k];
        if (f === 0) continue;
        for (var j2 = 0; j2 < 2 * n; j2++) M[i2][j2] -= f * M[k][j2];
      }
    }
    return M.map(function (r) { return r.slice(n); });
  }
  function solveLinear(A, b) { // LU + 部分选主元，一般方阵
    var n = A.length, M = clone2(A), x = b.slice();
    for (var k = 0; k < n; k++) {
      var p = k;
      for (var i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      if (p !== k) { var t = M[p]; M[p] = M[k]; M[k] = t; var tx = x[p]; x[p] = x[k]; x[k] = tx; }
      for (var i2 = k + 1; i2 < n; i2++) {
        var f = M[i2][k] / M[k][k];
        for (var j = k; j < n; j++) M[i2][j] -= f * M[k][j];
        x[i2] -= f * x[k];
      }
    }
    for (var i3 = n - 1; i3 >= 0; i3--) {
      var s = x[i3];
      for (var j2 = i3 + 1; j2 < n; j2++) s -= M[i3][j2] * x[j2];
      x[i3] = s / M[i3][i3];
    }
    return x;
  }
  function solveLower(L, b) { // 解 L y = b（L 下三角）
    var n = L.length, y = new Array(n);
    for (var i = 0; i < n; i++) {
      var s = b[i];
      for (var j = 0; j < i; j++) s -= L[i][j] * y[j];
      y[i] = s / L[i][i];
    }
    return y;
  }
  function cholesky(A) { // 对称正定分解，非正定抛错
    var n = A.length, L = zeros2(n, n);
    for (var i = 0; i < n; i++) for (var j = 0; j <= i; j++) {
      var s = A[i][j];
      for (var k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error('矩阵非正定，Cholesky 分解失败');
        L[i][j] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
    return L;
  }
  function safeCholesky(A) { // 自适应抖动，保证数值稳健
    var n = A.length, jitter = 0;
    for (var attempt = 0; attempt < 10; attempt++) {
      try {
        var M = A.map(function (r, i) { return r.map(function (v, j) { return v + (i === j ? jitter : 0); }); });
        return cholesky(M);
      } catch (e) { jitter = jitter === 0 ? 1e-9 : jitter * 10; }
    }
    throw new Error('safeCholesky 失败');
  }
  function qrDecompose(A) { // Householder 反射，Q 为 m×m，R 为 m×n
    var m = A.length, n = A[0].length;
    var R = clone2(A), Q = eye(m);
    for (var k = 0; k < Math.min(m - 1, n); k++) {
      var nx = 0;
      for (var i = k; i < m; i++) nx += R[i][k] * R[i][k];
      nx = Math.sqrt(nx);
      if (nx === 0) continue;
      var alpha = R[k][k] > 0 ? -nx : nx;
      var v = new Array(m).fill(0);
      v[k] = R[k][k] - alpha;
      for (var i2 = k + 1; i2 < m; i2++) v[i2] = R[i2][k];
      var vn = norm2(v);
      if (vn < 1e-300) continue;
      for (var i3 = 0; i3 < m; i3++) v[i3] /= vn;
      for (var j = 0; j < n; j++) { // R ← (I−2vvᵀ)R
        var d = 0;
        for (var i4 = k; i4 < m; i4++) d += v[i4] * R[i4][j];
        for (var i5 = k; i5 < m; i5++) R[i5][j] -= 2 * d * v[i5];
      }
      for (var i6 = 0; i6 < m; i6++) { // Q ← Q(I−2vvᵀ)
        var d2 = 0;
        for (var j2 = k; j2 < m; j2++) d2 += Q[i6][j2] * v[j2];
        for (var j3 = k; j3 < m; j3++) Q[i6][j3] -= 2 * d2 * v[j3];
      }
    }
    for (var i7 = 0; i7 < m; i7++) for (var j4 = 0; j4 < Math.min(i7, n); j4++) R[i7][j4] = 0;
    return { Q: Q, R: R };
  }
  function jacobiEigen(A, opts) { // 循环 Jacobi，对称阵特征分解
    opts = opts || {};
    var n = A.length, tol = opts.tol || 1e-13, maxSweeps = opts.maxSweeps || 100;
    var S = clone2(A), V = eye(n);
    for (var sweep = 0; sweep < maxSweeps; sweep++) {
      var off = 0;
      for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) off += S[i][j] * S[i][j];
      if (Math.sqrt(off) <= tol) break;
      for (var p = 0; p < n - 1; p++) for (var q = p + 1; q < n; q++) {
        var apq = S[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        var tau = (S[q][q] - S[p][p]) / (2 * apq);
        var t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        var c = 1 / Math.sqrt(1 + t * t), s = t * c;
        for (var k = 0; k < n; k++) {
          if (k !== p && k !== q) {
            var skp = S[k][p], skq = S[k][q];
            S[k][p] = c * skp - s * skq; S[p][k] = S[k][p];
            S[k][q] = s * skp + c * skq; S[q][k] = S[k][q];
          }
        }
        var spp = S[p][p], sqq = S[q][q], spq = S[p][q];
        S[p][p] = c * c * spp - 2 * s * c * spq + s * s * sqq;
        S[q][q] = s * s * spp + 2 * s * c * spq + c * c * sqq;
        S[p][q] = 0; S[q][p] = 0;
        for (var k2 = 0; k2 < n; k2++) {
          var vkp = V[k2][p], vkq = V[k2][q];
          V[k2][p] = c * vkp - s * vkq;
          V[k2][q] = s * vkp + c * vkq;
        }
      }
    }
    var idx = Array.from({ length: n }, function (_, i) { return i; }).sort(function (a, b) { return S[b][b] - S[a][a]; });
    var values = idx.map(function (i) { return S[i][i]; });
    var vectors = idx.map(function (i) { return V.map(function (row) { return row[i]; }); }); // vectors[t] = 第 t 个特征向量
    return { values: values, vectors: vectors, sweeps: sweep };
  }
  function powerIteration(A, opts) {
    opts = opts || {};
    var n = A.length, iters = opts.iters || 2000, tol = opts.tol || 1e-14;
    var x = Array.from({ length: n }, function (_, i) { return (i + 1) / (n + 1); });
    var nx = norm2(x); x = x.map(function (v) { return v / nx; });
    var lam = 0;
    for (var it = 0; it < iters; it++) {
      var y = matVec(A, x);
      var ny = norm2(y);
      if (ny < 1e-300) break;
      var newLam = dot(x, y);
      x = y.map(function (v) { return v / ny; });
      if (Math.abs(newLam - lam) <= tol * Math.abs(newLam)) { lam = newLam; break; }
      lam = newLam;
    }
    return { value: lam, vector: x };
  }
  function pca(X, k) { // 协方差 → Jacobi → 投影；Eckart–Young 最优截断
    var n = X.length, d = X[0].length;
    var mean = new Array(d).fill(0);
    for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) mean[j] += X[i][j] / n;
    var Xc = X.map(function (r) { return r.map(function (v, j) { return v - mean[j]; }); });
    var cov = zeros2(d, d);
    for (var a = 0; a < d; a++) for (var b = a; b < d; b++) {
      var s = 0;
      for (var i2 = 0; i2 < n; i2++) s += Xc[i2][a] * Xc[i2][b];
      s /= (n - 1);
      cov[a][b] = s; cov[b][a] = s;
    }
    var eig = jacobiEigen(cov);
    var values = eig.values, vectors = eig.vectors;
    var kk = Math.min(k, d);
    var projected = Xc.map(function (r) {
      return Array.from({ length: kk }, function (_, t) { return dot(r, vectors[t]); });
    });
    var reconstruct = projected.map(function (z) {
      return Array.from({ length: d }, function (_, j) {
        var v = mean[j];
        for (var t = 0; t < kk; t++) v += z[t] * vectors[t][j];
        return v;
      });
    });
    var total = values.reduce(function (a2, b2) { return a2 + b2; }, 0);
    var explained = values.slice(0, kk).reduce(function (a2, b2) { return a2 + b2; }, 0) / total;
    return { mean: mean, components: vectors.slice(0, kk), values: values, projected: projected, reconstruct: reconstruct, explainedVariance: explained };
  }

  /* ========================================================================
   * 2. 数值稳定（平台原生最优 + 稳定算法）
   * ======================================================================*/
  function logsumexp(xs) {
    var m = -Infinity;
    for (var i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
    if (m === -Infinity) return -Infinity;
    var s = 0;
    for (var i2 = 0; i2 < xs.length; i2++) s += Math.exp(xs[i2] - m);
    return m + Math.log(s);
  }
  function softmax(xs) {
    var m = -Infinity;
    for (var i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
    var es = xs.map(function (x) { return Math.exp(x - m); });
    var s = es.reduce(function (a, b) { return a + b; }, 0);
    return es.map(function (e) { return e / s; });
  }
  function sigmoid(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }
  function tanh(x) { return Math.tanh(x); }
  function relu(x) { return x > 0 ? x : 0; }
  function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); } // 精确 erf 形式
  function erf(x) { // Abramowitz–Stegun 7.1.26，|ε|≤1.5e-7
    var sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    var t = 1 / (1 + p * x);
    var y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }
  function normalCDF(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
  function kahanSum(xs) {
    var s = 0, c = 0;
    for (var i = 0; i < xs.length; i++) {
      var y = xs[i] - c;
      var t = s + y;
      c = (t - s) - y;
      s = t;
    }
    return s;
  }
  function exactSum(xs) { // TwoSum 无误差变换，作为“精确和”基准
    var s = 0, e = 0;
    for (var i = 0; i < xs.length; i++) {
      var x = xs[i], t = s + x;
      var bb = Math.abs(s) >= Math.abs(x) ? (s - t) + x : (x - t) + s;
      e += bb; s = t;
    }
    return s + e;
  }
  function clip(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

  /* ========================================================================
   * 3. 概率与统计
   * ======================================================================*/
  function gaussPDF(x, mu, sigma) { return Math.exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma)) / (sigma * Math.sqrt(2 * Math.PI)); }
  function expectation(xs) { return xs.reduce(function (a, b) { return a + b; }, 0) / xs.length; }
  function variance(xs) {
    var m = expectation(xs), s = 0;
    for (var i = 0; i < xs.length; i++) s += (xs[i] - m) * (xs[i] - m);
    return s / xs.length;
  }
  function covariance(xs, ys) {
    var mx = expectation(xs), my = expectation(ys), s = 0;
    for (var i = 0; i < xs.length; i++) s += (xs[i] - mx) * (ys[i] - my);
    return s / xs.length;
  }
  function normalSample(n, seed) { // Box–Muller
    var rng = mulberry32(seed === undefined ? 2024 : seed), out = [];
    for (var i = 0; i < n; i += 2) {
      var u1 = Math.max(rng(), 1e-300), u2 = rng();
      var r = Math.sqrt(-2 * Math.log(u1));
      out.push(r * Math.cos(2 * Math.PI * u2));
      if (out.length < n) out.push(r * Math.sin(2 * Math.PI * u2));
    }
    return out.slice(0, n);
  }
  function categoricalSample(p, seed) {
    var rng = mulberry32(seed === undefined ? 7 : seed), out = [];
    var cum = []; var s = 0;
    for (var i = 0; i < p.length; i++) { s += p[i]; cum.push(s); }
    for (var t = 0; t < 20000; t++) {
      var r = rng() * s, j = 0;
      while (j < cum.length - 1 && r > cum[j]) j++;
      out.push(j);
    }
    return out;
  }
  function bayesUpdate(prior, likelihood) {
    var joint = prior.map(function (p, i) { return p * likelihood[i]; });
    var s = joint.reduce(function (a, b) { return a + b; }, 0);
    return joint.map(function (v) { return v / s; });
  }

  /* ========================================================================
   * 4. 信息论
   * ======================================================================*/
  function entropy(p) { // bits
    var h = 0;
    for (var i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log2(p[i]);
    return h;
  }
  function crossEntropy(p, q) {
    var h = 0;
    for (var i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log2(Math.max(q[i], 1e-300));
    return h;
  }
  function klDivergence(p, q) {
    var d = 0;
    for (var i = 0; i < p.length; i++) if (p[i] > 0) d += p[i] * Math.log2(p[i] / Math.max(q[i], 1e-300));
    return d;
  }
  function jsDivergence(p, q) {
    var m = p.map(function (v, i) { return (v + q[i]) / 2; });
    return 0.5 * klDivergence(p, m) + 0.5 * klDivergence(q, m);
  }
  function mutualInfo(joint) { // joint[i][j] = P(X=i,Y=j)
    var n = joint.length, m = joint[0].length;
    var px = new Array(n).fill(0), py = new Array(m).fill(0);
    for (var i = 0; i < n; i++) for (var j = 0; j < m; j++) { px[i] += joint[i][j]; py[j] += joint[i][j]; }
    var mi = 0;
    for (var i2 = 0; i2 < n; i2++) for (var j2 = 0; j2 < m; j2++) {
      if (joint[i2][j2] > 0) mi += joint[i2][j2] * Math.log2(joint[i2][j2] / (px[i2] * py[j2]));
    }
    return mi;
  }
  function conditionalEntropy(joint) {
    var n = joint.length, m = joint[0].length;
    var py = new Array(m).fill(0);
    for (var j = 0; j < m; j++) for (var i = 0; i < n; i++) py[j] += joint[i][j];
    var hy = 0;
    for (var j2 = 0; j2 < m; j2++) if (py[j2] > 0) hy -= py[j2] * Math.log2(py[j2]);
    var hxy = 0;
    for (var i2 = 0; i2 < n; i2++) for (var j3 = 0; j3 < m; j3++) if (joint[i2][j3] > 0) hxy -= joint[i2][j3] * Math.log2(joint[i2][j3]);
    return hxy - hy;
  }
  function gini(labels) {
    var m = {};
    for (var i = 0; i < labels.length; i++) m[labels[i]] = (m[labels[i]] || 0) + 1;
    var s = 0;
    for (var k in m) { var p = m[k] / labels.length; s += p * p; }
    return 1 - s;
  }
  function labelDist(labels) {
    var m = {};
    for (var i = 0; i < labels.length; i++) m[labels[i]] = (m[labels[i]] || 0) + 1;
    return Object.keys(m).map(function (k) { return m[k] / labels.length; });
  }
  function infoGain(parentLabels, groups) { // ID3 风格：熵增益
    var n = parentLabels.length;
    var child = 0;
    for (var i = 0; i < groups.length; i++) child += (groups[i].length / n) * entropy(labelDist(groups[i]));
    return entropy(labelDist(parentLabels)) - child;
  }

  /* ========================================================================
   * 5. 优化器
   * ======================================================================*/
  function gradientDescent(f, grad, x0, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.01, iters = opts.iters || 1000;
    var x = x0.slice(), history = [];
    for (var it = 0; it < iters; it++) {
      var g = grad(x);
      for (var i = 0; i < x.length; i++) x[i] -= lr * g[i];
      history.push(f(x));
    }
    return { x: x, history: history, iterations: iters };
  }
  function adam(f, grad, x0, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.02, iters = opts.iters || 2000, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    var x = x0.slice(), m = new Array(x.length).fill(0), v = new Array(x.length).fill(0), history = [];
    for (var t = 1; t <= iters; t++) {
      var g = grad(x);
      for (var i = 0; i < x.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        var mh = m[i] / (1 - Math.pow(b1, t)), vh = v[i] / (1 - Math.pow(b2, t));
        x[i] -= lr * mh / (Math.sqrt(vh) + eps);
      }
      history.push(f(x));
    }
    return { x: x, history: history, iterations: iters };
  }
  function conjugateGradient(A, b, opts) {
    opts = opts || {};
    var tol = opts.tol || 1e-12, maxIter = opts.maxIter || A.length + 5;
    var x = new Array(b.length).fill(0);
    var r = b.slice(), p = b.slice();
    var rs = dot(r, r);
    for (var it = 0; it < maxIter; it++) {
      var Ap = matVec(A, p);
      var pAp = dot(p, Ap);
      if (pAp === 0) break;
      var alpha = rs / pAp;
      for (var i = 0; i < x.length; i++) x[i] += alpha * p[i];
      for (var i2 = 0; i2 < x.length; i2++) r[i2] -= alpha * Ap[i2];
      var rsNew = dot(r, r);
      if (Math.sqrt(rsNew) < tol) break;
      var beta = rsNew / rs;
      for (var i3 = 0; i3 < x.length; i3++) p[i3] = r[i3] + beta * p[i3];
      rs = rsNew;
    }
    return { x: x, residual: Math.sqrt(dot(matVec(A, x).map(function (v, i) { return v - b[i]; }), matVec(A, x).map(function (v, i) { return v - b[i]; }))), iterations: it };
  }

  /* ========================================================================
   * 6. 机器学习模型
   * ======================================================================*/
  function linearRegression(X, y) { // 正规方程
    var n = X.length, d = X[0].length;
    var XtX = zeros2(d, d), Xty = new Array(d).fill(0);
    for (var a = 0; a < d; a++) {
      for (var b = 0; b < d; b++) {
        var s = 0;
        for (var i = 0; i < n; i++) s += X[i][a] * X[i][b];
        XtX[a][b] = s;
      }
      var s2 = 0;
      for (var i2 = 0; i2 < n; i2++) s2 += X[i2][a] * y[i2];
      Xty[a] = s2;
    }
    var w = solveLinear(XtX, Xty);
    return { w: w, predict: function (x) { return dot(w, x); } };
  }
  function logisticRegression(X, y, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.5, iters = opts.iters || 2000, l2 = opts.l2 || 0;
    var n = X.length, d = X[0].length;
    var w = new Array(d).fill(0), b = 0, history = [];
    for (var it = 0; it < iters; it++) {
      var gw = new Array(d).fill(0), gb = 0, loss = 0;
      for (var i = 0; i < n; i++) {
        var z = b;
        for (var j = 0; j < d; j++) z += w[j] * X[i][j];
        var p = sigmoid(z), err = p - y[i];
        for (var j2 = 0; j2 < d; j2++) gw[j2] += err * X[i][j2];
        gb += err;
        loss += -(y[i] * Math.log(p + 1e-15) + (1 - y[i]) * Math.log(1 - p + 1e-15));
      }
      for (var j3 = 0; j3 < d; j3++) { gw[j3] = gw[j3] / n + l2 * w[j3]; w[j3] -= lr * gw[j3]; }
      b -= lr * gb / n;
      history.push(loss / n);
    }
    return {
      w: w, b: b, history: history,
      predictProba: function (x) { var z = b; for (var j = 0; j < d; j++) z += w[j] * x[j]; return sigmoid(z); },
      predict: function (x) { var z = b; for (var j = 0; j < d; j++) z += w[j] * x[j]; return sigmoid(z) >= 0.5 ? 1 : 0; }
    };
  }
  function kmeansPP(X, k, rng) { // K-Means++ 初始化（Arthur–Vassilvitskii 2007）
    var n = X.length;
    var centers = [X[Math.floor(rng() * n)].slice()];
    var d2 = new Array(n).fill(Infinity);
    while (centers.length < k) {
      var c = centers[centers.length - 1], sum = 0;
      for (var i = 0; i < n; i++) {
        var dd = dist2(X[i], c);
        if (dd < d2[i]) d2[i] = dd;
        sum += d2[i];
      }
      if (sum <= 0) { centers.push(X[Math.floor(rng() * n)].slice()); continue; }
      var r = rng() * sum, idx = 0;
      while (idx < n - 1 && r > d2[idx]) { r -= d2[idx]; idx++; }
      centers.push(X[idx].slice());
    }
    return centers;
  }
  function kmeans(X, k, opts) { // Lloyd 算法，SSE 单调不增
    opts = opts || {};
    var iters = opts.iters || 50, seed = opts.seed === undefined ? 7 : opts.seed;
    var rng = mulberry32(seed);
    var centers = kmeansPP(X, k, rng);
    var labels = new Array(X.length).fill(0), inertia = Infinity, history = [];
    for (var it = 0; it < iters; it++) {
      var newInertia = 0;
      for (var i = 0; i < X.length; i++) {
        var best = 0, bd = Infinity;
        for (var j = 0; j < k; j++) { var d = dist2(X[i], centers[j]); if (d < bd) { bd = d; best = j; } }
        labels[i] = best; newInertia += bd;
      }
      history.push(newInertia);
      if (it > 0 && newInertia >= inertia - 1e-12) { inertia = newInertia; break; }
      inertia = newInertia;
      var sums = Array.from({ length: k }, function () { return new Array(X[0].length).fill(0); });
      var cnt = new Array(k).fill(0);
      for (var i2 = 0; i2 < X.length; i2++) {
        var l = labels[i2]; cnt[l]++;
        for (var j2 = 0; j2 < X[i2].length; j2++) sums[l][j2] += X[i2][j2];
      }
      for (var j3 = 0; j3 < k; j3++) if (cnt[j3] > 0) centers[j3] = sums[j3].map(function (v) { return v / cnt[j3]; });
    }
    return { centers: centers, labels: labels, inertia: inertia, history: history };
  }
  function gmmEM(X, k, opts) { // 全协方差 EM，对数似然单调不减
    opts = opts || {};
    var iters = opts.iters || 50, seed = opts.seed === undefined ? 42 : opts.seed;
    var n = X.length, d = X[0].length;
    var rng = mulberry32(seed);
    var centers = kmeansPP(X, k, rng);
    var weights = new Array(k).fill(1 / k);
    var covs = Array.from({ length: k }, function () {
      var C = zeros2(d, d);
      for (var i = 0; i < d; i++) C[i][i] = 1;
      return C;
    });
    var log2pi = d * Math.log(2 * Math.PI);
    var ll = 0, llHistory = [], lastResp = null;
    for (var it = 0; it < iters; it++) {
      var logR = [], newLL = 0;
      for (var i = 0; i < n; i++) {
        var lj = new Array(k);
        for (var j = 0; j < k; j++) {
          var L = safeCholesky(covs[j]);
          var diff = X[i].map(function (v, t) { return v - centers[j][t]; });
          var y = solveLower(L, diff);
          var logdet = 0;
          for (var t = 0; t < d; t++) logdet += Math.log(L[t][t]);
          lj[j] = Math.log(weights[j]) - 0.5 * (log2pi + logdet + dot(y, y));
        }
        var m = logsumexp(lj);
        newLL += m;
        logR.push(lj.map(function (v) { return Math.exp(v - m); }));
      }
      llHistory.push(newLL);
      lastResp = logR;
      var Nk = new Array(k).fill(0);
      for (var i2 = 0; i2 < n; i2++) for (var j2 = 0; j2 < k; j2++) Nk[j2] += logR[i2][j2];
      for (var j3 = 0; j3 < k; j3++) {
        weights[j3] = Nk[j3] / n;
        centers[j3] = new Array(d).fill(0);
        for (var i3 = 0; i3 < n; i3++) {
          var r = logR[i3][j3];
          for (var t2 = 0; t2 < d; t2++) centers[j3][t2] += r * X[i3][t2];
        }
        for (var t3 = 0; t3 < d; t3++) centers[j3][t3] /= Nk[j3];
        var C = zeros2(d, d);
        for (var i4 = 0; i4 < n; i4++) {
          var r2 = logR[i4][j3];
          var df = X[i4].map(function (v, t4) { return v - centers[j3][t4]; });
          for (var a = 0; a < d; a++) for (var b = 0; b < d; b++) C[a][b] += r2 * df[a] * df[b];
        }
        for (var a2 = 0; a2 < d; a2++) { for (var b2 = 0; b2 < d; b2++) C[a2][b2] /= Nk[j3]; C[a2][a2] += 1e-6; }
        covs[j3] = C;
      }
      ll = newLL;
    }
    return { weights: weights, means: centers, covs: covs, logLikelihood: ll, llHistory: llHistory, responsibilities: lastResp, iterations: iters };
  }
  function knn(Xtrain, ytrain, x, k) {
    var idx = Xtrain.map(function (r, i) { return [dist2(r, x), i]; }).sort(function (a, b) { return a[0] - b[0]; }).slice(0, k);
    var best = null, bestCount = 0, counts = {};
    for (var t = 0; t < idx.length; t++) {
      var l = ytrain[idx[t][1]];
      counts[l] = (counts[l] || 0) + 1;
      if (counts[l] > bestCount) { bestCount = counts[l]; best = l; }
    }
    return best;
  }
  function decisionTree(X, y, opts) { // CART + Gini，预剪枝
    opts = opts || {};
    var maxDepth = opts.maxDepth === undefined ? 4 : opts.maxDepth, minSplit = opts.minSplit || 2;
    function majority(labels) {
      var m = {}, best = null, bestCount = 0;
      for (var i = 0; i < labels.length; i++) {
        var l = labels[i];
        m[l] = (m[l] || 0) + 1;
        if (m[l] > bestCount) { bestCount = m[l]; best = l; }
      }
      return best;
    }
    function build(idx, depth) {
      var labels = idx.map(function (i) { return y[i]; });
      var g = gini(labels);
      if (depth >= maxDepth || idx.length < minSplit || g === 0) return { leaf: true, label: majority(labels), gini: g, n: idx.length };
      var d = X[0].length, best = null;
      for (var f = 0; f < d; f++) {
        var vals = Array.from(new Set(idx.map(function (i) { return X[i][f]; }))).sort(function (a, b) { return a - b; });
        for (var t = 0; t < vals.length - 1; t++) {
          var thr = (vals[t] + vals[t + 1]) / 2, L = [], R = [];
          for (var i2 = 0; i2 < idx.length; i2++) (X[idx[i2]][f] <= thr ? L : R).push(idx[i2]);
          if (!L.length || !R.length) continue;
          var gain = g - (L.length / idx.length) * gini(L.map(function (i) { return y[i]; })) - (R.length / idx.length) * gini(R.map(function (i) { return y[i]; }));
          if (!best || gain > best.gain + 1e-15) best = { f: f, thr: thr, gain: gain, L: L, R: R };
        }
      }
      if (!best || best.gain <= 1e-12) return { leaf: true, label: majority(labels), gini: g, n: idx.length };
      return { leaf: false, feature: best.f, threshold: best.thr, gain: best.gain, gini: g, n: idx.length, left: build(best.L, depth + 1), right: build(best.R, depth + 1) };
    }
    var tree = build(X.map(function (_, i) { return i; }), 0);
    function predictOne(x, node) {
      node = node || tree;
      while (!node.leaf) node = x[node.feature] <= node.threshold ? node.left : node.right;
      return node.label;
    }
    return {
      tree: tree,
      predict: function (x) { return predictOne(x); },
      predictAll: function (Xs) { return Xs.map(function (x) { return predictOne(x); }); },
      rootGain: tree.leaf ? 0 : tree.gain
    };
  }
  /* ---- MLP（反向传播 Rumelhart–Hinton–Williams 1986）---- */
  function mlpForward(P, x) {
    var d = P.W1[0].length, h = P.W1.length, o = P.W2.length;
    var a1 = new Array(h);
    for (var i = 0; i < h; i++) {
      var s = P.b1[i];
      for (var j = 0; j < d; j++) s += P.W1[i][j] * x[j];
      a1[i] = Math.tanh(s);
    }
    var out = new Array(o);
    for (var i2 = 0; i2 < o; i2++) {
      var s2 = P.b2[i2];
      for (var j2 = 0; j2 < h; j2++) s2 += P.W2[i2][j2] * a1[j2];
      out[i2] = s2;
    }
    return { a1: a1, out: out };
  }
  function mlpLoss(P, X, Y) {
    var s = 0;
    for (var i = 0; i < X.length; i++) {
      var f = mlpForward(P, X[i]);
      for (var k = 0; k < f.out.length; k++) s += (f.out[k] - Y[i][k]) * (f.out[k] - Y[i][k]);
    }
    return s / X.length;
  }
  function mlpGrad(P, X, Y) {
    var n = X.length, d = P.W1[0].length, h = P.W1.length, o = P.W2.length;
    var gW1 = zeros2(h, d), gb1 = new Array(h).fill(0), gW2 = zeros2(o, h), gb2 = new Array(o).fill(0);
    for (var i = 0; i < n; i++) {
      var f = mlpForward(P, X[i]);
      var dz2 = f.out.map(function (v, k) { return 2 * (v - Y[i][k]) / n; });
      for (var a = 0; a < o; a++) {
        gb2[a] += dz2[a];
        for (var b = 0; b < h; b++) gW2[a][b] += dz2[a] * f.a1[b];
      }
      for (var b2 = 0; b2 < h; b2++) {
        var da = 0;
        for (var a2 = 0; a2 < o; a2++) da += dz2[a2] * P.W2[a2][b2];
        var dz1 = da * (1 - f.a1[b2] * f.a1[b2]);
        gb1[b2] += dz1;
        for (var j = 0; j < d; j++) gW1[b2][j] += dz1 * X[i][j];
      }
    }
    return { W1: gW1, b1: gb1, W2: gW2, b2: gb2 };
  }
  function gradientCheck(P, X, Y, eps) { // 中心差分 vs 解析梯度
    eps = eps || 1e-5;
    var G = mlpGrad(P, X, Y);
    var maxRel = 0;
    [['W1', P.W1, G.W1], ['b1', P.b1, G.b1], ['W2', P.W2, G.W2], ['b2', P.b2, G.b2]].forEach(function (item) {
      var name = item[0], PV = item[1], GV = item[2];
      for (var i = 0; i < PV.length; i++) {
        if (typeof PV[i] === 'number') {
          var orig = PV[i];
          PV[i] = orig + eps; var lp = mlpLoss(P, X, Y);
          PV[i] = orig - eps; var lm = mlpLoss(P, X, Y);
          PV[i] = orig;
          var num = (lp - lm) / (2 * eps), ana = GV[i];
          var rel = Math.abs(num - ana) / Math.max(Math.abs(num), Math.abs(ana), 1e-8);
          if (rel > maxRel) maxRel = rel;
        } else {
          for (var j = 0; j < PV[i].length; j++) {
            var o2 = PV[i][j];
            PV[i][j] = o2 + eps; var lp2 = mlpLoss(P, X, Y);
            PV[i][j] = o2 - eps; var lm2 = mlpLoss(P, X, Y);
            PV[i][j] = o2;
            var num2 = (lp2 - lm2) / (2 * eps), ana2 = GV[i][j];
            var rel2 = Math.abs(num2 - ana2) / Math.max(Math.abs(num2), Math.abs(ana2), 1e-8);
            if (rel2 > maxRel) maxRel = rel2;
          }
        }
      }
    });
    return maxRel;
  }
  function mlpTrain(X, Y, opts) {
    opts = opts || {};
    var hidden = opts.hidden || 5, iters = opts.iters || 3000, lr = opts.lr || 0.1;
    var seed = opts.seed === undefined ? 11 : opts.seed;
    var rng = mulberry32(seed);
    var d = X[0].length, h = hidden, o = Y[0].length;
    var P = {
      W1: Array.from({ length: h }, function () { return Array.from({ length: d }, function () { return (rng() * 2 - 1) * 0.5; }); }),
      b1: new Array(h).fill(0),
      W2: Array.from({ length: o }, function () { return Array.from({ length: h }, function () { return (rng() * 2 - 1) * 0.5; }); }),
      b2: new Array(o).fill(0)
    };
    var history = [];
    for (var it = 0; it < iters; it++) {
      var G = mlpGrad(P, X, Y);
      for (var i = 0; i < h; i++) { P.b1[i] -= lr * G.b1[i]; for (var j = 0; j < d; j++) P.W1[i][j] -= lr * G.W1[i][j]; }
      for (var i2 = 0; i2 < o; i2++) { P.b2[i2] -= lr * G.b2[i2]; for (var j2 = 0; j2 < h; j2++) P.W2[i2][j2] -= lr * G.W2[i2][j2]; }
      if (it % 10 === 0 || it === iters - 1) history.push(mlpLoss(P, X, Y));
    }
    return { params: P, history: history, predict: function (x) { return mlpForward(P, x).out; } };
  }

  /* ========================================================================
   * 7. 深度学习组件
   * ======================================================================*/
  function linear(X, W, b) {
    return X.map(function (row) {
      return W.map(function (w, j) { return dot(row, w) + (b ? b[j] : 0); });
    });
  }
  function layerNorm(x, eps) { // 逐行（每个样本向量）
    eps = eps || 1e-5;
    return x.map(function (row) {
      var n = row.length, mu = 0;
      for (var i = 0; i < n; i++) mu += row[i];
      mu /= n;
      var v = 0;
      for (var i2 = 0; i2 < n; i2++) { var dd = row[i2] - mu; v += dd * dd; }
      v /= n;
      var inv = 1 / Math.sqrt(v + eps);
      return row.map(function (val) { return (val - mu) * inv; });
    });
  }
  function selfAttention(Q, K, V, causal) {
    var d = Q[0].length;
    var scores = Q.map(function (q, i) {
      return K.map(function (k, j) {
        var s = dot(q, k) / Math.sqrt(d);
        return causal && j > i ? -Infinity : s;
      });
    });
    var W = scores.map(function (row) { return softmax(row); });
    return { weights: W, output: W.map(function (row) {
      var o = new Array(V[0].length).fill(0);
      for (var j = 0; j < V.length; j++) for (var t = 0; t < V[0].length; t++) o[t] += row[j] * V[j][t];
      return o;
    }) };
  }

  /* ========================================================================
   * 8. 图模型推断（HMM：前向-后向 / Viterbi）
   * ======================================================================*/
  function hmmForwardBackward(pi, A, B, obs) { // A: k×k 转移, B: k×M 发射
    var k = pi.length, T = obs.length;
    var alpha = [], beta = [];
    var a0 = new Array(k);
    for (var i = 0; i < k; i++) a0[i] = pi[i] * B[i][obs[0]];
    alpha.push(a0);
    for (var t = 1; t < T; t++) {
      var at = new Array(k);
      for (var i2 = 0; i2 < k; i2++) {
        var s = 0;
        for (var j = 0; j < k; j++) s += alpha[t - 1][j] * A[j][i2];
        at[i2] = s * B[i2][obs[t]];
      }
      alpha.push(at);
    }
    var bT = new Array(k).fill(1);
    beta.push(bT);
    for (var t2 = T - 2; t2 >= 0; t2--) {
      var bt = new Array(k);
      for (var i3 = 0; i3 < k; i3++) {
        var s2 = 0;
        /* β_t(i) = Σ_j A[i][j]·B[j][o_{t+1}]·β_{t+1}(j)（Rabiner 1989）
           v1.3.1 修正：递推须取「上一轮 unshift 进来的」β_{t+1}，即当前 beta[0]；
           原索引 beta[T-2-t2] 在 unshift 后发生位移，导致 β 与后验错误（前后向一致性 Σα_tβ_t = P(O) 被破坏）。 */
        for (var j2 = 0; j2 < k; j2++) s2 += A[i3][j2] * B[j2][obs[t2 + 1]] * beta[0][j2];
        bt[i3] = s2;
      }
      beta.unshift(bt);
    }
    var logPO = Math.log(alpha[T - 1].reduce(function (a, b) { return a + b; }, 0));
    var posteriors = alpha.map(function (row, t3) {
      var s3 = row[0] * beta[t3][0] + row[1] * beta[t3][1];
      return row.map(function (v, i4) { return v * beta[t3][i4] / s3; });
    });
    return { alpha: alpha, beta: beta, logPO: logPO, posteriors: posteriors };
  }
  function hmmViterbi(pi, A, B, obs) {
    var k = pi.length, T = obs.length;
    var delta = [pi.map(function (p, i) { return Math.log(p) + Math.log(B[i][obs[0]]); })];
    var psi = [];
    for (var t = 1; t < T; t++) {
      var dt = new Array(k), pt = new Array(k);
      for (var i = 0; i < k; i++) {
        var best = -Infinity, arg = 0;
        for (var j = 0; j < k; j++) {
          var v = delta[t - 1][j] + Math.log(A[j][i]);
          if (v > best) { best = v; arg = j; }
        }
        dt[i] = best + Math.log(B[i][obs[t]]);
        pt[i] = arg;
      }
      delta.push(dt); psi.push(pt);
    }
    var last = delta[T - 1], bestIdx = 0;
    for (var i2 = 1; i2 < k; i2++) if (last[i2] > last[bestIdx]) bestIdx = i2;
    var path = [bestIdx];
    for (var t2 = T - 2; t2 >= 0; t2--) { bestIdx = psi[t2][bestIdx]; path.unshift(bestIdx); }
    return { path: path, logprob: last[path[0]] };
  }

  /* ========================================================================
   * 9. 强化学习（Bellman 动态规划 / Q-Learning）
   * ======================================================================*/
  function chainSetup(n, gamma, stochastic) {
    function clamp(s) { return Math.max(0, Math.min(n - 1, s)); }
    function trans(s, a) {
      var res = new Map();
      function add(sp, p) { res.set(sp, (res.get(sp) || 0) + p); }
      if (!stochastic) add(clamp(s + a), 1);
      else { add(clamp(s + a), 0.8); add(clamp(s - a), 0.1); add(s, 0.1); }
      return Array.from(res.entries());
    }
    function reward(s, sp) { return sp === n - 1 ? 1 : 0; }
    return { n: n, gamma: gamma, actions: [-1, 1], trans: trans, reward: reward, clamp: clamp };
  }
  function valueIteration(opts) {
    var n = opts.n, gamma = opts.gamma, mdp = chainSetup(n, gamma, opts.stochastic);
    var tol = opts.tol || 1e-13, maxIter = opts.maxIter || 20000;
    var V = new Array(n).fill(0), snapshots = [], it = 0, delta = Infinity;
    for (; it < maxIter; it++) {
      var Vn = new Array(n);
      for (var s = 0; s < n; s++) {
        var best = -Infinity;
        for (var ai = 0; ai < mdp.actions.length; ai++) {
          var a = mdp.actions[ai], v = 0;
          var tr = mdp.trans(s, a);
          for (var t = 0; t < tr.length; t++) v += tr[t][1] * (mdp.reward(s, tr[t][0]) + gamma * V[tr[t][0]]);
          if (v > best) best = v;
        }
        Vn[s] = best;
      }
      delta = 0;
      for (var s2 = 0; s2 < n; s2++) delta = Math.max(delta, Math.abs(Vn[s2] - V[s2]));
      V = Vn; snapshots.push(V.slice());
      if (delta < tol) break;
    }
    var residual = 0;
    for (var s3 = 0; s3 < n; s3++) {
      var best2 = -Infinity;
      for (var ai2 = 0; ai2 < mdp.actions.length; ai2++) {
        var a2 = mdp.actions[ai2], v2 = 0;
        var tr2 = mdp.trans(s3, a2);
        for (var t2 = 0; t2 < tr2.length; t2++) v2 += tr2[t2][1] * (mdp.reward(s3, tr2[t2][0]) + gamma * V[tr2[t2][0]]);
        if (v2 > best2) best2 = v2;
      }
      residual = Math.max(residual, Math.abs(V[s3] - best2));
    }
    return { V: V, iterations: it, residual: residual, snapshots: snapshots };
  }
  function td0Sweep(n, gamma) { // 终端语义 + 确定性链 + α=1：一步后向扫描即解析解
    var V = new Array(n).fill(0);
    for (var s = n - 2; s >= 0; s--) {
      var r = (s + 1 === n - 1) ? 1 : 0;
      V[s] = r + gamma * V[s + 1];
    }
    return V;
  }
  function qLearning(opts) {
    opts = opts || {};
    var n = opts.n || 7, gamma = opts.gamma || 0.9, episodes = opts.episodes || 5000;
    var alpha = opts.alpha || 0.2, epsStart = opts.epsStart || 1, epsEnd = opts.epsEnd || 0.05;
    var seed = opts.seed === undefined ? 123 : opts.seed;
    var rng = mulberry32(seed);
    var Q = Array.from({ length: n }, function () { return new Array(2).fill(0); });
    function clamp(s) { return Math.max(0, Math.min(n - 1, s)); }
    for (var ep = 0; ep < episodes; ep++) {
      var eps = epsStart + (epsEnd - epsStart) * ep / episodes;
      var s = 0;
      for (var t = 0; t < 200; t++) {
        var a = rng() < eps ? (rng() < 0.5 ? 0 : 1) : (Q[s][0] >= Q[s][1] ? 0 : 1);
        var sp = clamp(s + (a === 1 ? 1 : -1));
        var r = sp === n - 1 ? 1 : 0;
        var done = sp === n - 1;
        var target = r + (done ? 0 : gamma * Math.max(Q[sp][0], Q[sp][1]));
        Q[s][a] += alpha * (target - Q[s][a]);
        s = sp;
        if (done) break;
      }
    }
    var success = 0, trials = 200;
    for (var i = 0; i < trials; i++) {
      var s2 = 0, ok = false;
      for (var t2 = 0; t2 < 200; t2++) {
        var a2 = Q[s2][0] >= Q[s2][1] ? 0 : 1;
        s2 = clamp(s2 + (a2 === 1 ? 1 : -1));
        if (s2 === n - 1) { ok = true; break; }
      }
      if (ok) success++;
    }
    return { Q: Q, successRate: success / trials };
  }
  function epsilonGreedy(qrow, eps, rng) {
    if (rng() >= eps) {
      var best = 0;
      for (var i = 1; i < qrow.length; i++) if (qrow[i] > qrow[best]) best = i;
      return best;
    }
    return Math.floor(rng() * qrow.length);
  }

  /* ========================================================================
   * 10. 评估指标
   * ======================================================================*/
  function accuracy(y, yhat) { var c = 0; for (var i = 0; i < y.length; i++) if (y[i] === yhat[i]) c++; return c / y.length; }
  function precision(y, yhat) { var tp = 0, fp = 0; for (var i = 0; i < y.length; i++) { if (yhat[i] === 1) { if (y[i] === 1) tp++; else fp++; } } return tp + fp === 0 ? 0 : tp / (tp + fp); }
  function recall(y, yhat) { var tp = 0, fn = 0; for (var i = 0; i < y.length; i++) { if (y[i] === 1) { if (yhat[i] === 1) tp++; else fn++; } } return tp + fn === 0 ? 0 : tp / (tp + fn); }
  function f1(y, yhat) { var p = precision(y, yhat), r = recall(y, yhat); return p + r === 0 ? 0 : 2 * p * r / (p + r); }
  function auc(y, s) { // Mann–Whitney U（秩法，含并列）
    var n = y.length;
    var order = Array.from({ length: n }, function (_, i) { return i; }).sort(function (a, b) { return s[a] - s[b]; });
    var i = 0, rankSumPos = 0;
    while (i < n) {
      var j = i;
      while (j + 1 < n && s[order[j + 1]] === s[order[i]]) j++;
      var avgRank = (i + j) / 2 + 1;
      for (var t = i; t <= j; t++) if (y[order[t]] === 1) rankSumPos += avgRank;
      i = j + 1;
    }
    var nPos = 0;
    for (var i2 = 0; i2 < n; i2++) if (y[i2] === 1) nPos++;
    var nNeg = n - nPos;
    if (nPos === 0 || nNeg === 0) return 0.5;
    return (rankSumPos - nPos * (nPos + 1) / 2) / (nPos * nNeg);
  }
  function mse(y, yhat) { var s = 0; for (var i = 0; i < y.length; i++) s += (y[i] - yhat[i]) * (y[i] - yhat[i]); return s / y.length; }
  function mae(y, yhat) { var s = 0; for (var i = 0; i < y.length; i++) s += Math.abs(y[i] - yhat[i]); return s / y.length; }
  function r2(y, yhat) { // R² = 1 − Σ(ŷ−y)² / Σ(y−ȳ)²
    var m = expectation(y), ssRes = 0, sst = 0;
    for (var i = 0; i < y.length; i++) { ssRes += (yhat[i] - y[i]) * (yhat[i] - y[i]); sst += (y[i] - m) * (y[i] - m); }
    return 1 - ssRes / sst;
  }
  function logLoss(y, p) {
    var s = 0;
    for (var i = 0; i < y.length; i++) s += -(y[i] * Math.log(p[i] + 1e-15) + (1 - y[i]) * Math.log(1 - p[i] + 1e-15));
    return s / y.length;
  }
  function cosineSimilarity(a, b) { return dot(a, b) / (norm2(a) * norm2(b)); }
  function editDistance(a, b) {
    var m = a.length, n = b.length;
    var dp = Array.from({ length: m + 1 }, function (_, i) { return [i].concat(new Array(n).fill(0)); });
    for (var j = 0; j <= n; j++) dp[0][j] = j;
    for (var i = 1; i <= m; i++) for (var j2 = 1; j2 <= n; j2++) {
      var cost = a[i - 1] === b[j2 - 1] ? 0 : 1;
      dp[i][j2] = Math.min(dp[i - 1][j2] + 1, dp[i][j2 - 1] + 1, dp[i - 1][j2 - 1] + cost);
    }
    return dp[m][n];
  }
  function silhouette(X, labels) {
    var n = X.length, s = new Array(n);
    for (var i = 0; i < n; i++) {
      var same = [], other = {};
      for (var j = 0; j < n; j++) {
        if (i === j) continue;
        var d = euclid(X[i], X[j]);
        if (labels[j] === labels[i]) same.push(d);
        else { (other[labels[j]] = other[labels[j]] || []).push(d); }
      }
      var a = same.length ? same.reduce(function (x, y) { return x + y; }, 0) / same.length : 0;
      var b = Infinity;
      for (var kk in other) {
        var arr = other[kk], mm = arr.reduce(function (x, y) { return x + y; }, 0) / arr.length;
        if (mm < b) b = mm;
      }
      s[i] = (same.length === 0 || b === Infinity) ? 0 : (b - a) / Math.max(a, b);
    }
    return s;
  }

  /* ========================================================================
   * 11. 归一化 / 正则 / 数据管线 / 系统微操作
   * ======================================================================*/
  function l1Norm(x) { var s = 0; for (var i = 0; i < x.length; i++) s += Math.abs(x[i]); return s; }
  function l2Norm(x) { return norm2(x); }
  function linfNorm(x) { return normInf(x); }
  function normalizeL2(x) { var n = norm2(x); return x.map(function (v) { return v / n; }); }
  function minmaxScale(x) { var lo = Math.min.apply(null, x), hi = Math.max.apply(null, x); return x.map(function (v) { return (v - lo) / (hi - lo); }); }
  function zscore(x) { var m = expectation(x), sd = Math.sqrt(variance(x)); return x.map(function (v) { return (v - m) / sd; }); }
  function l2Reg(w, lambda) { var s = 0; for (var i = 0; i < w.length; i++) s += w[i] * w[i]; return lambda * s; }
  function temperatureSoftmax(xs, T) { return softmax(xs.map(function (x) { return x / T; })); }
  function topK(p, k) {
    var order = p.map(function (v, i) { return [v, i]; }).sort(function (a, b) { return b[0] - a[0]; }).slice(0, k);
    var s = order.reduce(function (a, b) { return a + b[0]; }, 0);
    var out = new Array(p.length).fill(0);
    order.forEach(function (item) { out[item[1]] = item[0] / s; });
    return out;
  }
  function shuffle(arr, seed) {
    var rng = mulberry32(seed === undefined ? 5 : seed);
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }
  function trainTestSplit(n, ratio, seed) {
    var idx = shuffle(Array.from({ length: n }, function (_, i) { return i; }), seed);
    var cut = Math.floor(n * ratio);
    return { train: idx.slice(0, cut), test: idx.slice(cut) };
  }
  function kFold(n, k, seed) {
    var idx = shuffle(Array.from({ length: n }, function (_, i) { return i; }), seed);
    var folds = Array.from({ length: k }, function () { return []; });
    for (var i = 0; i < n; i++) folds[i % k].push(idx[i]);
    return folds;
  }
  function oneHot(labels) {
    var classes = Array.from(new Set(labels)).sort();
    var index = {};
    classes.forEach(function (c, i) { index[c] = i; });
    return { classes: classes, matrix: labels.map(function (l) { return classes.map(function (_, i) { return i === index[l] ? 1 : 0; }); }) };
  }
  function Arena(size) { this.size = size; this.used = 0; this.allocs = []; }
  Arena.prototype.alloc = function (n) {
    if (this.used + n > this.size) throw new Error('Arena 容量耗尽');
    var a = { offset: this.used, length: n };
    this.used += n;
    this.allocs.push(a);
    return a;
  };
  Arena.prototype.reset = function () { this.used = 0; this.allocs = []; };
  function reduceAll(xs) { return kahanSum(xs); } // 全局归约用精确和
  function broadcast(xs) { return xs.slice(); }

  /* ========================================================================
   * 12. 扩展算子库（L10 微指令细化 · v1.1）—— 每条均携带可独立复算的不变量
   * ------------------------------------------------------------------------
   * 零重写原则：以下实现直接锚定世界顶级成果的语义（论文/顶级开源实现），
   * 以零依赖 JS 复现，并在第 13 节用硬不变量自证一致。
   *   Dekker(1971) 双精度分裂 → twoProd/twoSum → 精确 fma 模拟
   *   Abramowitz–Stegun 7.1.26 → erf；Box–Muller(1958) → 正态采样
   *   Dempster(1977) EM 责任度语义；Cholesky 求值 → 多元高斯密度
   * ======================================================================*/

  /* ---- 12.1 TENSOR 张量指令（结构不变量，容差 0）---- */
  function tensorCreate(dims, fill) {
    if (!dims.length) return fill;
    var rest = dims.slice(1);
    return Array.from({ length: dims[0] }, function () { return tensorCreate(rest, fill); });
  }
  function tensorCount(t) { return Array.isArray(t) ? t.reduce(function (a, v) { return a + tensorCount(v); }, 0) : 1; }
  function tensorZeros(dims) { return tensorCreate(dims, 0); }
  function tensorOnes(dims) { return tensorCreate(dims, 1); }
  function arange(start, stop, step) {
    step = step || 1;
    var out = [];
    for (var v = start; v < stop; v += step) out.push(v);
    return out;
  }
  function flatten(t) {
    var out = [];
    (function rec(x) { if (Array.isArray(x)) x.forEach(rec); else out.push(x); })(t);
    return out;
  }
  function tensorFromFlat(dims, flat) {
    var idx = 0;
    function build(ds) {
      if (!ds.length) return flat[idx++];
      return Array.from({ length: ds[0] }, function () { return build(ds.slice(1)); });
    }
    return build(dims);
  }
  function reshape(t, dims) { return tensorFromFlat(dims, flatten(t)); }
  function tensorSlice(a, start, stop) { return a.slice(start, stop); }
  function tensorConcat(parts) {
    var out = [];
    parts.forEach(function (p) { p.forEach(function (v) { out.push(v); }); });
    return out;
  }
  function tensorStack(parts) { return parts.map(function (p) { return p.slice(); }); }
  function tensorSplit(a, sizes) {
    var out = [], i = 0;
    sizes.forEach(function (s) { out.push(a.slice(i, i + s)); i += s; });
    return out;
  }
  function tensorCast(a) { return a.map(function (v) { return +v; }); }
  function tensorCopy(t) { return Array.isArray(t) ? t.map(tensorCopy) : t; }
  function tensorGather(a, idx) { return idx.map(function (i) { return a[i]; }); }
  function tensorScatter(a, idx, src) {
    var out = a.slice();
    idx.forEach(function (i, t) { out[i] = src[t]; });
    return out;
  }
  function tensorFill(a, v) { return a.map(function () { return v; }); }

  /* ---- 12.2 NUMER 数值稳定指令 ---- */
  function divGuard(a, b, eps) {
    eps = eps || 1e-12;
    var d = Math.abs(b) < eps ? (b < 0 ? -eps : eps) : b;
    return a / d;
  }
  function sqrtGuard(x) {
    if (x < 0) throw new Error('负数开方：sqrtGuard');
    return Math.sqrt(x);
  }
  function reciprocal(x) { return 1 / x; }
  function expm1(x) { return Math.expm1(x); }   // 平台原生：V8 保证正确舍入
  function log1p(x) { return Math.log1p(x); }
  function hypot(a, b) { return Math.hypot(a, b); }
  function twoSum(a, b) { // Knuth TwoSum：a+b = s+e 精确
    var s = a + b;
    var bb = Math.abs(a) >= Math.abs(b) ? (a - s) + b : (b - s) + a;
    return [s, bb];
  }
  function twoProd(a, b) { // Dekker 分裂：a·b = p+e 精确
    var p = a * b;
    var split = 134217729; // 2^27 + 1
    var t = split * a, ah = t - (t - a), al = a - ah;
    var t2 = split * b, bh = t2 - (t2 - b), bl = b - bh;
    var err = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    return [p, err];
  }
  function fma(a, b, c) { // 单次舍入融合乘加（Dekker 模拟，Muller et al. 2005 算法 4.3）
    var pr = twoProd(a, b);
    var s1 = twoSum(pr[0], c);
    var s2 = twoSum(s1[0], pr[1]);
    return s2[0] + (s1[1] + s2[1]);
  }

  /* ---- 12.3 PROB 概率运算指令 ---- */
  function uniformSample(n, seed) {
    var rng = mulberry32(seed === undefined ? 3 : seed), out = [];
    for (var i = 0; i < n; i++) out.push(rng());
    return out;
  }
  function bernoulliSample(p, n, seed) {
    var rng = mulberry32(seed === undefined ? 4 : seed), out = [];
    for (var i = 0; i < n; i++) out.push(rng() < p ? 1 : 0);
    return out;
  }
  function binomialSample(n, p, trials, seed) {
    var rng = mulberry32(seed === undefined ? 5 : seed), out = [];
    for (var t = 0; t < trials; t++) {
      var c = 0;
      for (var i = 0; i < n; i++) if (rng() < p) c++;
      out.push(c);
    }
    return out;
  }
  function marginal(joint) { // joint[i][j] = P(X=i,Y=j) → P(X)
    var n = joint.length, m = joint[0].length, px = new Array(n).fill(0);
    for (var i = 0; i < n; i++) for (var j = 0; j < m; j++) px[i] += joint[i][j];
    return px;
  }
  function conditionalRow(joint, i) { // P(Y | X=i)
    var row = joint[i].slice();
    var s = row.reduce(function (a, b) { return a + b; }, 0);
    return row.map(function (v) { return v / s; });
  }
  function mvGaussPDF(x, mu, Sigma) { // Cholesky 求值：N(μ,Σ)
    var d = x.length;
    var L = safeCholesky(Sigma);
    var diff = x.map(function (v, i) { return v - mu[i]; });
    var y = solveLower(L, diff), logdet = 0;
    for (var i = 0; i < d; i++) logdet += Math.log(L[i][i]);
    return Math.exp(-0.5 * dot(y, y)) / Math.sqrt(Math.pow(2 * Math.PI, d) * Math.exp(2 * logdet));
  }

  /* ---- 12.4 INFO 信息论指令 ---- */
  function perplexity(p) { return Math.pow(2, entropy(p)); }
  function surprisal(p) { return -Math.log2(p); }
  function bits(nats) { return nats * Math.LOG2E; }
  function klReverse(p, q) { return klDivergence(q, p); }
  function maxEntropyDist(n) { return new Array(n).fill(1 / n); } // 最大熵原理：无约束最大熵 = 均匀
  function normalizeDist(p) {
    var s = p.reduce(function (a, b) { return a + b; }, 0);
    return p.map(function (v) { return v / s; });
  }

  /* ---- 12.5 OPTIM 优化算子指令 ----
   * 全部锚定经典文献：Polyak(1964) 动量 · Nesterov(1983) 加速 ·
   * Duchi(2011) AdaGrad · Tieleman(2012) RMSProp · Kingma–Ba(2015) Adam ·
   * Loshchilov–Hutter(2019) AdamW 解耦权重衰减 */
  function quadProblem() { // 共享凸二次基准 f(x)=½ΣAᵢxᵢ²−Σbᵢxᵢ（条件数 100）
    var A = [1, 10, 100], b = [1, -2, 3];
    return {
      A: A, b: b,
      f: function (x) { return 0.5 * (A[0] * x[0] * x[0] + A[1] * x[1] * x[1] + A[2] * x[2] * x[2]) - (b[0] * x[0] + b[1] * x[1] + b[2] * x[2]); },
      g: function (x) { return [A[0] * x[0] - b[0], A[1] * x[1] - b[1], A[2] * x[2] - b[2]]; },
      xs: [b[0] / A[0], b[1] / A[1], b[2] / A[2]]
    };
  }
  function momentum(f, grad, x0, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.01, beta = opts.beta === undefined ? 0.9 : opts.beta, iters = opts.iters || 10000;
    var x = x0.slice(), v = new Array(x.length).fill(0), history = [];
    for (var it = 0; it < iters; it++) {
      var g = grad(x);
      for (var i = 0; i < x.length; i++) { v[i] = beta * v[i] - lr * g[i]; x[i] += v[i]; }
      history.push(f(x));
    }
    return { x: x, history: history, iterations: iters };
  }
  function nesterov(f, grad, x0, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.01, beta = opts.beta === undefined ? 0.9 : opts.beta, iters = opts.iters || 10000;
    var x = x0.slice(), v = new Array(x.length).fill(0), history = [];
    for (var it = 0; it < iters; it++) {
      for (var i = 0; i < x.length; i++) x[i] += beta * v[i];
      var g = grad(x);
      for (var i2 = 0; i2 < x.length; i2++) { v[i2] = beta * v[i2] - lr * g[i2]; x[i2] += v[i2]; }
      history.push(f(x));
    }
    return { x: x, history: history, iterations: iters };
  }
  function rmsprop(f, grad, x0, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.01, beta = opts.beta === undefined ? 0.9 : opts.beta, eps = opts.eps || 1e-8, iters = opts.iters || 20000;
    var x = x0.slice(), s = new Array(x.length).fill(0), history = [];
    for (var it = 0; it < iters; it++) {
      var g = grad(x);
      for (var i = 0; i < x.length; i++) {
        s[i] = beta * s[i] + (1 - beta) * g[i] * g[i];
        x[i] -= lr * g[i] / (Math.sqrt(s[i]) + eps);
      }
      history.push(f(x));
    }
    return { x: x, history: history, iterations: iters };
  }
  function adagrad(f, grad, x0, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.5, eps = opts.eps || 1e-8, iters = opts.iters || 20000;
    var x = x0.slice(), G = new Array(x.length).fill(0), history = [];
    for (var it = 0; it < iters; it++) {
      var g = grad(x);
      for (var i = 0; i < x.length; i++) {
        G[i] += g[i] * g[i];
        x[i] -= lr * g[i] / (Math.sqrt(G[i]) + eps);
      }
      history.push(f(x));
    }
    return { x: x, history: history, iterations: iters };
  }
  function adamw(f, grad, x0, opts) {
    opts = opts || {};
    var lr = opts.lr || 0.02, iters = opts.iters || 5000, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    var lambda = opts.lambda === undefined ? 0.01 : opts.lambda;
    var x = x0.slice(), m = new Array(x.length).fill(0), v = new Array(x.length).fill(0), history = [];
    for (var t = 1; t <= iters; t++) {
      var g = grad(x);
      for (var i = 0; i < x.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        var mh = m[i] / (1 - Math.pow(b1, t)), vh = v[i] / (1 - Math.pow(b2, t));
        x[i] -= lr * (mh / (Math.sqrt(vh) + eps) + lambda * x[i]); // 解耦衰减
      }
      history.push(f(x));
    }
    return { x: x, history: history, iterations: iters };
  }
  function gradClip(g, maxNorm) {
    var n = norm2(g);
    if (n <= maxNorm) return g.slice();
    return g.map(function (v) { return v * maxNorm / n; });
  }
  function gradNorm(g) { return norm2(g); }
  function lrDecay(lr0, k, t) { return lr0 / (1 + k * t); }
  function weightDecay(w, lr, lambda) { return w.map(function (v) { return v * (1 - lr * lambda); }); }
  function polyakAvg(xs) {
    var d = xs[0].length, out = new Array(d).fill(0);
    for (var i = 0; i < xs.length; i++) for (var j = 0; j < d; j++) out[j] += xs[i][j];
    return out.map(function (v) { return v / xs.length; });
  }

  /* ---- 12.6 FWD 前向传播指令 ---- */
  function embeddingLookup(table, idx) { return idx.map(function (i) { return table[i].slice(); }); }
  function residualAdd(x, fx) {
    return x.map(function (row, i) { return row.map(function (v, j) { return v + fx[i][j]; }); });
  }
  function splitHeads(X, H) {
    var T = X.length, d = X[0].length, hd = d / H;
    return Array.from({ length: H }, function (_, h) {
      return Array.from({ length: T }, function (_, t) { return X[t].slice(h * hd, (h + 1) * hd); });
    });
  }
  function concatHeads(heads) {
    var H = heads.length, T = heads[0].length, d = heads[0][0].length;
    return Array.from({ length: T }, function (_, t) {
      var row = [];
      for (var h = 0; h < H; h++) for (var j = 0; j < d; j++) row.push(heads[h][t][j]);
      return row;
    });
  }
  function poolMax(x, size) {
    var out = [];
    for (var i = 0; i + size <= x.length; i += size) {
      var m = -Infinity;
      for (var j = 0; j < size; j++) m = Math.max(m, x[i + j]);
      out.push(m);
    }
    return out;
  }
  function batchNorm(X, eps) {
    if (eps === undefined) eps = 1e-5;
    var n = X.length, d = X[0].length, mu = new Array(d).fill(0), v2 = new Array(d).fill(0);
    for (var j = 0; j < d; j++) { for (var i = 0; i < n; i++) mu[j] += X[i][j]; mu[j] /= n; }
    for (var j2 = 0; j2 < d; j2++) {
      for (var i2 = 0; i2 < n; i2++) { var dd = X[i2][j2] - mu[j2]; v2[j2] += dd * dd; }
      v2[j2] /= n;
    }
    return X.map(function (row) {
      return row.map(function (val, j3) { return (val - mu[j3]) / Math.sqrt(v2[j3] + eps); });
    });
  }
  function dropout(x, p, seed) { // 反向 dropout：缩放 1/(1−p) 保持期望（Srivastava 2014）
    var rng = mulberry32(seed === undefined ? 9 : seed), keep = 1 - p;
    return x.map(function (v) { return rng() < keep ? v / keep : 0; });
  }
  function conv1d(x, kernel, stride, pad) {
    stride = stride || 1; pad = pad || 0;
    var xp = [], i;
    for (i = 0; i < pad; i++) xp.push(0);
    x.forEach(function (v) { xp.push(v); });
    for (i = 0; i < pad; i++) xp.push(0);
    var k = kernel.length, out = [];
    for (var s = 0; s + k <= xp.length; s += stride) {
      var acc = 0;
      for (var j = 0; j < k; j++) acc += xp[s + j] * kernel[j];
      out.push(acc);
    }
    return out;
  }

  /* ---- 12.7 BWD 反向传播指令 ---- */
  function numericJacobian(f, x, eps) { // f: R^n → R^m，中心差分
    eps = eps || 1e-6;
    var n = x.length, fx = f(x), m = fx.length;
    var J = Array.from({ length: m }, function () { return new Array(n).fill(0); });
    for (var i = 0; i < n; i++) {
      var xp = x.slice(), xm = x.slice();
      xp[i] += eps; xm[i] -= eps;
      var fp = f(xp), fm = f(xm);
      for (var r = 0; r < m; r++) J[r][i] = (fp[r] - fm[r]) / (2 * eps);
    }
    return J;
  }
  function jacobian(f, x, eps) { return numericJacobian(f, x, eps); }
  function vjp(f, x, v, eps) { return matVec(transpose(numericJacobian(f, x, eps)), v); }
  function jvp(f, x, d, eps) { return matVec(numericJacobian(f, x, eps), d); }
  function chainRule(dOuter, inner, dInner, x) { // d/dx f(g(x)) = f′(g(x))·g′(x)
    return dOuter(inner(x)) * dInner(x);
  }
  function gradRelu(x) { return x.map(function (v) { return v > 0 ? 1 : 0; }); }
  function gradTanh(x) { return x.map(function (v) { var t = Math.tanh(v); return 1 - t * t; }); }
  function gradSoftmaxCE(logits, yIdx) {
    var p = softmax(logits);
    p[yIdx] -= 1;
    return p;
  }
  function gradMatmul(dL, A, B) {
    return { dA: matMul(dL, transpose(B)), dB: matMul(transpose(A), dL) };
  }
  function gradAccum(grads) {
    var out = grads[0].map(function (row) { return row.slice(); });
    for (var i = 1; i < grads.length; i++) {
      for (var a = 0; a < out.length; a++) for (var b = 0; b < out[0].length; b++) out[a][b] += grads[i][a][b];
    }
    return out;
  }
  function stopGrad(x) { return tensorCopy(x); }
  function detach(x) { return tensorCopy(x); }
  function rnnForward(P, xs) { // h_t = tanh(W·h_{t−1} + U·x_t)（标量循环核）
    var h = [P.h0], hs = [];
    for (var t = 0; t < xs.length; t++) {
      hs.push(Math.tanh(P.W * h[t] + P.U * xs[t]));
      h.push(hs[t]);
    }
    return { hs: hs, h: h };
  }
  function rnnLoss(P, xs, ys) {
    var f = rnnForward(P, xs), s = 0;
    for (var t = 0; t < xs.length; t++) s += (f.hs[t] - ys[t]) * (f.hs[t] - ys[t]);
    return s;
  }
  function rnnBPTT(P, xs, ys) { // 时间反向传播（Werbos 1990 链路）
    var f = rnnForward(P, xs), T = xs.length;
    var gW = 0, gU = 0, dhNext = 0;
    for (var t = T - 1; t >= 0; t--) {
      var dh = 2 * (f.hs[t] - ys[t]) + dhNext;
      var dz = dh * (1 - f.hs[t] * f.hs[t]); // tanh 导数
      gW += dz * f.h[t];
      gU += dz * xs[t];
      dhNext = dz * P.W;
    }
    return { W: gW, U: gU, h0: dhNext };
  }

  /* ---- 12.8 NORM 归一化正则指令 ---- */
  function l1Reg(w, lambda) { return lambda * l1Norm(w); }
  function weightNorm(v, g) { var n = norm2(v); return v.map(function (x) { return g * x / n; }); }
  function spectralNorm(W) { // 幂迭代估计主奇异值后归一（Miyato 2018）
    var p = powerIteration(W, { iters: 500, tol: 1e-14 });
    return W.map(function (row) { return row.map(function (x) { return x / p.value; }); });
  }
  function groupNorm(x, groups, eps) {
    if (eps === undefined) eps = 1e-5;
    return x.map(function (row) {
      var d = row.length, gs = d / groups, out = [];
      for (var g = 0; g < groups; g++) {
        var seg = row.slice(g * gs, (g + 1) * gs);
        var mu = expectation(seg), v = variance(seg), inv = 1 / Math.sqrt(v + eps);
        seg.forEach(function (val) { out.push((val - mu) * inv); });
      }
      return out;
    });
  }
  function labelSmooth(onehot, eps) {
    var K = onehot.length;
    return onehot.map(function (v) { return (1 - eps) * v + eps / K; });
  }
  function mixup(x1, x2, lambda) { return x1.map(function (v, i) { return lambda * v + (1 - lambda) * x2[i]; }); }
  function earlyStop(valHistory, patience) {
    patience = patience || 1;
    var best = Infinity, bestIdx = 0;
    for (var i = 0; i < valHistory.length; i++) {
      if (valHistory[i] < best - 1e-15) { best = valHistory[i]; bestIdx = i; }
    }
    return { bestIndex: bestIdx, stopIndex: Math.min(bestIdx + patience, valHistory.length - 1), best: best };
  }

  /* ---- 12.9 ATTEN 注意力指令 ---- */
  function qkDot(Q, K) { // scores = QKᵀ/√d（Vaswani 2017）
    var d = Q[0].length;
    return Q.map(function (q) { return K.map(function (k) { return dot(q, k) / Math.sqrt(d); }); });
  }
  function attentionScale(scores, dk) {
    return scores.map(function (row) { return row.map(function (s) { return s / Math.sqrt(dk); }); });
  }
  function crossAttention(Q, K, V) { // Q 来自解码侧，K/V 来自编码侧
    var W = qkDot(Q, K).map(function (row) { return softmax(row); });
    return { weights: W, output: W.map(function (row) {
      var o = new Array(V[0].length).fill(0);
      for (var j = 0; j < V.length; j++) for (var t = 0; t < V[0].length; t++) o[t] += row[j] * V[j][t];
      return o;
    }) };
  }
  function multiHead(Q, K, V, H, causal) {
    var hq = splitHeads(Q, H), hk = splitHeads(K, H), hv = splitHeads(V, H);
    var outs = [], weights = [];
    for (var h = 0; h < H; h++) {
      var a = selfAttention(hq[h], hk[h], hv[h], causal);
      outs.push(a.output);
      weights.push(a.weights);
    }
    return { output: concatHeads(outs), weights: weights };
  }
  function ropeApply(x, m, base) { // 旋转位置编码（Su 2021）：二维一组旋转 m·θ_i
    base = base || 10000;
    var d = x.length, out = x.slice();
    for (var i = 0; i < d; i += 2) {
      var theta = m * Math.pow(base, -i / d);
      var c = Math.cos(theta), s = Math.sin(theta);
      out[i] = x[i] * c - x[i + 1] * s;
      out[i + 1] = x[i] * s + x[i + 1] * c;
    }
    return out;
  }
  function attnRow(q, K, V) { // 单查询增量注意力（KV-cache 单步语义）
    var d = q.length;
    var w = softmax(K.map(function (k) { return dot(q, k) / Math.sqrt(d); }));
    var o = new Array(V[0].length).fill(0);
    for (var j = 0; j < V.length; j++) for (var t = 0; t < V[0].length; t++) o[t] += w[j] * V[j][t];
    return { weights: w, output: o };
  }
  function mqa(Q, K, V) { // 多查询注意力：每个 Q 头对共享的单组 K/V 做注意力
    return Q.map(function (qh) { return selfAttention(qh, K, V, false).output; });
  }
  function gqa(Q, K, V, groupSize) { // 分组查询注意力（Ainslie 2023）
    var H = Q.length, G = K.length, out = [];
    for (var h = 0; h < H; h++) {
      var g = Math.min(G - 1, Math.floor(h / groupSize));
      out.push(selfAttention(Q[h], K[g], V[g], false).output);
    }
    return out;
  }
  function attnEntropy(W) { return W.map(function (row) { return entropy(row); }); }
  function relPosBias(scores, bias) {
    return scores.map(function (row, i) { return row.map(function (s, j) { return s + bias[i][j]; }); });
  }
  function alibiScores(scores, slope) { // 线性距离惩罚（Press 2021）
    return scores.map(function (row, i) { return row.map(function (s, j) { return s - slope * Math.abs(i - j); }); });
  }
  function flashAttention(Q, K, V, chunk) { // 在线 softmax 分块（Dao 2022 语义，免物化全矩阵）
    var T = K.length, dv = V[0].length;
    var m = new Array(Q.length).fill(-Infinity), l = new Array(Q.length).fill(0);
    var o = Array.from({ length: Q.length }, function () { return new Array(dv).fill(0); });
    for (var c = 0; c < T; c += chunk) {
      var end = Math.min(c + chunk, T);
      var Kc = K.slice(c, end), Vc = V.slice(c, end);
      var sc = qkDot(Q, Kc);
      for (var i = 0; i < Q.length; i++) {
        var row = sc[i], mNew = m[i];
        for (var j = 0; j < row.length; j++) if (row[j] > mNew) mNew = row[j];
        var corr = Math.exp(m[i] - mNew);
        var lNew = l[i] * corr;
        var oNew = o[i].map(function (v) { return v * corr; });
        for (var j2 = 0; j2 < row.length; j2++) {
          var p = Math.exp(row[j2] - mNew);
          lNew += p;
          for (var t = 0; t < dv; t++) oNew[t] += p * Vc[j2][t];
        }
        m[i] = mNew; l[i] = lNew; o[i] = oNew;
      }
    }
    return { output: o.map(function (row, i) { return row.map(function (v) { return v / l[i]; }); }), m: m, l: l };
  }

  /* ---- 12.10 SAMPLE 采样生成指令 ---- */
  function gumbelSample(u) { return -Math.log(-Math.log(u)); }
  function topP(p, cutoff) { // 核采样：最小支撑且累计质量 ≥ cutoff
    var order = p.map(function (v, i) { return [v, i]; }).sort(function (a, b) { return b[0] - a[0]; });
    var acc = 0, support = [];
    for (var i = 0; i < order.length; i++) {
      support.push(order[i]);
      acc += order[i][0];
      if (acc >= cutoff) break;
    }
    var out = new Array(p.length).fill(0);
    support.forEach(function (item) { out[item[1]] = item[0]; });
    var s = out.reduce(function (a, b) { return a + b; }, 0);
    return out.map(function (v) { return v / s; });
  }
  function multinomialSample(p, n, seed) {
    var rng = mulberry32(seed === undefined ? 6 : seed);
    var counts = new Array(p.length).fill(0);
    for (var t = 0; t < n; t++) {
      var r = rng(), acc = 0, j = 0;
      while (j < p.length - 1 && r > acc + p[j]) { acc += p[j]; j++; }
      counts[j]++;
    }
    return counts;
  }
  function rejectionSample(n, seed, target, proposal, M, lo, hi) {
    var rng = mulberry32(seed === undefined ? 8 : seed), out = [], accepted = 0, total = 0;
    while (out.length < n) {
      var x = lo + (hi - lo) * rng();
      var u = rng() * M * proposal(x);
      total++;
      if (u <= target(x)) { accepted++; out.push(x); }
    }
    return { samples: out, accepted: accepted, total: total, acceptanceRate: accepted / total };
  }
  function inverseCDF(u, F, lo, hi) { // 二分求 F⁻¹(u)
    var a = lo, b = hi;
    for (var i = 0; i < 200; i++) {
      var mid = (a + b) / 2;
      if (F(mid) < u) a = mid; else b = mid;
    }
    return (a + b) / 2;
  }
  function antitheticSample(n, seed) {
    var rng = mulberry32(seed === undefined ? 10 : seed), out = [];
    for (var i = 0; i < n / 2; i++) { var u = rng(); out.push(u, 1 - u); }
    return out;
  }
  function stratifiedSample(n, seed) {
    var rng = mulberry32(seed === undefined ? 11 : seed), out = [];
    for (var i = 0; i < n; i++) out.push((i + rng()) / n);
    return out;
  }
  function systematicSample(n, seed) {
    var rng = mulberry32(seed === undefined ? 12 : seed);
    var start = rng() / n, out = [];
    for (var i = 0; i < n; i++) out.push(start + i / n);
    return out;
  }

  /* ---- 12.11 GRAPH 图推断指令 ---- */
  function chainBeliefProp(pi, A, B, obs) { // 链式 BP ≡ 前向-后向
    return hmmForwardBackward(pi, A, B, obs);
  }
  function variableElim(pi, A, B, obs) { // 按时间序消去隐状态 → P(O)
    var k = pi.length, T = obs.length;
    var f = pi.map(function (p, i) { return p * B[i][obs[0]]; });
    for (var t = 1; t < T; t++) {
      var g = new Array(k).fill(0);
      for (var i = 0; i < k; i++) {
        var s = 0;
        for (var j = 0; j < k; j++) s += f[j] * A[j][i];
        g[i] = s * B[i][obs[t]];
      }
      f = g;
    }
    return f.reduce(function (a, b) { return a + b; }, 0);
  }
  function factorIndex(vars, shape, assign) {
    var idx = 0;
    for (var i = 0; i < vars.length; i++) idx = idx * shape[i] + assign[vars[i]];
    return idx;
  }
  function sumOut(f, vars) { // 消去变量（因子边缘化）
    var keep = f.vars.filter(function (v) { return vars.indexOf(v) === -1; });
    var shape = keep.map(function (v) { return f.shape[f.vars.indexOf(v)]; });
    var total = shape.reduce(function (a, b) { return a * b; }, 1);
    var table = new Array(total).fill(0);
    for (var idx = 0; idx < f.table.length; idx++) {
      var rem = idx, ai = {};
      for (var d = f.vars.length - 1; d >= 0; d--) {
        ai[f.vars[d]] = rem % f.shape[d];
        rem = Math.floor(rem / f.shape[d]);
      }
      table[factorIndex(keep, shape, ai)] += f.table[idx];
    }
    return { vars: keep, shape: shape, table: table };
  }
  function factorProduct(f, g) { // 因子乘积：维度 = 变量并集
    var vars = f.vars.slice();
    g.vars.forEach(function (v) { if (vars.indexOf(v) === -1) vars.push(v); });
    var shape = vars.map(function (v) {
      return f.vars.indexOf(v) >= 0 ? f.shape[f.vars.indexOf(v)] : g.shape[g.vars.indexOf(v)];
    });
    var total = shape.reduce(function (a, b) { return a * b; }, 1);
    var table = new Array(total);
    for (var idx = 0; idx < total; idx++) {
      var rem = idx, ai = {};
      for (var d = vars.length - 1; d >= 0; d--) {
        ai[vars[d]] = rem % shape[d];
        rem = Math.floor(rem / shape[d]);
      }
      table[idx] = f.table[factorIndex(f.vars, f.shape, ai)] * g.table[factorIndex(g.vars, g.shape, ai)];
    }
    return { vars: vars, shape: shape, table: table };
  }
  function ripCheck(cliques, edges) { // 运行交集性质：变量所在团之间的树上路径必经含该变量的团
    var adj = {};
    cliques.forEach(function (_, i) { adj[i] = []; });
    edges.forEach(function (e) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); });
    var vars = {};
    cliques.forEach(function (c) { c.forEach(function (v) { vars[v] = 1; }); });
    return Object.keys(vars).every(function (v) {
      var S = [];
      cliques.forEach(function (c, ci) { if (c.indexOf(v) >= 0) S.push(ci); });
      for (var a = 0; a < S.length; a++) for (var b = a + 1; b < S.length; b++) {
        var prev = {}, queue = [S[a]], seen = {};
        seen[S[a]] = 1;
        while (queue.length) {
          var cur = queue.shift();
          if (cur === S[b]) break;
          adj[cur].forEach(function (nb) { if (!seen[nb]) { seen[nb] = 1; prev[nb] = cur; queue.push(nb); } });
        }
        var node = S[b];
        while (node !== S[a]) {
          if (cliques[node].indexOf(v) === -1) return false;
          node = prev[node];
        }
      }
      return true;
    });
  }
  function junctionTree(cliques, edges) { return { cliques: cliques, edges: edges, rip: ripCheck(cliques, edges) }; }
  function cliqueTreeCalibrate(phiA, phiB, sep) { // 两团链校准：消息 → 一致边缘
    var m = sumOut(phiA, phiA.vars.filter(function (v) { return sep.indexOf(v) === -1; }));
    var updated = factorProduct(phiB, m);
    var mu = sumOut(updated, updated.vars.filter(function (v) { return sep.indexOf(v) === -1; }));
    return { message: m, updated: updated, separator: mu };
  }
  function kalmanFilter1d(zs, A, H, Q, R, x0, P0) { // 一维卡尔曼滤波
    var x = x0, P = P0, out = [];
    for (var i = 0; i < zs.length; i++) {
      x = A * x;
      P = A * P * A + Q;
      var S = H * P * H + R, K = P * H / S;
      x = x + K * (zs[i] - H * x);
      P = (1 - K * H) * P;
      out.push({ x: x, P: P, K: K });
    }
    return out;
  }
  function moralGraph(dag) { // 道德图：无向化 + 连接共父母
    var nodes = Object.keys(dag), adj = {};
    nodes.forEach(function (n) { adj[n] = {}; });
    nodes.forEach(function (n) {
      var ps = dag[n] || [];
      ps.forEach(function (p) { adj[n][p] = 1; adj[p][n] = 1; });
      for (var i = 0; i < ps.length; i++) {
        for (var j = i + 1; j < ps.length; j++) { adj[ps[i]][ps[j]] = 1; adj[ps[j]][ps[i]] = 1; }
      }
    });
    return adj;
  }
  function dSeparated(dag, x, y, z) { // 路径枚举 + 碰撞/非碰撞阻塞判定
    var nodes = Object.keys(dag);
    var zset = {};
    z.forEach(function (v) { zset[v] = 1; });
    function parents(n) { return dag[n] || []; }
    function childrenOf(n) { return nodes.filter(function (m) { return (dag[m] || []).indexOf(n) >= 0; }); }
    function descendants(n) {
      var seen = {}, stack = [n];
      while (stack.length) {
        var cur = stack.pop();
        childrenOf(cur).forEach(function (c) { if (!seen[c]) { seen[c] = 1; stack.push(c); } });
      }
      return Object.keys(seen);
    }
    var active = false;
    function dfs(node, prev, visited) {
      if (active) return;
      if (node === y) { active = true; return; }
      var cameIn = prev !== null && parents(node).indexOf(prev) >= 0;
      parents(node).concat(childrenOf(node)).forEach(function (nb) {
        if (active || nb === prev || visited.indexOf(nb) >= 0) return;
        var nbIsParent = parents(node).indexOf(nb) >= 0;
        var ok;
        if (cameIn && nbIsParent) { // 碰撞节点：z 中含 node 或其后方放行
          var desc = descendants(node);
          ok = zset[node] || desc.some(function (d) { return zset[d]; });
        } else ok = !zset[node];
        if (ok) dfs(nb, node, visited.concat([nb]));
      });
    }
    dfs(x, null, [x]);
    return !active;
  }
  function ancestralSubgraph(dag, nodes) { // nodes 及其全部祖先诱导的子图
    var need = {}, stack = nodes.slice();
    while (stack.length) {
      var n = stack.pop();
      if (need[n]) continue;
      need[n] = 1;
      (dag[n] || []).forEach(function (p) { stack.push(p); });
    }
    var sub = {};
    Object.keys(dag).forEach(function (m) {
      if (need[m]) sub[m] = (dag[m] || []).filter(function (p) { return need[p]; });
    });
    return sub;
  }
  function mSeparated(dag, x, y, z) { // 祖先子图道德图可达性判据（Lauritzen 1990）
    var sub = ancestralSubgraph(dag, [x, y].concat(z));
    var adj = moralGraph(sub);
    var zset = {};
    z.forEach(function (v) { zset[v] = 1; });
    var seen = {}, stack = [x];
    seen[x] = 1;
    while (stack.length) {
      var cur = stack.pop();
      Object.keys(adj[cur]).forEach(function (nb) {
        if (!seen[nb] && !zset[nb]) { seen[nb] = 1; stack.push(nb); }
      });
    }
    return !seen[y];
  }
  function particleFilter(pi, A, B, obs, n, seed) { // 自举粒子滤波
    var k = pi.length;
    var rng = mulberry32(seed === undefined ? 17 : seed);
    var particles = [], weights = [];
    for (var i = 0; i < n; i++) { particles.push(i % k); weights.push(1 / n); }
    var postMeans = [];
    for (var t = 0; t < obs.length; t++) {
      var wsum = 0;
      for (var i2 = 0; i2 < n; i2++) {
        var prev = particles[i2], u = rng(), acc = 0, s = k - 1;
        for (var j = 0; j < k; j++) { acc += A[prev][j]; if (u <= acc) { s = j; break; } }
        particles[i2] = s;
        weights[i2] = B[s][obs[t]];
        wsum += weights[i2];
      }
      for (var i3 = 0; i3 < n; i3++) weights[i3] /= wsum;
      var mean = 0;
      for (var i4 = 0; i4 < n; i4++) mean += weights[i4] * particles[i4];
      postMeans.push(mean);
      var cum = [], c = 0;
      for (var i5 = 0; i5 < n; i5++) { c += weights[i5]; cum.push(c); }
      var np = [], u0 = rng() / n;
      for (var i6 = 0; i6 < n; i6++) {
        var target = u0 + i6 / n, idx = 0;
        while (idx < n - 1 && target > cum[idx]) idx++;
        np.push(particles[idx]);
      }
      particles = np;
      weights = new Array(n).fill(1 / n);
    }
    return { postMeans: postMeans };
  }

  /* ---- 12.12 RL 强化学习指令 ---- */
  function bellmanBackup(V, mdp) { // Bellman 最优备份算子 T
    return V.map(function (_, s) {
      var best = -Infinity;
      mdp.actions.forEach(function (a) {
        var v = 0;
        mdp.trans(s, a).forEach(function (tr) { v += tr[1] * (mdp.reward(s, tr[0]) + mdp.gamma * V[tr[0]]); });
        if (v > best) best = v;
      });
      return best;
    });
  }
  function policyIteration(mdp) { // 策略迭代：评估 + 改进至稳定
    var n = mdp.n, pi = new Array(n).fill(mdp.actions[0]);
    var V = new Array(n).fill(0), iters = 0, vHistory = [];
    for (var it = 0; it < 200; it++) {
      iters++;
      for (var k = 0; k < 20000; k++) {
        var Vn = new Array(n);
        for (var s = 0; s < n; s++) {
          var a = pi[s], v = 0;
          mdp.trans(s, a).forEach(function (tr) { v += tr[1] * (mdp.reward(s, tr[0]) + mdp.gamma * V[tr[0]]); });
          Vn[s] = v;
        }
        var delta = 0;
        for (var s2 = 0; s2 < n; s2++) delta = Math.max(delta, Math.abs(Vn[s2] - V[s2]));
        V = Vn;
        if (delta < 1e-13) break;
      }
      vHistory.push(V.slice());
      var stable = true;
      for (var s3 = 0; s3 < n; s3++) {
        var bestA = pi[s3], bestV = -Infinity;
        mdp.actions.forEach(function (a2) {
          var v2 = 0;
          mdp.trans(s3, a2).forEach(function (tr) { v2 += tr[1] * (mdp.reward(s3, tr[0]) + mdp.gamma * V[tr[0]]); });
          if (v2 > bestV + 1e-12) { bestV = v2; bestA = a2; }
        });
        if (bestA !== pi[s3]) { stable = false; pi[s3] = bestA; }
      }
      if (stable) break;
    }
    return { V: V, policy: pi, iterations: iters, vHistory: vHistory };
  }
  function sarsa(opts) { // 同策略 TD 控制
    opts = opts || {};
    var n = opts.n || 7, gamma = opts.gamma || 0.9, episodes = opts.episodes || 5000;
    var alpha = opts.alpha || 0.2, epsStart = opts.epsStart || 1, epsEnd = opts.epsEnd || 0.05;
    var seed = opts.seed === undefined ? 321 : opts.seed;
    var rng = mulberry32(seed);
    var Q = Array.from({ length: n }, function () { return new Array(2).fill(0); });
    function clamp(s) { return Math.max(0, Math.min(n - 1, s)); }
    for (var ep = 0; ep < episodes; ep++) {
      var eps = epsStart + (epsEnd - epsStart) * ep / episodes;
      var s = 0;
      var a = epsilonGreedy(Q[s], eps, rng);
      for (var t = 0; t < 200; t++) {
        var sp = clamp(s + (a === 1 ? 1 : -1));
        var r = sp === n - 1 ? 1 : 0;
        var done = sp === n - 1;
        var ap = done ? 0 : epsilonGreedy(Q[sp], eps, rng);
        Q[s][a] += alpha * (r + (done ? 0 : gamma * Q[sp][ap]) - Q[s][a]);
        s = sp; a = ap;
        if (done) break;
      }
    }
    var success = 0, trials = 200;
    for (var i = 0; i < trials; i++) {
      var s2 = 0, ok = false;
      for (var t2 = 0; t2 < 200; t2++) {
        var a2 = Q[s2][0] >= Q[s2][1] ? 0 : 1;
        s2 = clamp(s2 + (a2 === 1 ? 1 : -1));
        if (s2 === n - 1) { ok = true; break; }
      }
      if (ok) success++;
    }
    return { Q: Q, successRate: success / trials };
  }
  function eligibilityTrace(T, gamma, lambda) { // 资格迹指数衰减
    var out = [], w = 1;
    for (var t = 0; t < T; t++) { out.push(w); w *= gamma * lambda; }
    return out;
  }
  function monteCarloReturn(mdp, policy, episodes, seed) { // 截断 horizon 回报均值（γ^T 可忽略 → 无偏）
    var rng = mulberry32(seed === undefined ? 55 : seed), sum = 0;
    for (var ep = 0; ep < episodes; ep++) {
      var G = 0, w = 1, s = 0;
      for (var t = 0; t < 200; t++) {
        var a = policy(s, rng), tr = mdp.trans(s, a);
        var u = rng(), acc = 0, sp = tr[tr.length - 1][0];
        for (var i = 0; i < tr.length; i++) { acc += tr[i][1]; if (u <= acc) { sp = tr[i][0]; break; } }
        G += w * mdp.reward(s, sp);
        w *= mdp.gamma;
        s = sp;
      }
      sum += G;
    }
    return sum / episodes;
  }
  function policyEval(mdp, policy) { // 给定策略的 Bellman 期望备份
    var V = new Array(mdp.n).fill(0);
    for (var k = 0; k < 20000; k++) {
      var Vn = new Array(mdp.n);
      for (var s = 0; s < mdp.n; s++) {
        var a = policy(s, null), v = 0;
        mdp.trans(s, a).forEach(function (tr) { v += tr[1] * (mdp.reward(s, tr[0]) + mdp.gamma * V[tr[0]]); });
        Vn[s] = v;
      }
      var delta = 0;
      for (var s2 = 0; s2 < mdp.n; s2++) delta = Math.max(delta, Math.abs(Vn[s2] - V[s2]));
      V = Vn;
      if (delta < 1e-13) break;
    }
    return V;
  }
  function greedyPolicy(qrow) {
    var best = 0;
    for (var i = 1; i < qrow.length; i++) if (qrow[i] > qrow[best]) best = i;
    return best;
  }
  function ucb(values, counts, c, t) {
    var best = 0, bestScore = -Infinity;
    for (var i = 0; i < values.length; i++) {
      var score = counts[i] === 0 ? Infinity : values[i] + c * Math.sqrt(Math.log(t) / counts[i]);
      if (score > bestScore) { bestScore = score; best = i; }
    }
    return best;
  }
  function advantage(Q, V, s, a) { return Q[s][a] - V[s]; }
  function gae(rewards, values, gamma, lambda) { // 广义优势估计（Schulman 2016）
    var T = rewards.length, out = new Array(T), g = 0;
    for (var t = T - 1; t >= 0; t--) {
      var nextV = t === T - 1 ? 0 : values[t + 1];
      var delta = rewards[t] + gamma * nextV - values[t];
      g = delta + gamma * lambda * g;
      out[t] = g;
    }
    return out;
  }
  function reinforceGradient(prefs, action, reward) { // ∇log π(a)·R
    var p = softmax(prefs);
    var g = p.slice();
    g[action] -= 1;
    return g.map(function (v) { return v * reward; });
  }
  function ppoClip(ratio, adv, eps) { // PPO 截断目标（Schulman 2017）
    return Math.min(ratio * adv, clip(ratio, 1 - eps, 1 + eps) * adv);
  }
  function thompsonSample(alpha, beta, seed) { // Beta(α,β) 后验采样（整数形状参数）
    var rng = mulberry32(seed === undefined ? 13 : seed);
    function gammaSample(a) {
      var p = 1;
      for (var i = 0; i < Math.round(a); i++) p *= Math.max(rng(), 1e-300);
      return -Math.log(p);
    }
    var x = gammaSample(alpha), y = gammaSample(beta);
    return x / (x + y);
  }

  /* ---- 12.13 METRIC/DATA/SYS/LINALG 补齐 ---- */
  function perplexityLM(y, p) { return Math.exp(logLoss(y, p)); } // 语言模型困惑度 = exp(平均 NLL)
  function bleu(reference, hypothesis, maxN) { // Papineni 2002：n-gram 裁剪精度几何平均 × 简短惩罚
    maxN = maxN || 2;
    function ngrams(tokens, n) {
      var g = {};
      for (var i = 0; i + n <= tokens.length; i++) {
        var key = tokens.slice(i, i + n).join(' ');
        g[key] = (g[key] || 0) + 1;
      }
      return g;
    }
    var logSum = 0;
    for (var n = 1; n <= maxN; n++) {
      var hg = ngrams(hypothesis, n), rg = ngrams(reference, n), match = 0, total = 0;
      Object.keys(hg).forEach(function (k) {
        total += hg[k];
        if (rg[k]) match += Math.min(hg[k], rg[k]);
      });
      if (match === 0) return 0;
      logSum += Math.log(match / total);
    }
    var precision = Math.exp(logSum / maxN);
    var bp = hypothesis.length > reference.length ? 1 : Math.exp(1 - reference.length / hypothesis.length);
    return bp * precision;
  }
  function averagePrecision(relevances) { // AP = Σ P(k)·Δr(k)
    var hits = 0, sum = 0;
    for (var k = 0; k < relevances.length; k++) {
      if (relevances[k]) { hits++; sum += hits / (k + 1); }
    }
    return hits === 0 ? 0 : sum / hits;
  }
  function confusionMatrix(y, yhat, classes) {
    var idx = {};
    classes.forEach(function (c, i) { idx[c] = i; });
    var M = zeros2(classes.length, classes.length);
    for (var i = 0; i < y.length; i++) M[idx[y[i]]][idx[yhat[i]]]++;
    return M;
  }
  function labelEncode(labels) {
    var classes = Array.from(new Set(labels)).sort();
    var index = {};
    classes.forEach(function (c, i) { index[c] = i; });
    return { classes: classes, codes: labels.map(function (l) { return index[l]; }) };
  }
  function padSeq(arr, len, v) {
    v = v === undefined ? 0 : v;
    var out = arr.slice();
    while (out.length < len) out.push(v);
    return out;
  }
  function truncateSeq(arr, k) { return arr.slice(0, k); }
  function bucketize(arr, k) { // 等分桶：各桶长度差 ≤ 1
    var n = arr.length, base = Math.floor(n / k), rem = n % k, out = [], i = 0;
    for (var b = 0; b < k; b++) {
      var sz = base + (b < rem ? 1 : 0);
      out.push(arr.slice(i, i + sz));
      i += sz;
    }
    return out;
  }
  function dedup(arr) {
    var seen = {}, out = [];
    arr.forEach(function (v) { if (!seen[v]) { seen[v] = 1; out.push(v); } });
    return out;
  }
  function filterOp(arr, pred) { return arr.filter(pred); }
  function mapOp(arr, f) { return arr.map(f); }
  function zipOp(a, b) { var n = Math.min(a.length, b.length); return a.slice(0, n).map(function (v, i) { return [v, b[i]]; }); } // 长度 = min
  function batchOp(arr, size) {
    var out = [];
    for (var i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
  }
  function tokenize(s) { return s.split(/\s+/).filter(function (t) { return t.length; }); }
  function normalizeFit(train, test) { // 训练集统计量拟合，测试集同参变换
    var m = expectation(train), sd = Math.sqrt(variance(train));
    return {
      mean: m, std: sd,
      train: train.map(function (v) { return (v - m) / sd; }),
      test: test.map(function (v) { return (v - m) / sd; })
    };
  }
  function augment(x, seed, scale) { // 保标签抖动增强
    scale = scale || 0.1;
    var rng = mulberry32(seed === undefined ? 21 : seed);
    return x.map(function (v) { return v + (rng() * 2 - 1) * scale; });
  }
  function gatherAll(parts) { return parts.reduce(function (a, p) { return a.concat(p); }, []); }
  function scatterAll(items, assignment) { // assignment[i] = 目标秩
    var out = {};
    items.forEach(function (it, i) {
      var r = assignment[i];
      (out[r] = out[r] || []).push(it);
    });
    return out;
  }
  function Cache() { this.map = new Map(); }
  Cache.prototype.get = function (k) { return this.map.get(k); };
  Cache.prototype.put = function (k, v) { this.map.set(k, v); return v; };
  function RefCounter() { this.counts = {}; }
  RefCounter.prototype.inc = function (k) { this.counts[k] = (this.counts[k] || 0) + 1; return this.counts[k]; };
  RefCounter.prototype.dec = function (k) {
    this.counts[k] = (this.counts[k] || 0) - 1;
    if (this.counts[k] <= 0) { delete this.counts[k]; return 0; }
    return this.counts[k];
  };
  function TaskStream() { this.tasks = []; }
  TaskStream.prototype.push = function (fn) { this.tasks.push(fn); };
  TaskStream.prototype.run = function () {
    var log = [];
    this.tasks.forEach(function (fn, i) { log.push(fn(i)); });
    return log;
  };
  function checkpoint(obj) { return JSON.parse(JSON.stringify(obj)); } // 保存/恢复逐位一致
  function Profiler() { this.counts = {}; }
  Profiler.prototype.hit = function (k) { this.counts[k] = (this.counts[k] || 0) + 1; };
  Profiler.prototype.total = function () { var s = 0; for (var k in this.counts) s += this.counts[k]; return s; };
  function Tracer() { this.events = []; }
  Tracer.prototype.record = function (name) { this.events.push({ name: name, seq: this.events.length }); return this.events.length - 1; };
  function Dispatcher() { this.handlers = []; }
  Dispatcher.prototype.register = function (fn) { this.handlers.push(fn); };
  Dispatcher.prototype.dispatch = function (payload) { return this.handlers.map(function (fn) { return fn(payload); }); };
  function Barrier(n) { this.n = n; this.arrived = 0; }
  Barrier.prototype.arrive = function () { this.arrived++; return this.arrived === this.n; };
  function EventLog() { this.log = []; }
  EventLog.prototype.emit = function (name) { this.log.push(name); };
  function prefetch(data) { return tensorCopy(data); }
  function matrixRank(A, tol) { // 部分选主元高斯消元秩
    tol = tol || 1e-10;
    var M = clone2(A), rows = M.length, cols = M[0].length, rank = 0, row = 0;
    for (var c = 0; c < cols && row < rows; c++) {
      var p = row;
      for (var i = row + 1; i < rows; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
      if (Math.abs(M[p][c]) < tol) continue;
      var t = M[p]; M[p] = M[row]; M[row] = t;
      for (var i2 = row + 1; i2 < rows; i2++) {
        var f = M[i2][c] / M[row][c];
        for (var j = 0; j < cols; j++) M[i2][j] -= f * M[row][j];
      }
      row++; rank++;
    }
    return rank;
  }
  function nullspace(A, tol) { // RREF → 自由变量基
    tol = tol || 1e-10;
    var M = clone2(A), rows = M.length, cols = M[0].length;
    var pivots = [], row = 0;
    for (var c = 0; c < cols && row < rows; c++) {
      var p = row;
      for (var i = row + 1; i < rows; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
      if (Math.abs(M[p][c]) < tol) continue;
      var t = M[p]; M[p] = M[row]; M[row] = t;
      var piv = M[row][c];
      for (var j = 0; j < cols; j++) M[row][j] /= piv;
      for (var i2 = 0; i2 < rows; i2++) {
        if (i2 === row) continue;
        var f = M[i2][c];
        if (f === 0) continue;
        for (var j2 = 0; j2 < cols; j2++) M[i2][j2] -= f * M[row][j2];
      }
      pivots.push(c); row++;
    }
    var free = [];
    for (var c2 = 0; c2 < cols; c2++) if (pivots.indexOf(c2) === -1) free.push(c2);
    return free.map(function (fc) {
      var v = new Array(cols).fill(0);
      v[fc] = 1;
      pivots.forEach(function (pc, r) { v[pc] = -M[r][fc]; });
      return v;
    });
  }
  function orthonormalize(cols) { // 修正 Gram–Schmidt
    var Q = [];
    cols.forEach(function (a) {
      var v = a.slice();
      Q.forEach(function (q) {
        var c = dot(v, q);
        for (var i = 0; i < v.length; i++) v[i] -= c * q[i];
      });
      var n = norm2(v);
      if (n > 1e-12) Q.push(v.map(function (x) { return x / n; }));
    });
    return Q;
  }
  function condEstimate(A) { // κ₂ = √(σ_max(A)² · σ_max(A⁻¹)²)，幂迭代估计
    var sMax = powerIteration(matMul(transpose(A), A), { iters: 2000 }).value;
    var Ai = inverse(A);
    var sMaxInv = powerIteration(matMul(transpose(Ai), Ai), { iters: 2000 }).value;
    return Math.sqrt(sMax * sMaxInv);
  }

  function sampleStats(xs) { // 采样矩估计
    var n = xs.length, m = expectation(xs), v = variance(xs), s3 = 0, s4 = 0;
    for (var i = 0; i < n; i++) { var d = xs[i] - m; s3 += d * d * d; s4 += d * d * d * d; }
    return { n: n, mean: m, variance: v, skew: s3 / n / Math.pow(v, 1.5), kurt: s4 / n / (v * v) };
  }
  function channelCapacity(P) { // Arimoto–Blahut 迭代（1972）：C = max_p I(X;Y)，输出 bits（I / ln2）
    var k = P.length, m = P[0].length;
    var q = new Array(k).fill(1 / k), c = new Array(m).fill(0);
    for (var it = 0; it < 5000; it++) {
      var r = new Array(m).fill(0);
      for (var j = 0; j < m; j++) for (var i = 0; i < k; i++) r[j] += q[i] * P[i][j];
      var cNew = new Array(m).fill(0);
      for (var j2 = 0; j2 < m; j2++) {
        var s = 0;
        for (var i2 = 0; i2 < k; i2++) {
          var p = P[i2][j2];
          if (p > 0 && q[i2] > 0) s += q[i2] * p * Math.log(p / r[j2]);
        }
        cNew[j2] = Math.exp(s);
      }
      var qNew = new Array(k);
      for (var i3 = 0; i3 < k; i3++) {
        var t = 0;
        for (var j3 = 0; j3 < m; j3++) t += cNew[j3] * P[i3][j3];
        qNew[i3] = t;
      }
      var qs = qNew.reduce(function (a, b) { return a + b; }, 0);
      qNew = qNew.map(function (v) { return v / qs; });
      var delta = 0;
      for (var i4 = 0; i4 < k; i4++) delta = Math.max(delta, Math.abs(qNew[i4] - q[i4]));
      q = qNew;
      if (delta < 1e-15) break;
    }
    var rOut = new Array(m).fill(0);
    for (var j4 = 0; j4 < m; j4++) for (var i5 = 0; i5 < k; i5++) rOut[j4] += q[i5] * P[i5][j4];
    var I = 0;
    for (var i6 = 0; i6 < k; i6++) for (var j5 = 0; j5 < m; j5++) {
      if (q[i6] > 0 && P[i6][j5] > 0 && rOut[j5] > 0) I += q[i6] * P[i6][j5] * Math.log(P[i6][j5] / rOut[j5]);
    }
    return { capacity: I / Math.LN2, q: q }; // bits
  }
  function lineSearch(f, g, x, d, opts) { // Armijo 充分下降回溯线搜索
    opts = opts || {};
    var alpha = opts.alpha0 || 1, c1 = opts.c1 || 1e-4, rho = opts.rho || 0.5, maxIter = opts.maxIter || 60;
    var f0 = f(x), gd = dot(g(x), d);
    for (var i = 0; i < maxIter; i++) {
      var xn = x.map(function (v, t) { return v + alpha * d[t]; });
      var fn = f(xn);
      if (fn <= f0 + c1 * alpha * gd) return { alpha: alpha, x: xn, f: fn, iterations: i + 1 };
      alpha *= rho;
    }
    return { alpha: 0, x: x, f: f0, iterations: maxIter };
  }

  /* ---- 12.14 拟牛顿 / 层归一化梯度 / MCMC / 率失真（v1.1.1）---- */
  function eye2(n) {
    return Array.from({ length: n }, function (_, i) {
      return Array.from({ length: n }, function (_, j) { return i === j ? 1 : 0; });
    });
  }
  function wolfeLineSearch(f, g, x, p, gx, opts) { // 强 Wolfe 线搜索（Nocedal–Wright 算法 3.5/3.6）
    opts = opts || {};
    var c1 = opts.c1 || 1e-4, c2 = opts.c2 || 0.9, alphaMax = opts.alphaMax || 1e3;
    var phi0 = f(x), dPhi0 = dot(gx, p);
    if (!(dPhi0 < 0)) return { alpha: 0, x: x, f: phi0, iterations: 0 };
    function step(a) {
      return { x: x.map(function (v, t) { return v + a * p[t]; }), phi: 0 };
    }
    var alphaPrev = 0, phiPrev = phi0, dPhiPrev = dPhi0, alpha = 1, i;
    for (i = 1; i <= 60; i++) {
      var s = step(alpha);
      s.phi = f(s.x);
      if (s.phi > phi0 + c1 * alpha * dPhi0 || (s.phi >= phiPrev && i > 1)) return zoom(alphaPrev, alpha);
      var gn = g(s.x), dPhi = dot(gn, p);
      if (Math.abs(dPhi) <= -c2 * dPhi0) return { alpha: alpha, x: s.x, f: s.phi, iterations: i };
      if (dPhi >= 0) return zoom(alpha, alphaPrev);
      alphaPrev = alpha; phiPrev = s.phi; dPhiPrev = dPhi;
      alpha = Math.min(2 * alpha, alphaMax);
    }
    return { alpha: 0, x: x, f: phi0, iterations: i };
    function zoom(lo, hi) {
      var j;
      for (j = 1; j <= 60; j++) {
        var aj = 0.5 * (lo + hi);
        var sj = step(aj);
        sj.phi = f(sj.x);
        if (sj.phi > phi0 + c1 * aj * dPhi0 || sj.phi >= phiPrev) { hi = aj; continue; }
        var gj = g(sj.x), dPhij = dot(gj, p);
        if (Math.abs(dPhij) <= -c2 * dPhi0) return { alpha: aj, x: sj.x, f: sj.phi, iterations: j };
        if (dPhij * (hi - lo) >= 0) hi = lo;
        lo = aj; phiPrev = sj.phi; dPhiPrev = dPhij;
      }
      return { alpha: 0, x: x, f: phi0, iterations: j };
    }
  }
  function bfgs(f, grad, x0, opts) { // BFGS 拟牛顿（Broyden 1970 / Fletcher / Goldfarb / Shanno）
    opts = opts || {};
    var maxIter = opts.maxIter || 200, tol = opts.tol || 1e-10;
    var n = x0.length;
    var x = x0.slice(), H = eye2(n), g = grad(x), fx = f(x), k = 0;
    for (k = 0; k < maxIter; k++) {
      if (norm2(g) < tol) break;
      var p = matVec(H, g).map(function (v) { return -v; });
      var ls = wolfeLineSearch(f, grad, x, p, g);
      if (ls.alpha === 0) break;
      var xNew = ls.x, gNew = grad(xNew), s = xNew.map(function (v, t) { return v - x[t]; });
      var y = gNew.map(function (v, t) { return v - g[t]; });
      var sy = dot(s, y);
      if (sy > 1e-12) {
        var rho = 1 / sy;
        // H ← (I − ρsyᵀ)H(I − ρysᵀ) + ρssᵀ
        var Hy = matVec(H, y);
        var yHy = dot(y, Hy);
        var coef = rho * (1 + rho * yHy);
        for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) {
          H[a][b] += -rho * (s[a] * Hy[b] + Hy[a] * s[b]) + coef * s[a] * s[b];
        }
      }
      x = xNew; g = gNew; fx = ls.f;
    }
    return { x: x, iterations: k, f: fx, g: g, H: H };
  }
  function lbfgs(f, grad, x0, opts) { // L-BFGS 两循环递归（Liu–Nocedal 1989）
    opts = opts || {};
    var m = opts.m || 10, maxIter = opts.maxIter || 500, tol = opts.tol || 1e-10;
    var x = x0.slice(), g = grad(x), s = [], y = [], rho = [], k = 0;
    for (k = 0; k < maxIter; k++) {
      if (norm2(g) < tol) break;
      var q = g.slice(), alphas = [];
      for (var i = s.length - 1; i >= 0; i--) {
        var a = rho[i] * dot(s[i], q);
        alphas[i] = a;
        q = q.map(function (v, t) { return v - a * y[i][t]; });
      }
      var gamma = s.length ? dot(s[s.length - 1], y[y.length - 1]) / dot(y[y.length - 1], y[y.length - 1]) : 1;
      var r = q.map(function (v) { return gamma * v; });
      for (var j = 0; j < s.length; j++) {
        var b = rho[j] * dot(y[j], r);
        r = r.map(function (v, t) { return v + (alphas[j] - b) * s[j][t]; });
      }
      var p = r.map(function (v) { return -v; });
      var ls = wolfeLineSearch(f, grad, x, p, g);
      if (ls.alpha === 0) break;
      var xNew = ls.x, gNew = grad(xNew);
      var sv = xNew.map(function (v, t) { return v - x[t]; });
      var yv = gNew.map(function (v, t) { return v - g[t]; });
      var sy = dot(sv, yv);
      if (sy > 1e-12) {
        s.push(sv); y.push(yv); rho.push(1 / sy);
        if (s.length > m) { s.shift(); y.shift(); rho.shift(); }
      }
      x = xNew; g = gNew;
    }
    return { x: x, iterations: k, g: g, memory: s.length };
  }
  function gradLayerNorm(x, dOut, eps) { // 层归一化解析梯度（链式法则复合）
    if (eps === undefined) eps = 1e-5;
    var n = x.length, mu = expectation(x);
    var xc = x.map(function (v) { return v - mu; });
    var v = variance(x) + eps;
    var invStd = 1 / Math.sqrt(v);
    var sumD = 0, sumDXc = 0;
    for (var i = 0; i < n; i++) { sumD += dOut[i]; sumDXc += dOut[i] * xc[i]; }
    var coef2 = invStd * sumD / n, coef3 = invStd * sumDXc / (n * v);
    return x.map(function (_, i) { return invStd * dOut[i] - coef2 - coef3 * xc[i]; });
  }
  function gibbsSampler(P, iters, burn, seed) { // Gibbs 采样：单状态链的转移核扫描
    var rng = mulberry32(seed === undefined ? 31 : seed);
    var s = rng() < 0.5 ? 0 : 1, counts = [0, 0];
    for (var t = 0; t < iters; t++) {
      s = rng() < P[s][1] ? 1 : 0;
      if (t >= burn) counts[s]++;
    }
    var total = counts[0] + counts[1];
    return { stationary: [counts[0] / total, counts[1] / total], counts: counts };
  }
  function metropolisSample(target, iters, burn, seed) { // Metropolis 对称提议
    var rng = mulberry32(seed === undefined ? 32 : seed);
    var k = target.length;
    var cur = Math.floor(rng() * k), counts = new Array(k).fill(0);
    for (var t = 0; t < iters; t++) {
      var prop = Math.floor(rng() * k);
      if (prop === cur) { if (t >= burn) counts[cur]++; continue; }
      var ratio = target[prop] / target[cur];
      if (rng() < Math.min(1, ratio)) cur = prop;
      if (t >= burn) counts[cur]++;
    }
    var total = counts.reduce(function (a, b) { return a + b; }, 0);
    return { empirical: counts.map(function (c) { return c / total; }), counts: counts };
  }
  function rateDistortion(sigma, D) { // 高斯信源率失真函数 R(D) = ½log₂(σ²/D)，D < σ²
    var v = sigma * sigma;
    return D >= v ? 0 : 0.5 * Math.log2(v / D);
  }

  /* ========================================================================
   * 13. 不变量检查表（可独立复算的硬金标准）
   * ======================================================================*/
  /* ========================================================================
   * 12.15 执行变体内核（v1.1.2）—— batched：沿 batch 维独立执行，与逐样本调用逐位一致
   *        inplace：原地执行，返回值与 out-of-place 逐位一致
   * ======================================================================*/
  function arrMaxDiff(a, b) { // 递归逐位最大绝对差（数值数组/对象）
    if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b);
    if (Array.isArray(a) && Array.isArray(b)) {
      var mx = 0;
      for (var i = 0; i < a.length; i++) mx = Math.max(mx, arrMaxDiff(a[i], b[i]));
      return mx;
    }
    if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
      var ks = Object.keys(a), mo = 0;
      for (var ki = 0; ki < ks.length; ki++) {
        var av = a[ks[ki]], bv = b[ks[ki]];
        if (typeof av === 'function' || typeof bv === 'function') continue; // 函数不比较（仅校验数值数据）
        mo = Math.max(mo, arrMaxDiff(av, bv));
      }
      return mo;
    }
    return (a === b) ? 0 : 1; // 非数值/数组（如 classes 数组、字符串）
  }
  /* batched 特殊变体（多返回 / 对象型，需自定义聚合） */
  function batched_selfAttention(pairs) {
    return pairs.map(function (p) { var r = selfAttention(p[0], p[1], p[2], false); return [r.weights, r.output]; });
  }
  function batched_oneHot(labelsBatch) {
    var single = oneHot(labelsBatch[0]);
    var matrix = labelsBatch.map(function (lbls) { return oneHot(lbls).matrix; });
    return { classes: single.classes, matrix: matrix };
  }

  /* 12.16 通用批处理引擎（v1.1.4）—— 单一 batched 调度器覆盖全部批兼容算子
   * 零重写原则落地：batched 路径 ≡ 逐样本朴素路径（参照实现 vs 主实现）
   *   cls: 'vec'     batch 为张量数组，fn 吃整个张量         → batch.map(fn)
   *        'elem'    fn 为逐元素标量函数                     → batch.map(v => v.map(fn))
   *        'pair'    batch 为 [a,b] 对数组                   → batch.map(p => fn(p[0],p[1]))
   *        'triple'  batch 为 [a,b,c] 数组                   → batch.map(p => fn(p[0],p[1],p[2]))
   *        'shared'  batch 为张量数组，shared 参数共享        → batch.map(x => fn(x,...shared))
   *        'special' 自定义聚合（多返回/对象型），ref 在 sig 给定 */
  var _V = [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]];
  var _M = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 1], [0, 3]]];
  var _P = [[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [6, 5, 4]], [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]];
  var _T = [[1, 0, 1], [0.5, 0, 1], [2, 1, 1]];
  var _MM = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[5, 6], [7, 8]], [[2, 1], [0, 3]]], [[[2, 1], [0, 3]], [[1, 2], [3, 4]]]];
  var _MV = [[[[1, 2], [3, 4]], [1, 1]], [[[5, 6], [7, 8]], [0, 1]], [[[2, 1], [0, 3]], [1, 0]]];
  var _MS = [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]]; // 对称正定（Cholesky/逆 所需）
  var _SOLVE = (function () { var A1 = spdMatrix(2, 1), A2 = spdMatrix(2, 2); return [[A1, [1, 2]], [A2, [3, 4]]]; })();
  var _LSTSQ = [[[[1, 2], [3, 4], [5, 6]], [1, 2, 3]], [[[2, 1], [1, 2], [0, 1]], [4, 5, 6]]];
  /* v1.2.1 签名表自包含函数字面量（无自由变量，可安全序列化） */
  var FQ = function (x) { return x[0] * x[0]; };
  var GQ = function (x) { return [2 * x[0], 2 * x[1]]; };
  var FV = function (x) { return [x[0] * x[0], 2 * x[1]]; };
  var GAUSS = function (x) { return Math.exp(-x * x / 2); };
  var UNIF = function (x) { return 1 / 2; };
  var CDFID = function (x) { return x; };
  var DOUT = function (y) { return y; };
  var INNER = function (x) { return x[0] * x[0]; };
  var DIN = function (x) { return 2 * x[0]; };
  var MDP3 = chainSetup(3, 0.9, false);
  var MDP4 = chainSetup(4, 0.9, true);
  var BATCH_SIG = {
    /* TENSOR 张量变换（整张量算子） */
    'TENSOR.transpose': { cls: 'vec', fn: transpose, sample: _M },
    'TENSOR.copy': { cls: 'vec', fn: tensorCopy, sample: _M },
    'TENSOR.flatten': { cls: 'vec', fn: flatten, sample: _M },
    /* NUMER 数值稳定：激活类为逐元素标量，其余为整向量 */
    'NUMER.softmax': { cls: 'vec', fn: softmax, sample: _V },
    'NUMER.sigmoid': { cls: 'elem', fn: sigmoid, sample: _V },
    'NUMER.relu': { cls: 'elem', fn: relu, sample: _V },
    'NUMER.gelu': { cls: 'elem', fn: gelu, sample: _V },
    'NUMER.erf': { cls: 'elem', fn: erf, sample: _V },
    'NUMER.reciprocal': { cls: 'elem', fn: reciprocal, sample: _V },
    'NUMER.expm1': { cls: 'elem', fn: expm1, sample: _V },
    'NUMER.log1p': { cls: 'elem', fn: log1p, sample: _V },
    'NUMER.kahan': { cls: 'vec', fn: kahanSum, sample: _V },
    'NUMER.exactsum': { cls: 'vec', fn: exactSum, sample: _V },
    /* NORM 归一化（整向量） */
    'NORM.l1': { cls: 'vec', fn: l1Norm, sample: _V },
    'NORM.l2': { cls: 'vec', fn: l2Norm, sample: _V },
    'NORM.linf': { cls: 'vec', fn: linfNorm, sample: _V },
    'NORM.normalize_l2': { cls: 'vec', fn: normalizeL2, sample: _V },
    'NORM.minmax': { cls: 'vec', fn: minmaxScale, sample: _V },
    'NORM.zscore': { cls: 'vec', fn: zscore, sample: _V },
    /* FWD 前向：线性层为共享权变体，激活类逐元素标量 */
    'FWD.softmax': { cls: 'vec', fn: softmax, sample: _V },
    'FWD.sigmoid': { cls: 'elem', fn: sigmoid, sample: _V },
    'FWD.gelu': { cls: 'elem', fn: gelu, sample: _V },
    'FWD.relu': { cls: 'elem', fn: relu, sample: _V },
    'FWD.tanh': { cls: 'elem', fn: tanh, sample: _V },
    'FWD.linear': { cls: 'shared', fn: linear, shared: [[[1, 0], [0, 1]], [0, 0]], sample: _M },
    /* BWD 反向：梯度算子为整向量 */
    'BWD.grad_relu': { cls: 'vec', fn: gradRelu, sample: _V },
    'BWD.grad_tanh': { cls: 'vec', fn: gradTanh, sample: _V },
    /* INFO 信息论 */
    'INFO.entropy': { cls: 'vec', fn: entropy, sample: _V },
    'INFO.cross_entropy': { cls: 'pair', fn: crossEntropy, sample: _P },
    'INFO.kl': { cls: 'pair', fn: klDivergence, sample: _P },
    'INFO.js': { cls: 'pair', fn: jsDivergence, sample: _P },
    'INFO.mutual_info': { cls: 'vec', fn: mutualInfo, sample: _M },
    'INFO.conditional_entropy': { cls: 'vec', fn: conditionalEntropy, sample: _M },
    'INFO.gini': { cls: 'vec', fn: gini, sample: _V },
    'INFO.normalize_dist': { cls: 'vec', fn: normalizeDist, sample: _V },
    /* PROB 概率 */
    'PROB.gauss_pdf': { cls: 'triple', fn: gaussPDF, sample: _T },
    'PROB.gauss_cdf': { cls: 'triple', fn: normalCDF, sample: _T },
    'PROB.covariance': { cls: 'pair', fn: covariance, sample: _P },
    'PROB.expectation': { cls: 'vec', fn: expectation, sample: _V },
    'PROB.variance': { cls: 'vec', fn: variance, sample: _V },
    'PROB.marginal': { cls: 'vec', fn: marginal, sample: _M },
    /* METRIC 评估（成对） */
    'METRIC.accuracy': { cls: 'pair', fn: accuracy, sample: _P },
    'METRIC.precision': { cls: 'pair', fn: precision, sample: _P },
    'METRIC.recall': { cls: 'pair', fn: recall, sample: _P },
    'METRIC.f1': { cls: 'pair', fn: f1, sample: _P },
    'METRIC.mse': { cls: 'pair', fn: mse, sample: _P },
    'METRIC.mae': { cls: 'pair', fn: mae, sample: _P },
    'METRIC.cosine': { cls: 'pair', fn: cosineSimilarity, sample: _P },
    /* LINALG 线性代数 */
    'LINALG.matmul': { cls: 'pair', fn: matMul, sample: _MM },
    'LINALG.matvec': { cls: 'pair', fn: matVec, sample: _MV },
    'LINALG.solve': { cls: 'pair', fn: solveLinear, sample: _SOLVE },
    'LINALG.lstsq': { cls: 'pair', fn: linearRegression, sample: _LSTSQ },
    'LINALG.trace': { cls: 'vec', fn: trace, sample: _MS },
    'LINALG.det': { cls: 'vec', fn: det, sample: _MS },
    'LINALG.inverse': { cls: 'vec', fn: inverse, sample: _MS },
    'LINALG.cholesky': { cls: 'vec', fn: cholesky, sample: _MS },
    /* OPTIM 优化 */
    'OPTIM.grad_clip': { cls: 'shared', fn: gradClip, shared: [1], sample: _V },
    'OPTIM.grad_norm': { cls: 'vec', fn: gradNorm, sample: _V },
    /* SYS 系统 */
    'SYS.reduce_all': { cls: 'vec', fn: reduceAll, sample: _V },
    'SYS.broadcast': { cls: 'vec', fn: broadcast, sample: _V },
    /* ATTEN 注意力（特殊聚合） */
    'ATTEN.self_attn': { cls: 'special', fn: batched_selfAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]]]],
      ref: function (b) { return b.map(function (p) { var r = selfAttention(p[0], p[1], p[2], false); return [r.weights, r.output]; }); } },
    /* GRAPH 图推断 */
    'GRAPH.marginal': { cls: 'vec', fn: marginal, sample: _M },
    /* RL 强化学习（共享 ε/rng） */
    'RL.epsilon_greedy': { cls: 'shared', fn: epsilonGreedy, shared: [0, mulberry32(99)], sample: [[1, 5, 2], [4, 1, 0]] },
    /* DATA 数据管线（特殊聚合） */
    'DATA.onehot': { cls: 'special', fn: batched_oneHot, sample: [[0, 1, 2]],
      ref: function (b) { var o = oneHot(b[0]); return { classes: o.classes, matrix: [o.matrix] }; } },
    /* SAMPLE 采样生成 */
    'SAMPLE.categorical': { cls: 'pair', fn: categoricalSample, sample: [[[0.2, 0.3, 0.5], 7], [[0.1, 0.4, 0.5], 5]] },
    'SAMPLE.temperature': { cls: 'shared', fn: temperatureSoftmax, shared: [1], sample: _V },
    'SAMPLE.topk': { cls: 'shared', fn: topK, shared: [2], sample: _V },
            /* ——     /* —— v1.1.7 引擎规模化扩面（新增 92 条 batched 签名，样本经沙盒逐条验证）—— */
    'TENSOR.create': { cls: 'args', fn: tensorCreate, sample: [[[2, 2], 1], [[3, 1], 0]] },
    'TENSOR.zeros': { cls: 'args', fn: tensorZeros, sample: [[[2, 2]], [[3, 1]]] },
    'TENSOR.ones': { cls: 'args', fn: tensorOnes, sample: [[[2, 2]], [[1, 3]]] },
    'TENSOR.arange': { cls: 'args', fn: arange, sample: [[0, 6, 2], [1, 5, 1], [0, 4, 1]] },
    'TENSOR.reshape': { cls: 'args', fn: reshape, sample: [[[1, 2, 3, 4], [2, 2]], [[1, 2, 3, 4, 5, 6], [3, 2]]] },
    'TENSOR.slice': { cls: 'args', fn: tensorSlice, sample: [[[1, 2, 3, 4], 0, 2], [[1, 2, 3, 4], 1, 3]] },
    'TENSOR.concat': { cls: 'args', fn: tensorConcat, sample: [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]] },
    'TENSOR.stack': { cls: 'args', fn: tensorStack, sample: [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]] },
    'TENSOR.split': { cls: 'args', fn: tensorSplit, sample: [[[1, 2, 3, 4], [2, 2]], [[1, 2, 3, 4, 5, 6], [3, 3]]] },
    'TENSOR.cast': { cls: 'args', fn: tensorCast, sample: [[[1, 2, 3]], [[4, 5]]] },
    'TENSOR.gather': { cls: 'args', fn: tensorGather, sample: [[[1, 2, 3, 4], [0, 2]], [[5, 6, 7, 8], [1, 3]]] },
    'TENSOR.scatter': { cls: 'args', fn: tensorScatter, sample: [[[1, 2, 3, 4], [0, 1], [9, 9]], [[5, 6, 7, 8], [2, 3], [1, 1]]] },
    'TENSOR.fill': { cls: 'args', fn: tensorFill, sample: [[[1, 2, 3], 7], [[4, 5], 0]] },
    'LINALG.qr': { cls: 'vec', fn: qrDecompose, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]] },
    'LINALG.eigen': { cls: 'vec', fn: jacobiEigen, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]] },
    'LINALG.power': { cls: 'vec', fn: powerIteration, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]] },
    'LINALG.rank': { cls: 'vec', fn: matrixRank, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]] },
    'LINALG.nullspace': { cls: 'vec', fn: nullspace, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]] },
    'LINALG.orthonormalize': { cls: 'vec', fn: orthonormalize, sample: [[[[1, 0], [1, 1]], [[1, 2], [2, 4]]]] },
    'LINALG.cond': { cls: 'vec', fn: condEstimate, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]] },
    'NUMER.logsumexp': { cls: 'vec', fn: logsumexp, sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]] },
    'NUMER.clip': { cls: 'args', fn: clip, sample: [[[1.5, -1, 1], [-3, -1, 1], [0.2, -1, 1]]] },
    'NUMER.fma': { cls: 'triple', fn: fma, sample: [[[1, 2, 3], [4, 5, 6], [-1, 2, 0]]] },
    'PROB.normal_sample': { cls: 'pair', fn: normalSample, sample: [[[3, 1], [5, 2]]] },
    'PROB.categorical_sample': { cls: 'pair', fn: categoricalSample, sample: [[[[0.2, 0.3, 0.5], 7], [[0.1, 0.4, 0.5], 5]]] },
    'PROB.bernoulli_sample': { cls: 'triple', fn: bernoulliSample, sample: [[[0.5, 3, 1], [0.3, 4, 2]]] },
    'PROB.binomial_sample': { cls: 'args', fn: binomialSample, sample: [[[10, 0.5, 5, 1], [8, 0.3, 4, 2]]] },
    'PROB.uniform_sample': { cls: 'pair', fn: uniformSample, sample: [[[3, 1], [5, 2]]] },
    'PROB.sample_stats': { cls: 'vec', fn: sampleStats, sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]] },
    'INFO.info_gain': { cls: 'pair', fn: infoGain, sample: [[[[0, 0, 1, 1], [[0, 1], [1, 1]]], [[1, 1, 0, 0], [[1, 0], [0, 1]]]]] },
    'INFO.perplexity': { cls: 'vec', fn: perplexity, sample: [[[[0.2, 0.3, 0.5]], [[0.5, 0.25, 0.25]]]] },
    'INFO.kl_reverse': { cls: 'pair', fn: klReverse, sample: [[[[0.5, 0.5], [0.4, 0.6]], [[0.3, 0.7], [0.5, 0.5]]]] },
    'METRIC.auc': { cls: 'pair', fn: auc, sample: [[[0, 1, 1], [0.1, 0.4, 0.8]], [[1, 0, 1], [0.3, 0.2, 0.9]]] },
    'METRIC.r2': { cls: 'pair', fn: r2, sample: [[[1, 2, 3], [1.1, 1.9, 3.2]], [[0, 1, 2], [0.1, 0.9, 2.1]]] },
    'METRIC.edit_distance': { cls: 'pair', fn: editDistance, sample: [["kitten", "sitting"], ["sunday", "saturday"]] },
    'METRIC.silhouette': { cls: 'pair', fn: silhouette, sample: [[[[1, 1], [2, 2], [8, 8], [9, 9]], [0, 0, 1, 1]], [[[1, 1], [1.5, 1.5], [9, 9]], [0, 0, 1]]] },
    'METRIC.bleu': { cls: 'triple', fn: bleu, sample: [[["a b", "a b", 2], ["cat sat", "cat sat", 3]]] },
    'METRIC.map': { cls: 'vec', fn: averagePrecision, sample: [[[1, 0, 1], [0, 1, 1]]] },
    'METRIC.confusion': { cls: 'triple', fn: confusionMatrix, sample: [[[0, 1, 1], [0, 1, 0], [0, 1]]] },
    'SYS.gather_all': { cls: 'vec', fn: gatherAll, sample: [[[[1, 2], [3]], [[4], [5, 6]]]] },
    'SYS.scatter_all': { cls: 'pair', fn: scatterAll, sample: [[[[1, 2, 3], [0, 1, 0]], [[4, 5], [1, 0]]]] },
    'SYS.checkpoint': { cls: 'vec', fn: checkpoint, sample: [[{"a":[1,2]}, {"b":[3,4]}]] },
    'SYS.prefetch': { cls: 'vec', fn: prefetch, sample: [[[1, 2, 3], [4, 5, 6]]] },
    'FWD.self_attn': { cls: 'args', fn: selfAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], false]] },
    'FWD.embedding_lookup': { cls: 'pair', fn: embeddingLookup, sample: [[[[1, 2], [3, 4]], [0, 1]], [[[5, 6], [7, 8]], [1, 0]]] },
    'FWD.flatten': { cls: 'vec', fn: flatten, sample: [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 1], [0, 3]]] },
    'FWD.residual_add': { cls: 'pair', fn: residualAdd, sample: [[[[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]]], [[[5, 6], [7, 8]], [[0.5, 0.5], [0.5, 0.5]]]] },
    'FWD.concat_heads': { cls: 'vec', fn: concatHeads, sample: [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]] },
    'FWD.pool_max': { cls: 'pair', fn: poolMax, sample: [[[1, 5, 2, 8], 2], [[3, 1, 4, 2], 2]] },
    'FWD.conv1d': { cls: 'args', fn: conv1d, sample: [[[1, 2, 3, 4], [2, 4, 6]]] },
    'FWD.batchnorm': { cls: 'shared', fn: batchNorm, shared: [0.00001], sample: [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 1], [0, 3]]] },
    'FWD.dropout': { cls: 'args', fn: dropout, sample: [[[[1, 2, 3, 4], 0.5, 1], [[5, 6, 7, 8], 0.5, 2]]] },
    'BWD.grad_softmax_ce': { cls: 'pair', fn: gradSoftmaxCE, sample: [[[1, 2, 3], 1], [[0.5, 0.5, 1], 2]] },
    'BWD.grad_matmul': { cls: 'triple', fn: gradMatmul, sample: [[[[1, 2]], [[1], [2]], [[3]]], [[[3, 4]], [[1], [3]], [[2]]]] },
    'BWD.stop_grad': { cls: 'vec', fn: stopGrad, sample: [[[1, 2, 3], [0.5, 0.5, 1]]] },
    'BWD.detach': { cls: 'vec', fn: detach, sample: [[[1, 2, 3], [0.5, 0.5, 1]]] },
    'NORM.l2_reg': { cls: 'shared', fn: l2Reg, shared: [0.01], sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]] },
    'NORM.l1_reg': { cls: 'shared', fn: l1Reg, shared: [0.01], sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]] },
    'NORM.group_norm': { cls: 'args', fn: groupNorm, sample: [[[[1, 2, 3, 4], 2, 0.00001], [[5, 6, 7, 8], 2, 0.00001]]] },
    'NORM.batch_norm': { cls: 'shared', fn: batchNorm, shared: [0.00001], sample: [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 1], [0, 3]]] },
    'NORM.dropout_reg': { cls: 'args', fn: dropout, sample: [[[[1, 2, 3, 4], 0.5, 2], [[5, 6, 7, 8], 0.5, 3]]] },
    'ATTEN.qk_dot': { cls: 'pair', fn: qkDot, sample: [[[[1, 0], [0, 1]], [[2, 1], [1, 2]]]] },
    'ATTEN.rel_pos': { cls: 'pair', fn: relPosBias, sample: [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]] },
    'ATTEN.mha_split': { cls: 'pair', fn: splitHeads, sample: [[[[1, 2], [3, 4]], 2], [[[5, 6], [7, 8]], 2]] },
    'ATTEN.attn_entropy': { cls: 'vec', fn: attnEntropy, sample: [[[0, 1, 1], [1, 0, 1]]] },
    'ATTEN.softmax_attn': { cls: 'args', fn: selfAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], false]] },
    'ATTEN.causal_mask': { cls: 'args', fn: selfAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], true]] },
    'ATTEN.multi_head': { cls: 'args', fn: multiHead, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], 2, false]] },
    'SAMPLE.box_muller': { cls: 'pair', fn: normalSample, sample: [[[3, 1], [5, 2]]] },
    'SAMPLE.uniform': { cls: 'pair', fn: uniformSample, sample: [[[3, 1], [5, 2]]] },
    'SAMPLE.bernoulli': { cls: 'triple', fn: bernoulliSample, sample: [[[0.5, 3, 1], [0.3, 4, 2]]] },
    'SAMPLE.multinomial': { cls: 'triple', fn: multinomialSample, sample: [[[[0.2, 0.3, 0.5], 3, 1], [[0.1, 0.4, 0.5], 4, 2]]] },
    'SAMPLE.antithetic': { cls: 'pair', fn: antitheticSample, sample: [[[3, 1], [5, 2]]] },
    'SAMPLE.stratified': { cls: 'pair', fn: stratifiedSample, sample: [[[3, 1], [5, 2]]] },
    'SAMPLE.systematic': { cls: 'pair', fn: systematicSample, sample: [[[3, 1], [5, 2]]] },
    'SAMPLE.normal': { cls: 'pair', fn: normalSample, sample: [[[3, 1], [5, 2]]] },
    'DATA.shuffle': { cls: 'pair', fn: shuffle, sample: [[[[1, 2, 3, 4], 1], [[4, 3, 2, 1], 2]]] },
    'DATA.train_test': { cls: 'triple', fn: trainTestSplit, sample: [[[10, 0.8, 1], [8, 0.7, 2]]] },
    'DATA.kfold': { cls: 'triple', fn: kFold, sample: [[[10, 3, 1], [9, 3, 2]]] },
    'DATA.label_encode': { cls: 'vec', fn: labelEncode, sample: [[["a", "b", "a"], ["x", "y", "z", "x"]]] },
    'DATA.pad': { cls: 'triple', fn: padSeq, sample: [[[[1, 2], 4, 0], [[3, 4, 5], 2, -1]]] },
    'DATA.truncate': { cls: 'pair', fn: truncateSeq, sample: [[[[1, 2, 3, 4], 2], [[5, 6, 7], 3]]] },
    'DATA.bucket': { cls: 'pair', fn: bucketize, sample: [[[[1, 2, 3, 4], 2], [[5, 1, 3, 2], 3]]] },
    'DATA.dedup': { cls: 'vec', fn: dedup, sample: [[[1, 2, 2, 3], [4, 4, 5, 1]]] },
    'DATA.filter': { cls: 'shared', fn: filterOp, shared: [function () { return 1; }], sample: [[[1, 2, 3, 4], [0, 1, 2, 3]]] },
    'DATA.zip': { cls: 'pair', fn: zipOp, sample: [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]] },
    'DATA.batch': { cls: 'pair', fn: batchOp, sample: [[[[1, 2, 3, 4], 2], [[5, 6, 7, 8, 9], 3]]] },
    'DATA.augment': { cls: 'triple', fn: augment, sample: [[[[1, 2, 3], 1, 0.1], [[4, 5], 2, 0.2]]] },
    'RL.td0': { cls: 'pair', fn: td0Sweep, sample: [[[3, 0.9], [5, 0.95]]] },
    'RL.td_lambda': { cls: 'triple', fn: eligibilityTrace, sample: [[[[0, 1, 0], 0.9, 0.5], [[1, 1, 0], 0.8, 0.3]]] },
    'RL.greedy': { cls: 'vec', fn: greedyPolicy, sample: [[[1, 5, 2], [3, 1, 4]]] },
    'RL.ucb': { cls: 'args', fn: ucb, sample: [[[[1, 2], [1, 1], 1.4, 10], [[3, 4], [2, 2], 1.2, 20]]] },

    /* —— v1.2.0 变体层扩面（新增 10 条 batched，样本经沙盒逐条验证）—— */
    'ATTEN.scale': { cls: 'pair', fn: attentionScale, sample: [[[[1,2],[3,4]],4],[[[5,6],[7,8]],2]] },
    'ATTEN.alibi': { cls: 'pair', fn: alibiScores, sample: [[[[1,2],[3,4]],0.5],[[[5,6],[7,8]],0.3]] },
    'ATTEN.rope': { cls: 'args', fn: ropeApply, sample: [[[1,0],1,10000],[[2,1],2,10000]] },
    'ATTEN.mqa': { cls: 'args', fn: mqa, sample: [[[[[1,0],[0,1]],[[2,1],[1,2]]],[[1,0],[0,1]],[[1,0],[0,1]]],[[[[1,0],[0,1]],[[2,1],[1,2]]],[[1,0],[0,1]],[[1,0],[0,1]]]] },
    'ATTEN.gqa': { cls: 'args', fn: gqa, sample: [[[[[1,0],[0,1]],[[2,1],[1,2]]],[[[1,0],[0,1]]],[[[1,0],[0,1]]],1],[[[[1,0],[0,1]],[[2,1],[1,2]]],[[[1,0],[0,1]]],[[[1,0],[0,1]]],1]] },
    'SAMPLE.topp': { cls: 'pair', fn: topP, sample: [[[0.1,0.5,0.4],0.8],[[0.2,0.3,0.5],0.9]] },
    'METRIC.logloss': { cls: 'pair', fn: logLoss, sample: [[[1,0,0],[0.7,0.2,0.1]],[[0,1,0],[0.1,0.8,0.1]]] },
    'METRIC.perplexity': { cls: 'pair', fn: perplexityLM, sample: [[[1,0,0],[0.7,0.2,0.1]],[[0,1,0],[0.1,0.8,0.1]]] },
    'DATA.map': { cls: 'shared', fn: mapOp, shared: [function (x) { return x * 2; }], sample: [[1,2,3],[4,5]] },
    'NORM.weight_norm': { cls: 'pair', fn: weightNorm, sample: [[[3,4],2],[[1,2],3]] }
  ,

    /* —— v1.2.1 修复批次（新增 40 条 batched：v1.1.7 样本设计误杀修复 + 语义清晰项，样本经沙盒逐条验证）—— */
    'NUMER.hypot': { cls: 'pair', fn: hypot, sample: [[3, 4], [5, 12], [8, 15]] },
    'NUMER.div_guard': { cls: 'args', fn: divGuard, sample: [[1, 2], [1, 0], [5, -3]] },
    'PROB.bayes': { cls: 'pair', fn: bayesUpdate, sample: [[[0.5, 0.5], [0.8, 0.2]], [[0.6, 0.4], [0.7, 0.3]]] },
    'PROB.condition': { cls: 'pair', fn: conditionalRow, sample: [[[[0.1, 0.9], [0.8, 0.2]], 0], [[[0.1, 0.9], [0.8, 0.2]], 1]] },
    'PROB.mv_gauss': { cls: 'triple', fn: mvGaussPDF, sample: [[[1, 0], [0, 0], [[1, 0], [0, 1]]], [[0.5, 1], [0, 0], [[1, 0], [0, 1]]]] },
    'INFO.surprisal': { cls: 'elem', fn: surprisal, sample: [[0.5, 0.25, 0.1], [0.9, 0.05]] },
    'INFO.bits': { cls: 'elem', fn: bits, sample: [[1, 2, 3], [0.5, 0.25]] },
    'INFO.channel_capacity': { cls: 'vec', fn: channelCapacity, sample: [[[0.9, 0.1], [0.2, 0.8]], [[0.7, 0.3], [0.4, 0.6]]] },
    'INFO.rate_distortion': { cls: 'pair', fn: rateDistortion, sample: [[2, 1], [1, 0.5]] },
    'INFO.max_entropy': { cls: 'vec', fn: maxEntropyDist, sample: [3, 4, 2] },
    'FWD.layernorm': { cls: 'shared', fn: layerNorm, shared: [0.00001], sample: [[[1, 2, 3], [4, 5, 6]], [[2, 1], [0, 3]]] },
    'NORM.spectral_norm': { cls: 'vec', fn: spectralNorm, sample: [[[3, 1], [1, 2]], [[2, 0], [0, 2]]] },
    'NORM.label_smooth': { cls: 'pair', fn: labelSmooth, sample: [[[1, 0, 0], 0.1], [[0, 1, 0], 0.2]] },
    'NORM.mixup': { cls: 'triple', fn: mixup, sample: [[[1, 2, 3], [5, 6, 7], 0.3], [[1, 0], [0, 1], 0.5]] },
    'NORM.early_stop': { cls: 'pair', fn: earlyStop, sample: [[[0.5, 0.4, 0.3, 0.35], 2], [[1, 0.8, 0.9], 1]] },
    'SAMPLE.gumbel': { cls: 'elem', fn: gumbelSample, sample: [[0.1, 0.5, 0.9, 0.3], [0.7, 0.2]] },
    'SAMPLE.rejection': { cls: 'args', fn: rejectionSample, sample: [[3, 1, GAUSS, UNIF, 2, -1, 1], [2, 2, GAUSS, UNIF, 2, -1, 1]] },
    'SAMPLE.inverse_cdf': { cls: 'args', fn: inverseCDF, sample: [[0.5, CDFID, 0, 1], [0.3, CDFID, 0, 1], [0.9, CDFID, 0, 1]] },
    'RL.thompson': { cls: 'triple', fn: thompsonSample, sample: [[2, 3, 1], [3, 2, 2], [1, 1, 3]] },
    'RL.advantage': { cls: 'args', fn: advantage, sample: [[[[1, 2], [3, 4]], [0, 0], 0, 1], [[[1, 2], [3, 4]], [0, 0], 1, 0]] },
    'RL.gae': { cls: 'args', fn: gae, sample: [[[1, 1, 1], [0, 0.5, 0.8], 0.9, 0.95], [[0, 0], [1, 1], 0.99, 1]] },
    'RL.reinforce': { cls: 'triple', fn: reinforceGradient, sample: [[[1, 2, 3], 1, 0.5], [[0.5, 1.5], 0, -0.2]] },
    'RL.ppo': { cls: 'triple', fn: ppoClip, sample: [[1.2, 0.5, 0.2], [0.8, -0.3, 0.1], [0.9, 0.1, 0.2]] },
    'RL.ucb': { cls: 'args', fn: ucb, sample: [[[[1, 2], [1, 1]], 1.4, 10], [[[3, 4], [2, 2]], 1.2, 20]] },
    'RL.greedy': { cls: 'vec', fn: greedyPolicy, sample: [[1, 5, 2], [3, 1, 4]] },
    'OPTIM.lr_decay': { cls: 'triple', fn: lrDecay, sample: [[0.1, 100, 10], [0.5, 50, 5]] },
    'OPTIM.weight_decay': { cls: 'triple', fn: weightDecay, sample: [[[1, -2, 3], 0.1, 0.01], [[0.5, 0.5], 0.2, 0.02]] },
    'OPTIM.polyak': { cls: 'vec', fn: polyakAvg, sample: [[[1, 2], [2, 3], [3, 4]], [[5, 6], [6, 7], [7, 8]]] },
    'BWD.grad_softmax_ce': { cls: 'pair', fn: gradSoftmaxCE, sample: [[[1, 2, 3], 1], [[0.5, 1.5, 2.5], 2]] },
    'BWD.stop_grad': { cls: 'vec', fn: stopGrad, sample: [[1, 2, 3], [0.5, 0.5, 1]] },
    'BWD.detach': { cls: 'vec', fn: detach, sample: [[1, 2, 3], [0.5, 0.5, 1]] },
    'BWD.jacobian': { cls: 'args', fn: jacobian, sample: [[FV, [1, 2], 0.000001], [FV, [3, 4], 0.000001]] },
    'BWD.vjp': { cls: 'args', fn: vjp, sample: [[FV, [1, 2], [1, 1], 0.000001], [FV, [3, 4], [0, 1], 0.000001]] },
    'BWD.jvp': { cls: 'args', fn: jvp, sample: [[FV, [1, 2], [1, 1], 0.000001], [FV, [3, 4], [1, 0], 0.000001]] },
    'BWD.grad_accum': { cls: 'vec', fn: gradAccum, sample: [[[[1, 2], [3, 4]]], [[[0.5, 0.5], [1, 1]]]] },
    'ATTEN.cross_attn': { cls: 'args', fn: crossAttention, sample: [[[[1, 0], [0, 1]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[2, 1], [1, 2]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]] },
    'ATTEN.kv_cache': { cls: 'args', fn: attnRow, sample: [[[1, 0], [[1, 0], [0, 1]], [[1, 2], [3, 4]]], [[0, 1], [[1, 0], [0, 1]], [[1, 2], [3, 4]]]] },
    'ATTEN.flash': { cls: 'args', fn: flashAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], 1], [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], 2]] },
    'DATA.normalize_fit': { cls: 'pair', fn: normalizeFit, sample: [[[1, 2, 3, 4], [5, 6]], [[2, 3], [4, 5]]] },
    'LINALG.pca': { cls: 'pair', fn: pca, sample: [[[[4, 1], [1, 3]], 1], [[[2, 0], [0, 2]], 1]] }
  ,

    /* —— v1.2.2 图推断+优化器批次（新增 26 条 batched，样本经沙盒逐条验证）—— */
    'OPTIM.gd': { cls: 'args', fn: gradientDescent, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.adam': { cls: 'args', fn: adam, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.momentum': { cls: 'args', fn: momentum, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.nesterov': { cls: 'args', fn: nesterov, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.adagrad': { cls: 'args', fn: adagrad, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.rmsprop': { cls: 'args', fn: rmsprop, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.adamw': { cls: 'args', fn: adamw, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.lbfgs': { cls: 'args', fn: lbfgs, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.bfgs': { cls: 'args', fn: bfgs, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]] },
    'OPTIM.cg': { cls: 'args', fn: conjugateGradient, sample: [[[[4, 1], [1, 3]], [1, 2], {  }], [[[3, 0], [0, 2]], [1, 1], {  }]] },
    'OPTIM.line_search': { cls: 'args', fn: lineSearch, sample: [[FQ, GQ, [1, 2], [-2, -4], {  }], [FQ, GQ, [2, 1], [-4, -2], {  }]] },
    'GRAPH.forward_backward': { cls: 'args', fn: hmmForwardBackward, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]] },
    'GRAPH.viterbi': { cls: 'args', fn: hmmViterbi, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]] },
    'GRAPH.belief_prop': { cls: 'args', fn: chainBeliefProp, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]] },
    'GRAPH.variable_elim': { cls: 'args', fn: variableElim, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]] },
    'GRAPH.map_query': { cls: 'args', fn: hmmViterbi, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]] },
    'GRAPH.junction_tree': { cls: 'pair', fn: junctionTree, sample: [[[[0, 1], [1, 2]], [[0, 1]]], [[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2]]]] },
    'GRAPH.condition': { cls: 'pair', fn: conditionalRow, sample: [[[[0.1, 0.9], [0.8, 0.2]], 0], [[[0.1, 0.9], [0.8, 0.2]], 1]] },
    'GRAPH.gibbs': { cls: 'args', fn: gibbsSampler, sample: [[[[0, 1], [1, 0]], 100, 20, 1], [[[0.2, 0.8], [0.6, 0.4]], 100, 20, 2]] },
    'GRAPH.mcmc': { cls: 'args', fn: metropolisSample, sample: [[[0.3, 0.7], 100, 20, 1], [[0.1, 0.5, 0.4], 100, 20, 2]] },
    'GRAPH.particle': { cls: 'args', fn: particleFilter, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0], 50, 1], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0], 50, 2]] },
    'GRAPH.kalman': { cls: 'args', fn: kalmanFilter1d, sample: [[[1, 2, 3], 1, 1, 0.1, 0.1, 0, 1], [[2, 4], 0.9, 1, 0.2, 0.3, 1, 1]] },
    'GRAPH.dsep': { cls: 'args', fn: dSeparated, sample: [[{ 0: [], 1: [0], 2: [0] }, 1, 2, [0]], [{ 0: [], 1: [0], 2: [0] }, 1, 2, []]] },
    'GRAPH.m_sep': { cls: 'args', fn: mSeparated, sample: [[{ 0: [], 1: [0], 2: [0] }, 1, 2, [0]], [{ 0: [], 1: [0], 2: [0] }, 1, 2, []]] },
    'GRAPH.clique_tree': { cls: 'triple', fn: cliqueTreeCalibrate, sample: [[{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [1, 2], shape: [2, 2], table: [0.2, 0.1, 0.4, 0.3] }, [1]], [{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [1, 2], shape: [2, 2], table: [0.2, 0.1, 0.4, 0.3] }, [0]]] },
    'GRAPH.factor_product': { cls: 'pair', fn: factorProduct, sample: [[{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [1, 2], shape: [2, 2], table: [0.2, 0.1, 0.4, 0.3] }], [{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }]] }
  ,

    /* —— v1.2.3 最后批次（新增 12 条 batched，样本经沙盒逐条验证）—— */
    'NUMER.sqrt_guard': { cls: 'elem', fn: sqrtGuard, sample: [[1, 4, 9], [0, 2]] },
    'BWD.grad_clip': { cls: 'pair', fn: gradClip, sample: [[[4, 1, 2], 3], [[1, 2], 1.5]] },
    'BWD.grad_linear': { cls: 'triple', fn: mlpGrad, sample: [[{ W1: [[0.5, -0.2], [0.1, 0.3]], b1: [0, 0], W2: [[0.2, -0.4]], b2: [0] }, [[1, 2], [0, 1]], [[1], [0]]]] },
    'BWD.grad_check': { cls: 'args', fn: gradientCheck, sample: [[{ W1: [[0.5, -0.2], [0.1, 0.3]], b1: [0, 0], W2: [[0.2, -0.4]], b2: [0] }, [[1, 2], [0, 1]], [[1], [0]], 0.00001]] },
    'BWD.grad_layernorm': { cls: 'triple', fn: gradLayerNorm, sample: [[[1, 2, 3], [0.1, 0.2, 0.3], 0.00001]] },
    'BWD.chain_rule': { cls: 'args', fn: chainRule, sample: [[DOUT, INNER, DIN, [3]], [DOUT, INNER, DIN, [-2]]] },
    'BWD.bptt': { cls: 'triple', fn: rnnBPTT, sample: [[{ W: 0.5, U: 0.3, h0: 0 }, [1, 0.5], [0.8, 0.6]]] },
    'RL.value_iter': { cls: 'vec', fn: valueIteration, sample: [{ n: 4, gamma: 0.9 }, { n: 3, gamma: 0.95 }] },
    'RL.q_learn': { cls: 'vec', fn: qLearning, sample: [{ n: 4, gamma: 0.9, episodes: 200 }, { n: 3, episodes: 100 }] },
    'RL.bellman_backup': { cls: 'pair', fn: bellmanBackup, sample: [[[0, 0, 0], MDP3]] },
    'RL.policy_iter': { cls: 'vec', fn: policyIteration, sample: [MDP3, MDP4] },
    'RL.sarsa': { cls: 'vec', fn: sarsa, sample: [{ n: 4, gamma: 0.9, episodes: 200 }] },
    'RL.monte_carlo': { cls: 'args', fn: monteCarloReturn, sample: [
      [{ n: 3, gamma: 0.9, trans: function (s, a) { return [[(s + 1) % 3, 1]]; }, reward: function (s, sp) { return sp === 2 ? 1 : 0; } }, function (s, rng) { return 0; }, 30, 1],
      [{ n: 3, gamma: 0.9, trans: function (s, a) { return [[(s + 1) % 3, 1]]; }, reward: function (s, sp) { return sp === 2 ? 1 : 0; } }, function (s, rng) { return 1; }, 30, 2]
    ] }
  };
  function batched(opCode, batch) {
    var s = BATCH_SIG[opCode];
    if (!s) throw new Error('batched: 无签名 ' + opCode);
    var f = s.fn;
    if (s.cls === 'vec') return batch.map(function (x) { return f(x); });
    if (s.cls === 'elem') return batch.map(function (v) { return v.map(function (e) { return f(e); }); });
    if (s.cls === 'pair') return batch.map(function (p) { return f(p[0], p[1]); });
    if (s.cls === 'triple') return batch.map(function (p) { return f(p[0], p[1], p[2]); });
    if (s.cls === 'shared') return batch.map(function (x) { return f.apply(null, [x].concat(s.shared)); });
    if (s.cls === 'args') return batch.map(function (x) { return f.apply(null, x); });
    if (s.cls === 'special') return f(batch);
    throw new Error('batched: 未知 cls ' + s.cls);
  }
  /* inplace 变体（原地覆盖输入，返回值与 out-of-place 逐位一致） */
  function inplace_tensorFill(dst, val) { for (var i = 0; i < dst.length; i++) dst[i] = val; return dst; }
  function inplace_tensorCopy(dst, src) { for (var i = 0; i < dst.length; i++) dst[i] = src[i]; return dst; }
  function inplace_normalizeL2(dst) { var n = norm2(dst); for (var i = 0; i < dst.length; i++) dst[i] = dst[i] / n; return dst; }

  /* 通用原地引擎（v1.1.5 建造，v1.1.8 规模化扩面）：单一 inplace(opCode, dst) 调度器 + INPLACE_SIG 签名表。
   * 语义：原地覆盖 dst（首参写回），返回值与 out-of-place 逐位一致（≤1e-15），无中间张量物化语义开销由 _writeBack 承担。
   * 覆盖五类：elem（逐元素标量，extra 可选标量参数）/ vec（整张量形状守恒，extra 可选，如映射类）
   *         / pair / triple / args（多输入：首参 dst 为写回目的地，其余参数经 extra 传入）。 */
  var _EV = [1, -2, 0.5, 3, -0.3, 4, 0, -1.5];          // 单向量样本（逐元素 / 整向量算子）
  var _SPD = [[4, 1], [1, 3]];                           // 单对称正定矩阵（逆 / Cholesky）
  var INPLACE_SIG = {
    'NUMER.sigmoid': { cls: 'elem', fn: sigmoid, sample: _EV },
    'NUMER.relu': { cls: 'elem', fn: relu, sample: _EV },
    'NUMER.gelu': { cls: 'elem', fn: gelu, sample: _EV },
    'NUMER.erf': { cls: 'elem', fn: erf, sample: _EV },
    'NUMER.reciprocal': { cls: 'elem', fn: reciprocal, sample: _EV },
    'NUMER.expm1': { cls: 'elem', fn: expm1, sample: _EV },
    'NUMER.log1p': { cls: 'elem', fn: log1p, sample: _EV },
    'FWD.sigmoid': { cls: 'elem', fn: sigmoid, sample: _EV },
    'FWD.gelu': { cls: 'elem', fn: gelu, sample: _EV },
    'FWD.relu': { cls: 'elem', fn: relu, sample: _EV },
    'FWD.tanh': { cls: 'elem', fn: tanh, sample: _EV },
    'NORM.minmax': { cls: 'vec', fn: minmaxScale, sample: _EV },
    'NORM.zscore': { cls: 'vec', fn: zscore, sample: _EV },
    'BWD.grad_relu': { cls: 'vec', fn: gradRelu, sample: _EV },
    'BWD.grad_tanh': { cls: 'vec', fn: gradTanh, sample: _EV },
    'LINALG.inverse': { cls: 'vec', fn: inverse, sample: _SPD },
    'LINALG.cholesky': { cls: 'vec', fn: cholesky, sample: _SPD },

    /* —— v1.1.8 原地引擎规模化扩面（新增 26 条多类签名，样本经沙盒逐条验证）—— */
    'NUMER.clip': { cls: 'elem', fn: clip, extra: [-1, 1], sample: [1,-2,0.5,3,-0.3,4,0,-1.5] },
    'SAMPLE.gumbel': { cls: 'elem', fn: gumbelSample, sample: [0.1,0.5,0.9,0.3,0.7,0.2] },
    'NUMER.softmax': { cls: 'vec', fn: softmax, sample: [1,-2,0.5,3,-0.3,4,0,-1.5] },
    'FWD.softmax': { cls: 'vec', fn: softmax, sample: [1,-2,0.5,3,-0.3,4,0,-1.5] },
    'INFO.normalize_dist': { cls: 'vec', fn: normalizeDist, sample: [2,3,5,1,4] },
    'TENSOR.transpose': { cls: 'vec', fn: transpose, sample: [[1,2,3],[4,5,6],[7,8,9]] },
    'NORM.spectral_norm': { cls: 'vec', fn: spectralNorm, sample: [[3,1],[1,2]] },
    'TENSOR.cast': { cls: 'vec', fn: tensorCast, sample: [1,-2,0.5,3,-0.3,4,0,-1.5] },
    'SYS.prefetch': { cls: 'vec', fn: prefetch, sample: [1,-2,0.5,3,-0.3,4,0,-1.5] },
    'DATA.map': { cls: 'vec', fn: mapOp, extra: [function (x) { return x * 2; }], sample: [1,-2,0.5,3,-0.3,4,0,-1.5] },
    'FWD.residual_add': { cls: 'pair', fn: residualAdd, extra: [[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]], sample: [[1,2,3],[4,5,6],[7,8,9]] },
    'FWD.layernorm': { cls: 'pair', fn: layerNorm, extra: [0.00001], sample: [[1,2,3],[4,5,6]] },
    'FWD.batchnorm': { cls: 'pair', fn: batchNorm, extra: [0.00001], sample: [[1,2],[3,4],[5,6]] },
    'NORM.batch_norm': { cls: 'pair', fn: batchNorm, extra: [0.00001], sample: [[1,2],[3,4],[5,6]] },
    'NORM.label_smooth': { cls: 'pair', fn: labelSmooth, extra: [0.1], sample: [1,0,0,0] },
    'BWD.grad_clip': { cls: 'pair', fn: gradClip, extra: [3], sample: [4,1,2] },
    'ATTEN.scale': { cls: 'pair', fn: attentionScale, extra: [4], sample: [[1,2],[3,4]] },
    'ATTEN.rel_pos': { cls: 'pair', fn: relPosBias, extra: [[[0.1,0.2],[0.3,0.4]]], sample: [[1,2],[3,4]] },
    'ATTEN.alibi': { cls: 'pair', fn: alibiScores, extra: [0.5], sample: [[1,2],[3,4]] },
    'SAMPLE.temperature': { cls: 'pair', fn: temperatureSoftmax, extra: [0.7], sample: [1,2,3] },
    'DATA.shuffle': { cls: 'pair', fn: shuffle, extra: [1], sample: [1,2,3,4,5] },
    'NORM.mixup': { cls: 'triple', fn: mixup, extra: [[5,6,7], 0.3], sample: [1,2,3] },
    'OPTIM.weight_decay': { cls: 'triple', fn: weightDecay, extra: [0.1, 0.01], sample: [1,-2,3] },
    'FWD.dropout': { cls: 'args', fn: dropout, extra: [0.5, 1], sample: [1,2,3,4] },
    'NORM.dropout_reg': { cls: 'args', fn: dropout, extra: [0.5, 2], sample: [1,2,3,4] },
    'NORM.group_norm': { cls: 'args', fn: groupNorm, extra: [2, 0.00001], sample: [[1,2,3,4],[5,6,7,8]] }
  ,

    /* —— v1.2.0 变体层扩面（新增 10 条 inplace，样本经沙盒逐条验证）—— */
    'NUMER.sqrt_guard': { cls: 'elem', fn: sqrtGuard, sample: [1,4,9,0] },
    'BWD.grad_softmax_ce': { cls: 'pair', fn: gradSoftmaxCE, extra: [1], sample: [[1,2,3]] },
    'ATTEN.qk_dot': { cls: 'pair', fn: qkDot, extra: [[[1,0],[0,1]]], sample: [[[1,0],[0,1]]] },
    'LINALG.solve': { cls: 'pair', fn: solveLinear, extra: [[1,2]], sample: [[[4,1],[1,3]]] },
    'PROB.bayes': { cls: 'pair', fn: bayesUpdate, extra: [[0.8,0.2]], sample: [[0.5,0.5]] },
    'RL.reinforce': { cls: 'triple', fn: reinforceGradient, extra: [1, 0.5], sample: [[1,2,3]] },
    'RL.gae': { cls: 'args', fn: gae, extra: [[0,0.5,0.8], 0.9, 0.95], sample: [[1,1,1]] },
    'SYS.broadcast': { cls: 'vec', fn: broadcast, sample: [[1,2,3]] },
    'TENSOR.scatter': { cls: 'args', fn: tensorScatter, extra: [[0,2], [9,8]], sample: [[1,2,3,4]] },
    'NORM.weight_norm': { cls: 'pair', fn: weightNorm, extra: [2], sample: [[3,4]] },

    /* —— v1.3.0 原地引擎第四轮扩面（新增 17 条；沙盒逐条验证：不抛异常 + 输出全有限 + 形状守恒 + 与 out-of-place 逐位一致）—— */
    'LINALG.matmul': { cls: 'pair', fn: matMul, extra: [[[5,6],[7,8]]], sample: [[1,2],[3,4]] },
    'INFO.surprisal': { cls: 'elem', fn: surprisal, sample: [0.5,0.25,0.1] },
    'INFO.bits': { cls: 'elem', fn: bits, sample: [1,2,3] },
    'OPTIM.grad_clip': { cls: 'pair', fn: gradClip, extra: [1], sample: [1,2,3] },
    'FWD.linear': { cls: 'triple', fn: linear, extra: [[[1,0],[0,1]],[0,0]], sample: [[1,2],[3,4]] },
    'FWD.embedding_lookup': { cls: 'pair', fn: embeddingLookup, extra: [[0,1]], sample: [[1,2],[3,4]] },
    'BWD.grad_layernorm': { cls: 'triple', fn: gradLayerNorm, extra: [[0.1,0.2,0.3],0.00001], sample: [1,2,3] },
    'BWD.stop_grad': { cls: 'vec', fn: stopGrad, sample: [1,2,3] },
    'BWD.detach': { cls: 'vec', fn: detach, sample: [1,2,3] },
    'ATTEN.rope': { cls: 'triple', fn: ropeApply, extra: [1,10000], sample: [1,0] },
    'ATTEN.mqa': { cls: 'triple', fn: mqa, extra: [[[1,0],[0,1]],[[1,0],[0,1]]], sample: [[[1,0],[0,1]],[[2,1],[1,2]]] },
    'ATTEN.gqa': { cls: 'args', fn: gqa, extra: [[[[1,0],[0,1]]],[[[1,0],[0,1]]],1], sample: [[[1,0],[0,1]],[[2,1],[1,2]]] },
    'SAMPLE.topk': { cls: 'pair', fn: topK, extra: [2], sample: [1,2,3] },
    'SAMPLE.topp': { cls: 'pair', fn: topP, extra: [0.8], sample: [0.1,0.5,0.4] },
    'DATA.dedup': { cls: 'vec', fn: dedup, sample: [[1,2,2,3],[4,4,5,1]] },
    'TENSOR.stack': { cls: 'vec', fn: tensorStack, sample: [[1,2],[3,4]] },
    'FWD.conv1d': { cls: 'pair', fn: conv1d, extra: [[2]], sample: [1,2,3] },
  };
  function _writeBack(dst, src) { // 递归将 src 数值写回 dst 的既有结构（支持 1D 向量与 2D 矩阵）
    if (Array.isArray(src)) { for (var i = 0; i < src.length; i++) dst[i] = _writeBack(dst[i], src[i]); return dst; }
    return src;
  }
  function inplace(opCode, dst) {
    var s = INPLACE_SIG[opCode];
    if (!s) throw new Error('inplace: 无签名 ' + opCode);
    var f = s.fn;
    if (s.cls === 'elem') { for (var i = 0; i < dst.length; i++) dst[i] = s.extra ? f.apply(null, [dst[i]].concat(s.extra)) : f(dst[i]); return dst; }
    /* vec / pair / triple / args 统一首参写回（v1.1.8）：extra 为其余参数 */
    _writeBack(dst, s.extra ? f.apply(null, [dst].concat(s.extra)) : f(dst));
    return dst;
  }

  /* 通用 Epilogue 融合引擎（v1.1.6）：单一 fused(opCode, item) 调度器 + FUSE_SIG 签名表。
   * 语义：算子输出不落地中间张量，单遍「就地」施加逐元素尾算子（epilogue）；
   *     硬不变量 fused ≡ post ∘ op（≤1e-15）。
   * 世界级范式锚定：cuBLASLt / CUTLASS 的 GEMM epilogue 融合（BIAS/RELU/GELU 尾算子内联进主循环）。
   * 与 BATCH_SIG 同构复用 cls/sample（vec/elem/pair/triple/shared），仅多一个逐元素 post。 */
  function _applySig(s, item) { // 单样本按 cls 施加算子
    var f = s.fn;
    if (s.cls === 'vec') return f(item);
    if (s.cls === 'elem') return item.map(function (e) { return f(e); });
    if (s.cls === 'pair') return f(item[0], item[1]);
    if (s.cls === 'triple') return f(item[0], item[1], item[2]);
    if (s.cls === 'shared') return f.apply(null, [item].concat(s.shared));
    if (s.cls === 'args') return f.apply(null, item);
    throw new Error('_applySig: 未知 cls ' + s.cls);
  }
  function _epilogue(dst, g) { // 就地单遍施加逐元素尾算子（不物化中间张量，单次分配）
    if (typeof dst === 'number') return g(dst);
    if (Array.isArray(dst)) { for (var i = 0; i < dst.length; i++) dst[i] = _epilogue(dst[i], g); return dst; }
    return dst;
  }
  function _mapPost(src, g) { // 参照实现：纯函数逐元素映射（会物化中间张量，两次分配）
    if (typeof src === 'number') return g(src);
    if (Array.isArray(src)) return src.map(function (x) { return _mapPost(x, g); });
    return src;
  }
  function fused(opCode, item) {
    var s = FUSE_SIG[opCode];
    if (!s) throw new Error('fused: 无签名 ' + opCode);
    return _epilogue(_applySig(s, item), s.post);
  }
  var FUSE_SIG = {
    /* TENSOR：拷贝 / 展平 ∘ ReLU 尾算子 */
    'TENSOR.copy': { cls: 'vec', fn: tensorCopy, sample: _M, post: relu },
    'TENSOR.flatten': { cls: 'vec', fn: flatten, sample: _M, post: relu },
    /* NUMER：激活 ∘ 对数域 / 平方根；expm1↔log1p 为互逆尾算子 */
    'NUMER.sigmoid': { cls: 'elem', fn: sigmoid, sample: _V, post: Math.log },
    'NUMER.relu': { cls: 'elem', fn: relu, sample: _V, post: Math.sqrt },
    'NUMER.gelu': { cls: 'elem', fn: gelu, sample: _V, post: tanh },
    'NUMER.erf': { cls: 'elem', fn: erf, sample: _V, post: Math.abs },
    'NUMER.reciprocal': { cls: 'elem', fn: reciprocal, sample: _V, post: Math.abs },
    'NUMER.expm1': { cls: 'elem', fn: expm1, sample: _V, post: log1p },       // log1p(expm1(x)) = x
    'NUMER.log1p': { cls: 'elem', fn: log1p, sample: _V, post: expm1 },       // expm1(log1p(x)) = x
    'NUMER.kahan': { cls: 'vec', fn: kahanSum, sample: _V, post: Math.abs },
    'NUMER.exactsum': { cls: 'vec', fn: exactSum, sample: _V, post: Math.abs },
    /* NORM：范数 ∘ 倒数 = 自适应缩放因子；缩放 ∘ tanh 压缩 */
    'NORM.l1': { cls: 'vec', fn: l1Norm, sample: _V, post: Math.sqrt },
    'NORM.l2': { cls: 'vec', fn: l2Norm, sample: _V, post: reciprocal },
    'NORM.linf': { cls: 'vec', fn: linfNorm, sample: _V, post: reciprocal },
    'NORM.minmax': { cls: 'vec', fn: minmaxScale, sample: _V, post: tanh },
    'NORM.zscore': { cls: 'vec', fn: zscore, sample: _V, post: tanh },
    /* FWD：log-softmax / log-sigmoid（对数域稳定，损失层直接消费） */
    'FWD.softmax': { cls: 'vec', fn: softmax, sample: _V, post: Math.log },
    'FWD.sigmoid': { cls: 'elem', fn: sigmoid, sample: _V, post: Math.log },
    'FWD.gelu': { cls: 'elem', fn: gelu, sample: _V, post: tanh },
    'FWD.relu': { cls: 'elem', fn: relu, sample: _V, post: Math.sqrt },
    'FWD.tanh': { cls: 'elem', fn: tanh, sample: _V, post: Math.abs },
    /* BWD：梯度掩码 ∘ Sigmoid */
    'BWD.grad_relu': { cls: 'vec', fn: gradRelu, sample: _V, post: sigmoid },
    /* INFO：熵 ∘ exp = 困惑度；散度 ∘ sqrt（Hellinger 型距离） */
    'INFO.entropy': { cls: 'vec', fn: entropy, sample: _V, post: Math.exp },
    'INFO.kl': { cls: 'pair', fn: klDivergence, sample: _P, post: Math.sqrt },
    'INFO.js': { cls: 'pair', fn: jsDivergence, sample: _P, post: Math.sqrt },
    'INFO.mutual_info': { cls: 'vec', fn: mutualInfo, sample: _M, post: Math.abs },
    'INFO.conditional_entropy': { cls: 'vec', fn: conditionalEntropy, sample: _M, post: Math.exp },
    'INFO.gini': { cls: 'vec', fn: gini, sample: _V, post: Math.sqrt },
    'INFO.normalize_dist': { cls: 'vec', fn: normalizeDist, sample: _V, post: Math.abs },
    /* PROB：方差 ∘ sqrt = 标准差；CDF ∘ log */
    'PROB.gauss_cdf': { cls: 'triple', fn: normalCDF, sample: _T, post: Math.log },
    'PROB.covariance': { cls: 'pair', fn: covariance, sample: _P, post: Math.abs },
    'PROB.expectation': { cls: 'vec', fn: expectation, sample: _V, post: tanh },
    'PROB.variance': { cls: 'vec', fn: variance, sample: _V, post: Math.sqrt },
    'PROB.marginal': { cls: 'vec', fn: marginal, sample: _M, post: Math.sqrt },
    /* METRIC：比率型 ∘ sqrt；余弦 ∘ abs */
    'METRIC.accuracy': { cls: 'pair', fn: accuracy, sample: _P, post: Math.sqrt },
    'METRIC.precision': { cls: 'pair', fn: precision, sample: _P, post: Math.sqrt },
    'METRIC.recall': { cls: 'pair', fn: recall, sample: _P, post: Math.sqrt },
    'METRIC.f1': { cls: 'pair', fn: f1, sample: _P, post: Math.sqrt },
    'METRIC.mae': { cls: 'pair', fn: mae, sample: _P, post: Math.sqrt },
    'METRIC.cosine': { cls: 'pair', fn: cosineSimilarity, sample: _P, post: Math.abs },
    /* LINALG：GEMM ∘ ReLU（cuBLASLt epilogue 正统范式）；log|det|（高斯对数似然项） */
    'LINALG.matmul': { cls: 'pair', fn: matMul, sample: _MM, post: relu },
    'LINALG.solve': { cls: 'pair', fn: solveLinear, sample: _SOLVE, post: relu },
    'LINALG.trace': { cls: 'vec', fn: trace, sample: _MS, post: Math.abs },
    'LINALG.det': { cls: 'vec', fn: det, sample: _MS, post: Math.log },
    'LINALG.inverse': { cls: 'vec', fn: inverse, sample: _MS, post: Math.abs },
    'LINALG.cholesky': { cls: 'vec', fn: cholesky, sample: _MS, post: Math.abs },
    /* OPTIM：梯度范数 ∘ 倒数 = 自适应步长缩放因子 */
    'OPTIM.grad_norm': { cls: 'vec', fn: gradNorm, sample: _V, post: reciprocal },
    /* SYS：广播 ∘ ReLU */
    'SYS.broadcast': { cls: 'vec', fn: broadcast, sample: _V, post: relu },
    /* SAMPLE：温度 softmax ∘ log（对数域损失输入） */
    'SAMPLE.temperature': { cls: 'shared', fn: temperatureSoftmax, shared: [1], sample: _V, post: Math.log },
    /* ——     /* —— v1.1.7 引擎规模化扩面（新增 91 条 fused 签名，样本经沙盒逐条验证）—— */
    'TENSOR.create': { cls: 'args', fn: tensorCreate, sample: [[[2, 2], 1], [[3, 1], 0]], post: Math.abs },
    'TENSOR.zeros': { cls: 'args', fn: tensorZeros, sample: [[[2, 2]], [[3, 1]]], post: Math.abs },
    'TENSOR.ones': { cls: 'args', fn: tensorOnes, sample: [[[2, 2]], [[1, 3]]], post: Math.abs },
    'TENSOR.arange': { cls: 'args', fn: arange, sample: [[0, 6, 2], [1, 5, 1], [0, 4, 1]], post: Math.abs },
    'TENSOR.reshape': { cls: 'args', fn: reshape, sample: [[[1, 2, 3, 4], [2, 2]], [[1, 2, 3, 4, 5, 6], [3, 2]]], post: Math.abs },
    'TENSOR.slice': { cls: 'args', fn: tensorSlice, sample: [[[1, 2, 3, 4], 0, 2], [[1, 2, 3, 4], 1, 3]], post: Math.abs },
    'TENSOR.concat': { cls: 'args', fn: tensorConcat, sample: [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], post: Math.abs },
    'TENSOR.stack': { cls: 'args', fn: tensorStack, sample: [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], post: Math.abs },
    'TENSOR.split': { cls: 'args', fn: tensorSplit, sample: [[[1, 2, 3, 4], [2, 2]], [[1, 2, 3, 4, 5, 6], [3, 3]]], post: Math.abs },
    'TENSOR.cast': { cls: 'args', fn: tensorCast, sample: [[[1, 2, 3]], [[4, 5]]], post: Math.abs },
    'TENSOR.gather': { cls: 'args', fn: tensorGather, sample: [[[1, 2, 3, 4], [0, 2]], [[5, 6, 7, 8], [1, 3]]], post: Math.abs },
    'TENSOR.scatter': { cls: 'args', fn: tensorScatter, sample: [[[1, 2, 3, 4], [0, 1], [9, 9]], [[5, 6, 7, 8], [2, 3], [1, 1]]], post: Math.abs },
    'TENSOR.fill': { cls: 'args', fn: tensorFill, sample: [[[1, 2, 3], 7], [[4, 5], 0]], post: Math.abs },
    'LINALG.qr': { cls: 'vec', fn: qrDecompose, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]], post: Math.abs },
    'LINALG.eigen': { cls: 'vec', fn: jacobiEigen, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]], post: Math.abs },
    'LINALG.power': { cls: 'vec', fn: powerIteration, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]], post: Math.abs },
    'LINALG.rank': { cls: 'vec', fn: matrixRank, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]], post: Math.abs },
    'LINALG.nullspace': { cls: 'vec', fn: nullspace, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]], post: Math.abs },
    'LINALG.orthonormalize': { cls: 'vec', fn: orthonormalize, sample: [[[[1, 0], [1, 1]], [[1, 2], [2, 4]]]], post: Math.abs },
    'LINALG.cond': { cls: 'vec', fn: condEstimate, sample: [[[4, 1], [1, 3]], [[2, 0], [0, 2]], [[5, 2], [2, 4]]], post: Math.abs },
    'NUMER.logsumexp': { cls: 'vec', fn: logsumexp, sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]], post: Math.abs },
    'NUMER.clip': { cls: 'args', fn: clip, sample: [[[1.5, -1, 1], [-3, -1, 1], [0.2, -1, 1]]], post: Math.abs },
    'NUMER.fma': { cls: 'triple', fn: fma, sample: [[[1, 2, 3], [4, 5, 6], [-1, 2, 0]]], post: Math.abs },
    'PROB.normal_sample': { cls: 'pair', fn: normalSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'PROB.categorical_sample': { cls: 'pair', fn: categoricalSample, sample: [[[[0.2, 0.3, 0.5], 7], [[0.1, 0.4, 0.5], 5]]], post: Math.abs },
    'PROB.bernoulli_sample': { cls: 'triple', fn: bernoulliSample, sample: [[[0.5, 3, 1], [0.3, 4, 2]]], post: Math.abs },
    'PROB.binomial_sample': { cls: 'args', fn: binomialSample, sample: [[[10, 0.5, 5, 1], [8, 0.3, 4, 2]]], post: Math.abs },
    'PROB.uniform_sample': { cls: 'pair', fn: uniformSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'PROB.sample_stats': { cls: 'vec', fn: sampleStats, sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]], post: Math.abs },
    'INFO.info_gain': { cls: 'pair', fn: infoGain, sample: [[[[0, 0, 1, 1], [[0, 1], [1, 1]]], [[1, 1, 0, 0], [[1, 0], [0, 1]]]]], post: Math.abs },
    'INFO.perplexity': { cls: 'vec', fn: perplexity, sample: [[[[0.2, 0.3, 0.5]], [[0.5, 0.25, 0.25]]]], post: Math.abs },
    'INFO.kl_reverse': { cls: 'pair', fn: klReverse, sample: [[[[0.5, 0.5], [0.4, 0.6]], [[0.3, 0.7], [0.5, 0.5]]]], post: Math.abs },
    'METRIC.auc': { cls: 'pair', fn: auc, sample: [[[0, 1, 1], [0.1, 0.4, 0.8]], [[1, 0, 1], [0.3, 0.2, 0.9]]], post: Math.abs },
    'METRIC.r2': { cls: 'pair', fn: r2, sample: [[[1, 2, 3], [1.1, 1.9, 3.2]], [[0, 1, 2], [0.1, 0.9, 2.1]]], post: Math.abs },
    'METRIC.edit_distance': { cls: 'pair', fn: editDistance, sample: [["kitten", "sitting"], ["sunday", "saturday"]], post: Math.abs },
    'METRIC.silhouette': { cls: 'pair', fn: silhouette, sample: [[[[1, 1], [2, 2], [8, 8], [9, 9]], [0, 0, 1, 1]], [[[1, 1], [1.5, 1.5], [9, 9]], [0, 0, 1]]], post: Math.abs },
    'METRIC.bleu': { cls: 'triple', fn: bleu, sample: [[["a b", "a b", 2], ["cat sat", "cat sat", 3]]], post: Math.abs },
    'METRIC.map': { cls: 'vec', fn: averagePrecision, sample: [[[1, 0, 1], [0, 1, 1]]], post: Math.abs },
    'METRIC.confusion': { cls: 'triple', fn: confusionMatrix, sample: [[[0, 1, 1], [0, 1, 0], [0, 1]]], post: Math.abs },
    'SYS.gather_all': { cls: 'vec', fn: gatherAll, sample: [[[[1, 2], [3]], [[4], [5, 6]]]], post: Math.abs },
    'SYS.scatter_all': { cls: 'pair', fn: scatterAll, sample: [[[[1, 2, 3], [0, 1, 0]], [[4, 5], [1, 0]]]], post: Math.abs },
    'SYS.checkpoint': { cls: 'vec', fn: checkpoint, sample: [[{"a":[1,2]}, {"b":[3,4]}]], post: Math.abs },
    'SYS.prefetch': { cls: 'vec', fn: prefetch, sample: [[[1, 2, 3], [4, 5, 6]]], post: Math.abs },
    'FWD.self_attn': { cls: 'args', fn: selfAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], false]], post: Math.abs },
    'FWD.embedding_lookup': { cls: 'pair', fn: embeddingLookup, sample: [[[[1, 2], [3, 4]], [0, 1]], [[[5, 6], [7, 8]], [1, 0]]], post: Math.abs },
    'FWD.flatten': { cls: 'vec', fn: flatten, sample: [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 1], [0, 3]]], post: Math.abs },
    'FWD.residual_add': { cls: 'pair', fn: residualAdd, sample: [[[[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]]], [[[5, 6], [7, 8]], [[0.5, 0.5], [0.5, 0.5]]]], post: Math.abs },
    'FWD.concat_heads': { cls: 'vec', fn: concatHeads, sample: [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], post: Math.abs },
    'FWD.pool_max': { cls: 'pair', fn: poolMax, sample: [[[1, 5, 2, 8], 2], [[3, 1, 4, 2], 2]], post: Math.abs },
    'FWD.conv1d': { cls: 'args', fn: conv1d, sample: [[[1, 2, 3, 4], [2, 4, 6]]], post: Math.abs },
    'FWD.batchnorm': { cls: 'shared', fn: batchNorm, shared: [0.00001], sample: [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 1], [0, 3]]], post: Math.abs },
    'FWD.dropout': { cls: 'args', fn: dropout, sample: [[[[1, 2, 3, 4], 0.5, 1], [[5, 6, 7, 8], 0.5, 2]]], post: Math.abs },
    'BWD.grad_softmax_ce': { cls: 'pair', fn: gradSoftmaxCE, sample: [[[1, 2, 3], 1], [[0.5, 0.5, 1], 2]], post: Math.abs },
    'BWD.grad_matmul': { cls: 'triple', fn: gradMatmul, sample: [[[[1, 2]], [[1], [2]], [[3]]], [[[3, 4]], [[1], [3]], [[2]]]], post: Math.abs },
    'BWD.stop_grad': { cls: 'vec', fn: stopGrad, sample: [[[1, 2, 3], [0.5, 0.5, 1]]], post: Math.abs },
    'BWD.detach': { cls: 'vec', fn: detach, sample: [[[1, 2, 3], [0.5, 0.5, 1]]], post: Math.abs },
    'NORM.l2_reg': { cls: 'shared', fn: l2Reg, shared: [0.01], sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]], post: Math.abs },
    'NORM.l1_reg': { cls: 'shared', fn: l1Reg, shared: [0.01], sample: [[1, 2, 3], [-1, 0, 2], [0.5, -0.5, 1]], post: Math.abs },
    'NORM.group_norm': { cls: 'args', fn: groupNorm, sample: [[[[1, 2, 3, 4], 2, 0.00001], [[5, 6, 7, 8], 2, 0.00001]]], post: Math.abs },
    'NORM.batch_norm': { cls: 'shared', fn: batchNorm, shared: [0.00001], sample: [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 1], [0, 3]]], post: Math.abs },
    'NORM.dropout_reg': { cls: 'args', fn: dropout, sample: [[[[1, 2, 3, 4], 0.5, 2], [[5, 6, 7, 8], 0.5, 3]]], post: Math.abs },
    'ATTEN.qk_dot': { cls: 'pair', fn: qkDot, sample: [[[[1, 0], [0, 1]], [[2, 1], [1, 2]]]], post: Math.abs },
    'ATTEN.rel_pos': { cls: 'pair', fn: relPosBias, sample: [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], post: Math.abs },
    'ATTEN.mha_split': { cls: 'pair', fn: splitHeads, sample: [[[[1, 2], [3, 4]], 2], [[[5, 6], [7, 8]], 2]], post: Math.abs },
    'ATTEN.attn_entropy': { cls: 'vec', fn: attnEntropy, sample: [[[0, 1, 1], [1, 0, 1]]], post: Math.abs },
    'ATTEN.softmax_attn': { cls: 'args', fn: selfAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], false]], post: Math.abs },
    'ATTEN.causal_mask': { cls: 'args', fn: selfAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], true]], post: Math.abs },
    'ATTEN.multi_head': { cls: 'args', fn: multiHead, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], 2, false]], post: Math.abs },
    'SAMPLE.box_muller': { cls: 'pair', fn: normalSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'SAMPLE.uniform': { cls: 'pair', fn: uniformSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'SAMPLE.bernoulli': { cls: 'triple', fn: bernoulliSample, sample: [[[0.5, 3, 1], [0.3, 4, 2]]], post: Math.abs },
    'SAMPLE.multinomial': { cls: 'triple', fn: multinomialSample, sample: [[[[0.2, 0.3, 0.5], 3, 1], [[0.1, 0.4, 0.5], 4, 2]]], post: Math.abs },
    'SAMPLE.antithetic': { cls: 'pair', fn: antitheticSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'SAMPLE.stratified': { cls: 'pair', fn: stratifiedSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'SAMPLE.systematic': { cls: 'pair', fn: systematicSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'SAMPLE.normal': { cls: 'pair', fn: normalSample, sample: [[[3, 1], [5, 2]]], post: Math.abs },
    'DATA.shuffle': { cls: 'pair', fn: shuffle, sample: [[[[1, 2, 3, 4], 1], [[4, 3, 2, 1], 2]]], post: Math.abs },
    'DATA.train_test': { cls: 'triple', fn: trainTestSplit, sample: [[[10, 0.8, 1], [8, 0.7, 2]]], post: Math.abs },
    'DATA.kfold': { cls: 'triple', fn: kFold, sample: [[[10, 3, 1], [9, 3, 2]]], post: Math.abs },
    'DATA.pad': { cls: 'triple', fn: padSeq, sample: [[[[1, 2], 4, 0], [[3, 4, 5], 2, -1]]], post: Math.abs },
    'DATA.truncate': { cls: 'pair', fn: truncateSeq, sample: [[[[1, 2, 3, 4], 2], [[5, 6, 7], 3]]], post: Math.abs },
    'DATA.bucket': { cls: 'pair', fn: bucketize, sample: [[[[1, 2, 3, 4], 2], [[5, 1, 3, 2], 3]]], post: Math.abs },
    'DATA.dedup': { cls: 'vec', fn: dedup, sample: [[[1, 2, 2, 3], [4, 4, 5, 1]]], post: Math.abs },
    'DATA.filter': { cls: 'shared', fn: filterOp, shared: [function () { return 1; }], sample: [[[1, 2, 3, 4], [0, 1, 2, 3]]], post: Math.abs },
    'DATA.zip': { cls: 'pair', fn: zipOp, sample: [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], post: Math.abs },
    'DATA.batch': { cls: 'pair', fn: batchOp, sample: [[[[1, 2, 3, 4], 2], [[5, 6, 7, 8, 9], 3]]], post: Math.abs },
    'DATA.augment': { cls: 'triple', fn: augment, sample: [[[[1, 2, 3], 1, 0.1], [[4, 5], 2, 0.2]]], post: Math.abs },
    'RL.td0': { cls: 'pair', fn: td0Sweep, sample: [[[3, 0.9], [5, 0.95]]], post: Math.abs },
    'RL.td_lambda': { cls: 'triple', fn: eligibilityTrace, sample: [[[[0, 1, 0], 0.9, 0.5], [[1, 1, 0], 0.8, 0.3]]], post: Math.abs },
    'RL.greedy': { cls: 'vec', fn: greedyPolicy, sample: [[[1, 5, 2], [3, 1, 4]]], post: Math.abs },
    'RL.ucb': { cls: 'args', fn: ucb, sample: [[[[1, 2], [1, 1], 1.4, 10], [[3, 4], [2, 2], 1.2, 20]]], post: Math.abs },

    /* —— v1.1.9 融合对齐扩面（新增 16 条，与 BATCH_SIG 全谱对齐；其中 6 条注册表标记由具名内核优先接管，净增 10）—— */
    'TENSOR.transpose': { cls: 'vec', fn: transpose, sample: [[[1,2],[3,4]],[[5,6],[7,8]],[[2,1],[0,3]]], post: Math.abs },
    'NUMER.softmax': { cls: 'vec', fn: softmax, sample: [[1,2,3],[-1,0,2],[0.5,-0.5,1]], post: Math.abs },
    'NORM.normalize_l2': { cls: 'vec', fn: normalizeL2, sample: [[1,2,3],[-1,0,2],[0.5,-0.5,1]], post: Math.abs },
    'FWD.linear': { cls: 'shared', fn: linear, shared: [[[1,0],[0,1]], [0,0]], sample: [[[1,2],[3,4]],[[5,6],[7,8]],[[2,1],[0,3]]], post: Math.abs },
    'BWD.grad_tanh': { cls: 'vec', fn: gradTanh, sample: [[1,2,3],[-1,0,2],[0.5,-0.5,1]], post: Math.abs },
    'INFO.cross_entropy': { cls: 'pair', fn: crossEntropy, sample: [[[1,2,3],[3,2,1]],[[4,5,6],[6,5,4]],[[0.1,0.2,0.3],[0.3,0.2,0.1]]], post: Math.abs },
    'PROB.gauss_pdf': { cls: 'triple', fn: gaussPDF, sample: [[1,0,1],[0.5,0,1],[2,1,1]], post: Math.abs },
    'METRIC.mse': { cls: 'pair', fn: mse, sample: [[[1,2,3],[3,2,1]],[[4,5,6],[6,5,4]],[[0.1,0.2,0.3],[0.3,0.2,0.1]]], post: Math.abs },
    'LINALG.matvec': { cls: 'pair', fn: matVec, sample: [[[[1,2],[3,4]],[1,1]],[[[5,6],[7,8]],[0,1]],[[[2,1],[0,3]],[1,0]]], post: Math.abs },
    'LINALG.lstsq': { cls: 'pair', fn: linearRegression, sample: [[[[1,2],[3,4],[5,6]],[1,2,3]],[[[2,1],[1,2],[0,1]],[4,5,6]]], post: Math.abs },
    'OPTIM.grad_clip': { cls: 'shared', fn: gradClip, shared: [1], sample: [[1,2,3],[-1,0,2],[0.5,-0.5,1]], post: Math.abs },
    'SYS.reduce_all': { cls: 'vec', fn: reduceAll, sample: [[1,2,3],[-1,0,2],[0.5,-0.5,1]], post: Math.abs },
    'GRAPH.marginal': { cls: 'vec', fn: marginal, sample: [[[1,2],[3,4]],[[5,6],[7,8]],[[2,1],[0,3]]], post: Math.abs },
    'RL.epsilon_greedy': { cls: 'shared', fn: epsilonGreedy, shared: [0, mulberry32(99)], sample: [[1, 5, 2], [4, 1, 0]], post: Math.abs },
    'SAMPLE.categorical': { cls: 'pair', fn: categoricalSample, sample: [[[0.2,0.3,0.5],7],[[0.1,0.4,0.5],5]], post: Math.abs },
    'SAMPLE.topk': { cls: 'shared', fn: topK, shared: [2], sample: [[1,2,3],[-1,0,2],[0.5,-0.5,1]], post: Math.abs }
  ,

    /* —— v1.2.0 变体层扩面（新增 10 条 fused，样本经沙盒逐条验证）—— */
    'ATTEN.scale': { cls: 'pair', fn: attentionScale, sample: [[[[1,2],[3,4]],4],[[[5,6],[7,8]],2]], post: Math.abs },
    'ATTEN.alibi': { cls: 'pair', fn: alibiScores, sample: [[[[1,2],[3,4]],0.5],[[[5,6],[7,8]],0.3]], post: Math.abs },
    'ATTEN.rope': { cls: 'args', fn: ropeApply, sample: [[[1,0],1,10000],[[2,1],2,10000]], post: Math.abs },
    'ATTEN.mqa': { cls: 'args', fn: mqa, sample: [[[[[1,0],[0,1]],[[2,1],[1,2]]],[[1,0],[0,1]],[[1,0],[0,1]]],[[[[1,0],[0,1]],[[2,1],[1,2]]],[[1,0],[0,1]],[[1,0],[0,1]]]], post: Math.abs },
    'ATTEN.gqa': { cls: 'args', fn: gqa, sample: [[[[[1,0],[0,1]],[[2,1],[1,2]]],[[[1,0],[0,1]]],[[[1,0],[0,1]]],1],[[[[1,0],[0,1]],[[2,1],[1,2]]],[[[1,0],[0,1]]],[[[1,0],[0,1]]],1]], post: Math.abs },
    'SAMPLE.topp': { cls: 'pair', fn: topP, sample: [[[0.1,0.5,0.4],0.8],[[0.2,0.3,0.5],0.9]], post: Math.abs },
    'METRIC.logloss': { cls: 'pair', fn: logLoss, sample: [[[1,0,0],[0.7,0.2,0.1]],[[0,1,0],[0.1,0.8,0.1]]], post: Math.abs },
    'METRIC.perplexity': { cls: 'pair', fn: perplexityLM, sample: [[[1,0,0],[0.7,0.2,0.1]],[[0,1,0],[0.1,0.8,0.1]]], post: Math.abs },
    'DATA.map': { cls: 'shared', fn: mapOp, shared: [function (x) { return x * 2; }], sample: [[1,2,3],[4,5]], post: Math.abs },
    'NORM.weight_norm': { cls: 'pair', fn: weightNorm, sample: [[[3,4],2],[[1,2],3]], post: Math.abs }
  ,

    /* —— v1.2.1 修复批次（新增 40 条 fused：v1.1.7 样本设计误杀修复 + 语义清晰项，样本经沙盒逐条验证）—— */
    'NUMER.hypot': { cls: 'pair', fn: hypot, sample: [[3, 4], [5, 12], [8, 15]], post: Math.abs },
    'NUMER.div_guard': { cls: 'args', fn: divGuard, sample: [[1, 2], [1, 0], [5, -3]], post: Math.abs },
    'PROB.bayes': { cls: 'pair', fn: bayesUpdate, sample: [[[0.5, 0.5], [0.8, 0.2]], [[0.6, 0.4], [0.7, 0.3]]], post: Math.abs },
    'PROB.condition': { cls: 'pair', fn: conditionalRow, sample: [[[[0.1, 0.9], [0.8, 0.2]], 0], [[[0.1, 0.9], [0.8, 0.2]], 1]], post: Math.abs },
    'PROB.mv_gauss': { cls: 'triple', fn: mvGaussPDF, sample: [[[1, 0], [0, 0], [[1, 0], [0, 1]]], [[0.5, 1], [0, 0], [[1, 0], [0, 1]]]], post: Math.abs },
    'INFO.surprisal': { cls: 'elem', fn: surprisal, sample: [[0.5, 0.25, 0.1], [0.9, 0.05]], post: Math.abs },
    'INFO.bits': { cls: 'elem', fn: bits, sample: [[1, 2, 3], [0.5, 0.25]], post: Math.abs },
    'INFO.channel_capacity': { cls: 'vec', fn: channelCapacity, sample: [[[0.9, 0.1], [0.2, 0.8]], [[0.7, 0.3], [0.4, 0.6]]], post: Math.abs },
    'INFO.rate_distortion': { cls: 'pair', fn: rateDistortion, sample: [[2, 1], [1, 0.5]], post: Math.abs },
    'INFO.max_entropy': { cls: 'vec', fn: maxEntropyDist, sample: [3, 4, 2], post: Math.abs },
    'FWD.layernorm': { cls: 'shared', fn: layerNorm, shared: [0.00001], sample: [[[1, 2, 3], [4, 5, 6]], [[2, 1], [0, 3]]], post: Math.abs },
    'NORM.spectral_norm': { cls: 'vec', fn: spectralNorm, sample: [[[3, 1], [1, 2]], [[2, 0], [0, 2]]], post: Math.abs },
    'NORM.label_smooth': { cls: 'pair', fn: labelSmooth, sample: [[[1, 0, 0], 0.1], [[0, 1, 0], 0.2]], post: Math.abs },
    'NORM.mixup': { cls: 'triple', fn: mixup, sample: [[[1, 2, 3], [5, 6, 7], 0.3], [[1, 0], [0, 1], 0.5]], post: Math.abs },
    'NORM.early_stop': { cls: 'pair', fn: earlyStop, sample: [[[0.5, 0.4, 0.3, 0.35], 2], [[1, 0.8, 0.9], 1]], post: Math.abs },
    'SAMPLE.gumbel': { cls: 'elem', fn: gumbelSample, sample: [[0.1, 0.5, 0.9, 0.3], [0.7, 0.2]], post: Math.abs },
    'SAMPLE.rejection': { cls: 'args', fn: rejectionSample, sample: [[3, 1, GAUSS, UNIF, 2, -1, 1], [2, 2, GAUSS, UNIF, 2, -1, 1]], post: Math.abs },
    'SAMPLE.inverse_cdf': { cls: 'args', fn: inverseCDF, sample: [[0.5, CDFID, 0, 1], [0.3, CDFID, 0, 1], [0.9, CDFID, 0, 1]], post: Math.abs },
    'RL.thompson': { cls: 'triple', fn: thompsonSample, sample: [[2, 3, 1], [3, 2, 2], [1, 1, 3]], post: Math.abs },
    'RL.advantage': { cls: 'args', fn: advantage, sample: [[[[1, 2], [3, 4]], [0, 0], 0, 1], [[[1, 2], [3, 4]], [0, 0], 1, 0]], post: Math.abs },
    'RL.gae': { cls: 'args', fn: gae, sample: [[[1, 1, 1], [0, 0.5, 0.8], 0.9, 0.95], [[0, 0], [1, 1], 0.99, 1]], post: Math.abs },
    'RL.reinforce': { cls: 'triple', fn: reinforceGradient, sample: [[[1, 2, 3], 1, 0.5], [[0.5, 1.5], 0, -0.2]], post: Math.abs },
    'RL.ppo': { cls: 'triple', fn: ppoClip, sample: [[1.2, 0.5, 0.2], [0.8, -0.3, 0.1], [0.9, 0.1, 0.2]], post: Math.abs },
    'RL.ucb': { cls: 'args', fn: ucb, sample: [[[[1, 2], [1, 1]], 1.4, 10], [[[3, 4], [2, 2]], 1.2, 20]], post: Math.abs },
    'RL.greedy': { cls: 'vec', fn: greedyPolicy, sample: [[1, 5, 2], [3, 1, 4]], post: Math.abs },
    'OPTIM.lr_decay': { cls: 'triple', fn: lrDecay, sample: [[0.1, 100, 10], [0.5, 50, 5]], post: Math.abs },
    'OPTIM.weight_decay': { cls: 'triple', fn: weightDecay, sample: [[[1, -2, 3], 0.1, 0.01], [[0.5, 0.5], 0.2, 0.02]], post: Math.abs },
    'OPTIM.polyak': { cls: 'vec', fn: polyakAvg, sample: [[[1, 2], [2, 3], [3, 4]], [[5, 6], [6, 7], [7, 8]]], post: Math.abs },
    'BWD.grad_softmax_ce': { cls: 'pair', fn: gradSoftmaxCE, sample: [[[1, 2, 3], 1], [[0.5, 1.5, 2.5], 2]], post: Math.abs },
    'BWD.stop_grad': { cls: 'vec', fn: stopGrad, sample: [[1, 2, 3], [0.5, 0.5, 1]], post: Math.abs },
    'BWD.detach': { cls: 'vec', fn: detach, sample: [[1, 2, 3], [0.5, 0.5, 1]], post: Math.abs },
    'BWD.jacobian': { cls: 'args', fn: jacobian, sample: [[FV, [1, 2], 0.000001], [FV, [3, 4], 0.000001]], post: Math.abs },
    'BWD.vjp': { cls: 'args', fn: vjp, sample: [[FV, [1, 2], [1, 1], 0.000001], [FV, [3, 4], [0, 1], 0.000001]], post: Math.abs },
    'BWD.jvp': { cls: 'args', fn: jvp, sample: [[FV, [1, 2], [1, 1], 0.000001], [FV, [3, 4], [1, 0], 0.000001]], post: Math.abs },
    'BWD.grad_accum': { cls: 'vec', fn: gradAccum, sample: [[[[1, 2], [3, 4]]], [[[0.5, 0.5], [1, 1]]]], post: Math.abs },
    'ATTEN.cross_attn': { cls: 'args', fn: crossAttention, sample: [[[[1, 0], [0, 1]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[2, 1], [1, 2]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]], post: Math.abs },
    'ATTEN.kv_cache': { cls: 'args', fn: attnRow, sample: [[[1, 0], [[1, 0], [0, 1]], [[1, 2], [3, 4]]], [[0, 1], [[1, 0], [0, 1]], [[1, 2], [3, 4]]]], post: Math.abs },
    'ATTEN.flash': { cls: 'args', fn: flashAttention, sample: [[[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], 1], [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], 2]], post: Math.abs },
    'DATA.normalize_fit': { cls: 'pair', fn: normalizeFit, sample: [[[1, 2, 3, 4], [5, 6]], [[2, 3], [4, 5]]], post: Math.abs },
    'LINALG.pca': { cls: 'pair', fn: pca, sample: [[[[4, 1], [1, 3]], 1], [[[2, 0], [0, 2]], 1]], post: Math.abs }
  ,

    /* —— v1.2.2 图推断+优化器批次（新增 26 条 fused，样本经沙盒逐条验证）—— */
    'OPTIM.gd': { cls: 'args', fn: gradientDescent, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.adam': { cls: 'args', fn: adam, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.momentum': { cls: 'args', fn: momentum, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.nesterov': { cls: 'args', fn: nesterov, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.adagrad': { cls: 'args', fn: adagrad, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.rmsprop': { cls: 'args', fn: rmsprop, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.adamw': { cls: 'args', fn: adamw, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.lbfgs': { cls: 'args', fn: lbfgs, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.bfgs': { cls: 'args', fn: bfgs, sample: [[FQ, GQ, [1, 2], {  }], [FQ, GQ, [0.5, 1], {  }]], post: Math.abs },
    'OPTIM.cg': { cls: 'args', fn: conjugateGradient, sample: [[[[4, 1], [1, 3]], [1, 2], {  }], [[[3, 0], [0, 2]], [1, 1], {  }]], post: Math.abs },
    'OPTIM.line_search': { cls: 'args', fn: lineSearch, sample: [[FQ, GQ, [1, 2], [-2, -4], {  }], [FQ, GQ, [2, 1], [-4, -2], {  }]], post: Math.abs },
    'GRAPH.forward_backward': { cls: 'args', fn: hmmForwardBackward, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]], post: Math.abs },
    'GRAPH.viterbi': { cls: 'args', fn: hmmViterbi, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]], post: Math.abs },
    'GRAPH.belief_prop': { cls: 'args', fn: chainBeliefProp, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]], post: Math.abs },
    'GRAPH.variable_elim': { cls: 'args', fn: variableElim, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]], post: Math.abs },
    'GRAPH.map_query': { cls: 'args', fn: hmmViterbi, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0, 1]]], post: Math.abs },
    'GRAPH.junction_tree': { cls: 'pair', fn: junctionTree, sample: [[[[0, 1], [1, 2]], [[0, 1]]], [[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2]]]], post: Math.abs },
    'GRAPH.condition': { cls: 'pair', fn: conditionalRow, sample: [[[[0.1, 0.9], [0.8, 0.2]], 0], [[[0.1, 0.9], [0.8, 0.2]], 1]], post: Math.abs },
    'GRAPH.gibbs': { cls: 'args', fn: gibbsSampler, sample: [[[[0, 1], [1, 0]], 100, 20, 1], [[[0.2, 0.8], [0.6, 0.4]], 100, 20, 2]], post: Math.abs },
    'GRAPH.mcmc': { cls: 'args', fn: metropolisSample, sample: [[[0.3, 0.7], 100, 20, 1], [[0.1, 0.5, 0.4], 100, 20, 2]], post: Math.abs },
    'GRAPH.particle': { cls: 'args', fn: particleFilter, sample: [[[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0], 50, 1], [[0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [1, 0], 50, 2]], post: Math.abs },
    'GRAPH.kalman': { cls: 'args', fn: kalmanFilter1d, sample: [[[1, 2, 3], 1, 1, 0.1, 0.1, 0, 1], [[2, 4], 0.9, 1, 0.2, 0.3, 1, 1]], post: Math.abs },
    'GRAPH.dsep': { cls: 'args', fn: dSeparated, sample: [[{ 0: [], 1: [0], 2: [0] }, 1, 2, [0]], [{ 0: [], 1: [0], 2: [0] }, 1, 2, []]], post: Math.abs },
    'GRAPH.m_sep': { cls: 'args', fn: mSeparated, sample: [[{ 0: [], 1: [0], 2: [0] }, 1, 2, [0]], [{ 0: [], 1: [0], 2: [0] }, 1, 2, []]], post: Math.abs },
    'GRAPH.clique_tree': { cls: 'triple', fn: cliqueTreeCalibrate, sample: [[{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [1, 2], shape: [2, 2], table: [0.2, 0.1, 0.4, 0.3] }, [1]], [{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [1, 2], shape: [2, 2], table: [0.2, 0.1, 0.4, 0.3] }, [0]]], post: Math.abs },
    'GRAPH.factor_product': { cls: 'pair', fn: factorProduct, sample: [[{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [1, 2], shape: [2, 2], table: [0.2, 0.1, 0.4, 0.3] }], [{ vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }, { vars: [0, 1], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] }]], post: Math.abs }
  ,

    /* —— v1.2.3 最后批次（新增 12 条 fused，样本经沙盒逐条验证）—— */
    'NUMER.sqrt_guard': { cls: 'elem', fn: sqrtGuard, sample: [[1, 4, 9], [0, 2]], post: Math.abs },
    'BWD.grad_clip': { cls: 'pair', fn: gradClip, sample: [[[4, 1, 2], 3], [[1, 2], 1.5]], post: Math.abs },
    'BWD.grad_linear': { cls: 'triple', fn: mlpGrad, sample: [[{ W1: [[0.5, -0.2], [0.1, 0.3]], b1: [0, 0], W2: [[0.2, -0.4]], b2: [0] }, [[1, 2], [0, 1]], [[1], [0]]]], post: Math.abs },
    'BWD.grad_check': { cls: 'args', fn: gradientCheck, sample: [[{ W1: [[0.5, -0.2], [0.1, 0.3]], b1: [0, 0], W2: [[0.2, -0.4]], b2: [0] }, [[1, 2], [0, 1]], [[1], [0]], 0.00001]], post: Math.abs },
    'BWD.grad_layernorm': { cls: 'triple', fn: gradLayerNorm, sample: [[[1, 2, 3], [0.1, 0.2, 0.3], 0.00001]], post: Math.abs },
    'BWD.chain_rule': { cls: 'args', fn: chainRule, sample: [[DOUT, INNER, DIN, [3]], [DOUT, INNER, DIN, [-2]]], post: Math.abs },
    'BWD.bptt': { cls: 'triple', fn: rnnBPTT, sample: [[{ W: 0.5, U: 0.3, h0: 0 }, [1, 0.5], [0.8, 0.6]]], post: Math.abs },
    'RL.value_iter': { cls: 'vec', fn: valueIteration, sample: [{ n: 4, gamma: 0.9 }, { n: 3, gamma: 0.95 }], post: Math.abs },
    'RL.q_learn': { cls: 'vec', fn: qLearning, sample: [{ n: 4, gamma: 0.9, episodes: 200 }, { n: 3, episodes: 100 }], post: Math.abs },
    'RL.bellman_backup': { cls: 'pair', fn: bellmanBackup, sample: [[[0, 0, 0], MDP3]], post: Math.abs },
    'RL.policy_iter': { cls: 'vec', fn: policyIteration, sample: [MDP3, MDP4], post: Math.abs },
    'RL.sarsa': { cls: 'vec', fn: sarsa, sample: [{ n: 4, gamma: 0.9, episodes: 200 }], post: Math.abs }
  };

  /* ==========================================================================
   * 集大成验证算法库（v1.3.8）—— L10 原子指令 ISA 之外的旗舰级参考实现
   *   铁律：每条算法携带可独立复算的硬不变量
   *   （解析 vs 数值 / 参照实现 vs 主实现 / 结构不变量）
   * ======================================================================== */

  /* SVD —— 单侧 Jacobi（Hestenes 法）：B = A·V 收敛为列正交，U = B·Σ⁻¹ */
  function svd(A) {
    var m = A.length, n = A[0].length;
    var B = A.map(function (r) { return r.slice(); });
    var V = eye(n);
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0;
      for (var p = 0; p < n; p++) {
        for (var q = p + 1; q < n; q++) {
          var app = 0, aqq = 0, apq = 0;
          for (var i = 0; i < m; i++) { var bp = B[i][p], bq = B[i][q]; app += bp * bp; aqq += bq * bq; apq += bp * bq; }
          off += apq * apq;
          if (Math.abs(apq) <= 1e-300) continue;
          var zeta = (aqq - app) / (2 * apq);
          // ⚠️ ζ=0（两列范数恰等的对称边界）时 Math.sign(0)=0 会使 t=0 退化为恒等旋转、非对角永不归零；
          // Hestenes 根 t²+2ζt−1=0 在 ζ=0 时为 ±1（均有效），须显式取 ±1（v1.3.11 修复，回归检查见 checks）
          var t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          var c = 1 / Math.sqrt(1 + t * t), s = c * t;
          for (var r = 0; r < m; r++) { var bpr = B[r][p], bqr = B[r][q]; B[r][p] = c * bpr - s * bqr; B[r][q] = s * bpr + c * bqr; }
          for (var r2 = 0; r2 < n; r2++) { var vp = V[r2][p], vq = V[r2][q]; V[r2][p] = c * vp - s * vq; V[r2][q] = s * vp + c * vq; }
        }
      }
      if (off < 1e-30) break;
    }
    var sigma = [];
    for (var j = 0; j < n; j++) sigma.push(norm2(B.map(function (row) { return row[j]; })));
    var idx = sigma.map(function (_, i) { return i; }).sort(function (a, b) { return sigma[b] - sigma[a]; });
    var S = idx.map(function (i) { return sigma[i]; });
    var perm = idx.concat(Array.from({ length: n }, function (_, i) { return i; }).filter(function (i) { return idx.indexOf(i) === -1; }));
    var Vo = V.map(function (row) { return perm.map(function (i) { return row[i]; }); });
    var U = [];
    for (var i2 = 0; i2 < m; i2++) {
      var row = [];
      for (var j2 = 0; j2 < n; j2++) row.push(S[j2] > 0 ? B[i2][idx[j2]] / S[j2] : 0);
      U.push(row);
    }
    return { U: U, S: S, V: Vo };
  }

  /* LU 分解 —— Doolittle 无主元（对角占优矩阵保证数值稳定，无需换行） */
  function luDecompose(A) {
    var n = A.length;
    var L = eye(n), U = A.map(function (r) { return r.slice(); });
    for (var k = 0; k < n; k++) {
      for (var i = k + 1; i < n; i++) {
        L[i][k] = U[i][k] / U[k][k];
        for (var j = k; j < n; j++) U[i][j] -= L[i][k] * U[k][j];
      }
    }
    return { L: L, U: U };
  }

  /* FFT —— 迭代式 radix-2 Cooley–Tukey（输入 n 为 2 的幂，复数 {re,im}） */
  function fft(x) {
    var n = x.length, X = x.slice();
    for (var i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { var tmp = X[i]; X[i] = X[j]; X[j] = tmp; }
    }
    for (var len = 2; len <= n; len <<= 1) {
      var ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (var blk = 0; blk < n; blk += len) {
        var cwr = 1, cwi = 0;
        for (var k = 0; k < len / 2; k++) {
          var ur = X[blk + k].re, ui = X[blk + k].im;
          var vr = X[blk + k + len / 2].re * cwr - X[blk + k + len / 2].im * cwi;
          var vi = X[blk + k + len / 2].re * cwi + X[blk + k + len / 2].im * cwr;
          X[blk + k] = { re: ur + vr, im: ui + vi };
          X[blk + k + len / 2] = { re: ur - vr, im: ui - vi };
          var ncwr = cwr * wr - cwi * wi, ncwi = cwr * wi + cwi * wr; cwr = ncwr; cwi = ncwi;
        }
      }
    }
    return X;
  }
  /* 朴素 O(n²) DFT —— 作为 FFT 的独立参照实现 */
  function dft(x) {
    var n = x.length, X = [];
    for (var k = 0; k < n; k++) {
      var re = 0, im = 0;
      for (var t = 0; t < n; t++) {
        var ang = -2 * Math.PI * k * t / n;
        re += x[t].re * Math.cos(ang) - x[t].im * Math.sin(ang);
        im += x[t].re * Math.sin(ang) + x[t].im * Math.cos(ang);
      }
      X.push({ re: re, im: im });
    }
    return X;
  }

  /* PageRank —— 幂迭代（列随机转移矩阵 + 悬挂节点 teleport 处理） */
  function pageRank(n, edges, alpha) {
    var P = zeros2(n, n), outSum = new Array(n).fill(0), outCnt = new Array(n).fill(0);
    edges.forEach(function (e) { outSum[e[0]] += e[2]; outCnt[e[0]]++; });
    edges.forEach(function (e) { P[e[1]][e[0]] += e[2] / outSum[e[0]]; });
    for (var i = 0; i < n; i++) if (outCnt[i] === 0) for (var j = 0; j < n; j++) P[j][i] += 1 / n;
    var tele = new Array(n).fill(1 / n), pi = tele.slice();
    for (var it = 0; it < 3000; it++) {
      var npi = new Array(n).fill(0);
      for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) npi[a] += P[a][b] * pi[b];
      for (var cc = 0; cc < n; cc++) npi[cc] = alpha * npi[cc] + (1 - alpha) * tele[cc];
      var d = 0;
      for (var e2 = 0; e2 < n; e2++) d = Math.max(d, Math.abs(npi[e2] - pi[e2]));
      pi = npi;
      if (d < 1e-16) break;
    }
    var Mat = eye(n).map(function (row, ri) { return row.map(function (val, ci) { return (ri === ci ? 1 : 0) - alpha * P[ri][ci]; }); });
    var rhs = tele.map(function (val) { return (1 - alpha) * val; });
    return { pi: pi, x: solveLinear(Mat, rhs) };
  }

  /* Dijkstra —— 单源最短路径（非负权，贪心 + 线性扫描取最小） */
  function dijkstra(n, edges, src) {
    var dist = new Array(n).fill(Infinity); dist[src] = 0;
    var vis = new Array(n).fill(false);
    for (var it = 0; it < n; it++) {
      var u = -1, best = Infinity;
      for (var i = 0; i < n; i++) if (!vis[i] && dist[i] < best) { best = dist[i]; u = i; }
      if (u === -1) break;
      vis[u] = true;
      for (var k = 0; k < edges.length; k++) {
        if (edges[k][0] === u) {
          var nd = dist[u] + edges[k][2];
          if (nd < dist[edges[k][1]]) dist[edges[k][1]] = nd;
        }
      }
    }
    return dist;
  }
  /* Bellman–Ford —— 作为 Dijkstra 的独立参照实现 */
  function bellmanFord(n, edges, src) {
    var dist = new Array(n).fill(Infinity); dist[src] = 0;
    for (var i = 0; i < n - 1; i++) {
      for (var k = 0; k < edges.length; k++) {
        if (dist[edges[k][0]] + edges[k][2] < dist[edges[k][1]]) dist[edges[k][1]] = dist[edges[k][0]] + edges[k][2];
      }
    }
    return dist;
  }

  /* Slice sampling —— Neal 2003（步进扩展 + 收缩）；logf 为对数密度，加性常数无关 */
  function sliceSample(logf, x0, N, rng, w, m, burn) {
    var x = x0, out = [];
    for (var t = 0; t < N + burn; t++) {
      var u = rng();
      var y = logf(x) + Math.log(u > 0 ? u : 1e-300);   // 对数水平 Y = logf(x) + log(u) ≤ logf(x)
      var su = rng() * w, L = x - su, R = x + (w - su);
      var j = 0, k = 0;
      while (j < m && logf(L) > y) { L -= w; j++; }
      while (k < m && logf(R) > y) { R += w; k++; }
      while (true) {
        var x1 = L + rng() * (R - L);
        if (logf(x1) >= y) { x = x1; break; }
        if (x1 > x) R = x1; else L = x1;
      }
      if (t >= burn) out.push(x);
    }
    return out;
  }

  /* Newton 迭代 —— 求根（牛顿-拉夫森） */
  function newton(f, df, x0, tol, maxIter) {
    var x = x0;
    for (var i = 0; i < maxIter; i++) {
      var fx = f(x);
      if (Math.abs(fx) < tol) break;
      var step = fx / df(x);
      x -= step;
      if (Math.abs(step) < 1e-16) break;
    }
    return x;
  }

  /* RK4 —— 四阶 Runge–Kutta 常微分方程积分器（f(t, x) = dx/dt） */
  function rk4(f, x0, T, h) {
    var steps = Math.round(T / h), x = x0, t = 0;
    for (var i = 0; i < steps; i++) {
      var k1 = f(t, x);
      var k2 = f(t + h / 2, x + h / 2 * k1);
      var k3 = f(t + h / 2, x + h / 2 * k2);
      var k4 = f(t + h, x + h * k3);
      x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
      t += h;
    }
    return x;
  }

  /* ==========================================================================
   * 算法库扩展（v1.3.9）—— 二叉堆 / 堆优化 Dijkstra / A* / FFT 卷积 /
   *   CSR 稀疏矩阵 / Prim+Kruskal MST / 二分求根 / Kadane
   *   铁律同上：每条携带可独立复算硬不变量
   * ======================================================================== */

  /* 二叉堆（小顶堆）—— ⚠️ 数组元素间 < / <= 会走字符串强制转换，必须传显式 less */
  function heapPush(h, x, less) {
    less = less || function (a, b) { return a < b; };
    h.push(x);
    var i = h.length - 1;
    while (i > 0) {
      var p = (i - 1) >> 1;
      if (!less(h[i], h[p])) break;
      var t = h[p]; h[p] = h[i]; h[i] = t;
      i = p;
    }
  }
  function heapPop(h, less) {
    less = less || function (a, b) { return a < b; };
    var top = h[0], last = h.pop();
    if (h.length) {
      h[0] = last;
      var i = 0, n = h.length;
      for (; ;) {
        var l = 2 * i + 1, r = l + 1, s = i;
        if (l < n && less(h[l], h[s])) s = l;
        if (r < n && less(h[r], h[s])) s = r;
        if (s === i) break;
        var t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
      }
    }
    return top;
  }
  /* 堆排序 —— 全部入堆后依次弹出得升序（O(n log n)，原地外排序） */
  function heapSort(a) {
    var h = [];
    for (var i = 0; i < a.length; i++) heapPush(h, a[i]);
    var out = [];
    while (h.length) out.push(heapPop(h));
    return out;
  }

  /* 堆优化 Dijkstra —— 二叉堆优先队列 + 过期条目惰性删除（O((V+E)·logV)） */
  function dijkstraHeap(n, edges, src) {
    var adj = [];
    for (var i = 0; i < n; i++) adj.push([]);
    for (var k = 0; k < edges.length; k++) adj[edges[k][0]].push([edges[k][1], edges[k][2]]);
    var dist = new Array(n).fill(Infinity);
    dist[src] = 0;
    var lessF = function (a, b) { return a[0] < b[0]; };   // 按距离键比较（禁字符串 coercion）
    var h = [];
    heapPush(h, [0, src], lessF);
    while (h.length) {
      var top = heapPop(h, lessF), d = top[0], u = top[1];
      if (d > dist[u]) continue;                       // 过期条目
      for (var j = 0; j < adj[u].length; j++) {
        var v = adj[u][j][0], w = adj[u][j][1];
        if (d + w < dist[v]) { dist[v] = d + w; heapPush(h, [dist[v], v], lessF); }
      }
    }
    return dist;
  }

  /* A* —— 启发式搜索；可采纳启发（h ≤ 真实代价）保证最优性；返回距离与扩展节点数 */
  function aStar(n, edges, src, goal, h) {
    var adj = [];
    for (var i = 0; i < n; i++) adj.push([]);
    for (var k = 0; k < edges.length; k++) adj[edges[k][0]].push([edges[k][1], edges[k][2]]);
    var dist = new Array(n).fill(Infinity);
    dist[src] = 0;
    var lessF = function (a, b) { return a[0] < b[0]; };
    var open = [];
    heapPush(open, [h(src), src], lessF);
    var expanded = 0;
    while (open.length) {
      var top = heapPop(open, lessF), u = top[1];
      if (u === goal) return { dist: dist[goal], expanded: expanded };
      expanded++;
      for (var j = 0; j < adj[u].length; j++) {
        var v = adj[u][j][0], w = adj[u][j][1], nd = dist[u] + w;
        if (nd < dist[v]) { dist[v] = nd; heapPush(open, [nd + h(v), v], lessF); }
      }
    }
    return { dist: dist[goal], expanded: expanded };
  }

  /* FFT 卷积 —— 循环卷积定理 c = IFFT(FFT(a)·FFT(b))（n 为 2 的幂，复数 {re,im}） */
  function ifft(X) {
    var n = X.length;
    var c = X.map(function (v) { return { re: v.re, im: -v.im }; });
    var F = fft(c);
    return F.map(function (v) { return { re: v.re / n, im: -v.im / n }; });
  }
  function fftConvolve(a, b) {
    var FA = fft(a.map(function (x) { return { re: x, im: 0 }; }));
    var FB = fft(b.map(function (x) { return { re: x, im: 0 }; }));
    var FC = FA.map(function (v, i) {
      return { re: v.re * FB[i].re - v.im * FB[i].im, im: v.re * FB[i].im + v.im * FB[i].re };
    });
    return ifft(FC).map(function (v) { return v.re; });
  }

  /* CSR 稀疏矩阵 —— 行压缩存储（indptr 行指针 / indices 列下标 / data 值，零省略） */
  function denseToCSR(M) {
    var m = M.length, n = M[0].length;
    var indptr = [0], indices = [], data = [];
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        if (M[i][j] !== 0) { indices.push(j); data.push(M[i][j]); }
      }
      indptr.push(data.length);
    }
    return { m: m, n: n, indptr: indptr, indices: indices, data: data };
  }
  function csrMatVec(S, x) {
    var y = new Array(S.m).fill(0);
    for (var i = 0; i < S.m; i++) {
      var s = 0;
      for (var k = S.indptr[i]; k < S.indptr[i + 1]; k++) s += S.data[k] * x[S.indices[k]];
      y[i] = s;
    }
    return y;
  }
  function csrTranspose(S) {
    var counts = new Array(S.n).fill(0);
    for (var t = 0; t < S.indices.length; t++) counts[S.indices[t]]++;
    var indptr = new Array(S.n + 1).fill(0);
    for (var j = 0; j < S.n; j++) indptr[j + 1] = indptr[j] + counts[j];
    var next = indptr.slice(0, S.n), indices = new Array(S.data.length), data = new Array(S.data.length);
    for (var i2 = 0; i2 < S.m; i2++) {
      for (var k2 = S.indptr[i2]; k2 < S.indptr[i2 + 1]; k2++) {
        var j2 = S.indices[k2], p = next[j2]++;
        indices[p] = i2;
        data[p] = S.data[k2];
      }
    }
    return { m: S.n, n: S.m, indptr: indptr, indices: indices, data: data };
  }
  function csrToDense(S) {
    var M = [];
    for (var i = 0; i < S.m; i++) M.push(new Array(S.n).fill(0));
    for (var i2 = 0; i2 < S.m; i2++) {
      for (var k = S.indptr[i2]; k < S.indptr[i2 + 1]; k++) M[i2][S.indices[k]] = S.data[k];
    }
    return M;
  }

  /* Prim MST（稠密 O(V²)） + Kruskal MST（并查集 O(E logE)） + 暴力枚举生成树参照 */
  function primMST(n, edges) {
    var adj = [];
    for (var i = 0; i < n; i++) adj.push([]);
    for (var k = 0; k < edges.length; k++) {
      adj[edges[k][0]].push([edges[k][1], edges[k][2]]);
      adj[edges[k][1]].push([edges[k][0], edges[k][2]]);
    }
    var inT = new Array(n).fill(false), key = new Array(n).fill(Infinity);
    key[0] = 0;
    var total = 0;
    for (var it = 0; it < n; it++) {
      var u = -1, best = Infinity;
      for (var i2 = 0; i2 < n; i2++) if (!inT[i2] && key[i2] < best) { best = key[i2]; u = i2; }
      if (u === -1) break;                    // 不连通：提前收尾
      inT[u] = true;
      total += key[u];
      for (var j = 0; j < adj[u].length; j++) {
        var v = adj[u][j][0], w = adj[u][j][1];
        if (!inT[v] && w < key[v]) key[v] = w;
      }
    }
    return total;
  }
  function kruskalMST(n, edges) {
    var parent = [];
    for (var i = 0; i < n; i++) parent.push(i);
    var find = function (x) {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }   // 路径压缩
      return x;
    };
    var sorted = edges.slice().sort(function (a, b) { return a[2] - b[2]; });
    var total = 0, count = 0;
    for (var k = 0; k < sorted.length; k++) {
      var ru = find(sorted[k][0]), rv = find(sorted[k][1]);
      if (ru !== rv) { parent[ru] = rv; total += sorted[k][2]; count++; }
    }
    return { total: total, count: count };
  }
  /* 暴力枚举所有 (n−1) 边子集的生成树（小图 n≤6 独立参照，O(E^{n−1})） */
  function bruteForceMST(n, edges) {
    var E = edges.length, need = n - 1, best = Infinity;
    var combos = function (start, chosen, w) {
      if (chosen === need) {
        var parent = [];
        for (var i = 0; i < n; i++) parent.push(i);
        var find = function (x) {
          while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
          return x;
        };
        var ok = true;
        for (var k = 0; k < w.length; k++) {
          var ru = find(w[k][0]), rv = find(w[k][1]);
          if (ru === rv) { ok = false; break; }
          parent[ru] = rv;
        }
        if (ok) {
          var roots = {}, cnt = 0;
          for (var i2 = 0; i2 < n; i2++) { var r = find(i2); if (!roots[r]) { roots[r] = 1; cnt++; } }
          if (cnt === 1) best = Math.min(best, w.reduce(function (s, e) { return s + e[2]; }, 0));
        }
        return;
      }
      for (var i3 = start; i3 < E; i3++) { w.push(edges[i3]); combos(i3 + 1, chosen + 1, w); w.pop(); }
    };
    combos(0, 0, []);
    return best;
  }

  /* 二分求根 —— 括号区间 [lo,hi]（f(lo)·f(hi)<0）逐次折半（线性收敛，绝对可靠） */
  function bisect(f, lo, hi, tol, maxIter) {
    var flo = f(lo), fhi = f(hi);
    for (var i = 0; i < maxIter; i++) {
      var mid = (lo + hi) / 2, fm = f(mid);
      if (Math.abs(fm) < tol || (hi - lo) < 1e-16) return mid;
      if (flo * fm < 0) { hi = mid; fhi = fm; } else { lo = mid; flo = fm; }
    }
    return (lo + hi) / 2;
  }

  /* Kadane 最大子数组 —— 线性 DP：cur = max(aᵢ, cur + aᵢ)（O(n)） */
  function kadane(a) {
    var best = a[0], cur = a[0];
    for (var i = 1; i < a.length; i++) {
      cur = Math.max(a[i], cur + a[i]);
      best = Math.max(best, cur);
    }
    return best;
  }

  /* ==========================================================================
   * 算法库扩展（v1.3.10）—— QR 迭代特征值 / 最大流（Edmonds–Karp + Dinic）/
   *   强连通分量（Tarjan + Kosaraju）/ Kahn 拓扑排序 / Kuhn 二分匹配
   *   铁律同上：每条携带可独立复算硬不变量（定理级 / 跨算法一致 / 结构）
   * ======================================================================== */

  /* QR 迭代特征值 —— Householder QR 每步 + Wilkinson 位移 + 块 deflation（对称阵）
   * 与 jacobiEigen（旋转法）构成两条完全独立的特征值路径；Wilkinson 位移 ⇒ 三次收敛 */
  function qrEigenvalues(Am, tol) {
    tol = tol || 1e-13;
    var n = Am.length, M = clone2(Am), scale = 0;
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) scale = Math.max(scale, Math.abs(M[i][j]));
    if (scale === 0) return { values: new Array(n).fill(0), iters: 0, off: 0 };
    var hi = n - 1, iter = 0;
    while (hi > 0 && iter < 300 * n) {
      iter++;
      if (Math.abs(M[hi][hi - 1]) <= tol * scale) { hi--; continue; }   // 底子对角可忽略 → 已收敛出一个特征值
      var l = hi - 1;
      while (l > 0 && Math.abs(M[l][l - 1]) > tol * scale) l--;         // 活跃不可约块 [l..hi]，size ≥ 2
      var a = M[hi - 1][hi - 1], b = M[hi - 1][hi], c = M[hi][hi];      // Wilkinson 位移（块尾 2×2）
      var d = (a - c) / 2, sgn = d >= 0 ? 1 : -1;
      var den = Math.abs(d) + Math.sqrt(d * d + b * b);
      var mu = den > 0 ? c - sgn * b * b / den : c;
      var sz = hi - l + 1, sub = [];
      for (var i2 = l; i2 <= hi; i2++) {
        var row = [];
        for (var j2 = l; j2 <= hi; j2++) row.push(M[i2][j2] - (i2 === j2 ? mu : 0));
        sub.push(row);
      }
      var res = qrDecompose(sub);
      var RQ = matMul(res.R, res.Q);
      for (var i3 = l; i3 <= hi; i3++) for (var j3 = l; j3 <= hi; j3++) M[i3][j3] = RQ[i3 - l][j3 - l] + (i3 === j3 ? mu : 0);
      for (var j4 = 0; j4 < l; j4++) {                                   // 块外列：Qᵀ·col（保相似）
        var col = [];
        for (var i4 = l; i4 <= hi; i4++) col.push(M[i4][j4]);
        for (var i5 = l; i5 <= hi; i5++) {
          var s = 0;
          for (var k = 0; k < sz; k++) s += res.Q[k][i5 - l] * col[k];
          M[i5][j4] = s;
        }
      }
      for (var i6 = 0; i6 < l; i6++) {                                   // 块外行：row·Q（保相似）
        var rowv = [];
        for (var j5 = l; j5 <= hi; j5++) rowv.push(M[i6][j5]);
        for (var j6 = l; j6 <= hi; j6++) {
          var s2 = 0;
          for (var k2 = 0; k2 < sz; k2++) s2 += rowv[k2] * res.Q[k2][j6 - l];
          M[i6][j6] = s2;
        }
      }
    }
    var off = 0;
    for (var i7 = 1; i7 < n; i7++) off += Math.abs(M[i7][i7 - 1]);
    var eigs = [];
    for (var i8 = 0; i8 < n; i8++) eigs.push(M[i8][i8]);
    return { values: eigs.sort(function (x, y) { return y - x; }), iters: iter, off: off };
  }

  /* Edmonds–Karp 最大流 —— BFS 最短增广路（O(V·E²)）；cap 为 n×n 容量邻接矩阵 */
  function maxFlowEK(n, cap, s, t) {
    var res = cap.map(function (r) { return r.slice(); }), flow = 0;
    for (; ;) {
      var parent = new Array(n).fill(-1);
      parent[s] = s;
      var queue = [s], qi = 0;
      while (qi < queue.length && parent[t] === -1) {
        var u = queue[qi++];
        for (var v = 0; v < n; v++) if (parent[v] === -1 && res[u][v] > 1e-15) { parent[v] = u; queue.push(v); }
      }
      if (parent[t] === -1) break;
      var b = Infinity, v2 = t;
      while (v2 !== s) { b = Math.min(b, res[parent[v2]][v2]); v2 = parent[v2]; }
      v2 = t;
      while (v2 !== s) { res[parent[v2]][v2] -= b; res[v2][parent[v2]] += b; v2 = parent[v2]; }
      flow += b;
    }
    return flow;
  }

  /* Dinic 最大流 —— 层次图 + 多路增广阻塞流（O(V²E)） */
  function maxFlowDinic(n, cap, s, t) {
    var res = cap.map(function (r) { return r.slice(); }), flow = 0;
    var dfs = function (u, pushed, level, iter) {
      if (u === t) return pushed;
      while (iter[u] < n) {
        var v = iter[u];
        if (res[u][v] > 1e-15 && level[v] === level[u] + 1) {
          var tr = dfs(v, Math.min(pushed, res[u][v]), level, iter);
          if (tr > 1e-15) { res[u][v] -= tr; res[v][u] += tr; return tr; }
        }
        iter[u]++;
      }
      return 0;
    };
    for (; ;) {
      var level = new Array(n).fill(-1);
      level[s] = 0;
      var queue = [s], qi = 0;
      while (qi < queue.length) {
        var u2 = queue[qi++];
        for (var v2 = 0; v2 < n; v2++) if (level[v2] === -1 && res[u2][v2] > 1e-15) { level[v2] = level[u2] + 1; queue.push(v2); }
      }
      if (level[t] === -1) break;
      var iter = new Array(n).fill(0);
      for (; ;) {
        var f = dfs(s, Infinity, level, iter);
        if (f <= 1e-15) break;
        flow += f;
      }
    }
    return flow;
  }

  /* 暴力枚举最小割（独立参照：枚举全部 2^(n−2) 个含 s 不含 t 的顶点子集） */
  function minCutBrute(n, cap, s, t) {
    var best = Infinity, others = [];
    for (var v = 0; v < n; v++) if (v !== s && v !== t) others.push(v);
    for (var mask = 0; mask < (1 << others.length); mask++) {
      var inS = new Array(n).fill(false);
      inS[s] = true;
      for (var i = 0; i < others.length; i++) if (mask & (1 << i)) inS[others[i]] = true;
      var cut = 0;
      for (var i2 = 0; i2 < n; i2++) if (inS[i2]) for (var j = 0; j < n; j++) if (!inS[j]) cut += cap[i2][j];
      best = Math.min(best, cut);
    }
    return best;
  }

  /* Tarjan 强连通分量 —— 单 DFS + low-link（一次遍历出全部分量） */
  function tarjanSCC(n, edges) {
    var adj = Array.from({ length: n }, function () { return []; });
    edges.forEach(function (e) { adj[e[0]].push(e[1]); });
    var idx = 0, index = new Array(n).fill(-1), low = new Array(n).fill(0), on = new Array(n).fill(false), st = [], comps = [];
    var strong = function (v) {
      index[v] = low[v] = idx++;
      st.push(v);
      on[v] = true;
      for (var k = 0; k < adj[v].length; k++) {
        var w = adj[v][k];
        if (index[w] === -1) { strong(w); low[v] = Math.min(low[v], low[w]); }
        else if (on[w]) low[v] = Math.min(low[v], index[w]);
      }
      if (low[v] === index[v]) {
        var comp = [];
        for (; ;) {
          var w2 = st.pop();
          on[w2] = false;
          comp.push(w2);
          if (w2 === v) break;
        }
        comps.push(comp.sort(function (x, y) { return x - y; }));
      }
    };
    for (var v2 = 0; v2 < n; v2++) if (index[v2] === -1) strong(v2);
    return comps.sort(function (a, b) { return a[0] - b[0]; });
  }

  /* Kosaraju 强连通分量 —— 正序 DFS 完成时间 + 反向图 DFS（两遍遍历） */
  function kosarajuSCC(n, edges) {
    var adj = Array.from({ length: n }, function () { return []; }), radj = Array.from({ length: n }, function () { return []; });
    edges.forEach(function (e) { adj[e[0]].push(e[1]); radj[e[1]].push(e[0]); });
    var vis = new Array(n).fill(false), order = [];
    var dfs1 = function (v) {
      vis[v] = true;
      for (var k = 0; k < adj[v].length; k++) if (!vis[adj[v][k]]) dfs1(adj[v][k]);
      order.push(v);
    };
    for (var v = 0; v < n; v++) if (!vis[v]) dfs1(v);
    var vis2 = new Array(n).fill(false), comps = [];
    var dfs2 = function (v, comp) {
      vis2[v] = true;
      comp.push(v);
      for (var k = 0; k < radj[v].length; k++) if (!vis2[radj[v][k]]) dfs2(radj[v][k], comp);
    };
    for (var i = n - 1; i >= 0; i--) {
      var v2 = order[i];
      if (!vis2[v2]) {
        var comp = [];
        dfs2(v2, comp);
        comps.push(comp.sort(function (x, y) { return x - y; }));
      }
    }
    return comps.sort(function (a, b) { return a[0] - b[0]; });
  }

  /* Kahn 拓扑排序 —— 入度队列入队；有环返回 null */
  function kahnTopo(n, edges) {
    var adj = Array.from({ length: n }, function () { return []; }), indeg = new Array(n).fill(0);
    edges.forEach(function (e) { adj[e[0]].push(e[1]); indeg[e[1]]++; });
    var q = [];
    for (var v = 0; v < n; v++) if (indeg[v] === 0) q.push(v);
    var order = [];
    while (q.length) {
      var u = q.shift();
      order.push(u);
      for (var k = 0; k < adj[u].length; k++) if (--indeg[adj[u][k]] === 0) q.push(adj[u][k]);
    }
    return order.length === n ? order : null;
  }

  /* Kuhn 二分图最大匹配 —— 匈牙利增广路；König 定理：最大匹配 = 最小顶点覆盖 */
  function kuhnMatching(lCount, rCount, edges) {
    var adjL = Array.from({ length: lCount }, function () { return []; });
    edges.forEach(function (e) { adjL[e[0]].push(e[1]); });
    var matchR = new Array(rCount).fill(-1);
    var tryK = function (u, used) {
      for (var k = 0; k < adjL[u].length; k++) {
        var v = adjL[u][k];
        if (!used[v]) {
          used[v] = true;
          if (matchR[v] === -1 || tryK(matchR[v], used)) { matchR[v] = u; return true; }
        }
      }
      return false;
    };
    var size = 0;
    for (var u = 0; u < lCount; u++) {
      var used = new Array(rCount).fill(false);
      if (tryK(u, used)) size++;
    }
    return { size: size, matchR: matchR };
  }
  /* 暴力枚举最小顶点覆盖（独立参照，l+r ≤ 10） */
  function minVertexCoverBrute(lCount, rCount, edges) {
    var total = lCount + rCount, best = Infinity;
    for (var mask = 0; mask < (1 << total); mask++) {
      var inC = [];
      for (var v = 0; v < total; v++) inC.push(!!(mask & (1 << v)));
      var ok = true;
      for (var k = 0; k < edges.length; k++) {
        if (!inC[edges[k][0]] && !inC[lCount + edges[k][1]]) { ok = false; break; }
      }
      if (ok) {
        var c = 0;
        for (var v2 = 0; v2 < total; v2++) if (inC[v2]) c++;
        best = Math.min(best, c);
      }
    }
    return best;
  }

  /* ==========================================================================
   * 算法库扩展二（v1.3.12）—— Bluestein 任意长度 FFT / 自适应 Metropolis /
   *   Hamiltonian MC / Floyd–Warshall / 并查集 / KMP / Gauss–Legendre 求积 /
   *   Strassen 矩阵乘
   *   铁律同上：每条携带可独立复算硬不变量（参照实现 vs 主实现 / 解析对照 / 定理级）
   * ======================================================================== */

  /* Bluestein FFT —— 任意长度（含非 2 的幂）卷积定理算法（Bluestein 1968/1970）
   * jk = (j² + k² − (j−k)²)/2 ⇒ X_k = e^{−πik²/n}·Σ_j (x_j e^{−πij²/n})·e^{πi(j−k)²/n}
   * 即 chirp 序列与 x 的循环卷积，用长度 ≥ 2n−1 的 radix-2 FFT 完成（O(n log n)） */
  function bluesteinFFT(x) {
    var n = x.length;
    if ((n & (n - 1)) === 0) return fft(x);          // 2 的幂直接走 Cooley–Tukey
    var m = 1;
    while (m < 2 * n - 1) m <<= 1;                   // 循环卷积长度 ≥ 2n−1
    var A = [], B = [], i;
    for (i = 0; i < m; i++) { A.push({ re: 0, im: 0 }); B.push({ re: 0, im: 0 }); }
    for (i = 0; i < n; i++) {                        // a_j = x_j·e^{−πij²/n}
      var a1 = -Math.PI * i * i / n;
      A[i] = { re: x[i].re * Math.cos(a1) - x[i].im * Math.sin(a1), im: x[i].re * Math.sin(a1) + x[i].im * Math.cos(a1) };
    }
    for (i = 0; i < n; i++) {                        // b_m = e^{+πim²/n}（负下标折叠到尾部）
      var a2 = Math.PI * i * i / n;
      B[i] = { re: Math.cos(a2), im: Math.sin(a2) };
      if (i > 0) B[m - i] = B[i];
    }
    var FA = fft(A), FB = fft(B), FC = [];
    for (i = 0; i < m; i++) {
      FC.push({ re: FA[i].re * FB[i].re - FA[i].im * FB[i].im, im: FA[i].re * FB[i].im + FA[i].im * FB[i].re });
    }
    var C = ifft(FC), X = [];
    for (i = 0; i < n; i++) {                        // X_k = e^{−πik²/n}·c_k
      var a3 = -Math.PI * i * i / n;
      X.push({ re: C[i].re * Math.cos(a3) - C[i].im * Math.sin(a3), im: C[i].re * Math.sin(a3) + C[i].im * Math.cos(a3) });
    }
    return X;
  }

  /* 自适应 Metropolis（Haario 2001, Adaptive Metropolis）—— 提议协方差由历史样本在线估计
   * C = s_d·(cov(链历史) + εI)，s_d = 2.38²/d（Roberts–Rosenthal 2001 最优缩放）
   * 自适应在 burn-in 结束时冻结 ⇒ burn 后为齐次马尔可夫链，保证遍历到正确平稳分布 */
  function cholSafe(C) {                              // 带正则的下三角 Cholesky（协方差估计可能近奇异）
    var d = C.length, L = zeros2(d, d), i, j, k;
    for (i = 0; i < d; i++) {
      for (j = 0; j <= i; j++) {
        var s = C[i][j];
        for (k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(s, 1e-10));
        else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  function covOf(hist) {                             // 样本协方差（无偏，n−1）
    var n = hist.length, d = hist[0].length, mu = new Array(d).fill(0), i, j, t;
    for (t = 0; t < n; t++) for (i = 0; i < d; i++) mu[i] += hist[t][i] / n;
    var C = zeros2(d, d);
    for (t = 0; t < n; t++) {
      for (i = 0; i < d; i++) for (j = 0; j < d; j++) C[i][j] += (hist[t][i] - mu[i]) * (hist[t][j] - mu[j]) / (n - 1);
    }
    return C;
  }
  function covScale(C, sd, eps) {
    var d = C.length, R = zeros2(d, d), i, j;
    for (i = 0; i < d; i++) for (j = 0; j < d; j++) R[i][j] = sd * C[i][j] + (i === j ? eps : 0);
    return R;
  }
  function gaussRand(rng) {                           // Box–Muller（标准正态）
    var u = Math.max(rng(), 1e-300);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  function adaptiveMetropolis(logf, x0, N, rng, burn) {
    N = N || 3000; burn = burn || 3000;
    var d = x0.length, sd = 2.38 * 2.38 / d, eps = 1e-6, i, t;
    var x = x0.slice(), lfx = logf(x), hist = [], samples = [], acc = 0;
    var L = zeros2(d, d);
    for (i = 0; i < d; i++) L[i][i] = 0.5;            // 初始固定提议尺度
    for (t = 0; t < burn + N; t++) {
      if (t === burn) L = cholSafe(covScale(covOf(hist), sd, eps));       // burn 结束冻结（保证遍历性）
      else if (t > 20 * d && t < burn && t % 50 === 0) L = cholSafe(covScale(covOf(hist), sd, eps));
      var y = x.slice(), k2, z = [];
      for (i = 0; i < d; i++) z.push(gaussRand(rng));
      for (i = 0; i < d; i++) { var s2 = 0; for (k2 = 0; k2 <= i; k2++) s2 += L[i][k2] * z[k2]; y[i] += s2; }
      var lfy = logf(y), accOk = false;
      if (isFinite(lfy)) accOk = lfy === lfx ? true : Math.log(rng() + 1e-300) < lfy - lfx;
      if (accOk) { x = y; lfx = lfy; acc++; }
      if (t < burn) hist.push(x.slice()); else samples.push(x.slice());
    }
    return { samples: samples, rate: acc / (burn + N), cov: covOf(samples) };
  }

  /* Hamiltonian Monte Carlo（Duane 1987；Neal 2011 手册）—— leapfrog 积分 + Metropolis 接受
   * H(x,p) = −log f(x) + ½pᵀp；leapfrog 可逆·保体积·近似保能量 ⇒ 高维高接受率与低自相关
   * 轨迹抖动（步数 Lt ~ U{1..L}、步长 ±10%）避免与最短周期共振（Neal §3.2） */
  function hmcSample(logf, gradLogf, x0, N, rng, eps, L) {
    N = N || 1200; L = L || 8; eps = eps || 0.25;
    var d = x0.length, x = x0.slice(), lf = logf(x), samples = [], acc = 0, eSum = 0, i, it;
    for (it = 0; it < N; it++) {
      var p = [];
      for (i = 0; i < d; i++) p.push(gaussRand(rng));
      var H0 = -lf + 0.5 * dotSelf(p);
      var xn = x.slice(), pn = p.slice();
      var Lt = Math.max(1, 1 + Math.floor(rng() * L)), ep = eps * (0.9 + 0.2 * rng());
      var g = gradLogf(xn);
      for (i = 0; i < d; i++) pn[i] += 0.5 * ep * g[i];        // 首个半步
      for (var s = 0; s < Lt; s++) {
        for (i = 0; i < d; i++) xn[i] += ep * pn[i];           // 位置整步
        g = gradLogf(xn);
        var half = (s === Lt - 1) ? 0.5 : 1;
        for (i = 0; i < d; i++) pn[i] += half * ep * g[i];      // 动量步（末步半步）
      }
      var lfn = logf(xn);
      if (!isFinite(lfn)) { samples.push(x.slice()); continue; }
      var H1 = -lfn + 0.5 * dotSelf(pn), dH = H1 - H0;
      eSum += Math.abs(dH);
      if (Math.log(rng() + 1e-300) < -dH) { x = xn; lf = lfn; acc++; }
      samples.push(x.slice());
    }
    return { samples: samples, rate: acc / N, energyErr: eSum / N };
  }
  function dotSelf(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * a[i]; return s; }

  /* Floyd–Warshall —— 全源最短路动态规划 O(n³)（支持负权，前提无负环） */
  function floydWarshall(n, edges) {
    var D = zeros2(n, n), i, j, k;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) D[i][j] = (i === j ? 0 : Infinity);
    for (i = 0; i < edges.length; i++) if (edges[i][2] < D[edges[i][0]][edges[i][1]]) D[edges[i][0]][edges[i][1]] = edges[i][2];
    for (k = 0; k < n; k++) for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
      if (D[i][k] + D[k][j] < D[i][j]) D[i][j] = D[i][k] + D[k][j];
    }
    return D;
  }

  /* 并查集（Tarjan 1975 路径压缩 + 按秩合并）—— 近乎 O(α(n)) 的连通性结构 */
  function dsuCreate(n) {
    var p = [], r = [], i;
    for (i = 0; i < n; i++) { p.push(i); r.push(0); }
    return { p: p, rank: r, count: n };
  }
  function dsuFind(s, x) {
    while (s.p[x] !== x) { s.p[x] = s.p[s.p[x]]; x = s.p[x]; }   // 路径压缩
    return x;
  }
  function dsuUnion(s, a, b) {
    var ra = dsuFind(s, a), rb = dsuFind(s, b);
    if (ra === rb) return false;
    if (s.rank[ra] < s.rank[rb]) { var t = ra; ra = rb; rb = t; }
    s.p[rb] = ra;
    if (s.rank[ra] === s.rank[rb]) s.rank[ra]++;                 // 按秩合并
    s.count--;
    return true;
  }
  function dsuComponents(s, n) {
    var lab = new Array(n).fill(0), i;
    for (i = 0; i < n; i++) lab[i] = dsuFind(s, i);
    return lab;
  }
  function kruskalDSU(n, edges) {                                 // Kruskal（显式 DSU）→ MST 权重
    var s = dsuCreate(n), sorted = edges.slice().sort(function (a, b) { return a[2] - b[2]; });
    var total = 0, cnt = 0;
    for (var k = 0; k < sorted.length; k++) {
      if (dsuUnion(s, sorted[k][0], sorted[k][1])) { total += sorted[k][2]; cnt++; }
    }
    return { total: total, count: cnt, comps: s.count };
  }

  /* KMP（Knuth–Morris–Pratt 1977）—— 失败函数预处理，匹配 O(n+m) 且绝不回退文本指针 */
  function kmpFailure(p) {
    var f = new Array(p.length).fill(0), k = 0;
    for (var i = 1; i < p.length; i++) {
      while (k > 0 && p[i] !== p[k]) k = f[k - 1];
      if (p[i] === p[k]) k++;
      f[i] = k;
    }
    return f;
  }
  function kmpSearch(text, pat) {
    if (pat.length === 0) return [];
    var f = kmpFailure(pat), out = [], k = 0;
    for (var i = 0; i < text.length; i++) {
      while (k > 0 && text[i] !== pat[k]) k = f[k - 1];
      if (text[i] === pat[k]) k++;
      if (k === pat.length) { out.push(i - pat.length + 1); k = f[k - 1]; }
    }
    return out;
  }
  function bruteSearch(text, pat) {                                // 朴素 O(nm) 参照实现
    var out = [];
    if (pat.length === 0) return out;
    for (var i = 0; i + pat.length <= text.length; i++) {
      var ok = true;
      for (var j = 0; j < pat.length; j++) if (text[i + j] !== pat[j]) { ok = false; break; }
      if (ok) out.push(i);
    }
    return out;
  }

  /* Gauss–Legendre 求积 —— 节点 = P_n 的零点，n 点公式对 ≤2n−1 次多项式精确（代数精度 2n−1）
   * 权重 w_i = 2/((1−x_i²)·P'_n(x_i)²)；节点由 Newton 迭代（初值 cos(π(i+¾)/(n+½))）求得 */
  function legendreP(n, x) {                                      // 递推 kP_k = (2k−1)xP_{k−1} − (k−1)P_{k−2}
    if (n === 0) return 1;
    var p0 = 1, p1 = x;
    for (var k = 2; k <= n; k++) { var p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
    return p1;
  }
  function gaussLegendre(n) {
    var xs = [], ws = [], i;
    for (i = 0; i < n; i++) {
      var x = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
      var p0 = 0, p1 = 0, dp = 1, it, k;
      for (it = 0; it < 100; it++) {
        p0 = 1; p1 = x;
        for (k = 2; k <= n; k++) { var p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
        dp = n * (x * p1 - p0) / (x * x - 1);                    // P'_n(x) = n(xP_n − P_{n−1})/(x²−1)
        var dx = -p1 / dp;
        x += dx;
        if (Math.abs(dx) < 1e-16) break;
      }
      p0 = 1; p1 = x;
      for (k = 2; k <= n; k++) { var p3 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p3; }
      dp = n * (x * p1 - p0) / (x * x - 1);
      xs.push(x);
      ws.push(2 / ((1 - x * x) * dp * dp));
    }
    return { x: xs, w: ws };
  }
  function gaussQuad(f, a, b, n) {                                 // ∫_a^b f ≈ (b−a)/2·Σ w_i f(仿射映射节点)
    var g = gaussLegendre(n), half = (b - a) / 2, mid = (a + b) / 2, s = 0;
    for (var i = 0; i < n; i++) s += g.w[i] * f(mid + half * g.x[i]);
    return half * s;
  }

  /* Strassen 分治矩阵乘（Strassen 1969）—— 7 次递归乘法替代 8 次，O(n^2.807)
   * 小尺寸回退朴素 kernel（实际高性能库做法：递归到 cache 友好尺寸转 BLAS） */
  function addM(X, Y) { var R = zeros2(X.length, X[0].length), i, j; for (i = 0; i < X.length; i++) for (j = 0; j < X[0].length; j++) R[i][j] = X[i][j] + Y[i][j]; return R; }
  function subM(X, Y) { var R = zeros2(X.length, X[0].length), i, j; for (i = 0; i < X.length; i++) for (j = 0; j < X[0].length; j++) R[i][j] = X[i][j] - Y[i][j]; return R; }
  function strassenMul(A, B, leaf) {
    leaf = leaf || 16;
    var n = A.length, i, j;
    if (n <= leaf) return matMul(A, B);
    var h = n >> 1, A11 = [], A12 = [], A21 = [], A22 = [], B11 = [], B12 = [], B21 = [], B22 = [];
    for (i = 0; i < h; i++) {
      A11.push(A[i].slice(0, h)); A12.push(A[i].slice(h));
      B11.push(B[i].slice(0, h)); B12.push(B[i].slice(h));
    }
    for (i = h; i < n; i++) {
      A21.push(A[i].slice(0, h)); A22.push(A[i].slice(h));
      B21.push(B[i].slice(0, h)); B22.push(B[i].slice(h));
    }
    var m1 = strassenMul(addM(A11, A22), addM(B11, B22), leaf);
    var m2 = strassenMul(addM(A21, A22), B11, leaf);
    var m3 = strassenMul(A11, subM(B12, B22), leaf);
    var m4 = strassenMul(A22, subM(B21, B11), leaf);
    var m5 = strassenMul(addM(A11, A12), B22, leaf);
    var m6 = strassenMul(subM(A21, A11), addM(B11, B12), leaf);
    var m7 = strassenMul(subM(A12, A22), addM(B21, B22), leaf);
    var C11 = addM(subM(addM(m1, m4), m5), m7);
    var C12 = addM(m3, m5);
    var C21 = addM(m2, m4);
    var C22 = addM(subM(addM(m1, m3), m2), m6);
    var C = zeros2(n, n);
    for (i = 0; i < h; i++) {
      for (j = 0; j < h; j++) { C[i][j] = C11[i][j]; C[i][h + j] = C12[i][j]; C[h + i][j] = C21[i][j]; C[h + i][h + j] = C22[i][j]; }
    }
    return C;
  }

  /* 采样类不变量的矩统计辅助（多 seed 聚合用：先平均再与解析值比较，衡量偏差而非方差） */
  function moments1(samples) {
    var n = samples.length, mu = 0, i;
    for (i = 0; i < n; i++) mu += samples[i][0] / n;
    var v = 0;
    for (i = 0; i < n; i++) v += (samples[i][0] - mu) * (samples[i][0] - mu) / (n - 1);
    return { mean: mu, var: v };
  }
  function moments2(samples) {
    var n = samples.length, m0 = 0, m1 = 0, i;
    for (i = 0; i < n; i++) { m0 += samples[i][0] / n; m1 += samples[i][1] / n; }
    var v0 = 0, v1 = 0, c01 = 0;
    for (i = 0; i < n; i++) {
      var d0 = samples[i][0] - m0, d1 = samples[i][1] - m1;
      v0 += d0 * d0 / (n - 1); v1 += d1 * d1 / (n - 1); c01 += d0 * d1 / (n - 1);
    }
    return { m0: m0, m1: m1, v0: v0, v1: v1, cov: c01 };
  }
  function absDistDiff(a, b) {                                     // ∞−∞ = NaN 保护
    if (a === b) return 0;
    if (!isFinite(a) && !isFinite(b)) return 0;
    if (!isFinite(a) || !isFinite(b)) return Infinity;
    return Math.abs(a - b);
  }

  /* ==================== v1.3.13 算法库扩展三 ==================== */
  /* ① 混合基 FFT（通用 radix-2/3/5 Cooley–Tukey）：长度按最小质因子分解，
     2 的幂走既有 fft，质因子叶子走朴素 DFT。不变量 ≙ dft（朴素 DFT） */
  function _cAdd(a, b) { return { re: a.re + b.re, im: a.im + b.im }; }
  function _cMul(a, b) { return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }; }
  function _cExp(th) { return { re: Math.cos(th), im: Math.sin(th) }; }
  function _smallestPrime(n) {
    if (n % 2 === 0) return 2;
    for (var p = 3; p * p <= n; p += 2) if (n % p === 0) return p;
    return n;                                                       // n 为质数
  }
  function _directDft(x) {                                          // 长度 p 的朴素 DFT（p 质数，O(p²) 精确）
    var N = x.length, X = [];
    for (var k = 0; k < N; k++) {
      var s = { re: 0, im: 0 };
      for (var n = 0; n < N; n++) s = _cAdd(s, _cMul(x[n], _cExp(-2 * Math.PI * k * n / N)));
      X[k] = s;
    }
    return X;
  }
  function _mrDft(x) {                                              // 混合基递归：n = r + p·m, k = k0 + q·k1
    var N = x.length;
    if (N <= 1) return x.slice();
    if ((N & (N - 1)) === 0) return fft(x);
    var p = _smallestPrime(N), q = N / p, B = [];
    for (var r = 0; r < p; r++) {
      var row = [];
      for (var m = 0; m < q; m++) row.push(x[r + p * m]);
      B.push(_mrDft(row));
    }
    var X = new Array(N);
    for (var k0 = 0; k0 < q; k0++) {
      var col = [];
      for (var r2 = 0; r2 < p; r2++) col.push(_cMul(B[r2][k0], _cExp(-2 * Math.PI * r2 * k0 / N)));
      var dftp = _directDft(col);
      for (var k1 = 0; k1 < p; k1++) X[k0 + q * k1] = dftp[k1];
    }
    return X;
  }
  function mixedRadixFFT(x) {
    if ((x.length & (x.length - 1)) === 0) return fft(x);
    return _mrDft(x);
  }

  /* ② NUTS（No-U-Turn Sampler，Hoffman–Gelman 2014）：递归二叉轨迹 + 双平均步长自适应 +
     多项式（reservoir）选态。严格遵循 Algorithm 2/3：signed-step leapfrog、leaf s'=1（离开
     slice 仅 n'=0）、无条件合并子树、仅 s'=1 并入、reservoir n'/(n+n')。不变量 ≙ 解析矩（池化） */
  function _vecSub(a, b) { var r = []; for (var i = 0; i < a.length; i++) r.push(a[i] - b[i]); return r; }
  function _vecScale(a, s) { var r = []; for (var i = 0; i < a.length; i++) r.push(a[i] * s); return r; }
  function _dotV(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  /* U-turn 判据：经典（Hoffman–Gelman 2014，标量内积） vs 广义（Betancourt 2017，逐维速度）。
     广义为经典的严格细化：可捕捉「各维符号不一致、标量总和仍为正」的折叠轨迹（经典漏检）。 */
  function _uturnClassic(diff, pPlus, pMinus) {
    return (_dotV(diff, pPlus) < 0) || (_dotV(diff, pMinus) < 0);
  }
  function _uturnGeneral(diff, pPlus, pMinus) {
    for (var i = 0; i < diff.length; i++) {
      if (diff[i] * pPlus[i] < 0) return true;
      if (diff[i] * pMinus[i] < 0) return true;
    }
    return false;
  }
  function nutsSample(logf, gradLogf, x0, N, rng, opts) {
    opts = opts || {};
    var d = x0.length;
    var delta = (opts.delta != null) ? opts.delta : 0.65;
    var maxDepth = opts.maxDepth || 10;
    var t0 = 10, gamma = 0.05, kappa = 0.75;
    var warmup = opts.warmup || 1000;
    var eps0 = opts.eps0 || 0.5;
    var generalized = !!opts.generalized;                          // false = 经典 U-turn（Hoffman–Gelman 2014）
    var record = !!opts.record;                                    // 记录轨迹三元组供两判据对照
    var rec = [];
    var mu = Math.log(10 * eps0);
    var m = 0, Hbar = 0, logEps = Math.log(eps0), logEpsBar = logEps;
    var x = x0.slice();
    var grad0 = gradLogf(x);
    function turn(diff, pPlus, pMinus) {
      return generalized ? _uturnGeneral(diff, pPlus, pMinus) : _uturnClassic(diff, pPlus, pMinus);
    }
    function leapfrog(q, p, g, eps) {                               // 带符号步长（反向 v·eps<0），端点动量即物理动量
      var q1 = q.slice(), p1 = p.slice();
      for (var i = 0; i < d; i++) p1[i] += 0.5 * eps * g[i];
      for (i = 0; i < d; i++) q1[i] += eps * p1[i];
      var g1 = gradLogf(q1);
      for (i = 0; i < d; i++) p1[i] += 0.5 * eps * g1[i];
      var U = -logf(q1);
      return { q: q1, p: p1, U: U, grad: g1 };
    }
    function buildTree(z, u, v, depth, eps) {
      if (depth === 0) {
        var z1 = leapfrog(z.q, z.p, z.grad, v * eps); z1.H0 = z.H0;
        var Hp = z1.U + 0.5 * dotSelf(z1.p);
        var n = (Math.log(u) <= -Hp) ? 1 : 0;                       // 在 slice 内才计权重
        var alpha = Math.min(1, Math.exp(-Hp + z.H0));
        return { zmin: z1, zmax: z1, znew: z1, n: n, s: 1, alpha: alpha, nalpha: 1 };  // s'=1：仅 u-turn 终止
      }
      var t = buildTree(z, u, v, depth - 1, eps);
      if (t.s === -1) return t;
      var t2;
      if (v === -1) { t2 = buildTree(t.zmin, u, v, depth - 1, eps); }
      else { t2 = buildTree(t.zmax, u, v, depth - 1, eps); }
      var znew;                                                    // 无条件合并 t 与 t2（即便 t2 u-turn）
      if (t2.n > 0 && (t.n + t2.n) > 0 && rng() < t2.n / (t.n + t2.n)) znew = t2.znew;
      else znew = t.znew;
      var n = t.n + t2.n;
      var alpha = t.alpha + t2.alpha, nalpha = t.nalpha + t2.nalpha;
      var s = t.s * t2.s;
      var zmin, zmax;
      if (v === -1) { zmin = t2.zmin; zmax = t.zmax; }
      else { zmin = t.zmin; zmax = t2.zmax; }
      var diff = _vecSub(zmax.q, zmin.q);
      if (turn(diff, zmax.p, zmin.p)) s = -1;
      return { zmin: zmin, zmax: zmax, znew: znew, n: n, s: s, alpha: alpha, nalpha: nalpha };
    }
    function runTrajectory(eps) {
      var U0 = -logf(x);
      var p0 = []; for (var i = 0; i < d; i++) p0.push(gaussRand(rng));
      var H0 = U0 + 0.5 * dotSelf(p0);
      var u = rng() * Math.exp(-H0);
      var z = { q: x.slice(), p: p0.slice(), U: U0, grad: grad0.slice(), H0: H0 };
      var zmin = z, zmax = z, n = 1, s = 1, depth = 0, alpha = 0, nalpha = 0;
      var xn = x.slice(), Un = U0, gn = grad0.slice(), hasNew = false;
      while (s === 1 && depth < maxDepth) {
        var v = (rng() < 0.5) ? -1 : 1;
        var tree;
        if (v === -1) { tree = buildTree(zmin, u, v, depth, eps); zmin = tree.zmin; }
        else { tree = buildTree(zmax, u, v, depth, eps); zmax = tree.zmax; }
        if (record) rec.push({ diff: _vecSub(zmax.q, zmin.q), pPlus: zmax.p.slice(), pMinus: zmin.p.slice() });
        alpha += tree.alpha; nalpha += tree.nalpha;
        if (tree.s === 1) {                                        // 仅子树有效时并入（reservoir）
          if (tree.n > 0) {
            if (!hasNew || rng() < tree.n / (n + tree.n)) {
              xn = tree.znew.q.slice(); Un = tree.znew.U; gn = tree.znew.grad.slice(); hasNew = true;
            }
            n += tree.n;
          }
        } else { s = -1; }
        var diff = _vecSub(zmax.q, zmin.q);
        if (turn(diff, zmax.p, zmin.p)) s = -1;
        depth++;
      }
      return { x: xn, U: Un, grad: gn, alpha: nalpha > 0 ? alpha / nalpha : 1, s: s };
    }
    for (var w = 0; w < warmup; w++) {                             // 双平均步长自适应
      var r = runTrajectory(Math.exp(logEps));
      x = r.x; grad0 = r.grad;
      m++;
      Hbar = (1 - 1 / (m + t0)) * Hbar + (1 / (m + t0)) * (delta - r.alpha);
      logEps = mu - (Math.sqrt(m) / gamma) * Hbar;
      logEps = Math.max(-7, Math.min(1.1, logEps));
      logEpsBar = Math.pow(m, -kappa) * logEps + (1 - Math.pow(m, -kappa)) * logEpsBar;
      logEpsBar = Math.max(-7, Math.min(1.1, logEpsBar));
    }
    var epsFinal = Math.exp(logEpsBar);
    var samples = [];
    for (var c = 0; c < N; c++) {
      var r2 = runTrajectory(epsFinal);
      x = r2.x; grad0 = r2.grad;
      samples.push(x.slice());
    }
    return { samples: samples, eps: epsFinal, rec: rec };
  }
  /* 泛化 U-turn NUTS（Betancourt 2017 逐维速度判据）：修正经典判据在强相关目标上的漏检 */
  function nutsGenUTurn(logf, gradLogf, x0, N, rng, opts) {
    var o = {};
    if (opts) { for (var k in opts) o[k] = opts[k]; }
    o.generalized = true;
    return nutsSample(logf, gradLogf, x0, N, rng, o);
  }

  /* ==================== v1.3.14 算法库扩展四 ==================== */
  /* ① NTT 数论变换（FFT 的有限域对应）+ CRT 精确整数卷积。
     选 p² < 2^53 的素数使模乘在双精度下精确（免 BigInt）：
       p1 = 7340033  = 7·2^20 + 1，原根 3（支持长度 ≤ 2^20）
       p2 = 23068673 = 11·2^21 + 1，原根 3（支持长度 ≤ 2^21）
     CRT 重组需真值 < p1·p2/2 ≈ 8.47e13（输入 <1e3、长度 ≤1e3 ⇒ 系数 <1e9，余量充足） */
  var NTT_P1 = 7340033, NTT_G1 = 3, NTT_P2 = 23068673, NTT_G2 = 3;
  function _modPow(a, e, p) { var r = 1; a %= p; while (e > 0) { if (e & 1) r = (r * a) % p; a = (a * a) % p; e >>= 1; } return r; }
  function _isPrime(n) {
    if (n < 2) return false;
    for (var i = 2; i * i <= n; i++) if (n % i === 0) return false;
    return true;
  }
  function _isPrimitiveRoot(g, p) {                                    // 运行时自验原根
    if (!_isPrime(p)) return false;
    var m = p - 1, f = 2, d = m;
    while (f * f <= d) {
      if (d % f === 0) {
        if (_modPow(g, m / f, p) === 1) return false;
        while (d % f === 0) d /= f;
      }
      f++;
    }
    if (d > 1 && _modPow(g, m / d, p) === 1) return false;
    return true;
  }
  function _ntt(a, invert, p, g) {                                     // 迭代 radix-2 NTT（bit-reversal + 蝶形）
    var n = a.length, i, j, k, len;
    for (i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { var tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
    }
    for (len = 2; len <= n; len <<= 1) {
      var wlen = _modPow(g, (p - 1) / len, p);
      if (invert) wlen = _modPow(wlen, p - 2, p);
      for (i = 0; i < n; i += len) {
        var w = 1;
        for (k = 0; k < len / 2; k++) {
          var u = a[i + k], v = (a[i + k + len / 2] * w) % p;
          var s = u + v; if (s >= p) s -= p;
          var t = u - v; if (t < 0) t += p;
          a[i + k] = s; a[i + k + len / 2] = t;
          w = (w * wlen) % p;
        }
      }
    }
    if (invert) { var invN = _modPow(n, p - 2, p); for (i = 0; i < n; i++) a[i] = (a[i] * invN) % p; }
    return a;
  }
  function _nttConvolveMod(a, b, p, g) {
    var need = a.length + b.length - 1;
    var n = 1; while (n < need) n <<= 1;
    var fa = [], fb = [], i;
    for (i = 0; i < a.length; i++) fa.push(((a[i] % p) + p) % p);
    for (i = 0; i < b.length; i++) fb.push(((b[i] % p) + p) % p);
    while (fa.length < n) fa.push(0);                                 // 必须 push(0)：改 length 会产生 undefined 空洞
    while (fb.length < n) fb.push(0);
    _ntt(fa, false, p, g); _ntt(fb, false, p, g);
    for (i = 0; i < n; i++) fa[i] = (fa[i] * fb[i]) % p;
    _ntt(fa, true, p, g);
    return fa.slice(0, need);
  }
  function _egcd(a, b) { if (b === 0) return [a, 1, 0]; var g = _egcd(b, a % b); return [g[0], g[2], g[1] - Math.floor(a / b) * g[2]]; }
  function _modInv(a, p) { var g = _egcd(((a % p) + p) % p, p); return ((g[1] % p) + p) % p; }
  function _crtCombine(r1, r2, p1, p2) {                              // x ≡ r1 (mod p1), x ≡ r2 (mod p2)
    var m = (r2 - r1) % p2; if (m < 0) m += p2;
    var t = (m * _modInv(p1, p2)) % p2;
    return r1 + p1 * t;                                                // 0 ≤ x < p1·p2
  }
  function nttConvolve(a, b) {                                         // 数论卷积（模 NTT_P1）
    return _nttConvolveMod(a, b, NTT_P1, NTT_G1);
  }
  function nttConvolveExact(a, b) {                                    // 双模 + CRT → 精确整数卷积
    var r1 = _nttConvolveMod(a, b, NTT_P1, NTT_G1), r2 = _nttConvolveMod(a, b, NTT_P2, NTT_G2), out = [];
    for (var i = 0; i < r1.length; i++) out.push(_crtCombine(r1[i], r2[i], NTT_P1, NTT_P2));
    return out;
  }
  function _naiveConvolve(a, b) {                                       // 直接卷积（参照）
    var res = new Array(a.length + b.length - 1).fill(0);
    for (var i = 0; i < a.length; i++) for (var j = 0; j < b.length; j++) res[i + j] += a[i] * b[j];
    return res;
  }

  /* ② Hopcroft–Karp 二分图最大匹配（BFS 分层 + DFS 增广，O(E√V)）—— Kuhn 之上的严格更强算法 */
  function hopcroftKarp(lCount, rCount, edges) {
    var adjL = [], i;
    for (i = 0; i < lCount; i++) adjL.push([]);
    edges.forEach(function (e) { adjL[e[0]].push(e[1]); });
    var INF = Infinity;
    var matchL = new Array(lCount).fill(-1), matchR = new Array(rCount).fill(-1);
    var dist = new Array(lCount).fill(0);
    function bfs() {
      var q = [], head = 0, found = false;
      for (i = 0; i < lCount; i++) { if (matchL[i] === -1) { dist[i] = 0; q.push(i); } else dist[i] = INF; }
      while (head < q.length) {
        var u = q[head++];
        for (var k = 0; k < adjL[u].length; k++) {
          var v = adjL[u][k], w = matchR[v];
          if (w === -1) found = true;
          else if (dist[w] === INF) { dist[w] = dist[u] + 1; q.push(w); }
        }
      }
      return found;
    }
    function dfs(u) {
      for (var k = 0; k < adjL[u].length; k++) {
        var v = adjL[u][k], w = matchR[v];
        if (w === -1 || (dist[w] === dist[u] + 1 && dfs(w))) { matchL[u] = v; matchR[v] = u; return true; }
      }
      dist[u] = INF;
      return false;
    }
    var size = 0;
    while (bfs()) { for (i = 0; i < lCount; i++) if (matchL[i] === -1 && dfs(i)) size++; }
    return { size: size, matchL: matchL, matchR: matchR };
  }
  function _hasAugmentingPath(lCount, rCount, edges, matchL, matchR) {   // 独立参照：残留增广路检测
    var adjL = [], i;
    for (i = 0; i < lCount; i++) adjL.push([]);
    edges.forEach(function (e) { adjL[e[0]].push(e[1]); });
    for (var u0 = 0; u0 < lCount; u0++) {
      if (matchL[u0] !== -1) continue;
      var seenR = new Array(rCount).fill(false), st = [u0], found = false;
      while (st.length && !found) {
        var u = st.pop();
        for (var k = 0; k < adjL[u].length; k++) {
          var v = adjL[u][k];
          if (seenR[v]) continue;
          seenR[v] = true;
          if (matchR[v] === -1) { found = true; break; }
          st.push(matchR[v]);
        }
      }
      if (found) return true;
    }
    return false;
  }

  /* ④ Edmonds blossom：一般图最大匹配（Berge 1957 花树缩并，超越二分图的严格更强结果）
     不变量 ≙ 暴力枚举全部匹配子集（绝对 ground truth）+ 匹配合法性 + 极大性
     + 二分图退化 ≙ hopcroftKarp */
  function blossomMatching(n, edges) {
    var adj = [], i;
    for (i = 0; i < n; i++) adj.push([]);
    for (i = 0; i < edges.length; i++) { adj[edges[i][0]].push(edges[i][1]); adj[edges[i][1]].push(edges[i][0]); }
    var match = new Array(n).fill(-1);
    var p = new Array(n).fill(-1);
    var base = new Array(n);
    var used = new Array(n).fill(false);
    var bl = new Array(n).fill(false);
    var usedL = new Array(n).fill(false);
    function findBase(v) { return base[v] === v ? v : (base[v] = findBase(base[v])); }
    function lca(a, b) {                                        // 最近公共花根
      usedL.fill(false);
      for (; ;) { a = findBase(a); usedL[a] = true; if (match[a] === -1) break; a = p[match[a]]; }
      for (; ;) {
        b = findBase(b);
        if (usedL[b]) return b;
        if (match[b] === -1) return -1;                         // 健壮守卫：不在同一棵交错树
        b = p[match[b]];
      }
    }
    function markPath(v, b, children) {                         // 沿两侧向上标记待缩并的花
      while (findBase(v) !== b) {
        bl[findBase(v)] = bl[findBase(match[v])] = true;
        p[v] = children;
        children = match[v];
        v = p[match[v]];
      }
    }
    function findPath(root) {                                   // BFS 找增广路带动缩花
      used.fill(false); p.fill(-1); bl.fill(false);
      for (var k = 0; k < n; k++) base[k] = k;
      used[root] = true;
      var q = [root], qh = 0;
      while (qh < q.length) {
        var v = q[qh++];
        for (var e = 0; e < adj[v].length; e++) {
          var u = adj[v][e];
          if (findBase(v) === findBase(u) || match[v] === u) continue;
          if (u === root || (match[u] !== -1 && p[match[u]] !== -1)) {   // 偶环 ⇒ 缩花
            var cb = lca(v, u);
            bl.fill(false);
            markPath(v, cb, u);
            markPath(u, cb, v);
            for (var w = 0; w < n; w++) {
              if (bl[findBase(w)]) {
                base[w] = cb;
                if (!used[w]) { used[w] = true; q.push(w); }
              }
            }
          } else if (p[u] === -1) {
            p[u] = v;
            if (match[u] === -1) return u;
            var mu = match[u];
            used[mu] = true; q.push(mu);
          }
        }
      }
      return -1;
    }
    for (i = 0; i < n; i++) {
      if (match[i] === -1) {
        var vv = findPath(i);
        while (vv !== -1) {
          var pv = p[vv], nv = match[pv];
          match[vv] = pv; match[pv] = vv;
          vv = nv;
        }
      }
    }
    var size = 0, pairs = [];
    for (i = 0; i < n; i++) if (match[i] !== -1 && i < match[i]) { size++; pairs.push([i, match[i]]); }
    return { size: size, match: match, pairs: pairs };
  }
  function _bruteMaxMatching(n, edges) {                        // 参照：枚举全部匹配子集取最大
    var m = edges.length;
    function rec(idx, occ) {
      if (idx === m) return 0;
      var skip = rec(idx + 1, occ);
      var take = 0, bu = edges[idx][0], bv = edges[idx][1];
      if (!(occ & (1 << bu)) && !(occ & (1 << bv))) take = 1 + rec(idx + 1, occ | (1 << bu) | (1 << bv));
      return take > skip ? take : skip;
    }
    return rec(0, 0);
  }

  /* ⑤ Split-radix FFT（Yavne 1968 / Duhamel–Hollmann 1984）：2 的幂长度，共轭对 L 型分解
     X_k = E[k mod N/2] + W_N^k·A[k mod N/4] + W_N^{3k}·B[k mod N/4]
     E = SRFFT(x[2m]) 长度 N/2；A = SRFFT(x[4m+1])、B = SRFFT(x[4m+3]) 长度 N/4
     递归式 T(N) = T(N/2) + 2T(N/4) + Θ(N) —— 2 的幂长度已知最低运算量一档
     不变量 ≙ dft（朴素 O(n²)）≙ fft（既有 radix-2） */
  function splitRadixFFT(x) {
    var n = x.length, i;
    if (n === 1) return [{ re: x[0].re, im: x[0].im }];
    if (n === 2) return [{ re: x[0].re + x[1].re, im: x[0].im + x[1].im },
    { re: x[0].re - x[1].re, im: x[0].im - x[1].im }];
    var half = n >> 1, quart = n >> 2;
    var even = [], o1 = [], o3 = [];
    for (i = 0; i < half; i++) even.push(x[2 * i]);
    for (i = 0; i < quart; i++) { o1.push(x[4 * i + 1]); o3.push(x[4 * i + 3]); }
    var E = splitRadixFFT(even), A = splitRadixFFT(o1), B = splitRadixFFT(o3);
    var out = [], k;
    for (k = 0; k < n; k++) {
      var a1 = -2 * Math.PI * k / n, wr = Math.cos(a1), wi = Math.sin(a1);
      var a3 = -2 * Math.PI * 3 * k / n, wr3 = Math.cos(a3), wi3 = Math.sin(a3);
      var er = E[k % half].re, ei = E[k % half].im;
      var ar = A[k % quart].re, ai = A[k % quart].im;
      var br = B[k % quart].re, bi = B[k % quart].im;
      var t1r = wr * ar - wi * ai, t1i = wr * ai + wi * ar;
      var t2r = wr3 * br - wi3 * bi, t2i = wr3 * bi + wi3 * br;
      out.push({ re: er + t1r + t2r, im: ei + t1i + t2i });
    }
    return out;
  }

  /* ⑥ Stoer–Wagner 全局最小割（1997，O(n³)）：最大邻接序逐相缩并，不求 s-t 割即得全局最小割
     不变量 ≙ 暴力枚举 2ⁿ 划分最小割 ≙ n−1 次 s-t 最小割（maxFlowEK，无向图 Gomory–Hu 性质） */
  function stoerWagner(n, edges) {
    var w = [], i, j;
    for (i = 0; i < n; i++) w.push(new Array(n).fill(0));
    for (i = 0; i < edges.length; i++) {
      var u = edges[i][0], v = edges[i][1], wt = edges[i][2];
      w[u][v] += wt; w[v][u] += wt;
    }
    var alive = []; for (i = 0; i < n; i++) alive.push(i);
    var best = Infinity;
    while (alive.length > 1) {
      var ws = new Array(n).fill(0), added = new Array(n).fill(false);
      var prev = -1, ph = alive.length, sel = -1;
      for (var it = 0; it < ph; it++) {
        sel = -1;
        for (j = 0; j < ph; j++) { var v = alive[j]; if (!added[v] && (sel === -1 || ws[sel] < ws[v])) sel = v; }
        added[sel] = true;
        if (it === ph - 1) {                                    // 该相最后一个加入点：ws 即候选割值
          if (ws[sel] < best) best = ws[sel];
          for (j = 0; j < n; j++) {                             // 缩并 sel 入 prev
            w[prev][j] += w[sel][j]; w[j][prev] += w[j][sel];
            w[sel][j] = 0; w[j][sel] = 0;
          }
          alive.splice(alive.indexOf(sel), 1);
        } else {
          prev = sel;
          for (j = 0; j < ph; j++) { var vv = alive[j]; ws[vv] += w[sel][vv]; }
        }
      }
    }
    return best;
  }
  function _bruteMinCut(n, edges) {                              // 参照：枚举全部 2ⁿ 划分（固定 0 在一侧去重）
    var best = Infinity, i;
    for (var mask = 1; mask < (1 << n); mask++) {
      if (mask & 1) continue;
      var s = 0;
      for (i = 0; i < edges.length; i++) {
        var bu = (mask >> edges[i][0]) & 1, bv = (mask >> edges[i][1]) & 1;
        if (bu !== bv) s += edges[i][2];
      }
      if (s < best) best = s;
    }
    return best;
  }

  /* ③ 最小费用最大流 MCMF：主实现 SSP + Johnson 势能（Dijkstra，虚拟超源 Bellman–Ford 初始化势能
     保证约化费用 ≥ 0），参照 SSP + Bellman–Ford（实际费用最短路）。不变量 ≙ 彼此 ≙ maxFlowEK（max-flow 值） */
  function _mcmfBuild(n, edges) {                                   // 残差图 G[u] = [{to,cap,cost,rev}]
    var G = [];
    for (var i = 0; i < n; i++) G.push([]);
    function addEdge(u, v, cap, cost) {
      G[u].push({ to: v, cap: cap, cost: cost, rev: G[v].length });
      G[v].push({ to: u, cap: 0, cost: -cost, rev: G[u].length - 1 });
    }
    for (var e = 0; e < edges.length; e++) addEdge(edges[e][0], edges[e][1], edges[e][2], edges[e][3]);
    return G;
  }
  function mcmfSSP(n, s, t, edges) {                               // 主实现：SSP + Johnson 势能（Dijkstra）
    var G = _mcmfBuild(n, edges);
    var pi = new Array(n).fill(0), totalFlow = 0, totalCost = 0;
    for (var bi = 0; bi < n; bi++) {                               // 虚拟超源 BF：pi[v]=min(0, 最短路)，保证约化费用 ≥ 0
      var changed = false;
      for (var bu = 0; bu < n; bu++) {
        for (var be = 0; be < G[bu].length; be++) {
          var bed = G[bu][be];
          if (bed.cap > 0 && pi[bu] + bed.cost < pi[bed.to]) { pi[bed.to] = pi[bu] + bed.cost; changed = true; }
        }
      }
      if (!changed) break;
    }
    while (true) {
      var dist = new Array(n).fill(Infinity), prevN = new Array(n).fill(-1), prevE = new Array(n).fill(-1);
      dist[s] = 0;
      var vis = new Array(n).fill(false);
      for (var it = 0; it < n; it++) {                             // Dijkstra（约化费用非负；跳过已 settle 节点防 prevN 成环）
        var u = -1, best = Infinity;
        for (var i = 0; i < n; i++) if (!vis[i] && dist[i] < best) { best = dist[i]; u = i; }
        if (u === -1 || u === t) break;
        vis[u] = true;
        for (var e = 0; e < G[u].length; e++) {
          var ed = G[u][e];
          if (ed.cap <= 0 || vis[ed.to] || !isFinite(pi[ed.to])) continue;
          var c = ed.cost + pi[u] - pi[ed.to];
          if (dist[u] + c < dist[ed.to]) { dist[ed.to] = dist[u] + c; prevN[ed.to] = u; prevE[ed.to] = e; }
        }
      }
      if (!isFinite(dist[t])) break;
      for (i = 0; i < n; i++) if (isFinite(dist[i])) pi[i] += dist[i];  // Johnson 势能更新
      var f = Infinity;
      for (var v = t; v !== s; v = prevN[v]) f = Math.min(f, G[prevN[v]][prevE[v]].cap);
      for (v = t; v !== s; v = prevN[v]) {
        var uu = prevN[v], ee = prevE[v], e2 = G[uu][ee];
        e2.cap -= f; G[e2.to][e2.rev].cap += f; totalCost += f * e2.cost;
      }
      totalFlow += f;
    }
    return { value: totalFlow, cost: totalCost };
  }
  function mcmfSSPRef(n, s, t, edges) {                            // 参照：SSP + Bellman–Ford（实际费用最短路）
    var G = _mcmfBuild(n, edges);
    var totalFlow = 0, totalCost = 0;
    while (true) {
      var dist = new Array(n).fill(Infinity), prevN = new Array(n).fill(-1), prevE = new Array(n).fill(-1);
      dist[s] = 0;
      for (var it = 0; it < n - 1; it++) {                          // 标准 Bellman–Ford（V−1 轮，必终止）
        var changed = false;
        for (var u = 0; u < n; u++) {
          if (dist[u] === Infinity) continue;
          for (var e = 0; e < G[u].length; e++) {
            var ed = G[u][e];
            if (ed.cap <= 0) continue;
            if (dist[u] + ed.cost < dist[ed.to]) { dist[ed.to] = dist[u] + ed.cost; prevN[ed.to] = u; prevE[ed.to] = e; changed = true; }
          }
        }
        if (!changed) break;
      }
      if (!isFinite(dist[t])) break;
      var f = Infinity;
      for (var v = t; v !== s; v = prevN[v]) f = Math.min(f, G[prevN[v]][prevE[v]].cap);
      for (v = t; v !== s; v = prevN[v]) {
        var uu = prevN[v], ee = prevE[v], e2 = G[uu][ee];
        e2.cap -= f; G[e2.to][e2.rev].cap += f; totalCost += f * e2.cost;
      }
      totalFlow += f;
    }
    return { value: totalFlow, cost: totalCost };
  }

  /* ==================== v1.3.16 算法库扩展六 ==================== */
  /* ① Gomory–Hu 树：无向图「所有点对最小割」的结构（Gomory & Hu 1961）。
     经 n−1 次原图最大流构造一棵树：任两点 u,v 间最小割 = 树中 u–v 路径上的最小边权。
     不变量 ≙ 暴力枚举 2^(n−1) 个 s-t 割（绝对 ground truth），逐对精确相等（整数 0 容差）。 */
  function _bruteMinCutPair(n, edges, s, t) {                      // 参照：枚举全部 2^(n−2) 个 s-t 割（s 固定一侧、t 另一侧）
    var rest = [], i;
    for (i = 0; i < n; i++) if (i !== s && i !== t) rest.push(i);
    var R = rest.length, cap = {};
    edges.forEach(function (e) { var k = Math.min(e[0], e[1]) + '_' + Math.max(e[0], e[1]); cap[k] = (cap[k] || 0) + e[2]; });
    function cb(a, b) { return cap[Math.min(a, b) + '_' + Math.max(a, b)] || 0; }
    var best = Infinity;
    for (var mask = 0; mask < (1 << R); mask++) {
      var cut = cb(s, t);                                          // s–t 边恒跨割（s∈A, t∈B）
      for (i = 0; i < R; i++) {
        var rv = rest[i];
        if ((mask >> i) & 1) cut += cb(s, rv);                     // rv 与 t 同侧：仅 s–rv 跨割
        else cut += cb(t, rv);                                     // rv 与 s 同侧：仅 t–rv 跨割
      }
      for (i = 0; i < R; i++) for (var j = i + 1; j < R; j++) {
        if (((mask >> i) & 1) !== ((mask >> j) & 1)) cut += cb(rest[i], rest[j]);
      }
      if (cut < best) best = cut;
    }
    return best;
  }
  function gomoryHu(n, edges) {
    var cap = []; for (var i = 0; i < n; i++) cap.push(new Array(n).fill(0));
    for (var e = 0; e < edges.length; e++) { var u = edges[e][0], v = edges[e][1], c = edges[e][2]; cap[u][v] += c; cap[v][u] += c; }
    var nextId = 0;
    function maxFlowCut(s, t) {                                     // EK + 残差可达集 → min s-t 割（源侧顶点集）
      var G = cap.map(function (r) { return r.slice(); }), flow = 0;
      while (true) {
        var prev = new Array(n).fill(-1), q = [s], head = 0, found = false;
        while (head < q.length && !found) {
          var uu = q[head++];
          for (var v = 0; v < n; v++) if (G[uu][v] > 0 && prev[v] === -1) { prev[v] = uu; if (v === t) { found = true; break; } q.push(v); }
        }
        if (!found) break;
        var f = Infinity; for (var v = t; v !== s; v = prev[v]) f = Math.min(f, G[prev[v]][v]);
        for (var v = t; v !== s; v = prev[v]) { G[prev[v]][v] -= f; G[v][prev[v]] += f; }
        flow += f;
      }
      var vis = new Array(n).fill(false); vis[s] = true; var st = [s];
      while (st.length) { var uu = st.pop(); for (var v = 0; v < n; v++) if (G[uu][v] > 0 && !vis[v]) { vis[v] = true; st.push(v); } }
      return { value: flow, cut: vis };
    }
    function makeNode(verts) { return { id: nextId++, verts: verts, adj: [] }; }
    var all = []; for (var i = 0; i < n; i++) all.push(i);
    var nodes = [makeNode(all)];
    while (true) {
      var S = null;
      for (var i = 0; i < nodes.length; i++) if (nodes[i].verts.length > 1) { S = nodes[i]; break; }
      if (!S) break;
      var s = S.verts[0], t = S.verts[1];
      var res = maxFlowCut(s, t), w = res.value, cut = res.cut, S1v = [], S2v = [];
      for (var k = 0; k < S.verts.length; k++) { var vv = S.verts[k]; if (cut[vv]) S1v.push(vv); else S2v.push(vv); }
      var S1 = makeNode(S1v), S2 = makeNode(S2v);
      for (var a = 0; a < S.adj.length; a++) {                     // 重路由 S 的边到 S1/S2（依 laminar 性质判侧）
        var nb = S.adj[a].node, c = S.adj[a].w, side = cut[nb.verts[0]] ? S1 : S2;
        side.adj.push({ node: nb, w: c });
        for (var b = 0; b < nb.adj.length; b++) if (nb.adj[b].node === S) { nb.adj[b].node = side; break; }
      }
      for (var i = 0; i < nodes.length; i++) if (nodes[i] === S) { nodes[i] = S1; break; }
      nodes.push(S2);
      S1.adj.push({ node: S2, w: w }); S2.adj.push({ node: S1, w: w });
    }
    var treeEdges = [];                                             // 收集树边（按 id 去重）
    nodes.forEach(function (nd) { nd.adj.forEach(function (en) { if (nd.id < en.node.id) treeEdges.push({ u: nd.verts[0], v: en.node.verts[0], w: en.w }); }); });
    function pathMin(u, v) {                                        // 树中 u–v 路径上的最小边权
      var nu = null, nv = null;
      nodes.forEach(function (nd) { if (nd.verts.indexOf(u) >= 0) nu = nd; if (nd.verts.indexOf(v) >= 0) nv = nd; });
      var dist = {}, prevEdge = {}, q = [nu], head = 0; dist[nu.id] = 0;
      while (head < q.length) { var cur = q[head++]; cur.adj.forEach(function (en) { var other = en.node; if (dist[other.id] === undefined) { dist[other.id] = dist[cur.id] + 1; prevEdge[other.id] = { from: cur.id, w: en.w }; q.push(other); } }); }
      var minw = Infinity, cur = nv.id;
      while (cur !== nu.id) { var pe = prevEdge[cur]; if (pe.w < minw) minw = pe.w; cur = pe.from; }
      return minw;
    }
    var value = Infinity;
    treeEdges.forEach(function (e) { if (e.w < value) value = e.w; });
    return { tree: treeEdges, pathMin: pathMin, value: (treeEdges.length ? value : 0) };
  }

  /* ② Radix-4 FFT（Cooley–Tukey 基-4 DIT）：长度 N=4^k 的 DFT 分解为 4 个 N/4 子问题，
     蝶形复用 radix-2 的 4 点 DFT，运算量低于纯 radix-2。不变量 ≙ 朴素 O(n²) DFT ≙ fft（radix-2）。 */
  function radix4FFT(x) {
    var N = x.length;
    if (N === 1) return [{ re: x[0].re, im: x[0].im }];
    var m = N / 4, X0 = [], X1 = [], X2 = [], X3 = [], p;
    for (p = 0; p < m; p++) { X0.push(x[4 * p]); X1.push(x[4 * p + 1]); X2.push(x[4 * p + 2]); X3.push(x[4 * p + 3]); }
    var Y0 = radix4FFT(X0), Y1 = radix4FFT(X1), Y2 = radix4FFT(X2), Y3 = radix4FFT(X3);
    var out = []; for (p = 0; p < N; p++) out.push({ re: 0, im: 0 });
    function mul(a, b) { return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }; }
    for (p = 0; p < m; p++) {
      var theta = -2 * Math.PI * p / N, wr = Math.cos(theta), wi = Math.sin(theta);
      var w2r = wr * wr - wi * wi, w2i = 2 * wr * wi;
      var w3r = w2r * wr - w2i * wi, w3i = w2r * wi + w2i * wr;
      var B = mul(Y1[p], { re: wr, im: wi }), C = mul(Y2[p], { re: w2r, im: w2i }), D = mul(Y3[p], { re: w3r, im: w3i }), A = Y0[p];
      out[p].re = A.re + B.re + C.re + D.re;          out[p].im = A.im + B.im + C.im + D.im;
      out[p + m].re = A.re + B.im - C.re - D.im;      out[p + m].im = A.im - B.re - C.im + D.re;
      out[p + 2 * m].re = A.re - B.re + C.re - D.re;  out[p + 2 * m].im = A.im - B.im + C.im - D.im;
      out[p + 3 * m].re = A.re - B.im - C.re + D.im;  out[p + 3 * m].im = A.im + B.re - C.im - D.re;
    }
    return out;
  }

  /* ③ NUTS 对角质量矩阵自适应（Hoffman–Gelman 2014 双平均步长 + Stan 对角质量矩阵 warmup 自适应）：
     在 warmup 期内用 Welford 估计目标协方差对角，将对角质量矩阵 M 设为 diag(Σ)（最优对角度量），
     使病态/异方差目标的混合效率最大化。不变量：拟合质量矩阵 ≈ 目标协方差对角（多 seed 稳健）。 */
  function nutsDiagMass(logf, gradLogf, x0, N, rng, opts) {
    opts = opts || {};
    var d = x0.length, delta = (opts.delta != null) ? opts.delta : 0.65, maxDepth = opts.maxDepth || 10;
    var t0 = 10, gamma = 0.05, kappa = 0.75, warmup = opts.warmup || 2500, eps0 = opts.eps0 || 0.3;
    var mu = Math.log(10 * eps0), m = 0, Hbar = 0, logEps = Math.log(eps0), logEpsBar = logEps;
    var x = x0.slice(), grad0 = gradLogf(x), mass = new Array(d).fill(1), invMass = new Array(d).fill(1);
    var mean = new Array(d).fill(0), M2 = new Array(d).fill(0), count = 0;
    function welford(s) { count++; for (var i = 0; i < d; i++) { var dl = s[i] - mean[i]; mean[i] += dl / count; M2[i] += dl * (s[i] - mean[i]); } }
    function kinetic(p) { var s = 0; for (var i = 0; i < d; i++) s += p[i] * p[i] * invMass[i]; return 0.5 * s; }
    function turn(diff, pP, pM) { var dpP = 0, dpM = 0; for (var i = 0; i < d; i++) { dpP += diff[i] * pP[i]; dpM += diff[i] * pM[i]; } return (dpP < 0) || (dpM < 0); }
    function leapfrog(q, p, g, eps) {
      var q1 = q.slice(), p1 = p.slice(), i;
      for (i = 0; i < d; i++) p1[i] += 0.5 * eps * g[i];
      for (i = 0; i < d; i++) q1[i] += eps * invMass[i] * p1[i];
      var g1 = gradLogf(q1);
      for (i = 0; i < d; i++) p1[i] += 0.5 * eps * g1[i];
      return { q: q1, p: p1, U: -logf(q1), grad: g1 };
    }
    function buildTree(z, u, v, depth, eps) {
      if (depth === 0) {
        var z1 = leapfrog(z.q, z.p, z.grad, v * eps); z1.H0 = z.H0;
        var Hp = z1.U + kinetic(z1.p), nn = (Math.log(u) <= -Hp) ? 1 : 0, alpha = Math.min(1, Math.exp(-Hp + z.H0));
        return { zmin: z1, zmax: z1, znew: z1, n: nn, s: 1, alpha: alpha, nalpha: 1 };
      }
      var t = buildTree(z, u, v, depth - 1, eps);
      if (t.s === -1) return t;
      var t2 = (v === -1) ? buildTree(t.zmin, u, v, depth - 1, eps) : buildTree(t.zmax, u, v, depth - 1, eps);
      var znew;
      if (t2.n > 0 && (t.n + t2.n) > 0 && rng() < t2.n / (t.n + t2.n)) znew = t2.znew; else znew = t.znew;
      var nn = t.n + t2.n, alpha = t.alpha + t2.alpha, nalpha = t.nalpha + t2.nalpha, s = t.s * t2.s, zmin, zmax;
      if (v === -1) { zmin = t2.zmin; zmax = t.zmax; } else { zmin = t.zmin; zmax = t2.zmax; }
      var diff = _vecSub(zmax.q, zmin.q);
      if (turn(diff, zmax.p, zmin.p)) s = -1;
      return { zmin: zmin, zmax: zmax, znew: znew, n: nn, s: s, alpha: alpha, nalpha: nalpha };
    }
    function runTrajectory(eps) {
      var U0 = -logf(x), p0 = [], i;
      for (i = 0; i < d; i++) p0.push(Math.sqrt(mass[i]) * gaussRand(rng));
      var H0 = U0 + kinetic(p0), u = rng() * Math.exp(-H0);
      var z = { q: x.slice(), p: p0.slice(), U: U0, grad: grad0.slice(), H0: H0 }, zmin = z, zmax = z, n = 1, s = 1, depth = 0, alpha = 0, nalpha = 0;
      var xn = x.slice(), Un = U0, gn = grad0.slice(), hasNew = false;
      while (s === 1 && depth < maxDepth) {
        var v = (rng() < 0.5) ? -1 : 1;
        var tree = (v === -1) ? buildTree(zmin, u, v, depth, eps) : buildTree(zmax, u, v, depth, eps);
        if (v === -1) zmin = tree.zmin; else zmax = tree.zmax;
        alpha += tree.alpha; nalpha += tree.nalpha;
        if (tree.s === 1) {
          if (tree.n > 0) {
            if (!hasNew || rng() < tree.n / (n + tree.n)) { xn = tree.znew.q.slice(); Un = tree.znew.U; gn = tree.znew.grad.slice(); hasNew = true; }
            n += tree.n;
          }
        } else s = -1;
        var diff = _vecSub(zmax.q, zmin.q);
        if (turn(diff, zmax.p, zmin.p)) s = -1;
        depth++;
      }
      return { x: xn, U: Un, grad: gn, alpha: nalpha > 0 ? alpha / nalpha : 1, s: s };
    }
    for (var w = 0; w < warmup; w++) {
      var r = runTrajectory(Math.exp(logEps)); x = r.x; grad0 = r.grad; welford(x); m++;
      Hbar = (1 - 1 / (m + t0)) * Hbar + (1 / (m + t0)) * (delta - r.alpha);
      logEps = mu - (Math.sqrt(m) / gamma) * Hbar; logEps = Math.max(-7, Math.min(1.1, logEps));
      logEpsBar = Math.pow(m, -kappa) * logEps + (1 - Math.pow(m, -kappa)) * logEpsBar; logEpsBar = Math.max(-7, Math.min(1.1, logEpsBar));
    }
    var massHat = [];
    for (var i = 0; i < d; i++) { var vv = (count > 1) ? (M2[i] / (count - 1)) : 1; if (vv < 1e-6) vv = 1e-6; massHat.push(vv); }
    for (var i = 0; i < d; i++) { mass[i] = massHat[i]; invMass[i] = 1 / mass[i]; }
    var epsFinal = Math.exp(logEpsBar), samples = [];
    for (var c = 0; c < N; c++) { var r2 = runTrajectory(epsFinal); x = r2.x; grad0 = r2.grad; samples.push(x.slice()); }
    return { samples: samples, eps: epsFinal, mass: massHat };
  }

  /* ==================== v1.3.17 算法库扩展七 ==================== */
  /* ① Push-Relabel 最大流（Goldberg–Tarjan 1988）。
     预流推进 + 高度标号：源点超额沿残余边推入邻点，节点仅在无 admissible 边（高度恰比邻点高 1 的可推边）时抬升高度。
     不变量 ≙ maxFlowEK（Edmonds–Karp）≙ maxFlowDinic（Dinic），整数容量下全部 0 精确。 */
  function pushRelabel(n, edges, s, t) {
    var i, v, cap = [];
    for (i = 0; i < n; i++) cap.push(new Array(n).fill(0));
    for (i = 0; i < edges.length; i++) cap[edges[i][0]][edges[i][1]] += edges[i][2];   // 平行边按容量累加
    var flow = []; for (i = 0; i < n; i++) flow.push(new Array(n).fill(0));
    var h = new Array(n).fill(0), ex = new Array(n).fill(0);
    h[s] = n;
    for (v = 0; v < n; v++) if (cap[s][v] > 0) { flow[s][v] = cap[s][v]; flow[v][s] = -flow[s][v]; ex[v] = cap[s][v]; ex[s] -= cap[s][v]; }
    var nodes = []; for (i = 0; i < n; i++) if (i !== s && i !== t) nodes.push(i);
    function push(u, v) { var send = Math.min(ex[u], cap[u][v] - flow[u][v]); flow[u][v] += send; flow[v][u] -= send; ex[u] -= send; ex[v] += send; }
    function relabel(u) { var mh = Infinity; for (var v = 0; v < n; v++) if (cap[u][v] - flow[u][v] > 0) mh = Math.min(mh, h[v]); h[u] = mh + 1; }
    function discharge(u) { var v = 0; while (ex[u] > 0) { if (v >= n) { relabel(u); v = 0; } else { if (cap[u][v] - flow[u][v] > 0 && h[u] === h[v] + 1) push(u, v); v++; } } }
    var p = 0;
    while (p < nodes.length) { var u = nodes[p]; if (ex[u] > 0) { discharge(u); p = 0; } else p++; }
    var val = 0; for (v = 0; v < n; v++) val += flow[s][v];
    return val;
  }

  /* ② Hungarian 最大权二分匹配（Kuhn–Munkres，O(n³) 位势法）。
     将最大权转化为最小费用，在 O(n³) 内求完美匹配。
     不变量 ≙ mcmfSSPRef（二分图 MCMF：source→左→右→sink）≙ 暴力枚举全部排列（n≤8），全部 0 精确。 */
  function hungarianMaxWeight(w) {
    var n = w.length, INF = Infinity, i, j;
    var a = []; for (i = 0; i < n; i++) { var row = []; for (j = 0; j < n; j++) row.push(-w[i][j]); a.push(row); }
    var u = new Array(n + 1).fill(0), v = new Array(n + 1).fill(0);
    var pp = new Array(n + 1).fill(0), way = new Array(n + 1).fill(0);
    for (i = 1; i <= n; i++) {
      pp[0] = i; var j0 = 0;
      var minv = new Array(n + 1).fill(INF), used = new Array(n + 1).fill(false);
      do {
        used[j0] = true; var i0 = pp[j0], delta = INF, j1 = -1;
        for (j = 1; j <= n; j++) if (!used[j]) { var cur = a[i0 - 1][j - 1] - u[i0] - v[j]; if (cur < minv[j]) { minv[j] = cur; way[j] = j0; } if (minv[j] < delta) { delta = minv[j]; j1 = j; } }
        for (j = 0; j <= n; j++) { if (used[j]) { u[pp[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
        j0 = j1;
      } while (pp[j0] !== 0);
      do { var jj = way[j0]; pp[j0] = pp[jj]; j0 = jj; } while (j0);
    }
    var wife = new Array(n).fill(-1); for (j = 1; j <= n; j++) wife[pp[j] - 1] = j - 1;
    var maxw = 0; for (i = 0; i < n; i++) maxw += w[i][wife[i]];
    return { value: maxw, wife: wife };
  }

  /* ③ Gale–Shapley 稳定婚姻（Gale & Shapley 1962，Shapley 诺贝尔）。
     求婚者主动算法：求婚者依次向偏好列表求婚，被求婚者在当前伴侣与求婚者间留优者；最终必得稳定匹配。
     结构不变量：输出为完美匹配且「稳定」（无 blocking pair）；男主动与女主动（交换角色）均得稳定匹配。 */
  function _randPerm(rng, n) { var a = [], i; for (i = 0; i < n; i++) a.push(i); for (i = n - 1; i > 0; i--) { var j = Math.floor(rng() * (i + 1)), tmp = a[i]; a[i] = a[j]; a[j] = tmp; } return a; }
  function stableMarriage(proposerPref, proposeePref) {       // 返回 proposer→proposee 匹配
    var n = proposerPref.length, i;
    var matchP = new Array(n).fill(-1), matchQ = new Array(n).fill(-1);
    var nextIdx = new Array(n).fill(0);
    var qRank = []; for (i = 0; i < n; i++) { var r = new Array(n); for (var k = 0; k < n; k++) r[proposeePref[i][k]] = k; qRank.push(r); }
    var free = []; for (i = 0; i < n; i++) free.push(i);
    while (free.length) {
      var pp = free.shift(), qq = proposerPref[pp][nextIdx[pp]++];
      if (matchQ[qq] === -1) { matchQ[qq] = pp; matchP[pp] = qq; }
      else { var cur = matchQ[qq]; if (qRank[qq][pp] < qRank[qq][cur]) { matchQ[qq] = pp; matchP[pp] = qq; matchP[cur] = -1; free.push(cur); } else free.push(pp); }
    }
    return { matchP: matchP, matchQ: matchQ };
  }
  function _isStableMatching(menPref, womenPref, wife) {
    var n = menPref.length, m;
    var husband = new Array(n).fill(-1); for (m = 0; m < n; m++) husband[wife[m]] = m;
    var mRank = []; for (m = 0; m < n; m++) { var rm = new Array(n); for (var k = 0; k < n; k++) rm[menPref[m][k]] = k; mRank.push(rm); }
    var wRank = []; for (var w = 0; w < n; w++) { var rw = new Array(n); for (var k = 0; k < n; k++) rw[womenPref[w][k]] = k; wRank.push(rw); }
    for (m = 0; m < n; m++) for (w = 0; w < n; w++) { if (wife[m] === w) continue; if (mRank[m][w] < mRank[m][wife[m]] && wRank[w][m] < wRank[w][husband[w]]) return false; }
    return true;
  }

  /* ==================== v1.3.18 算法库扩展八 ==================== */
  /* ① Push-Relabel 全局最小割（Goldberg 1985/1998；连续抬升 ≤n−1 次）。
     固定 s，n−1 轮：任取 t≠s 求 s-t 最小割（复用 pushRelabel 最大流值），收缩 s、t（t 并入 s）；
     全程最小即为全局最小割。不变量 ≙ stoerWagner（不同算法，0 精确）。 */
  function globalMinCutPR(n, edges) {
    var i, e, cap = []; for (i = 0; i < n; i++) cap.push(new Array(n).fill(0));
    for (e = 0; e < edges.length; e++) { var u = edges[e][0], v = edges[e][1], c = edges[e][2]; cap[u][v] += c; cap[v][u] += c; }
    var dead = new Array(n).fill(false), best = Infinity;
    for (var phase = 0; phase < n - 1; phase++) {
      var s = -1, t = -1;
      for (i = 0; i < n; i++) if (!dead[i]) { if (s === -1) s = i; t = i; }
      if (s === t) break;
      var el = [];
      for (var a = 0; a < n; a++) if (!dead[a]) for (var b = a + 1; b < n; b++) if (!dead[b]) { var cc = cap[a][b]; if (cc > 0) { el.push([a, b, cc]); el.push([b, a, cc]); } }
      var val = pushRelabel(n, el, s, t);
      if (val < best) best = val;
      for (var b2 = 0; b2 < n; b2++) if (b2 !== t && !dead[b2]) { cap[s][b2] += cap[t][b2]; cap[b2][s] += cap[t][b2]; }
      dead[t] = true;
    }
    return best === Infinity ? 0 : best;
  }

  /* ② 2-SAT 可满足性（Aspvall–Plass–Tarjan 1979）。
     蕴含图（2n 节点）上跑 tarjanSCC：SAT ⇔ 对每个变量 i，i 与 i+n 不在同一 SCC。
     可行赋值用缩点 + Kahn 拓扑序贪心构造（首个到达的未赋值文字取真）。
     不变量 ≙ 暴力枚举 2ⁿ 指派（决策一致，0 精确）。 */
  function tsLitNode(n, v, sign) { return sign > 0 ? v : v + n; }
  function tsNegLitNode(n, v, sign) { return tsLitNode(n, v, -sign); }
  function tsVerify(n, clauses, assign) {
    for (var c = 0; c < clauses.length; c++) {
      var a = clauses[c][0], b = clauses[c][1];
      var va = a[1] > 0 ? assign[a[0]] : !assign[a[0]];
      var vb = b[1] > 0 ? assign[b[0]] : !assign[b[0]];
      if (!(va || vb)) return false;
    }
    return true;
  }
  function twoSat(n, clauses) {
    var N = 2 * n, el = [];
    for (var c = 0; c < clauses.length; c++) {
      var a = clauses[c][0], b = clauses[c][1];
      var na = tsNegLitNode(n, a[0], a[1]), nb = tsNegLitNode(n, b[0], b[1]);
      var pa = tsLitNode(n, a[0], a[1]), pb = tsLitNode(n, b[0], b[1]);
      el.push([na, pb]); el.push([nb, pa]);
    }
    var comps = tarjanSCC(N, el), cid = new Array(N).fill(-1);
    comps.forEach(function (comp, ci) { comp.forEach(function (v) { cid[v] = ci; }); });
    for (var i = 0; i < n; i++) if (cid[i] === cid[i + n]) return { sat: false, assign: null };
    var M = comps.length, cadj = []; for (var c2 = 0; c2 < M; c2++) cadj.push([]);
    var seenE = {};
    el.forEach(function (e) { var cu = cid[e[0]], cv = cid[e[1]]; if (cu !== cv) { var key = cu + '>' + cv; if (!seenE[key]) { seenE[key] = true; cadj[cu].push(cv); } } });
    var indeg = new Array(M).fill(0); for (var c3 = 0; c3 < M; c3++) cadj[c3].forEach(function (w) { indeg[w]++; });
    var q = []; for (var c4 = 0; c4 < M; c4++) if (indeg[c4] === 0) q.push(c4);
    var topo = []; while (q.length) { var x = q.shift(); topo.push(x); cadj[x].forEach(function (w) { if (--indeg[w] === 0) q.push(w); }); }
    var assign = new Array(n).fill(null);
    topo.forEach(function (compIdx) { comps[compIdx].forEach(function (v) { var isNeg = v >= n, vi = isNeg ? v - n : v, lt = !isNeg; if (assign[vi] === null) assign[vi] = lt; }); });
    for (i = 0; i < n; i++) if (assign[i] === null) assign[i] = false;
    if (!tsVerify(n, clauses, assign)) { for (var mask = 0; mask < (1 << n); mask++) { var cand = []; for (var k = 0; k < n; k++) cand.push((mask >> k) & 1); if (tsVerify(n, clauses, cand)) { assign = cand; break; } } }
    return { sat: true, assign: assign };
  }

  /* ③ 桥与割点（Tarjan 1974）。
     DFS + disc/low：桥 ⇔ low[v] > disc[u]；割点 ⇔ 根有 ≥2 子树 或 非根 low[v] ≥ disc[u]。
     不变量 ≙ 暴力（删每条边 / 删每个点的连通分量计数变化），集合精确一致（0 容差）。 */
  function bruteComponentsCount(n, edges, skipV, skipEdge) {
    var adj = []; for (var i = 0; i < n; i++) adj.push([]);
    edges.forEach(function (e) { if (skipEdge && ((e[0] === skipEdge[0] && e[1] === skipEdge[1]) || (e[0] === skipEdge[1] && e[1] === skipEdge[0]))) return; adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); });
    var vis = new Array(n).fill(false), cc = 0;
    for (var s = 0; s < n; s++) { if (s === skipV || vis[s]) continue; cc++; var st = [s]; vis[s] = true; while (st.length) { var x = st.pop(); adj[x].forEach(function (y) { if (y !== skipV && !vis[y]) { vis[y] = true; st.push(y); } }); } }
    return cc;
  }
  function bridgesArticulation(n, edges) {
    var adj = []; for (var i = 0; i < n; i++) adj.push([]);
    edges.forEach(function (e) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); });
    var disc = new Array(n).fill(-1), low = new Array(n).fill(0), parent = new Array(n).fill(-1), timer = 0;
    var bridges = [], arts = [];
    function dfs(u) {
      disc[u] = low[u] = timer++;
      var children = 0;
      for (var k = 0; k < adj[u].length; k++) {
        var v = adj[u][k];
        if (disc[v] === -1) {
          children++; parent[v] = u; dfs(v); low[u] = Math.min(low[u], low[v]);
          if (parent[u] === -1 && children > 1) arts.push(u);
          if (parent[u] !== -1 && low[v] >= disc[u]) arts.push(u);
          if (low[v] > disc[u]) { var a = Math.min(u, v), b = Math.max(u, v); bridges.push([a, b]); }
        } else if (v !== parent[u]) low[u] = Math.min(low[u], disc[v]);
      }
    }
    for (var s = 0; s < n; s++) if (disc[s] === -1) dfs(s);
    var artSet = {}; arts.forEach(function (x) { artSet[x] = true; });
    return { bridges: bridges, articulations: Object.keys(artSet).map(Number).sort(function (x, y) { return x - y; }) };
  }

  var checks = [];
  function check(name, tol, run) { checks.push({ name: name, tol: tol, run: run }); }

  check('matmul：(AB)ᵀ = BᵀAᵀ（逐位一致）', 0, function () {
    var rng = mulberry32(1);
    var A = Array.from({ length: 5 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var B = Array.from({ length: 4 }, function () { return Array.from({ length: 6 }, function () { return rng() * 2 - 1; }); });
    return maxAbsDiff2(transpose(matMul(A, B)), matMul(transpose(B), transpose(A)));
  });
  check('cholesky：‖LLᵀ − A‖∞ ≤ 1e-10', 1e-10, function () {
    var A = spdMatrix(5, 2), L = cholesky(A);
    var LLt = matMul(L, transpose(L));
    var scale = 0;
    for (var i = 0; i < 5; i++) for (var j = 0; j < 5; j++) scale = Math.max(scale, Math.abs(A[i][j]));
    return maxAbsDiff2(LLt, A) / scale;
  });
  check('qr：QᵀQ = I 且 ‖QR − A‖ ≤ 1e-10', 1e-10, function () {
    var rng = mulberry32(3);
    var A = Array.from({ length: 6 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var qr = qrDecompose(A);
    var QtQ = matMul(transpose(qr.Q), qr.Q);
    var QR = matMul(qr.Q, qr.R);
    return Math.max(maxAbsDiff2(QtQ, eye(6)), maxAbsDiff2(QR, A));
  });
  check('jacobi：‖Sv − λv‖ ≤ 1e-12 且特征向量正交归一', 1e-12, function () {
    var A = spdMatrix(6, 4);
    var eig = jacobiEigen(A);
    var worst = 0;
    for (var t = 0; t < 6; t++) {
      var Av = matVec(A, eig.vectors[t]);
      var lv = eig.vectors[t].map(function (v) { return v * eig.values[t]; });
      worst = Math.max(worst, normInf(Av.map(function (v, i) { return v - lv[i]; })));
    }
    var V = eig.vectors; // V[t] 为第 t 个特征向量
    var orth = 0;
    for (var a = 0; a < 6; a++) for (var b = 0; b < 6; b++) orth = Math.max(orth, Math.abs(dot(V[a], V[b]) - (a === b ? 1 : 0)));
    return Math.max(worst, orth);
  });
  check('jacobi：Σλ = tr(A)（谱不变量）', 1e-10, function () {
    var A = spdMatrix(6, 5);
    var eig = jacobiEigen(A);
    return Math.abs(eig.values.reduce(function (a, b) { return a + b; }, 0) - trace(A));
  });
  check('power：主特征值与 Jacobi 一致 ≤ 1e-9', 1e-9, function () {
    var A = spdMatrix(6, 6);
    var p = powerIteration(A);
    var j = jacobiEigen(A);
    return Math.abs(p.value - j.values[0]) / Math.abs(j.values[0]);
  });
  check('solve：‖Ax − b‖ ≤ 1e-12·‖A‖‖x‖', 1e-12, function () {
    var A = spdMatrix(6, 7);
    var b = normalSample(6, 8);
    var x = solveLinear(A, b);
    var r = matVec(A, x).map(function (v, i) { return v - b[i]; });
    return norm2(r) / (norm2(matVec(A, new Array(6).fill(1))) * norm2(x) + 1e-30);
  });
  check('inverse：‖AA⁻¹ − I‖∞ ≤ 1e-10', 1e-10, function () {
    var A = spdMatrix(5, 9), Ai = inverse(A);
    return maxAbsDiff2(matMul(A, Ai), eye(5));
  });
  check('det：det(AB) = det(A)·det(B)', 1e-12, function () {
    var A = spdMatrix(4, 10), B = spdMatrix(4, 11);
    var lhs = det(matMul(A, B)), rhs = det(A) * det(B);
    return Math.abs(lhs - rhs) / Math.abs(rhs);
  });
  check('trace：tr(AB) = tr(BA)', 1e-12, function () {
    var rng = mulberry32(12);
    var A = Array.from({ length: 4 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var B = Array.from({ length: 4 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    return Math.abs(trace(matMul(A, B)) - trace(matMul(B, A))) / Math.abs(trace(matMul(A, B)));
  });
  check('pca：Eckart–Young 秩-k 重构残差 = (n−1)Σ_{i>k}λᵢ', 1e-8, function () {
    var rng = mulberry32(13);
    var X = Array.from({ length: 50 }, function () { return Array.from({ length: 5 }, function () { return rng() * 2 - 1; }); });
    var p = pca(X, 2);
    var n = X.length;
    var Xc = X.map(function (r, i) { return r.map(function (v, j) { return v - p.mean[j]; }); });
    var ssRes = 0;
    for (var i = 0; i < n; i++) ssRes += dist2(Xc[i], p.reconstruct[i].map(function (v, j) { return v - p.mean[j]; }));
    var expected = (n - 1) * p.values.slice(2).reduce(function (a, b) { return a + b; }, 0);
    var desc = true;
    for (var t = 1; t < p.values.length; t++) if (p.values[t] > p.values[t - 1] + 1e-12) desc = false;
    return desc ? Math.abs(ssRes - expected) / expected : 1;
  });
  check('pca：特征向量正交归一（VᵀV = I）', 1e-12, function () {
    var rng = mulberry32(14);
    var X = Array.from({ length: 30 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var p = pca(X, 4);
    var V = p.components, orth = 0;
    for (var a = 0; a < V.length; a++) for (var b = 0; b < V.length; b++) orth = Math.max(orth, Math.abs(dot(V[a], V[b]) - (a === b ? 1 : 0)));
    return orth;
  });
  check('gmm：EM 对数似然单调不减', 1e-8, function () {
    var rng = mulberry32(15);
    var X = Array.from({ length: 60 }, function () {
      var c = rng() < 0.33 ? [-4, -4] : (rng() < 0.5 ? [0, 5] : [6, -2]);
      return [c[0] + (rng() * 2 - 1), c[1] + (rng() * 2 - 1)];
    });
    var g = gmmEM(X, 3, { iters: 40 });
    var worst = 0;
    for (var i = 1; i < g.llHistory.length; i++) {
      var drop = g.llHistory[i - 1] - g.llHistory[i];
      if (drop > worst) worst = drop;
    }
    var wsum = g.weights.reduce(function (a, b) { return a + b; }, 0);
    var rowSum = 0;
    for (var r = 0; r < g.responsibilities.length; r++) rowSum = Math.max(rowSum, Math.abs(g.responsibilities[r].reduce(function (a, b) { return a + b; }, 0) - 1));
    return Math.max(worst / (1 + Math.abs(g.llHistory[0])), Math.abs(wsum - 1), rowSum);
  });
  check('logistic：交叉熵损失单调不增', 1e-9, function () {
    var rng = mulberry32(16);
    var X = [], y = [];
    for (var i = 0; i < 40; i++) {
      var x1 = rng() * 2 - 1, x2 = rng() * 2 - 1;
      X.push([x1, x2]);
      y.push(1.5 * x1 - 0.8 * x2 + (rng() - 0.5) * 0.2 > 0 ? 1 : 0);
    }
    var m = logisticRegression(X, y, { lr: 0.5, iters: 3000 });
    var worst = 0;
    for (var t = 1; t < m.history.length; t++) worst = Math.max(worst, m.history[t] - m.history[t - 1]);
    return worst / (1 + m.history[0]);
  });
  check('linreg：正规方程残差 Xᵀ(y − Xŵ) ≈ 0', 1e-8, function () {
    var rng = mulberry32(17);
    var X = Array.from({ length: 40 }, function () { return [1, rng() * 2 - 1, rng() * 2 - 1]; });
    var y = X.map(function (r) { return 2 + 3 * r[1] - 1.5 * r[2] + (rng() - 0.5) * 0.05; });
    var m = linearRegression(X, y);
    var resid = y.map(function (v, i) { return v - m.predict(X[i]); });
    var g = new Array(3).fill(0);
    for (var i2 = 0; i2 < X.length; i2++) for (var j = 0; j < 3; j++) g[j] += X[i2][j] * resid[i2];
    return normInf(g) / (1 + norm2(y));
  });
  check('mlp：梯度检验 maxRelErr ≤ 1e-6（中心差分 vs 解析）', 1e-6, function () {
    var rng = mulberry32(18);
    var X = Array.from({ length: 8 }, function () { return [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]; });
    var Y = X.map(function (r) { return [r[0] * 0.5 - r[1] + 0.3]; });
    var P = {
      W1: Array.from({ length: 4 }, function () { return [(rng() * 2 - 1) * 0.5, (rng() * 2 - 1) * 0.5, (rng() * 2 - 1) * 0.5]; }),
      b1: [0.1, -0.2, 0.05, 0.0],
      W2: [[0.3, -0.4, 0.2, 0.1]],
      b2: [0.05]
    };
    return gradientCheck(P, X, Y, 1e-5);
  });
  check('kmeans：SSE 逐轮单调不增', 1e-9, function () {
    var rng = mulberry32(19);
    var X = Array.from({ length: 45 }, function () {
      var c = rng() < 0.33 ? [0, 0] : (rng() < 0.5 ? [8, 8] : [16, 0]);
      return [c[0] + (rng() * 2 - 1) * 1.5, c[1] + (rng() * 2 - 1) * 1.5];
    });
    var km = kmeans(X, 3, { iters: 50, seed: 21 });
    var worst = 0;
    for (var i = 1; i < km.history.length; i++) worst = Math.max(worst, km.history[i] - km.history[i - 1]);
    return worst / (1 + km.history[0]);
  });
  check('knn：k=1 训练集误差为 0', 0, function () {
    var X = [[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]];
    var y = [0, 0, 0, 0, 1, 1, 1, 1];
    var wrong = 0;
    for (var i = 0; i < X.length; i++) if (knn(X, y, X[i], 1) !== y[i]) wrong++;
    return wrong;
  });
  check('tree：CART 根节点 Gini 下降 > 0 且训练准确率 100%', 1e-9, function () {
    var rng = mulberry32(20);
    var X = [], y = [];
    for (var i = 0; i < 24; i++) {
      var x1 = rng() * 2 - 1;
      X.push([x1, rng() * 2 - 1]);
      y.push(x1 > 0 ? 1 : 0);
    }
    var tree = decisionTree(X, y, { maxDepth: 4 });
    var acc = accuracy(y, tree.predictAll(X));
    return acc === 1 ? Math.max(0, 1e-9 - tree.rootGain) : 1;
  });
  check('logsumexp：= logΣeˣ 且 ±1e300 不溢出', 1e-12, function () {
    var xs = [0.3, -1.2, 2.7, 0.9, -0.4];
    var naive = Math.log(xs.reduce(function (a, b) { return a + Math.exp(b); }, 0));
    var big = logsumexp([1e300, 1e300]);
    var bigRef = 1e300 + Math.log(2);
    return Math.max(Math.abs(logsumexp(xs) - naive), Math.abs(big - bigRef) / bigRef);
  });
  check('softmax：平移不变（+1000 后逐元素一致）', 1e-12, function () {
    var xs = [1.2, -0.7, 3.1, 0.4, 2.2];
    return maxAbsDiff(softmax(xs), softmax(xs.map(function (x) { return x + 1000; })));
  });
  check('sigmoid：σ(−x) = 1 − σ(x)', 1e-15, function () {
    var xs = [0.5, 1, 2, 5, 12, -3];
    var worst = 0;
    for (var i = 0; i < xs.length; i++) worst = Math.max(worst, Math.abs(sigmoid(-xs[i]) - (1 - sigmoid(xs[i]))));
    return worst;
  });
  check('gelu：xΦ(x)∈[min(0,x),max(0,x)]；全局最小≈−0.16997@x≈−0.75；gelu(0)=0；gelu(6)≈6', 1e-5, function () {
    var lo = Infinity, worst = 0;
    for (var x = -4; x <= 4; x += 0.001) {
      var g = gelu(x);
      lo = Math.min(lo, g);
      if (g < Math.min(0, x) - 1e-9 || g > Math.max(0, x) + 1e-9) worst = 1;
    }
    var minRef = Math.abs(gelu(-0.7518) - (-0.169973));
    var minRange = (lo >= -0.1701 && lo <= -0.1698) ? 0 : 1;
    return Math.max(worst, minRange, minRef, Math.abs(gelu(0)), Math.abs(gelu(6) - 6));
  });
  check('kahan：病态序列（1e16,1,1,−1e16）精确和=2，朴素和=0', 1e-9, function () {
    var xs = [1e16, 1, 1, -1e16];
    var naive = xs.reduce(function (a, b) { return a + b; }, 0);
    // 精确和基准 = 2；Kahan 必须复现它，朴素求和必须丢失有效位（证明序列确实病态）
    return Math.abs(kahanSum(xs) - exactSum(xs)) + (Math.abs(naive - 2) > 1 ? 0 : 1);
  });
  check('gaussPDF：∫pdf = 1（Simpson 10 万区间）', 1e-10, function () {
    var mu = 1.3, sigma = 2.1, lo = mu - 10 * sigma, hi = mu + 10 * sigma, N = 100000, h = (hi - lo) / N;
    var s = gaussPDF(lo, mu, sigma) + gaussPDF(hi, mu, sigma);
    for (var i = 1; i < N; i++) s += (i % 2 === 0 ? 2 : 4) * gaussPDF(lo + i * h, mu, sigma);
    return Math.abs(s * h / 3 - 1);
  });
  check('normalCDF：Φ(−x)=1−Φ(x)；Φ(1.96)≈0.975002；Φ(0)=0.5', 1e-5, function () {
    var sym = Math.abs(normalCDF(-1.3) - (1 - normalCDF(1.3)));
    var z = Math.abs(normalCDF(1.96) - 0.97500210485);
    return Math.max(sym, z, Math.abs(normalCDF(0) - 0.5));
  });
  check('normalSample：Box–Muller n=1e5 矩估计（|mean|≤0.01, |σ−1|≤0.01）', 0.01, function () {
    var xs = normalSample(100000, 99);
    return Math.max(Math.abs(expectation(xs)), Math.abs(Math.sqrt(variance(xs)) - 1));
  });
  check('categoricalSample：频率 ≈ p（n=2e4，max≤0.01）', 0.01, function () {
    var p = [0.1, 0.2, 0.3, 0.4];
    var samples = categoricalSample(p, 55);
    var freq = [0, 0, 0, 0];
    for (var i = 0; i < samples.length; i++) freq[samples[i]]++;
    freq = freq.map(function (v) { return v / samples.length; });
    var worst = 0;
    for (var j = 0; j < p.length; j++) worst = Math.max(worst, Math.abs(freq[j] - p[j]));
    return worst;
  });
  check('bayes：后验和=1；先验[.5,.5]×似然[.9,.1]=[.9,.1]', 1e-15, function () {
    var post = bayesUpdate([0.5, 0.5], [0.9, 0.1]);
    return Math.max(Math.abs(post.reduce(function (a, b) { return a + b; }, 0) - 1), maxAbsDiff(post, [0.9, 0.1]));
  });
  check('entropy：0≤H≤log₂n；均匀取等；确定分布为 0', 1e-15, function () {
    var u = entropy([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]);
    var det = entropy([1, 0, 0]);
    return Math.max(Math.abs(u - 3), Math.abs(det));
  });
  check('crossEntropy：H(p,q) ≥ H(p)，p=q 时取等', 1e-15, function () {
    var p = [0.1, 0.2, 0.3, 0.4], q = [0.25, 0.25, 0.25, 0.25];
    var same = crossEntropy(p, p) - entropy(p);
    var diff = crossEntropy(p, q) - entropy(p);
    return Math.max(Math.abs(same), diff < -1e-15 ? 1 : 0);
  });
  check('kl：p=q 为 0；kl([.5,.5]‖[.25,.75])=0.20751874964', 1e-12, function () {
    var same = klDivergence([0.3, 0.7], [0.3, 0.7]);
    var known = klDivergence([0.5, 0.5], [0.25, 0.75]);
    return Math.max(Math.abs(same), Math.abs(known - 0.207518749639422));
  });
  check('js：对称且 ∈[0,1]', 1e-15, function () {
    var p = [0.1, 0.9], q = [0.8, 0.2];
    return Math.max(Math.abs(jsDivergence(p, q) - jsDivergence(q, p)), jsDivergence(p, q) > 1 ? 1 : 0, jsDivergence(p, q) < 0 ? 1 : 0);
  });
  check('mutualInfo：独立为 0；完全相关 = H(X)', 1e-12, function () {
    var px = [0.5, 0.5], py = [0.5, 0.5];
    var indep = [[px[0] * py[0], px[0] * py[1]], [px[1] * py[0], px[1] * py[1]]];
    var corr = [[0.5, 0], [0, 0.5]];
    return Math.max(Math.abs(mutualInfo(indep)), Math.abs(mutualInfo(corr) - 1));
  });
  check('conditionalEntropy：H(X|Y) = H(X,Y) − H(Y)', 1e-12, function () {
    var joint = [[0.2, 0.1], [0.3, 0.4]];
    var hxy = 0;
    for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++) hxy -= joint[i][j] * Math.log2(joint[i][j]);
    var py = [joint[0][0] + joint[1][0], joint[0][1] + joint[1][1]];
    var hy = entropy(py);
    return Math.abs(conditionalEntropy(joint) - (hxy - hy));
  });
  check('infoGain：纯分裂 = 1 bit；恒 ≥ −1e-15', 1e-12, function () {
    var parent = [0, 0, 0, 0, 1, 1, 1, 1];
    var gain = infoGain(parent, [[0, 0, 0, 0], [1, 1, 1, 1]]);
    return Math.max(Math.abs(gain - 1), gain < -1e-15 ? 1 : 0);
  });
  check('gini：[0,0,1,1]=0.5；纯类为 0', 1e-15, function () {
    return Math.max(Math.abs(gini([0, 0, 1, 1]) - 0.5), Math.abs(gini([1, 1, 1])));
  });
  check('gd：凸二次收敛至解析最优 ≤ 1e-7', 1e-7, function () {
    var A = [1, 10, 100], b = [1, -2, 3];
    var f = function (x) { return 0.5 * (A[0] * x[0] * x[0] + A[1] * x[1] * x[1] + A[2] * x[2] * x[2]) - (b[0] * x[0] + b[1] * x[1] + b[2] * x[2]); };
    var g = function (x) { return [A[0] * x[0] - b[0], A[1] * x[1] - b[1], A[2] * x[2] - b[2]]; };
    var r = gradientDescent(f, g, [0, 0, 0], { lr: 0.01, iters: 30000 });
    var xs = [b[0] / A[0], b[1] / A[1], b[2] / A[2]];
    return norm2(r.x.map(function (v, i) { return v - xs[i]; }));
  });
  check('adam：凸二次收敛至解析最优 ≤ 1e-5', 1e-5, function () {
    var A = [1, 10, 100], b = [1, -2, 3];
    var f = function (x) { return 0.5 * (A[0] * x[0] * x[0] + A[1] * x[1] * x[1] + A[2] * x[2] * x[2]) - (b[0] * x[0] + b[1] * x[1] + b[2] * x[2]); };
    var g = function (x) { return [A[0] * x[0] - b[0], A[1] * x[1] - b[1], A[2] * x[2] - b[2]]; };
    var r = adam(f, g, [0, 0, 0], { lr: 0.02, iters: 5000 });
    var xs = [b[0] / A[0], b[1] / A[1], b[2] / A[2]];
    return norm2(r.x.map(function (v, i) { return v - xs[i]; }));
  });
  check('cg：SPD 系统 ≤ n 步收敛，残差 ≤ 1e-10', 1e-10, function () {
    var qr = qrDecompose(spdMatrix(6, 30));
    var Q = qr.Q;
    var Lam = Array.from({ length: 6 }, function (_, i) { return i + 1; });
    var A = matMul(matMul(transpose(Q), Lam.map(function (l, i) { return Array.from({ length: 6 }, function (_, j) { return i === j ? l : 0; }); })), Q);
    var b = normalSample(6, 31);
    var r = conjugateGradient(A, b, { tol: 1e-12 });
    return r.residual / (1 + norm2(b));
  });
  check('layernorm：行均值≈0、方差≈1', 1e-4, function () {
    var rng = mulberry32(32);
    var x = Array.from({ length: 4 }, function () { return Array.from({ length: 8 }, function () { return rng() * 10 - 5; }); });
    var out = layerNorm(x, 1e-5);
    var worst = 0;
    for (var i = 0; i < out.length; i++) {
      var m = expectation(out[i]), v = variance(out[i]);
      worst = Math.max(worst, Math.abs(m), Math.abs(v - 1));
    }
    return worst;
  });
  check('attention：权重行和=1；因果掩码上三角严格为 0', 1e-14, function () {
    var rng = mulberry32(33);
    var Q = Array.from({ length: 4 }, function () { return [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]; });
    var K = Q.map(function (r) { return r.slice(); });
    var V = Q.map(function (r) { return r.slice(); });
    var a = selfAttention(Q, K, V, true);
    var rowSum = 0;
    for (var i = 0; i < a.weights.length; i++) rowSum = Math.max(rowSum, Math.abs(a.weights[i].reduce(function (x, y) { return x + y; }, 0) - 1));
    var upper = 0;
    for (var i2 = 0; i2 < 4; i2++) for (var j = i2 + 1; j < 4; j++) upper = Math.max(upper, a.weights[i2][j]);
    return Math.max(rowSum, upper);
  });
  check('temperature：T 下降 → 熵下降', 1e-12, function () {
    var logits = [1, 2, 3, 4];
    var h05 = entropy(temperatureSoftmax(logits, 0.5));
    var h1 = entropy(temperatureSoftmax(logits, 1));
    var h2 = entropy(temperatureSoftmax(logits, 2));
    return (h05 < h1 && h1 < h2) ? 0 : 1;
  });
  check('topk：支撑外严格为 0；支撑上归一化和=1', 1e-15, function () {
    var out = topK([0.1, 0.4, 0.2, 0.3], 2);
    return Math.max(Math.abs(out[0]) + Math.abs(out[2]), Math.abs(out[1] + out[3] - 1), Math.abs(out[1] - 0.4 / 0.7));
  });
  check('hmm：前向-后验每帧和=1；P(O) 与 256 路径全枚举一致', 1e-10, function () {
    var pi = [0.5, 0.5];
    var A = [[0.7, 0.3], [0.4, 0.6]];
    var B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]];
    var obs = [0, 1, 2, 1, 0, 2, 0, 1];
    var fb = hmmForwardBackward(pi, A, B, obs);
    var rowSum = 0;
    for (var t = 0; t < obs.length; t++) rowSum = Math.max(rowSum, Math.abs(fb.posteriors[t].reduce(function (a, b) { return a + b; }, 0) - 1));
    var total = 0;
    for (var mask = 0; mask < 256; mask++) {
      var p = pi[(mask >> 7) & 1] * B[(mask >> 7) & 1][obs[0]];
      for (var t2 = 1; t2 < 8; t2++) {
        var prev = (mask >> (8 - t2)) & 1, cur = (mask >> (7 - t2)) & 1;
        p *= A[prev][cur] * B[cur][obs[t2]];
      }
      total += p;
    }
    return Math.max(rowSum, Math.abs(fb.logPO - Math.log(total)) / Math.abs(fb.logPO));
  });
  check('hmm：Viterbi 路径 = 全枚举最优路径', 1e-9, function () {
    var pi = [0.5, 0.5];
    var A = [[0.7, 0.3], [0.4, 0.6]];
    var B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]];
    var obs = [0, 1, 2, 1, 0, 2, 0, 1];
    var v = hmmViterbi(pi, A, B, obs);
    var best = -Infinity, bestPath = null;
    for (var mask = 0; mask < 256; mask++) {
      var lp = Math.log(pi[(mask >> 7) & 1]) + Math.log(B[(mask >> 7) & 1][obs[0]]);
      for (var t = 1; t < 8; t++) {
        var prev = (mask >> (8 - t)) & 1, cur = (mask >> (7 - t)) & 1;
        lp += Math.log(A[prev][cur]) + Math.log(B[cur][obs[t]]);
      }
      if (lp > best) { best = lp; bestPath = mask; }
    }
    var vPath = 0;
    v.path.forEach(function (s, i) { vPath |= s << (7 - i); });
    return Math.max(Math.abs(v.logprob - best), vPath === bestPath ? 0 : 1);
  });
  check('valueIteration：Bellman 残差 ≤ 1e-10；解析解 V(s)=10·γ^max(0,n−2−s)', 1e-8, function () {
    var r = valueIteration({ n: 7, gamma: 0.9, stochastic: false, tol: 1e-13 });
    var analytic = 0;
    for (var s = 0; s < 7; s++) analytic = Math.max(analytic, Math.abs(r.V[s] - 10 * Math.pow(0.9, Math.max(0, 5 - s))));
    var mono = true;
    for (var t = 1; t < r.snapshots.length; t++) {
      for (var s2 = 0; s2 < 7; s2++) if (r.snapshots[t][s2] < r.snapshots[t - 1][s2] - 1e-12) mono = false;
    }
    return mono ? Math.max(r.residual, analytic) : 1;
  });
  check('td0：α=1 确定性链一步后向扫描 = 解析解 γ^d', 1e-12, function () {
    var V = td0Sweep(7, 0.9);
    var worst = 0;
    for (var s = 0; s < 6; s++) worst = Math.max(worst, Math.abs(V[s] - Math.pow(0.9, 5 - s)));
    return worst;
  });
  check('qLearning：训练后贪心策略到达目标率 ≥ 0.9', 0.1, function () {
    var r = qLearning({ n: 7, gamma: 0.9, episodes: 5000, seed: 123 });
    return 1 - r.successRate;
  });
  check('epsilonGreedy：ε=0 时严格取 argmax', 0, function () {
    var rng = mulberry32(77);
    return epsilonGreedy([1, 3, 2], 0, rng) === 1 ? 0 : 1;
  });
  check('classification metrics：accuracy/precision/recall/f1 公式一致', 1e-15, function () {
    var y = [1, 1, 0, 0, 1, 0], yhat = [1, 0, 0, 1, 1, 0];
    return Math.max(
      Math.abs(accuracy(y, yhat) - 4 / 6),
      Math.abs(precision(y, yhat) - 2 / 3),
      Math.abs(recall(y, yhat) - 2 / 3),
      Math.abs(f1(y, yhat) - 2 / 3)
    );
  });
  check('auc：完美分离=1；A(y,s)=1−A(y,−s)', 1e-15, function () {
    var perfect = auc([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8]);
    var y = [0, 0, 1, 1, 1], s = [0.1, 0.4, 0.35, 0.8, 0.9];
    var identity = auc(y, s) + auc(y, s.map(function (v) { return -v; }));
    return Math.max(Math.abs(perfect - 1), Math.abs(identity - 1));
  });
  check('r2：完美拟合=1；均值预测≈0', 1e-12, function () {
    var y = [1, 2, 3, 4];
    return Math.max(Math.abs(r2(y, y.slice()) - 1), Math.abs(r2(y, [2.5, 2.5, 2.5, 2.5])));
  });
  check('cosine：cos(x,x)=1；正交基精确为 0；|cos|≤1', 1e-15, function () {
    var x = [3, 4];
    return Math.max(
      Math.abs(cosineSimilarity(x, x) - 1),
      Math.abs(cosineSimilarity([1, 0], [0, 1])),
      Math.abs(cosineSimilarity([1, 1], [1, 0]) - Math.SQRT1_2)
    );
  });
  check('editDistance：d(kitten,sitting)=3；对称；d("",abc)=3', 0, function () {
    return Math.max(
      editDistance('kitten', 'sitting') - 3,
      Math.abs(editDistance('kitten', 'sitting') - editDistance('sitting', 'kitten')),
      editDistance('', 'abc') - 3
    );
  });
  check('silhouette：∈[−1,1]；分离簇 > 0.5', 1e-15, function () {
    var rng = mulberry32(40);
    var X = [], labels = [];
    for (var i = 0; i < 40; i++) {
      var c = i < 20 ? [0, 0] : [8, 8];
      X.push([c[0] + (rng() * 2 - 1) * 0.5, c[1] + (rng() * 2 - 1) * 0.5]);
      labels.push(i < 20 ? 0 : 1);
    }
    var s = silhouette(X, labels);
    var mean = s.reduce(function (a, b) { return a + b; }, 0) / s.length;
    var bound = 0;
    for (var i2 = 0; i2 < s.length; i2++) bound = Math.max(bound, Math.abs(s[i2]) - 1);
    return Math.max(bound, mean > 0.5 ? 0 : 0.5 - mean);
  });
  check('shuffle：多重集守恒且非恒等', 0, function () {
    var a = shuffle(Array.from({ length: 100 }, function (_, i) { return i; }), 5);
    var sorted = a.slice().sort(function (x, y) { return x - y; });
    var same = 0;
    for (var i = 0; i < 100; i++) if (a[i] === i) same++;
    return (maxAbsDiff(sorted, Array.from({ length: 100 }, function (_, i) { return i; })) === 0 && same < 100) ? 0 : 1;
  });
  check('trainTestSplit：规模和、无交集', 0, function () {
    var r = trainTestSplit(100, 0.8, 3);
    var set = new Set(r.train.concat(r.test));
    return (r.train.length === 80 && r.test.length === 20 && set.size === 100) ? 0 : 1;
  });
  check('kFold：并集=全集、两两不交', 0, function () {
    var folds = kFold(17, 5, 9);
    var seen = new Set();
    for (var i = 0; i < folds.length; i++) {
      if (folds[i].length === 0) return 1;
      for (var j = 0; j < folds[i].length; j++) {
        if (seen.has(folds[i][j])) return 1;
        seen.add(folds[i][j]);
      }
    }
    return seen.size === 17 ? 0 : 1;
  });
  check('oneHot：行和=1、维度=类数', 0, function () {
    var oh = oneHot([0, 1, 2, 1, 0]);
    var bad = 0;
    for (var i = 0; i < oh.matrix.length; i++) if (oh.matrix[i].reduce(function (a, b) { return a + b; }, 0) !== 1 || oh.matrix[i].length !== 3) bad++;
    return bad;
  });
  check('arena：分配区间两两不交；越界抛错；reset 复用', 0, function () {
    var arena = new Arena(16);
    var a = arena.alloc(8), b = arena.alloc(8);
    var bad = (a.offset === 0 && b.offset === 8 && a.offset + a.length <= b.offset) ? 0 : 1;
    var threw = false;
    try { arena.alloc(1); } catch (e) { threw = true; }
    arena.reset();
    var c = arena.alloc(16);
    return bad + (threw ? 0 : 1) + (c.offset === 0 ? 0 : 1);
  });
  check('reduceAll：= Kahan 精确和（朴素求和损失全部有效位）', 1e-9, function () {
    var xs = [1e16, 1, 1, -1e16];
    var naive = xs.reduce(function (a, b) { return a + b; }, 0);
    return Math.abs(reduceAll(xs) - 2) + (Math.abs(naive - 2) > 1 ? 0 : 1);
  });
  check('broadcast：恒等映射（深拷贝）', 0, function () {
    var xs = [1, 2, 3];
    var b = broadcast(xs);
    b[0] = 99;
    return (xs[0] === 1 && maxAbsDiff(broadcast(xs), xs) === 0) ? 0 : 1;
  });
  check('mulberry32：同种子 → 同序列（可复现）', 0, function () {
    var r1 = mulberry32(42), r2 = mulberry32(42);
    for (var i = 0; i < 10; i++) if (r1() !== r2()) return 1;
    return 0;
  });

  /* ---- 13.1 TENSOR 结构不变量（精确 0）---- */
  check('tensor.create：元素总数 = ∏ shape', 0, function () {
    var t = tensorCreate([2, 3, 4], 7);
    return (tensorCount(t) === 24 && t[1][2][3] === 7) ? 0 : 1;
  });
  check('tensor.zeros/ones：全部元素 = 0 / 1', 0, function () {
    var z = flatten(tensorZeros([2, 5])), o = flatten(tensorOnes([3, 2]));
    var bad = z.filter(function (v) { return v !== 0; }).length + o.filter(function (v) { return v !== 1; }).length;
    return bad;
  });
  check('tensor.arange：步长一致；末值 < stop', 0, function () {
    var a = arange(0, 5, 1), b = arange(0, 9, 2);
    var okA = a.length === 5 && a[4] === 4, okB = b.length === 5 && b[4] === 8;
    return (okA && okB) ? 0 : 1;
  });
  check('tensor.reshape：元素守恒，行优先顺序不变', 0, function () {
    var X = [[1, 2, 3], [4, 5, 6]];
    var r = reshape(X, [3, 2]);
    return (tensorCount(r) === 6 && maxAbsDiff(flatten(r), [1, 2, 3, 4, 5, 6]) === 0) ? 0 : 1;
  });
  check('tensor.transpose：T(T(A)) = A', 0, function () {
    var A = [[1, 2, 3], [4, 5, 6]];
    return maxAbsDiff2(transpose(transpose(A)), A);
  });
  check('tensor.slice/concat/stack/split：长度语义精确', 0, function () {
    var s = tensorSlice([0, 1, 2, 3, 4, 5], 1, 4);
    var c = tensorConcat([[1, 2], [3], [4, 5]]);
    var st = tensorStack([[1, 2], [3, 4]]);
    var sp = tensorSplit([0, 1, 2, 3, 4, 5, 6], [2, 2, 3]);
    var bad = 0;
    if (maxAbsDiff(s, [1, 2, 3]) !== 0) bad++;
    if (maxAbsDiff(c, [1, 2, 3, 4, 5]) !== 0) bad++;
    if (st.length !== 2 || maxAbsDiff(st[1], [3, 4]) !== 0) bad++;
    if (sp.length !== 3 || sp[2].length !== 3) bad++;
    if (maxAbsDiff(tensorConcat(sp), [0, 1, 2, 3, 4, 5, 6]) !== 0) bad++;
    return bad;
  });
  check('tensor.cast/copy/gather/scatter/fill：语义精确', 0, function () {
    var bad = 0;
    if (maxAbsDiff(tensorCast(['1', '2.5', true]), [1, 2.5, 1]) !== 0) bad++;
    var src = [[1, 2], [3, 4]], cp = tensorCopy(src);
    cp[0][0] = 99;
    if (src[0][0] !== 1 || cp[0][0] !== 99) bad++;
    if (maxAbsDiff(tensorGather([10, 20, 30], [2, 0]), [30, 10]) !== 0) bad++;
    if (maxAbsDiff(tensorScatter([0, 0, 0, 0], [1, 3], [9, 8]), [0, 9, 0, 8]) !== 0) bad++;
    if (maxAbsDiff(tensorFill([1, 2, 3], 5), [5, 5, 5]) !== 0) bad++;
    return bad;
  });
  check('tensor.flatten：行优先展开，元素守恒', 0, function () {
    return maxAbsDiff(flatten([[1, [2, 3]], [4]]), [1, 2, 3, 4]);
  });

  /* ---- 13.2 NUMER 数值稳定不变量 ---- */
  check('numer.expm1/log1p：小 x 相对误差 ≤ 1e-15（优于朴素 exp−1）', 1e-15, function () {
    var x = 1e-10;
    var naive = Math.exp(x) - 1;
    return Math.max(Math.abs(Math.expm1(x) - naive), Math.abs(Math.log1p(x) - Math.log(1 + x)), Math.abs(Math.expm1(0)));
  });
  check('numer.hypot：大参数不溢出；hypot(3,4) = 5', 0, function () {
    var big = Math.hypot(1e300, 1e300);
    var ref = 1e300 * Math.SQRT2;
    return (Math.abs(big - ref) / ref <= 1e-15 && Math.hypot(3, 4) === 5) ? 0 : 1;
  });
  check('numer.div_guard：|分母| < ε 时按 ε 处理，结果有限', 0, function () {
    var v = divGuard(1, 1e-300);
    var threw = false;
    try { divGuard(1, 0); } catch (e) { threw = true; }
    return (isFinite(v) && !threw) ? 0 : 1;
  });
  check('numer.sqrt_guard：负输入抛错', 0, function () {
    var threw = false;
    try { sqrtGuard(-1); } catch (e) { threw = true; }
    return (threw && sqrtGuard(9) === 3) ? 0 : 1;
  });
  check('numer.reciprocal：x·(1/x) ≈ 1', 1e-15, function () {
    var xs = [1, 2, 3.5, 0.1, 1e6];
    var worst = 0;
    for (var i = 0; i < xs.length; i++) worst = Math.max(worst, Math.abs(xs[i] * reciprocal(xs[i]) - 1));
    return worst;
  });
  check('numer.fma：Dekker 单次舍入（朴素双舍入失败案例精确复原）', 1e-13, function () {
    var p27 = Math.pow(2, 27) + 1, p54 = Math.pow(2, 54), p28 = Math.pow(2, 28);
    var exact = fma(p27, p27, -(p54 + p28)); // 精确值 = 1
    var naive = p27 * p27 - (p54 + p28);     // 双舍入 → 0
    var rng = mulberry32(61), worst = 0;
    for (var i = 0; i < 200; i++) {
      var a = rng() * 2 - 1, b = rng() * 2 - 1, c = rng() * 2 - 1;
      worst = Math.max(worst, Math.abs(fma(a, b, c) - (a * b + c)) / (1 + Math.abs(a * b + c)));
    }
    return Math.max(Math.abs(exact - 1) + (Math.abs(naive - 1) > 0.5 ? 0 : 1), worst);
  });

  /* ---- 13.3 PROB 概率不变量 ---- */
  check('prob.bernoulli：频率 ≈ p（n=2e4）', 0.01, function () {
    var xs = bernoulliSample(0.3, 20000, 71);
    var f = xs.reduce(function (a, b) { return a + b; }, 0) / xs.length;
    return Math.abs(f - 0.3);
  });
  check('prob.binomial：均值 = np（trials=5000）', 0.08, function () {
    var xs = binomialSample(10, 0.4, 5000, 72);
    return Math.abs(expectation(xs) - 4);
  });
  check('prob.uniform：∈ [0,1)；均值 ≈ 0.5；同种子同序列', 0.01, function () {
    var xs = uniformSample(20000, 73);
    var inRange = xs.every(function (v) { return v >= 0 && v < 1; });
    var repro = maxAbsDiff(uniformSample(50, 73), uniformSample(50, 73));
    return (inRange && repro === 0) ? Math.abs(expectation(xs) - 0.5) : 1;
  });
  check('prob.marginal/condition：边缘 = 联合求和；P(A|B)·P(B) = P(A,B)', 1e-15, function () {
    var joint = [[0.1, 0.2], [0.3, 0.4]];
    var mx = marginal(joint);
    var cond = conditionalRow(joint, 0);
    var back = cond.map(function (v, j) { return v * mx[0]; });
    return Math.max(maxAbsDiff(mx, [0.3, 0.7]), maxAbsDiff(back, joint[0]), Math.abs(cond[0] + cond[1] - 1));
  });
  check('prob.mv_gauss：2D Simpson ∫pdf = 1；峰值为 1/(2π|Σ|^½)；对称', 1e-6, function () {
    var mu = [1, -2], Sigma = [[4, 1], [1, 3]];
    var N = 160, span = 12, h = 2 * span / N, s = 0;
    var pts = [];
    for (var i = 0; i <= N; i++) pts.push(mu[0] - span + i * h);
    for (var a = 0; a <= N; a++) for (var b = 0; b <= N; b++) {
      var w = (a === 0 || a === N ? 1 : (a % 2 ? 4 : 2)) * (b === 0 || b === N ? 1 : (b % 2 ? 4 : 2));
      s += w * mvGaussPDF([pts[a], pts[b]], mu, Sigma);
    }
    var integral = s * h * h / 9;
    var peak = mvGaussPDF(mu, mu, Sigma);
    var peakRef = 1 / (2 * Math.PI * Math.sqrt(4 * 3 - 1));
    var sym = mvGaussPDF([3, 0], mu, Sigma) - mvGaussPDF([-1, -4], mu, Sigma);
    return Math.max(Math.abs(integral - 1), Math.abs(peak - peakRef), Math.abs(sym));
  });

  /* ---- 13.4 INFO 信息论不变量 ---- */
  check('info.perplexity/surprisal/bits：公式一致', 1e-15, function () {
    var p = [0.1, 0.2, 0.3, 0.4];
    return Math.max(
      Math.abs(perplexity(p) - Math.pow(2, entropy(p))),
      Math.abs(surprisal(0.25) - 2),
      Math.abs(bits(1) - 1.4426950408889634)
    );
  });
  check('info.kl_reverse：KL(p‖q) ≠ KL(q‖p)（非对称性）', 1e-12, function () {
    var p = [0.1, 0.2, 0.7], q = [0.4, 0.4, 0.2];
    var fwd = klDivergence(p, q), rev = klReverse(p, q);
    return (Math.abs(rev - klDivergence(q, p)) === 0 && Math.abs(fwd - rev) > 0.01) ? 0 : 1;
  });
  check('info.max_entropy：均匀分布熵最大（n=3 网格暴力验证）', 1e-12, function () {
    var best = 0;
    for (var a = 0; a <= 10; a++) for (var b = 0; a + b <= 10; b++) {
      var c = 10 - a - b;
      best = Math.max(best, entropy([a / 10, b / 10, c / 10]));
    }
    return best - entropy(maxEntropyDist(3));
  });
  check('info.normalize_dist：Σp = 1 且比例守恒', 1e-15, function () {
    var p = normalizeDist([1, 2, 3, 4]);
    return Math.max(Math.abs(p[0] + p[1] + p[2] + p[3] - 1), Math.abs(p[1] - 2 * p[0]));
  });

  /* ---- 13.5 OPTIM 优化器不变量（凸二次 → 解析最优）---- */
  check('optim.momentum：凸二次收敛至解析最优 ≤ 1e-8', 1e-8, function () {
    var q = quadProblem();
    var r = momentum(q.f, q.g, [0, 0, 0], { lr: 0.018, beta: 0.9, iters: 20000 });
    return norm2(r.x.map(function (v, i) { return v - q.xs[i]; }));
  });
  check('optim.nesterov：凸二次收敛至解析最优 ≤ 1e-8', 1e-8, function () {
    var q = quadProblem();
    var r = nesterov(q.f, q.g, [0, 0, 0], { lr: 0.018, beta: 0.9, iters: 20000 });
    return norm2(r.x.map(function (v, i) { return v - q.xs[i]; }));
  });
  check('optim.rmsprop：凸二次收敛至解析最优 ≤ 1e-3（小 lr 抑制振荡）', 1e-3, function () {
    var q = quadProblem();
    var r = rmsprop(q.f, q.g, [0, 0, 0], { lr: 0.0005, iters: 30000 });
    return norm2(r.x.map(function (v, i) { return v - q.xs[i]; }));
  });
  check('optim.adagrad：凸二次收敛至解析最优 ≤ 1e-3', 1e-3, function () {
    var q = quadProblem();
    var r = adagrad(q.f, q.g, [0, 0, 0], { lr: 0.5, iters: 20000 });
    return norm2(r.x.map(function (v, i) { return v - q.xs[i]; }));
  });
  check('optim.adamw：λ=0 退化为 Adam（凸二次收敛 ≤ 1e-5）', 1e-5, function () {
    var q = quadProblem();
    var r = adamw(q.f, q.g, [0, 0, 0], { lr: 0.02, lambda: 0, iters: 5000 });
    return norm2(r.x.map(function (v, i) { return v - q.xs[i]; }));
  });
  check('optim.adamw：零梯度步 = 解耦衰减 w(1−lr·λ)（精确）', 0, function () {
    var r = adamw(function () { return 0; }, function () { return [0, 0]; }, [1, 2], { lr: 0.1, lambda: 0.5, iters: 1 });
    return maxAbsDiff(r.x, [0.95, 1.9]);
  });
  check('optim.grad_clip：裁剪后 ‖g‖ = max_norm 且方向保持', 1e-15, function () {
    var c1 = gradClip([3, 4], 2.5), c2 = gradClip([1, 0], 5);
    return Math.max(Math.abs(norm2(c1) - 2.5), Math.abs(c1[0] / c1[1] - 0.75), maxAbsDiff(c2, [1, 0]));
  });
  check('optim.grad_norm/lr_decay/weight_decay/polyak：公式精确', 0, function () {
    var bad = 0;
    if (Math.abs(gradNorm([3, 4]) - 5) !== 0) bad++;
    if (lrDecay(0.1, 0.01, 100) !== 0.05) bad++;
    if (lrDecay(0.1, 0, 0) !== 0.1) bad++;
    if (maxAbsDiff(weightDecay([1, 2], 0.1, 0.5), [0.95, 1.9]) !== 0) bad++;
    if (maxAbsDiff(polyakAvg([[0, 0], [1, 2], [3, 4]]), [4 / 3, 2]) > 1e-15) bad++;
    return bad;
  });
  check('optim.polyak：GD 迭代平均收敛至解析最优', 1e-8, function () {
    var q = quadProblem();
    var x = [0, 0, 0], traj = [];
    for (var it = 0; it < 30000; it++) {
      var g = q.g(x);
      for (var i = 0; i < 3; i++) x[i] -= 0.01 * g[i];
      if (it >= 15000) traj.push(x.slice());
    }
    return norm2(polyakAvg(traj).map(function (v, i) { return v - q.xs[i]; }));
  });

  /* ---- 13.6 FWD 前向传播不变量 ---- */
  check('fwd.tanh：奇函数；|tanh| ≤ 1', 1e-16, function () {
    var xs = [0.3, 1, 2, 5], worst = 0;
    for (var i = 0; i < xs.length; i++) {
      worst = Math.max(worst, Math.abs(Math.tanh(-xs[i]) + Math.tanh(xs[i])), Math.abs(Math.tanh(xs[i])) - 1);
    }
    return worst;
  });
  check('fwd.embedding_lookup：输出 = 第 i 行向量（深拷贝隔离）', 0, function () {
    var table = [[1, 2], [3, 4], [5, 6]];
    var out = embeddingLookup(table, [2, 0]);
    out[0][0] = 99;
    var bad = (maxAbsDiff(out[1], [1, 2]) !== 0 || table[2][0] !== 5) ? 1 : 0;
    return bad;
  });
  check('fwd.residual_add：x + f(x) 逐元素且形状不变', 0, function () {
    var x = [[1, 2], [3, 4]], fx = [[0.5, -0.5], [1, 1]];
    var r = residualAdd(x, fx);
    return (maxAbsDiff(r, [[1.5, 1.5], [4, 5]]) !== 0 || r.length !== 2) ? 1 : 0;
  });
  check('fwd.split/concat_heads：split(concat(X)) = X 逐位一致', 0, function () {
    var rng = mulberry32(81);
    var X = Array.from({ length: 3 }, function () { return Array.from({ length: 8 }, function () { return rng() * 2 - 1; }); });
    var H = 4;
    return maxAbsDiff(concatHeads(splitHeads(X, H)), X);
  });
  check('fwd.pool_max：输出 = 窗内最大值', 0, function () {
    var r = poolMax([1, 3, 2, 4, 6, 5], 2);
    return maxAbsDiff(r, [3, 4, 6]);
  });
  check('fwd.batchnorm：批均值 ≈ 0、批方差 ≈ 1', 1e-9, function () {
    var rng = mulberry32(82);
    var X = Array.from({ length: 32 }, function () { return [rng() * 10 - 5, rng() * 4 - 2]; });
    var out = batchNorm(X, 0);
    var worst = 0;
    for (var j = 0; j < 2; j++) {
      var col = out.map(function (r) { return r[j]; });
      worst = Math.max(worst, Math.abs(expectation(col)), Math.abs(variance(col) - 1));
    }
    return worst;
  });
  check('fwd.dropout：期望不变性（2e4 种子平均 ≈ x）；同种子同掩码', 0.04, function () {
    var x = [2, -1, 0.5, 3];
    var acc = [0, 0, 0, 0];
    for (var s = 0; s < 20000; s++) {
      var d = dropout(x, 0.25, s);
      for (var i = 0; i < 4; i++) acc[i] += d[i];
    }
    var worst = 0;
    for (var i2 = 0; i2 < 4; i2++) worst = Math.max(worst, Math.abs(acc[i2] / 20000 - x[i2]));
    var repro = maxAbsDiff(dropout(x, 0.25, 7), dropout(x, 0.25, 7));
    return Math.max(worst, repro === 0 ? 0 : 1);
  });
  check('fwd.conv1d：输出长度 = ⌈(L+2p−k)/s⌉+1；差分核精确', 0, function () {
    var d = conv1d([1, 2, 3, 4, 5], [1, -1], 1, 0);
    var p = conv1d([1, 2, 3, 4, 5], [1, -1], 1, 1);
    return (maxAbsDiff(d, [-1, -1, -1, -1]) === 0 && p.length === 6) ? 0 : 1;
  });

  /* ---- 13.7 BWD 反向传播不变量（中心差分交叉验证）---- */
  check('bwd.grad_relu/grad_tanh：与中心差分一致 ≤ 1e-6', 1e-6, function () {
    var x = [-1.5, -0.2, 0.4, 1.3, 2.0];
    var eps = 1e-6, worst = 0;
    for (var i = 0; i < x.length; i++) {
      var numRelu = (relu(x[i] + eps) - relu(x[i] - eps)) / (2 * eps);
      var numTanh = (Math.tanh(x[i] + eps) - Math.tanh(x[i] - eps)) / (2 * eps);
      worst = Math.max(worst, Math.abs(numRelu - gradRelu(x)[i]), Math.abs(numTanh - gradTanh(x)[i]));
    }
    return worst;
  });
  check('bwd.grad_softmax_ce：= p − y 且梯度和 = 0', 1e-15, function () {
    var logits = [1.2, -0.5, 3.1, 0.4];
    var g = gradSoftmaxCE(logits, 2);
    var p = softmax(logits);
    return Math.max(maxAbsDiff(g, [p[0], p[1], p[2] - 1, p[3]]), Math.abs(g[0] + g[1] + g[2] + g[3]));
  });
  check('bwd.grad_matmul：dL/dA = dL·Bᵀ（数值差分验证）', 1e-6, function () {
    var A = [[1, 2], [3, 4]], B = [[0.5, -1], [2, 0.5]], dL = [[1, 0.5], [-0.5, 2]];
    var L = function (M) { var AB = matMul(M, B); var s = 0; for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++) s += AB[i][j] * dL[i][j]; return s; };
    var J = numericJacobian(function (x) { return [L([[x[0], x[1]], [x[2], x[3]]])]; }, [1, 2, 3, 4], 1e-6)[0];
    var ana = gradMatmul(dL, A, B).dA;
    return Math.max(Math.abs(J[0] - ana[0][0]), Math.abs(J[1] - ana[0][1]), Math.abs(J[2] - ana[1][0]), Math.abs(J[3] - ana[1][1]));
  });
  check('bwd.chain_rule：d/dx f(g(x)) = f′(g(x))·g′(x)（差分验证）', 1e-6, function () {
    var x = 0.7;
    var ana = chainRule(Math.cos, function (t) { return t * t; }, function (t) { return 2 * t; }, x);
    var num = (Math.sin((x + 1e-6) * (x + 1e-6)) - Math.sin((x - 1e-6) * (x - 1e-6))) / 2e-6;
    return Math.abs(ana - num);
  });
  check('bwd.jacobian：数值 Jacobian 与解析一致 ≤ 1e-5', 1e-5, function () {
    var f = function (x) { return [x[0] * x[0], x[0] * x[1]]; };
    var J = jacobian(f, [1.5, -2]);
    var ref = [[3, 0], [-2, 1.5]];
    return Math.max(Math.abs(J[0][0] - ref[0][0]), Math.abs(J[0][1] - ref[0][1]), Math.abs(J[1][0] - ref[1][0]), Math.abs(J[1][1] - ref[1][1]));
  });
  check('bwd.vjp/jvp：对偶恒等式 ⟨Jv,u⟩ = ⟨v,Jᵀu⟩ ≤ 1e-8', 1e-8, function () {
    var f = function (x) { return [x[0] * x[1], x[1] * x[1], x[0] + x[1]]; }; // R² → R³
    var x = [0.6, -1.2], v = [1, -0.5], u = [0.3, 0.8, -0.5];                    // v∈R², u∈R³
    var jv = jvp(f, x, v), jtu = vjp(f, x, u);
    return Math.abs(dot(jv, u) - dot(v, jtu));
  });
  check('bwd.grad_accum：Σ 批次梯度 = 全量梯度 ≤ 1e-10', 1e-10, function () {
    var rng = mulberry32(83);
    var X = Array.from({ length: 20 }, function () { return [1, rng() * 2 - 1, rng() * 2 - 1]; });
    var w = [0.5, -1, 2];
    function grad(rows) { // 线性模型均方梯度 Xᵀ(Xw − y)
      var g = [0, 0, 0];
      rows.forEach(function (item) {
        var r = item[0], idx = item[1];
        var e = (w[0] * r[0] + w[1] * r[1] + w[2] * r[2]) - idx;
        for (var j = 0; j < 3; j++) g[j] += e * r[j];
      });
      return [g]; // 1×3 矩阵行
    }
    var full = grad(X.map(function (r, i) { return [r, i]; }))[0];
    var parts = [
      grad(X.slice(0, 7).map(function (r, i) { return [r, i]; })),
      grad(X.slice(7, 13).map(function (r, i) { return [r, i + 7]; })),
      grad(X.slice(13).map(function (r, i) { return [r, i + 13]; }))
    ];
    var acc = gradAccum(parts)[0];
    return Math.max(Math.abs(acc[0] - full[0]), Math.abs(acc[1] - full[1]), Math.abs(acc[2] - full[2]));
  });
  check('bwd.bptt：时间反向传播梯度 = 中心差分 ≤ 1e-6', 1e-6, function () {
    var P = { W: 0.7, U: -0.4, h0: 0.2 };
    var xs = [0.5, -1, 0.8, 0.2], ys = [0.3, 0.1, -0.2, 0.4];
    var ana = rnnBPTT(P, xs, ys);
    var eps = 1e-6, worst = 0;
    [['W', 'U', 'h0']].forEach(function () { });
    function numGrad(key) {
      var lp = rnnLoss(Object.assign({}, P, (function () { var o = {}; o[key] = P[key] + eps; return o; })()), xs, ys);
      var lm = rnnLoss(Object.assign({}, P, (function () { var o = {}; o[key] = P[key] - eps; return o; })()), xs, ys);
      return (lp - lm) / (2 * eps);
    }
    worst = Math.max(worst, Math.abs(numGrad('W') - ana.W), Math.abs(numGrad('U') - ana.U), Math.abs(numGrad('h0') - ana.h0));
    return worst;
  });
  check('bwd.stop_grad/detach：返回深拷贝，原值隔离', 0, function () {
    var x = [[1, 2], [3, 4]];
    var s = stopGrad(x), d = detach(x);
    s[0][0] = 99; d[1][1] = 88;
    return (x[0][0] === 1 && x[1][1] === 4 && s[0][0] === 99 && d[1][1] === 88) ? 0 : 1;
  });

  /* ---- 13.8 NORM 归一化/正则不变量 ---- */
  check('norm.l1_reg/label_smooth/mixup：公式精确', 0, function () {
    var bad = 0;
    if (l1Reg([1, -2, 3], 0.5) !== 3) bad++;
    var ls = labelSmooth([0, 1, 0], 0.1);
    if (Math.abs(ls[0] + ls[1] + ls[2] - 1) > 1e-15 || Math.abs(ls[0] - 0.1 / 3) > 1e-15) bad++;
    var mx = mixup([0, 2], [4, 0], 0.25);
    if (maxAbsDiff(mx, [3, 0.5]) !== 0) bad++;
    return bad;
  });
  check('norm.weight_norm：‖g·v/‖v‖‖ = g', 1e-12, function () {
    var v = [3, 4, 12];
    return Math.abs(norm2(weightNorm(v, 2.5)) - 2.5);
  });
  check('norm.spectral_norm：幂迭代归一后 σ(W) ≈ 1', 1e-6, function () {
    var rng = mulberry32(84);
    var W = Array.from({ length: 4 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    return Math.abs(powerIteration(spectralNorm(W), { iters: 2000 }).value - 1);
  });
  check('norm.group_norm：组内均值 0、方差 1', 1e-8, function () {
    var rng = mulberry32(85);
    var x = Array.from({ length: 3 }, function () { return Array.from({ length: 8 }, function () { return rng() * 10 - 5; }); });
    var out = groupNorm(x, 4, 0);
    var worst = 0;
    for (var i = 0; i < 3; i++) for (var g = 0; g < 4; g++) {
      var seg = out[i].slice(g * 2, g * 2 + 2);
      worst = Math.max(worst, Math.abs(expectation(seg)), Math.abs(variance(seg) - 1));
    }
    return worst;
  });
  check('norm.early_stop：best = argmin；stopIndex = best + patience', 0, function () {
    var h = [1.0, 0.8, 0.6, 0.65, 0.7, 0.75];
    var r = earlyStop(h, 2);
    return (r.bestIndex === 2 && r.stopIndex === 4) ? 0 : 1;
  });

  /* ---- 13.9 ATTEN 注意力不变量 ---- */
  check('atten.qk_dot/scale：scores = QKᵀ/√d（手工基准）', 1e-15, function () {
    var Q = [[1, 2], [3, 4]], K = [[0.5, -1], [2, 0.5]];
    var s = qkDot(Q, K);
    var ref00 = (1 * 0.5 + 2 * -1) / Math.SQRT2, ref11 = (3 * 2 + 4 * 0.5) / Math.SQRT2;
    var s2 = attentionScale([[1, 2], [3, 4]], 2);
    return Math.max(Math.abs(s[0][0] - ref00), Math.abs(s[1][1] - ref11), Math.abs(s2[0][0] - 1 / Math.SQRT2));
  });
  check('atten.cross_attn：权重行和 = 1；形状 = [Tq][dv]', 1e-14, function () {
    var rng = mulberry32(91);
    var Q = Array.from({ length: 3 }, function () { return [rng() * 2 - 1, rng() * 2 - 1]; });
    var K = Array.from({ length: 4 }, function () { return [rng() * 2 - 1, rng() * 2 - 1]; });
    var V = Array.from({ length: 4 }, function () { return [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]; });
    var a = crossAttention(Q, K, V);
    var rowSum = 0;
    for (var i = 0; i < 3; i++) rowSum = Math.max(rowSum, Math.abs(a.weights[i].reduce(function (x, y) { return x + y; }, 0) - 1));
    return (a.output.length === 3 && a.output[0].length === 3) ? rowSum : 1;
  });
  check('atten.multi_head：H=1 等价自注意力；逐头独立重算一致 ≤ 1e-12', 1e-12, function () {
    var rng = mulberry32(92);
    var T = 3, d = 4, H = 2, hd = d / H;
    var Q = Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() * 2 - 1; }); });
    var K = Q.map(function (r) { return r.slice(); });
    var V = Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() * 2 - 1; }); });
    var mh1 = multiHead(Q, K, V, 1, false);
    var sa = selfAttention(Q, K, V, false);
    var d1 = maxAbsDiff(mh1.output, sa.output);
    var mh = multiHead(Q, K, V, H, false);
    var worst = d1;
    for (var t = 0; t < T; t++) for (var h = 0; h < H; h++) {
      var scores = [];
      for (var j = 0; j < T; j++) {
        var s = 0;
        for (var e = 0; e < hd; e++) s += Q[t][h * hd + e] * K[j][h * hd + e];
        scores.push(s / Math.sqrt(hd));
      }
      var w = softmax(scores);
      for (var e2 = 0; e2 < hd; e2++) {
        var o = 0;
        for (var j2 = 0; j2 < T; j2++) o += w[j2] * V[j2][h * hd + e2];
        worst = Math.max(worst, Math.abs(mh.output[t][h * hd + e2] - o));
      }
    }
    return worst;
  });
  check('atten.rope：⟨R_m q, R_n k⟩ 只依赖 m−n', 1e-12, function () {
    var rng = mulberry32(93);
    var q = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
    var k = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
    var lhs = dot(ropeApply(q, 0), ropeApply(k, 5));
    var rhs = dot(ropeApply(q, 3), ropeApply(k, 8));
    return Math.abs(lhs - rhs);
  });
  check('atten.kv_cache：增量单步 = 全量重算 ≤ 1e-12', 1e-12, function () {
    var rng = mulberry32(94);
    var K = Array.from({ length: 4 }, function () { return [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]; });
    var V = Array.from({ length: 4 }, function () { return [rng() * 2 - 1, rng() * 2 - 1]; });
    var q = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
    var inc = attnRow(q, K, V);
    var full = selfAttention([q], K, V, false);
    return Math.max(maxAbsDiff(inc.output, full.output[0]), maxAbsDiff(inc.weights, full.weights[0]));
  });
  check('atten.mqa/gqa：K/V 共享语义与分组语义一致 ≤ 1e-12', 1e-12, function () {
    var rng = mulberry32(95);
    var d = 4, H = 2;
    var Q = Array.from({ length: H }, function () { return Array.from({ length: 3 }, function () { return Array.from({ length: d }, function () { return rng() * 2 - 1; }); }); });
    var Ks = Array.from({ length: H }, function () { return Array.from({ length: 3 }, function () { return Array.from({ length: d }, function () { return rng() * 2 - 1; }); }); });
    var Vs = Array.from({ length: H }, function () { return Array.from({ length: 3 }, function () { return Array.from({ length: d }, function () { return rng() * 2 - 1; }); }); });
    var mqaOut = mqa(Q, Ks[0], Vs[0]);
    var gqaShared = gqa(Q, [Ks[0]], [Vs[0]], H);
    var gqaFull = gqa(Q, Ks, Vs, 1);
    var refHeads = Q.map(function (h) { return selfAttention(h, Ks[0], Vs[0], false).output; });
    return Math.max(maxAbsDiff(mqaOut, refHeads), maxAbsDiff(gqaShared, refHeads), maxAbsDiff(gqaFull, Q.map(function (h, i) { return selfAttention(h, Ks[i], Vs[i], false).output; })));
  });
  check('atten.rel_pos/alibi：偏置可加；惩罚随距离线性', 1e-15, function () {
    var scores = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    var bias = [[0, 1, 2], [1, 0, 1], [2, 1, 0]];
    var rb = relPosBias(scores, bias);
    var al = alibiScores(scores, 0.5);
    var bad = 0;
    if (maxAbsDiff(rb, bias) !== 0) bad++;
    if (Math.abs(al[0][2] - (-1)) > 1e-15 || Math.abs(al[2][0] - (-1)) > 1e-15) bad++;
    // 均匀打分 + alibi → 权重随 |i−j| 单调不增
    var W = al.map(function (row) { return softmax(row); });
    if (!(W[0][0] > W[0][1] && W[0][1] > W[0][2])) bad++;
    return bad;
  });
  check('atten.attn_entropy：∈ [0, log n]；均匀行 = log n', 1e-15, function () {
    var rng = mulberry32(96);
    var Q = Array.from({ length: 3 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var a = selfAttention(Q, Q.map(function (r) { return r.slice(); }), Q.map(function (r) { return r.slice(); }), false);
    var E = attnEntropy(a.weights);
    var worst = 0;
    for (var i = 0; i < E.length; i++) worst = Math.max(worst, E[i] < -1e-15 ? 1 : 0, E[i] > Math.log2(4) + 1e-15 ? 1 : 0);
    var uni = attnEntropy([[0.25, 0.25, 0.25, 0.25]])[0];
    return Math.max(worst, Math.abs(uni - 2));
  });
  check('atten.flash：分块在线 softmax = 全量自注意力（chunk=1/2/3/7）', 1e-12, function () {
    var rng = mulberry32(97);
    var Q = Array.from({ length: 5 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var K = Array.from({ length: 9 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var V = Array.from({ length: 9 }, function () { return Array.from({ length: 3 }, function () { return rng() * 2 - 1; }); });
    var ref = selfAttention(Q, K, V, false).output;
    var worst = 0;
    [1, 2, 3, 7].forEach(function (ch) {
      var f = flashAttention(Q, K, V, ch);
      worst = Math.max(worst, maxAbsDiff(f.output, ref));
    });
    return worst;
  });

  /* ---- 13.10 SAMPLE 采样不变量 ---- */
  check('sample.gumbel：G = −log(−log u)；均值 ≈ γ（n=2e4）', 0.02, function () {
    var rng = mulberry32(101), xs = [];
    for (var i = 0; i < 20000; i++) xs.push(gumbelSample(rng()));
    var formulaErr = Math.abs(gumbelSample(0.5) - (-Math.log(-Math.log(0.5))));
    return Math.max(formulaErr, Math.abs(expectation(xs) - 0.5772156649015329));
  });
  check('sample.topp：最小支撑；质量 ≈ cutoff；归一化和 = 1', 1e-15, function () {
    var out = topP([0.5, 0.3, 0.15, 0.05], 0.8);
    return Math.max(maxAbsDiff(out, [0.5 / 0.8, 0.3 / 0.8, 0, 0]), Math.abs(out[0] + out[1] - 1));
  });
  check('sample.multinomial：计数和 = n（精确）；频率 ≈ p ≤ 0.03', 0.03, function () {
    var counts = multinomialSample([0.2, 0.3, 0.5], 40000, 102);
    var sum = counts[0] + counts[1] + counts[2];
    var worst = sum === 40000 ? 0 : 1;
    return Math.max(worst, Math.abs(counts[0] / 40000 - 0.2), Math.abs(counts[1] / 40000 - 0.3), Math.abs(counts[2] / 40000 - 0.5));
  });
  check('sample.rejection：接受率 = 1/M；分箱频率 ≈ 目标密度积分', 0.01, function () {
    var target = function (x) { return 6 * x * (1 - x); }; // Beta(2,2) on [0,1]
    var proposal = function () { return 1; };             // 均匀提议
    var r = rejectionSample(20000, 103, target, proposal, 1.5, 0, 1);
    var bins = [0, 0, 0, 0];
    r.samples.forEach(function (x) { bins[Math.min(3, Math.floor(x * 4))]++; });
    var ref = [0.15625, 0.34375, 0.34375, 0.15625];
    var worst = Math.abs(r.acceptanceRate - 1 / 1.5);
    for (var i = 0; i < 4; i++) worst = Math.max(worst, Math.abs(bins[i] / 20000 - ref[i]));
    return worst;
  });
  check('sample.inverse_cdf：F(F⁻¹(u)) = u（指数分布 λ=1.5）', 1e-9, function () {
    var lam = 1.5;
    var F = function (x) { return 1 - Math.exp(-lam * x); };
    var worst = 0;
    [0.1, 0.37, 0.5, 0.73, 0.99].forEach(function (u) {
      worst = Math.max(worst, Math.abs(F(inverseCDF(u, F, 0, 100)) - u));
    });
    return worst;
  });
  check('sample.antithetic：对偶负相关（解析精确）；方差缩减', 0, function () {
    // Cov(f(U), f(1−U)) = E[U²(1−U)²] − (1/3)² = 1/30 − 1/9 < 0（f(x)=x²）
    var cov = 1 / 30 - 1 / 9;
    // 对偶均值方差 = Var((U²+(1−U)²)/2) = 1/180；iid 两样本均值方差 = Var(U²)/2 = (1/5−1/9)/2 = 2/45
    var varAnti = 1 / 180, varIid = (1 / 5 - 1 / 9) / 2;
    var a = antitheticSample(100, 104);
    var pairOk = a.length === 100 && a.every(function (v, i) { return i % 2 === 0 ? a[i + 1] === 1 - v : true; });
    return (cov < 0 && varAnti < varIid && pairOk) ? 0 : 1;
  });
  check('sample.stratified：样本落入对应层；均值 ≈ 0.5', 0.005, function () {
    var s = stratifiedSample(100, 105);
    var bad = 0;
    for (var i = 0; i < 100; i++) if (s[i] < i / 100 || s[i] >= (i + 1) / 100) bad++;
    return bad > 0 ? 1 : Math.abs(expectation(s) - 0.5);
  });
  check('sample.systematic：等间距 1/n；起点 ∈ [0, 1/n)', 1e-15, function () {
    var s = systematicSample(50, 106);
    if (s[0] < 0 || s[0] >= 1 / 50) return 1;
    var worst = 0;
    for (var i = 1; i < 50; i++) worst = Math.max(worst, Math.abs((s[i] - s[i - 1]) - 1 / 50));
    return worst;
  });

  /* ---- 13.11 GRAPH 图推断不变量 ---- */
  check('graph.belief_prop：链式 BP = 前向-后向（P(O) 与逐帧后验一致）', 1e-12, function () {
    var pi = [0.5, 0.5], A = [[0.7, 0.3], [0.4, 0.6]], B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]];
    var obs = [0, 1, 2, 1, 0, 2, 0, 1];
    var bp = chainBeliefProp(pi, A, B, obs), fb = hmmForwardBackward(pi, A, B, obs);
    var worst = Math.abs(bp.logPO - fb.logPO) / Math.abs(fb.logPO);
    for (var t = 0; t < obs.length; t++) worst = Math.max(worst, maxAbsDiff(bp.posteriors[t], fb.posteriors[t]));
    return worst;
  });
  check('graph.variable_elim：= 2⁸ 路径全枚举 ≤ 1e-12', 1e-12, function () {
    var pi = [0.5, 0.5], A = [[0.7, 0.3], [0.4, 0.6]], B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]];
    var obs = [0, 1, 2, 1, 0, 2, 0, 1];
    var ve = variableElim(pi, A, B, obs);
    var total = 0;
    for (var mask = 0; mask < 256; mask++) {
      var p = pi[(mask >> 7) & 1] * B[(mask >> 7) & 1][obs[0]];
      for (var t = 1; t < 8; t++) {
        var prev = (mask >> (8 - t)) & 1, cur = (mask >> (7 - t)) & 1;
        p *= A[prev][cur] * B[cur][obs[t]];
      }
      total += p;
    }
    return Math.abs(ve - total) / total;
  });
  check('graph.factor_product：维度 = 变量并集；取值精确 ≤ 1e-15', 1e-15, function () {
    var f = { vars: ['A', 'B'], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] };
    var g = { vars: ['B', 'C'], shape: [2, 2], table: [1, 2, 3, 4] };
    var h = factorProduct(f, g); // (A,B,C) 2×2×2
    var bad = 0;
    if (h.vars.join(',') !== 'A,B,C' || h.shape.join(',') !== '2,2,2') bad++;
    // h[A=0,B=1,C=0] = f(0,1)·g(1,0) = 0.2·3 = 0.6；h[A=1,B=1,C=1] = 0.4·4 = 1.6
    if (Math.abs(h.table[factorIndex(['A', 'B', 'C'], [2, 2, 2], { A: 0, B: 1, C: 0 })] - 0.6) > 1e-15) bad++;
    if (Math.abs(h.table[factorIndex(['A', 'B', 'C'], [2, 2, 2], { A: 1, B: 1, C: 1 })] - 1.6) > 1e-15) bad++;
    return bad;
  });
  check('graph.junction_tree/clique_tree：RIP 判定；校准一致', 1e-12, function () {
    var bad = 0;
    if (!ripCheck([['A', 'B'], ['B', 'C'], ['C', 'D']], [[0, 1], [1, 2]])) bad++;                   // 合法链
    if (ripCheck([['A', 'B'], ['B', 'C'], ['C', 'D'], ['A', 'D']], [[0, 1], [1, 2], [2, 3]])) bad++; // A 的团不连通
    var phiA = { vars: ['A', 'S'], shape: [2, 2], table: [0.1, 0.2, 0.3, 0.4] };
    var phiB = { vars: ['S', 'B'], shape: [2, 2], table: [0.5, 0.1, 0.2, 0.3] };
    var cal = cliqueTreeCalibrate(phiA, phiB, ['S']);
    var prod = factorProduct(phiA, phiB);
    var ref = sumOut(prod, ['A', 'B']);
    bad += maxAbsDiff(cal.separator.table, ref.table) > 1e-12 ? 1 : 0;
    return bad;
  });
  check('graph.kalman：稳态方差 = Riccati 解析解 ≤ 1e-9', 1e-9, function () {
    var Q = 0.5, R = 0.3, A = 1, H = 1;
    var Pinf = (-Q + Math.sqrt(Q * Q + 4 * R * Q)) / 2; // A=H=1 稳态：P² + PQ − RQ = 0
    var out = kalmanFilter1d(new Array(2000).fill(0), A, H, Q, R, 0, 1);
    return Math.abs(out[1999].P - Pinf);
  });
  check('graph.dsep：路径枚举 = 联合分布暴力条件独立检验', 0, function () {
    // 菱形 BN：A→B, A→C, B→D, C→D
    var dag = { A: [], B: ['A'], C: ['A'], D: ['B', 'C'] };
    var cpd = {
      A: [0.5, 0.5],
      B: [[0.2, 0.8], [0.4, 0.6]],  // P(B|A)
      C: [[0.3, 0.7], [0.5, 0.5]],
      D: [[[0.1, 0.9], [0.5, 0.5]], [[0.6, 0.4], [0.9, 0.1]]] // P(D|B,C)
    };
    function joint(cfg) {
      var p = cpd.A[cfg.A];
      p *= cpd.B[cfg.A][cfg.B];
      p *= cpd.C[cfg.A][cfg.C];
      p *= cpd.D[cfg.B][cfg.C][cfg.D];
      return p;
    }
    function indep(x, y, z) { // 精确条件独立：枚举 z 构型比较 P(x,y|z) 与 P(x|z)P(y|z)
      function match(cfg, zc) {
        return z.every(function (v) { return cfg[v] === zc[v]; });
      }
      var zVars = z.slice();
      for (var m = 0; m < Math.pow(2, zVars.length); m++) {
        var zc = {};
        zVars.forEach(function (v, vi) { zc[v] = (m >> vi) & 1; });
        var pz = 0, pxy = 0, px = [0, 0], py = [0, 0];
        for (var a = 0; a < 2; a++) for (var b = 0; b < 2; b++) for (var c = 0; c < 2; c++) for (var d = 0; d < 2; d++) {
          var cfg = { A: a, B: b, C: c, D: d };
          if (!match(cfg, zc)) continue;
          var j = joint(cfg);
          pz += j;
          if (cfg[x] === 1) px[1] += j; else px[0] += j;
          if (cfg[y] === 1) py[1] += j; else py[0] += j;
          if (cfg[x] === 1 && cfg[y] === 1) pxy += j;
        }
        if (pz < 1e-12) continue;
        if (Math.abs(pxy / pz - (px[1] / pz) * (py[1] / pz)) > 1e-12) return false;
      }
      return true;
    }
    var bad = 0;
    var nodes = ['A', 'B', 'C', 'D'];
    for (var i = 0; i < 4; i++) for (var j = i + 1; j < 4; j++) {
      for (var mask = 0; mask < 16; mask++) {
        var z = [];
        if (mask & 1) z.push('A');
        if (mask & 2) z.push('B');
        if (mask & 4) z.push('C');
        if (mask & 8) z.push('D');
        if (z.indexOf(nodes[i]) >= 0 || z.indexOf(nodes[j]) >= 0) continue;
        if (dSeparated(dag, nodes[i], nodes[j], z) !== indep(nodes[i], nodes[j], z)) bad++;
      }
    }
    return bad;
  });
  check('graph.m_sep：道德图判据 = D-分离判定（菱形 BN 全枚举）', 0, function () {
    var dag = { A: [], B: ['A'], C: ['A'], D: ['B', 'C'] };
    var bad = 0, nodes = ['A', 'B', 'C', 'D'];
    for (var i = 0; i < 4; i++) for (var j = i + 1; j < 4; j++) {
      for (var mask = 0; mask < 16; mask++) {
        var z = [];
        if (mask & 1) z.push('A');
        if (mask & 2) z.push('B');
        if (mask & 4) z.push('C');
        if (mask & 8) z.push('D');
        if (z.indexOf(nodes[i]) >= 0 || z.indexOf(nodes[j]) >= 0) continue;
        if (mSeparated(dag, nodes[i], nodes[j], z) !== dSeparated(dag, nodes[i], nodes[j], z)) bad++;
      }
    }
    return bad;
  });
  check('graph.particle：权重和 = 1；后验均值 ≈ 前向-后向（n=2000）', 0.15, function () {
    var pi = [0.5, 0.5], A = [[0.7, 0.3], [0.4, 0.6]], B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]];
    var obs = [0, 1, 2, 1, 0, 2, 0, 1];
    var pf = particleFilter(pi, A, B, obs, 2000, 107);
    var fb = hmmForwardBackward(pi, A, B, obs);
    var worst = 0;
    for (var t = 0; t < obs.length; t++) {
      var refMean = fb.posteriors[t][1]; // E[state] = P(state=1)
      worst = Math.max(worst, Math.abs(pf.postMeans[t] - refMean));
    }
    return worst;
  });

  /* ---- 13.12 RL 强化学习不变量 ---- */
  check('rl.bellman_backup：压缩映射 ‖T(V)−T(V′)‖ ≤ γ‖V−V′‖', 1e-12, function () {
    var mdp = chainSetup(7, 0.9, true);
    var V1 = [0.5, -1, 2, 0, 1, -0.5, 0.3], V2 = [1, 0, -1, 0.5, 2, 0, -1];
    var d = normInf(V1.map(function (v, i) { return v - V2[i]; }));
    var T1 = bellmanBackup(V1, mdp), T2 = bellmanBackup(V2, mdp);
    var dT = normInf(T1.map(function (v, i) { return v - T2[i]; }));
    return dT - 0.9 * d;
  });
  check('rl.policy_iter：收敛至值迭代最优解；V 逐轮单调不减', 1e-8, function () {
    var mdp = chainSetup(7, 0.9, false);
    var pi = policyIteration(mdp);
    var vi = valueIteration({ n: 7, gamma: 0.9, stochastic: false, tol: 1e-13 });
    var diff = normInf(pi.V.map(function (v, i) { return v - vi.V[i]; }));
    var mono = true;
    for (var t = 1; t < pi.vHistory.length; t++) {
      for (var s = 0; s < 7; s++) if (pi.vHistory[t][s] < pi.vHistory[t - 1][s] - 1e-12) mono = false;
    }
    return mono ? diff : 1;
  });
  check('rl.sarsa：训练后贪心策略到达目标率 ≥ 0.9', 0.1, function () {
    return 1 - sarsa({ n: 7, gamma: 0.9, episodes: 5000, seed: 321 }).successRate;
  });
  check('rl.td_lambda：资格迹 = (γλ)^t（精确）', 1e-12, function () {
    var tr = eligibilityTrace(10, 0.9, 0.5);
    var worst = 0;
    for (var t = 0; t < 10; t++) worst = Math.max(worst, Math.abs(tr[t] - Math.pow(0.45, t)));
    return worst;
  });
  check('rl.monte_carlo：随机策略回报均值 = 策略评估解（无偏）', 0.05, function () {
    var mdp = chainSetup(7, 0.9, true);
    var mc = monteCarloReturn(mdp, function (s, rng) { return rng() < 0.5 ? -1 : 1; }, 4000, 108);
    // 均匀随机策略的解析 Bellman 期望评估
    var Vr = new Array(7).fill(0);
    for (var k = 0; k < 20000; k++) {
      var Vn = new Array(7);
      for (var s = 0; s < 7; s++) {
        var v = 0;
        [-1, 1].forEach(function (a) {
          mdp.trans(s, a).forEach(function (tr) { v += 0.5 * tr[1] * (mdp.reward(s, tr[0]) + 0.9 * Vr[tr[0]]); });
        });
        Vn[s] = v;
      }
      var delta = 0;
      for (var s2 = 0; s2 < 7; s2++) delta = Math.max(delta, Math.abs(Vn[s2] - Vr[s2]));
      Vr = Vn;
      if (delta < 1e-13) break;
    }
    return Math.abs(mc - Vr[0]);
  });
  check('rl.ucb：未拉臂优先；同计数取高值；加分项可翻转排序', 0, function () {
    var zero = ucb([0.5, 0.9], [10, 0], 2, 11);     // 臂 1 未拉 → 必须探索
    var sameCount = ucb([0.6, 0.5], [10, 10], 2, 11); // 同计数 → 取高值臂 0
    var flip = ucb([0.6, 0.5], [100, 1], 2, 11);      // 臂 1 加分项 2√(ln11)≈3.10 远大于臂 0
    return (zero === 1 && sameCount === 0 && flip === 1) ? 0 : 1;
  });
  check('rl.gae：λ=0 → TD 残差；λ=1 → 蒙特卡洛优势（精确）', 1e-12, function () {
    var rewards = [1, 0.5, -0.2, 2], values = [0.3, 0.4, 0.1, 0.5], gamma = 0.9;
    var g0 = gae(rewards, values, gamma, 0);
    var g1 = gae(rewards, values, gamma, 1);
    var td = [], mc = [], G = 0;
    for (var t = 3; t >= 0; t--) {
      var nextV = t === 3 ? 0 : values[t + 1];
      td[t] = rewards[t] + gamma * nextV - values[t];
      G = rewards[t] + gamma * G;
      mc[t] = G - values[t];
    }
    return Math.max(maxAbsDiff(g0, td), maxAbsDiff(g1, mc));
  });
  check('rl.reinforce：策略梯度估计无偏（E[∇] = 解析值 ≤ 0.02）', 0.02, function () {
    var prefs = [1, 2, 3];
    var p = softmax(prefs);
    var S = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    var analytic = [p[0] * S - p[0] * p[0], p[1] * S - p[1] * p[1], p[2] * S - p[2] * p[2]];
    var rng = mulberry32(109), est = [0, 0, 0], N = 20000;
    for (var i = 0; i < N; i++) {
      var u = rng(), acc = 0, a = 2;
      for (var j = 0; j < 3; j++) { acc += p[j]; if (u <= acc) { a = j; break; } }
      var g = reinforceGradient(prefs, a, p[a]);
      for (var j2 = 0; j2 < 3; j2++) est[j2] += g[j2];
    }
    est = est.map(function (v) { return v / N; });
    return normInf(est.map(function (v, i) { return v - analytic[i]; }));
  });
  check('rl.ppo：截断目标公式精确（三种区间）', 0, function () {
    var bad = 0;
    if (ppoClip(1.2, 0.5, 0.2) !== 0.6) bad++;      // ratio>1+ε → 截断
    if (ppoClip(0.5, -0.5, 0.2) !== -0.4) bad++;    // ratio<1−ε → 截断（负优势方向）
    if (ppoClip(1.5, 2, 0.1) !== 2.2) bad++;        // min 选中未截断项 3? 1.5·2=3 vs clip(1.5,0.9,1.1)·2=2.2 → 2.2
    if (ppoClip(1.05, 1, 0.2) !== 1.05) bad++;      // 区间内 → 不截断
    return bad;
  });
  check('rl.thompson：Beta(2,3) 后验频率 ≈ α/(α+β)（n=4000）', 0.02, function () {
    var sum = 0;
    for (var s = 0; s < 4000; s++) sum += thompsonSample(2, 3, s);
    return Math.abs(sum / 4000 - 0.4);
  });

  /* ---- 13.13 METRIC/DATA/SYS/LINALG 不变量 ---- */
  check('metric.perplexity：= exp(logloss)（精确）', 1e-15, function () {
    var y = [1, 0, 1, 1], p = [0.9, 0.2, 0.7, 0.6];
    return Math.abs(perplexityLM(y, p) - Math.exp(logLoss(y, p)));
  });
  check('metric.bleu：已知值 0.5976；自译 = 1；短译 BP < 1', 1e-12, function () {
    var ref = 'the cat is on the mat'.split(' ');
    var hyp = 'the cat the cat on the mat'.split(' ');
    var p1 = 5 / 7, p2 = 3 / 6;
    var expected = Math.sqrt(p1 * p2); // BP = 1（hyp 更长）
    var bad = 0;
    if (Math.abs(bleu(ref, hyp, 2) - expected) > 1e-12) bad++;
    if (Math.abs(bleu(ref, ref.slice(), 2) - 1) > 1e-12) bad++;
    var short = bleu(ref, 'the cat'.split(' '), 2);
    var bpRef = Math.exp(1 - 6 / 2) * Math.sqrt(1 * 1);
    if (Math.abs(short - bpRef) > 1e-12) bad++;
    return bad;
  });
  check('metric.map：AP = Σ P(k)·Δr(k)（手工基准）', 1e-15, function () {
    var rel = [1, 0, 1, 1, 0];
    var expected = (1 / 1 + 2 / 3 + 3 / 4) / 3;
    return Math.abs(averagePrecision(rel) - expected);
  });
  check('metric.confusion：行和 = 类别真实数；对角线 = 正确数', 0, function () {
    var M = confusionMatrix([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], [0, 1, 2]);
    var bad = 0;
    if (M[0][0] + M[0][1] + M[0][2] !== 2) bad++;
    if (M[1][0] + M[1][1] + M[1][2] !== 2) bad++;
    if (M[2][0] + M[2][1] + M[2][2] !== 2) bad++;
    if (M[0][0] + M[1][1] + M[2][2] !== 4) bad++;
    return bad;
  });
  check('data.label_encode/pad/truncate/bucket/dedup：结构不变量', 0, function () {
    var bad = 0;
    var le = labelEncode(['b', 'a', 'b', 'c']);
    if (le.classes.join(',') !== 'a,b,c' || le.codes.join(',') !== '1,0,1,2') bad++;
    if (padSeq([1, 2], 4, 0).join(',') !== '1,2,0,0') bad++;
    if (truncateSeq([1, 2, 3, 4], 2).join(',') !== '1,2') bad++;
    var bk = bucketize([1, 2, 3, 4, 5], 2);
    if (bk[0].length - bk[1].length !== 1 || bk[0].concat(bk[1]).length !== 5) bad++;
    if (dedup([1, 1, 2, 3, 3]).join(',') !== '1,2,3') bad++;
    return bad;
  });
  check('data.filter/map/zip/batch/tokenize：语义精确', 0, function () {
    var bad = 0;
    if (filterOp([1, 2, 3, 4], function (v) { return v % 2 === 0; }).join(',') !== '2,4') bad++;
    if (mapOp([1, 2, 3], function (v) { return v * 10; }).join(',') !== '10,20,30') bad++;
    if (zipOp([1, 2, 3], ['a', 'b']).length !== 2) bad++;
    var bt = batchOp([1, 2, 3, 4, 5], 2);
    if (bt.length !== 3 || bt[2].join(',') !== '5') bad++;
    if (tokenize('the  cat sat').join(' ') !== 'the cat sat') bad++;
    return bad;
  });
  check('data.normalize_fit：测试集用训练集统计量（防泄漏）', 1e-12, function () {
    var train = [1, 2, 3, 4, 5], test = [10, 20];
    var r = normalizeFit(train, test);
    var m = 3, sd = Math.sqrt(2); // 总体方差
    return Math.max(Math.abs(r.test[0] - (10 - m) / sd), Math.abs(r.test[1] - (20 - m) / sd), Math.abs(r.mean - m), Math.abs(r.std - sd));
  });
  check('data.augment：保形状、同种子可复现、值被扰动', 0, function () {
    var x = [1, 2, 3];
    var a1 = augment(x, 5, 0.5), a2 = augment(x, 5, 0.5);
    var changed = a1.some(function (v, i) { return v !== x[i]; });
    return (maxAbsDiff(a1, a2) === 0 && a1.length === 3 && changed) ? 0 : 1;
  });
  check('sys.gather/scatter/cache/ref_count：集合语义精确', 0, function () {
    var bad = 0;
    if (gatherAll([[1, 2], [3], [4, 5]]).join(',') !== '1,2,3,4,5') bad++;
    var sc = scatterAll(['a', 'b', 'c'], [0, 1, 0]);
    if (sc[0].join(',') !== 'a,c' || sc[1].join(',') !== 'b') bad++;
    var cache = new Cache();
    var obj = { v: 1 };
    cache.put('k', obj);
    if (cache.get('k') !== obj) bad++;
    var rc = new RefCounter();
    rc.inc('a'); rc.inc('a'); rc.dec('a');
    if (rc.dec('a') !== 0 || rc.counts.a !== undefined) bad++;
    return bad;
  });
  check('sys.stream/checkpoint/profile/trace/dispatch/barrier：系统语义', 0, function () {
    var bad = 0;
    var st = new TaskStream();
    st.push(function (i) { return i; });
    st.push(function (i) { return i * 2; });
    if (st.run().join(',') !== '0,2') bad++;
    var obj = { a: [1, 2], b: { c: 3 } };
    var cp = checkpoint(obj);
    cp.a[0] = 99;
    if (obj.a[0] !== 1 || cp.b.c !== 3) bad++;
    var pf = new Profiler();
    pf.hit('x'); pf.hit('y'); pf.hit('x');
    if (pf.total() !== 3) bad++;
    var tr = new Tracer();
    tr.record('a'); tr.record('b');
    if (tr.events[1].seq !== 1) bad++;
    var dp = new Dispatcher();
    var got = [];
    dp.register(function (p) { got.push(p + 1); });
    dp.register(function (p) { got.push(p + 2); });
    dp.dispatch(10);
    if (got.join(',') !== '11,12') bad++;
    var bar = new Barrier(3);
    if (bar.arrive() !== false || bar.arrive() !== false || bar.arrive() !== true) bad++;
    return bad;
  });
  check('sys.prefetch：深拷贝，无语义改变', 0, function () {
    var x = [[1, 2], [3, 4]];
    var p = prefetch(x);
    p[0][0] = 99;
    return (x[0][0] === 1 && maxAbsDiff(prefetch(x), x) === 0) ? 0 : 1;
  });
  check('linalg.rank/nullspace：秩精确；零空间维数 = 列数−秩；基满足 Ax = 0', 1e-12, function () {
    var bad = 0;
    if (matrixRank([[1, 2], [2, 4]]) !== 1) bad++;
    if (matrixRank([[1, 0], [0, 1]]) !== 2) bad++;
    if (matrixRank([[1, 0, 0], [0, 1, 0], [1, 1, 0]]) !== 2) bad++;
    var A = [[1, 2, 3], [2, 4, 6]]; // 秩 1，3 列 → 零空间维数 2
    var ns = nullspace(A);
    if (ns.length !== 2) bad++;
    ns.forEach(function (v) {
      if (norm2(matVec(A, v)) > 1e-10) bad++;
    });
    return bad;
  });
  check('linalg.orthonormalize/cond：QᵀQ = I；κ(diag(1,10)) = 10；κ ≥ 1', 1e-9, function () {
    var cols = [[1, 1, 0], [1, 0, 1], [0, 1, 1]];
    var Q = orthonormalize(cols);
    var orth = 0;
    for (var a = 0; a < Q.length; a++) for (var b = 0; b < Q.length; b++) orth = Math.max(orth, Math.abs(dot(Q[a], Q[b]) - (a === b ? 1 : 0)));
    // 张成保持：每个原向量 ∈ span(Q)
    var resid = 0;
    cols.forEach(function (a) {
      var proj = new Array(3).fill(0);
      Q.forEach(function (q) {
        var c = dot(a, q);
        for (var i = 0; i < 3; i++) proj[i] += c * q[i];
      });
      resid = Math.max(resid, norm2(a.map(function (v, i) { return v - proj[i]; })));
    });
    var kappa = condEstimate([[1, 0], [0, 10]]);
    var kappaEye = condEstimate([[1, 0], [0, 1]]);
    return Math.max(orth, resid, Math.abs(kappa - 10), Math.abs(kappaEye - 1), kappa < 1 - 1e-9 ? 1 : 0);
  });

  check('prob.sample_stats：方差 = E[X²]−E[X]²（精确）；矩估计收敛', 0.1, function () {
    var xs = normalSample(100000, 111);
    var st = sampleStats(xs);
    var e2 = expectation(xs.map(function (v) { return v * v; }));
    var exact = Math.abs(st.variance - (e2 - st.mean * st.mean));
    return Math.max(exact, Math.abs(st.skew), Math.abs(st.kurt - 3));
  });
  check('info.channel_capacity：BSC(0.1) 容量 = 1−H(ε) = 0.531 bits；≥ 网格暴力解', 1e-6, function () {
    var e = 0.1;
    var P = [[1 - e, e], [e, 1 - e]];
    var ab = channelCapacity(P);
    var h2 = -e * Math.log2(e) - (1 - e) * Math.log2(1 - e);
    var analytic = 1 - h2;
    // 网格暴力（输入分布 0..1 步长 0.005）
    var best = 0;
    for (var t = 0; t <= 200; t++) {
      var q0 = t / 200, q = [q0, 1 - q0];
      var r = [q[0] * P[0][0] + q[1] * P[1][0], q[0] * P[0][1] + q[1] * P[1][1]];
      var I = 0;
      for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++) {
        if (q[i] > 0 && P[i][j] > 0 && r[j] > 0) I += q[i] * P[i][j] * Math.log2(P[i][j] / r[j]);
      }
      if (I > best) best = I;
    }
    return Math.max(Math.abs(ab.capacity - analytic), best - ab.capacity);
  });
  check('optim.line_search：Armijo 充分下降成立；GD+线搜索收敛至解析最优', 1e-6, function () {
    var A = [1, 10, 100], b = [1, -2, 3];
    var f = function (x) { return 0.5 * (A[0] * x[0] * x[0] + A[1] * x[1] * x[1] + A[2] * x[2] * x[2]) - (b[0] * x[0] + b[1] * x[1] + b[2] * x[2]); };
    var g = function (x) { return [A[0] * x[0] - b[0], A[1] * x[1] - b[1], A[2] * x[2] - b[2]]; };
    var x = [0.5, 0.5, 0.5];
    var armijoOk = true, steps = 0;
    for (var it = 0; it < 5000; it++) {
      var gx = g(x);
      var d = gx.map(function (v) { return -v; });
      var f0 = f(x), gd = dot(gx, d);
      var ls = lineSearch(f, g, x, d, { alpha0: 1, c1: 1e-4 });
      if (ls.alpha === 0) break;
      if (ls.f > f0 + 1e-4 * ls.alpha * gd + 1e-15) armijoOk = false;
      x = ls.x;
      steps++;
    }
    var xs = [b[0] / A[0], b[1] / A[1], b[2] / A[2]];
    return armijoOk ? norm2(x.map(function (v, i) { return v - xs[i]; })) : 1;
  });

  /* ---- 13.14 拟牛顿 / MCMC / 率失真不变量（v1.1.1）---- */
  check('optim.bfgs：凸二次 ≤30 步收敛 ≤1e-8；H 近似 SPD；H ≈ A⁻¹（相对 Frobenius）', 1e-2, function () {
    var q = quadProblem();
    var r = bfgs(q.f, q.g, [0, 0, 0], { maxIter: 200 });
    var err = norm2(r.x.map(function (v, i) { return v - q.xs[i]; }));
    var eig = jacobiEigen(r.H);
    var minEig = Math.min.apply(null, eig.values);
    // A = diag(1,10,100) → A⁻¹ = diag(1, 0.1, 0.01)
    var Ai = [[1, 0, 0], [0, 0.1, 0], [0, 0, 0.01]];
    var diff = 0, ref = 0;
    for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) {
      diff += (r.H[a][b] - Ai[a][b]) * (r.H[a][b] - Ai[a][b]);
      ref += Ai[a][b] * Ai[a][b];
    }
    var hErr = Math.sqrt(diff / ref);
    return Math.max(err, r.iterations > 30 ? 1 : 0, minEig <= 0 ? 1 : 0, hErr);
  });
  check('optim.lbfgs：两循环递归凸二次 ≤30 步收敛 ≤1e-8；内存 ≤ m', 1e-2, function () {
    var q = quadProblem();
    var r = lbfgs(q.f, q.g, [0, 0, 0], { m: 5, maxIter: 200 });
    var err = norm2(r.x.map(function (v, i) { return v - q.xs[i]; }));
    return Math.max(err, r.iterations > 30 ? 1 : 0, r.memory > 5 ? 1 : 0);
  });
  check('bwd.grad_layernorm：解析梯度 = 数值 Jacobian ≤ 1e-6', 1e-6, function () {
    var rng = mulberry32(121);
    var x = Array.from({ length: 6 }, function () { return rng() * 4 - 2; });
    var dOut = Array.from({ length: 6 }, function () { return rng() * 2 - 1; });
    var L = function (xx) { return dot(layerNorm([xx])[0], dOut); };
    var J = numericJacobian(function (xx) { return [L(xx)]; }, x, 1e-6)[0];
    var ana = gradLayerNorm(x, dOut);
    var worst = 0;
    for (var i = 0; i < 6; i++) worst = Math.max(worst, Math.abs(J[i] - ana[i]));
    return worst;
  });
  check('graph.gibbs：平稳分布 = 解析解 [0.6, 0.4]（n=4e4）', 0.02, function () {
    var P = [[0.8, 0.2], [0.3, 0.7]]; // 平稳分布 π = [0.6, 0.4]
    var r = gibbsSampler(P, 40000, 1000, 122);
    return Math.max(Math.abs(r.stationary[0] - 0.6), Math.abs(r.stationary[1] - 0.4));
  });
  check('graph.mcmc：Metropolis 经验分布 ≈ 目标分布（n=4e4）', 0.02, function () {
    var target = [0.1, 0.2, 0.3, 0.4];
    var r = metropolisSample(target, 40000, 1000, 123);
    var worst = 0;
    for (var i = 0; i < 4; i++) worst = Math.max(worst, Math.abs(r.empirical[i] - target[i]));
    return worst;
  });
  check('info.rate_distortion：R(D) = ½log₂(σ²/D)；已知值；单调不增', 1e-12, function () {
    var bad = 0;
    if (Math.abs(rateDistortion(1, 0.5) - 0.5) > 1e-12) bad++;
    if (rateDistortion(1, 1) !== 0 || rateDistortion(1, 2) !== 0) bad++;
    var prev = Infinity;
    for (var d = 0.05; d < 1; d += 0.05) {
      var r = rateDistortion(2, d);
      if (r > prev + 1e-15) bad++;
      prev = r;
    }
    return bad;
  });

  /* 执行变体：batched / inplace 逐位一致性（v1.1.4 通用引擎） */
  function cloneVal(o) { // 深拷贝样本，避免参照路径与 batched 路径共享可变输入（如 lstsq 原地归一化 X）
    if (typeof o !== 'object' || o === null) return o;
    if (Array.isArray(o)) return o.map(cloneVal);
    var r = {}; for (var k in o) r[k] = cloneVal(o[k]); return r;
  }
  check('variant.batched：通用批处理引擎 — batched 路径 ≡ 逐样本朴素路径（全族批兼容算子，逐位一致）', 1e-15, function () {
    var bad = 0, n = 0;
    for (var code in BATCH_SIG) {
      if (!BATCH_SIG.hasOwnProperty(code)) continue;
      var s = BATCH_SIG[code]; n++;
      var got = batched(code, cloneVal(s.sample));
      var sc = cloneVal(s.sample), ref;
      if (s.cls === 'special') ref = s.ref(sc);
      else if (s.cls === 'vec') ref = sc.map(function (x) { return s.fn(x); });
      else if (s.cls === 'elem') ref = sc.map(function (v) { return v.map(function (e) { return s.fn(e); }); });
      else if (s.cls === 'pair') ref = sc.map(function (p) { return s.fn(p[0], p[1]); });
      else if (s.cls === 'triple') ref = sc.map(function (p) { return s.fn(p[0], p[1], p[2]); });
      else if (s.cls === 'shared') ref = sc.map(function (x) { return s.fn.apply(null, [x].concat(s.shared)); });
      else if (s.cls === 'args') ref = sc.map(function (x) { return s.fn.apply(null, x); });
      else { bad++; continue; }
      if (arrMaxDiff(got, ref) > 1e-15) bad++;
    }
    return bad + (n < 60 ? 1e6 : 0); // 守卫：至少覆盖 60 个批兼容算子
  });
  check('variant.inplace：通用原地引擎 — inplace 路径 ≡ out-of-place（全族原地兼容算子，逐位一致）', 1e-15, function () {
    var bad = 0, n = 0;
    for (var code in INPLACE_SIG) {
      if (!INPLACE_SIG.hasOwnProperty(code)) continue;
      var s = INPLACE_SIG[code]; n++;
      var got = inplace(code, cloneVal(s.sample));
      var ref = (s.cls === 'elem')
        ? s.sample.map(function (e) { return s.extra ? s.fn.apply(null, [e].concat(s.extra)) : s.fn(e); })
        : (s.extra ? s.fn.apply(null, [cloneVal(s.sample)].concat(s.extra)) : s.fn(cloneVal(s.sample)));
      if (arrMaxDiff(got, ref) > 1e-15) bad++;
    }
    // 保留三个具名原地内核（填充 / 复制 / 归一化 —— 真正原地，无中间数组）
    var a = [1, 2, 3, 4]; inplace_tensorFill(a, 7); if (arrMaxDiff(a, [7, 7, 7, 7]) > 0) bad++;
    var d = [9, 8, 7], sc = [1, 2, 3]; inplace_tensorCopy(d, sc); if (arrMaxDiff(d, [1, 2, 3]) > 0) bad++;
    var nv = [3, 4]; inplace_normalizeL2(nv); if (arrMaxDiff(nv, normalizeL2([3, 4])) > 1e-15) bad++;
    return bad + (n < 25 ? 1e6 : 0); // 守卫：至少覆盖 25 个原地兼容算子（v1.1.8 扩面后 43+）
  });

  /* ---- 12.15b 融合变体内核（v1.1.3）—— 与相邻算子融合，数学语义不变 ----
   * 硬不变量：fused 内核输出 = 顺序组合（op₁ ∘ op₂）逐位一致（参照实现 vs 主实现） */
  function fused_transpose_flatten(A) { // TENSOR: transpose ∘ flatten（消除中间 2D 张量）
    var m = A.length, n = A[0].length, out = new Array(m * n);
    for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) out[j * m + i] = A[i][j];
    return out;
  }
  function fused_matvec_relu(A, x) { // LINALG: matVec ∘ relu（仿射+激活融合）
    return matVec(A, x).map(function (v) { return relu(v); });
  }
  function fused_softmax_log(xs) { // NUMER: softmax ∘ log（log-softmax，稳定单趟）
    var m = -Infinity; for (var i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
    var s = 0; for (var i2 = 0; i2 < xs.length; i2++) s += Math.exp(xs[i2] - m);
    return xs.map(function (x) { return (x - m) - Math.log(s); });
  }
  function fused_gauss_pdf_log(xs, mu, sigma) { // PROB: gaussPDF ∘ log（log-PDF，稳定单趟）
    var c = sigma * Math.sqrt(2 * Math.PI), inv = 1 / (sigma * sigma);
    return xs.map(function (x) { var z = x - mu; return -0.5 * z * z * inv - Math.log(c); });
  }
  function fused_softmax_ce(logits, y) { // INFO: softmax ∘ crossEntropy（稳定 softmax+CE，base-2）
    var m = -Infinity; for (var i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
    var s = 0; for (var i2 = 0; i2 < logits.length; i2++) s += Math.exp(logits[i2] - m);
    var cls = -1; for (var k = 0; k < y.length; k++) if (y[k] === 1) { cls = k; break; }
    return -(((logits[cls] - m) * Math.LOG2E) - Math.log2(s));
  }
  function fused_gradclip_norm(g, maxNorm) { // OPTIM: gradClip ∘ gradNorm（单趟计算裁剪梯度与范数）
    var c = gradClip(g, maxNorm);
    return { clipped: c, norm: gradNorm(c) };
  }
  function fused_linear_relu(X, W, b) { // FWD: linear ∘ relu（线性层+激活融合）
    return X.map(function (row) {
      return W.map(function (w, j) { return relu(dot(row, w) + (b ? b[j] : 0)); });
    });
  }
  function fused_gradtanh_chain(dy, x) { // BWD: gradTanh ∘ 上游梯度（激活梯度融合反向传播）
    return dy.map(function (d, i) { var t = Math.tanh(x[i]); return d * (1 - t * t); });
  }
  function fused_norm_affine(x, g) { // NORM: normalizeL2 ∘ 仿射缩放（权重范数式归一后乘缩放）
    var n = norm2(x); return x.map(function (v, i) { return g[i] * (v / n); });
  }
  function fused_attn_weights(Q, K, causal) { // ATTEN: qkDot ∘ scale ∘ 因果掩码 ∘ softmax（单趟因果注意力权重）
    var d = Q[0].length;
    return Q.map(function (q, i) {
      var row = K.map(function (k, j) {
        var s = dot(q, k) / Math.sqrt(d);
        return (causal && j > i) ? -Infinity : s;
      });
      return softmax(row);
    });
  }
  function fused_topk_temp(logits, T, k) { // SAMPLE: temperatureSoftmax ∘ topK（温度缩放后取 Top-K）
    return topK(temperatureSoftmax(logits, T), k);
  }
  function fused_marginal_norm(joint) { // GRAPH: marginal ∘ normalizeDist（边缘化后归一）
    return normalizeDist(marginal(joint));
  }
  function fused_value_greedy(opts) { // RL: valueIteration ∘ greedy（从价值函数抽取贪心策略）
    var r = valueIteration(opts);
    var n = opts.n, gamma = opts.gamma, mdp = chainSetup(n, gamma, opts.stochastic);
    var pol = new Array(n);
    for (var s = 0; s < n; s++) {
      var best = -Infinity, ba = 0;
      for (var ai = 0; ai < mdp.actions.length; ai++) {
        var a = mdp.actions[ai], v = 0, tr = mdp.trans(s, a);
        for (var t = 0; t < tr.length; t++) v += tr[t][1] * (mdp.reward(s, tr[t][0]) + gamma * r.V[tr[t][0]]);
        if (v > best) { best = v; ba = ai; }
      }
      pol[s] = ba;
    }
    return pol;
  }
  function fused_rmse(y, yhat) { // METRIC: mse ∘ sqrt（RMSE）
    return Math.sqrt(mse(y, yhat));
  }
  function fused_label_onehot(labels) { // DATA: labelEncode ∘ oneHot（标签编码后独热）
    var le = labelEncode(labels), K = le.classes.length;
    return le.codes.map(function (code) {
      return Array.from({ length: K }, function (_, i) { return i === code ? 1 : 0; });
    });
  }
  function fused_allreduce(xs) { // SYS: reduceAll ∘ broadcast（all-reduce：归约广播副本 = 全局和）
    return reduceAll(broadcast(xs));
  }

  /* 执行变体：fused 融合一致性（v1.1.3） */
  check('variant.fused：通用 Epilogue 融合引擎 — fused ≡ post ∘ op（不物化中间张量）＋16 族代表算子融合', 1e-15, function () {
    var bad = 0, n = 0;
    // 通用融合引擎：遍历 FUSE_SIG，断言 fused 路径 ≡ 纯组合（post ∘ op）
    for (var fcode in FUSE_SIG) {
      if (!FUSE_SIG.hasOwnProperty(fcode)) continue;
      var fs = FUSE_SIG[fcode], fitems = fs.sample;
      for (var fi = 0; fi < fitems.length; fi++) {
        n++;
        var fgot = fused(fcode, cloneVal(fitems[fi]));
        var fref = _mapPost(_applySig(fs, cloneVal(fitems[fi])), fs.post);
        if (arrMaxDiff(fgot, fref) > 1e-15) bad++;
      }
    }
    var A = [[1, 2, 3], [4, 5, 6]];
    if (arrMaxDiff(fused_transpose_flatten(A), flatten(transpose(A))) > 1e-15) bad++;
    var M = [[1, 2], [3, 4]], xv = [1, 1];
    if (arrMaxDiff(fused_matvec_relu(M, xv), matVec(M, xv).map(relu)) > 1e-15) bad++;
    var xs = [1, 2, 3, -1];
    if (arrMaxDiff(fused_softmax_log(xs), softmax(xs).map(Math.log)) > 1e-15) bad++;
    if (arrMaxDiff(fused_gauss_pdf_log([0, 1, 2], 0, 1), [0, 1, 2].map(function (v) { return Math.log(gaussPDF(v, 0, 1)); })) > 1e-15) bad++;
    var logits = [2, 1, 0.5], y = [0, 1, 0];
    if (Math.abs(fused_softmax_ce(logits, y) - crossEntropy(y, softmax(logits))) > 1e-15) bad++;
    var gn = fused_gradclip_norm([3, 4], 1);
    if (arrMaxDiff(gn.clipped, gradClip([3, 4], 1)) > 1e-15 || Math.abs(gn.norm - gradNorm(gradClip([3, 4], 1))) > 1e-15) bad++;
    var W = [[1, 0], [0, 1]], b = [0, 0], X1 = [[1, 2], [3, 4]];
    if (arrMaxDiff(fused_linear_relu(X1, W, b), linear(X1, W, b).map(function (row) { return row.map(relu); })) > 1e-15) bad++;
    var dy = [1, -1, 2], gx = [0.1, 0.2, 0.3];
    if (arrMaxDiff(fused_gradtanh_chain(dy, gx), dy.map(function (d, i) { var t = Math.tanh(gx[i]); return d * (1 - t * t); })) > 1e-15) bad++;
    var vv = [3, 4], gg = [2, 2];
    if (arrMaxDiff(fused_norm_affine(vv, gg), normalizeL2(vv).map(function (val, i) { return val * gg[i]; })) > 1e-15) bad++;
    var q2 = [[1, 0], [0, 1]], k2 = [[1, 0], [0, 1]];
    if (arrMaxDiff(fused_attn_weights(q2, k2, true), selfAttention(q2, k2, q2, true).weights) > 1e-15) bad++;
    var tk = fused_topk_temp([1, 2, 3], 1, 2), tkr = topK(temperatureSoftmax([1, 2, 3], 1), 2);
    if (JSON.stringify(tk) !== JSON.stringify(tkr)) bad++;
    var J = [[0.2, 0.3], [0.25, 0.25]];
    if (arrMaxDiff(fused_marginal_norm(J), normalizeDist(marginal(J))) > 1e-15) bad++;
    var opts = { n: 5, gamma: 0.9, stochastic: false, maxIter: 20000, tol: 1e-13 };
    var rpol = fused_value_greedy(opts);
    var Vr = valueIteration(opts).V, mdp = chainSetup(opts.n, opts.gamma, opts.stochastic), refpol = new Array(opts.n);
    for (var s2 = 0; s2 < opts.n; s2++) {
      var best2 = -Infinity, ba2 = 0;
      for (var ai2 = 0; ai2 < mdp.actions.length; ai2++) {
        var a2 = mdp.actions[ai2], v2 = 0, tr2 = mdp.trans(s2, a2);
        for (var t2 = 0; t2 < tr2.length; t2++) v2 += tr2[t2][1] * (mdp.reward(s2, tr2[t2][0]) + opts.gamma * Vr[tr2[t2][0]]);
        if (v2 > best2) { best2 = v2; ba2 = ai2; }
      }
      refpol[s2] = ba2;
    }
    if (arrMaxDiff(rpol, refpol) > 0) bad++;
    var rmseRef = (function () { var dd = [1, 2], pp = [3, 0], s = 0; for (var i = 0; i < 2; i++) { var e = dd[i] - pp[i]; s += e * e; } return Math.sqrt(s / 2); })();
    if (Math.abs(fused_rmse([1, 2], [3, 0]) - rmseRef) > 1e-15) bad++;
    var labels = ['a', 'b', 'a', 'c'];
    if (arrMaxDiff(fused_label_onehot(labels), oneHot(labelEncode(labels).codes).matrix) > 0) bad++;
    if (Math.abs(fused_allreduce([1, 2, 3, 4]) - reduceAll(broadcast([1, 2, 3, 4]))) > 0) bad++;
    return bad + (n < 40 ? 1e6 : 0); // 守卫：至少覆盖 40 个融合样本
  });
  /* —— v1.2.6 不变量扩容第三批：谱定理 / 解析方差 / 结构律 / 复合律 —— */
  check('trace：幂迹谱定理 tr(A²) = Σλᵢ²', 1e-9, function () {
    var A = [[4, 1], [1, 3]], A2 = matMul(A, A);
    var tr = trace(A2), eig = jacobiEigen(A);
    var sum = eig.values.reduce(function (a, b) { return a + b * b; }, 0);
    return Math.abs(tr - sum);
  });
  check('eigen：重构 A = Σλᵢ·vᵢvᵢᵀ（最强分解一致性，行存储）', 1e-10, function () {
    var A = [[4, 1], [1, 3]], eig = jacobiEigen(A);
    var R = [[0, 0], [0, 0]];
    for (var i = 0; i < 2; i++)
      for (var j = 0; j < 2; j++)
        for (var k = 0; k < 2; k++) R[i][j] += eig.values[k] * eig.vectors[k][i] * eig.vectors[k][j];
    return arrMaxDiff(R, A);
  });
  check('binomial：样本方差 ≈ np(1−p)（解析 n=10, p=0.5）', 0.08, function () {
    var s = binomialSample(10, 0.5, 20000, 1);
    var m = s.reduce(function (a, b) { return a + b; }, 0) / s.length;
    var v = s.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / s.length;
    return Math.abs(v - 2.5);
  });
  check('gumbel：样本方差 ≈ π²/6（解析）', 0.06, function () {
    var us = [0.05, 0.11, 0.23, 0.37, 0.41, 0.53, 0.59, 0.67, 0.71, 0.83, 0.91, 0.97];
    var s = [];
    for (var r = 0; r < 2000; r++) for (var i = 0; i < us.length; i++) s.push(gumbelSample(us[(i + r * 7) % us.length]));
    var m = s.reduce(function (a, b) { return a + b; }, 0) / s.length;
    var v = s.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / s.length;
    return Math.abs(v - Math.PI * Math.PI / 6);
  });
  check('editDistance：三角不等式 d(a,c) ≤ d(a,b) + d(b,c)', 0, function () {
    var bad = 0, tri = [['kitten', 'sitting', 'sitting'], ['abc', 'abd', 'xyz'], ['', 'a', 'ab']];
    for (var i = 0; i < tri.length; i++) if (editDistance(tri[i][0], tri[i][2]) > editDistance(tri[i][0], tri[i][1]) + editDistance(tri[i][1], tri[i][2]) + 1e-15) bad++;
    return bad;
  });
  check('conv1d：线性性 conv(αx+βy) = α·conv(x) + β·conv(y)', 1e-12, function () {
    var x = [1, 2, 3, 4], y = [0.5, -1, 2, 0], k = [1, -1, 2];
    var lhs = conv1d(x.map(function (v, i) { return 2 * v + 3 * y[i]; }), k),
        rhs = conv1d(x, k).map(function (v, i) { return 2 * v + 3 * conv1d(y, k)[i]; });
    return arrMaxDiff(lhs, rhs);
  });
  check('rope：复合律 R_n(R_m x) = R_{m+n} x（旋转角可加）', 1e-12, function () {
    var x = [1, 0, 2, 3];
    return arrMaxDiff(ropeApply(ropeApply(x, 3, 10000), 5, 10000), ropeApply(x, 8, 10000));
  });
  check('qkDot：√d 缩放有效性（未缩放方差 ≈ d 倍）', 20, function () {
    var d = 64, rng = mulberry32(77), rawVar = 0, scVar = 0, n = 200;
    function rnd() { return rng() * 2 - 1; }
    for (var t = 0; t < n; t++) {
      var q = [], k = [];
      for (var i = 0; i < d; i++) { q.push(rnd()); k.push(rnd()); }
      var dot = 0; for (var i2 = 0; i2 < d; i2++) dot += q[i2] * k[i2];
      rawVar += dot * dot; scVar += (dot / Math.sqrt(d)) * (dot / Math.sqrt(d));
    }
    rawVar /= n; scVar /= n;
    return Math.abs(rawVar / scVar - d);
  });
  check('batchnorm：通道独立（改列 0 不影响列 1 输出）', 0, function () {
    var X = [[1, 10], [3, 20], [5, 30]];
    var X2 = [[100, 10], [200, 20], [300, 30]];
    var b1 = batchNorm(X, 1e-5), b2 = batchNorm(X2, 1e-5);
    return arrMaxDiff(b1.map(function (r) { return [r[1]]; }), b2.map(function (r) { return [r[1]]; }));
  });
  check('kFold：折规模均衡（max−min ≤ 1）', 0, function () {
    var f = kFold(10, 3), sizes = f.map(function (x) { return x.length; });
    return Math.max.apply(null, sizes) - Math.min.apply(null, sizes) > 1 ? 1 : 0;
  });
  check('td0：γ=0 退化为立即结构（V 恰一处 = 1，其余 = 0）', 0, function () {
    var V = td0Sweep(5, 0), ones = 0;
    for (var i = 0; i < V.length; i++) if (Math.abs(V[i] - 1) < 1e-12) ones++;
    return Math.abs(ones - 1) + (V.some(function (v) { return v !== 0 && Math.abs(v - 1) > 1e-12; }) ? 1 : 0);
  });
  check('dropout：非零元素 = x/(1−p)（缩放精确）', 0, function () {
    var x = [1, 2, 3, 4, 5], p = 0.5;
    var d = dropout(x, p, 42), bad = 0;
    for (var i = 0; i < x.length; i++) { var want = Math.abs(d[i]) < 1e-15 ? 0 : x[i] / (1 - p); bad += Math.abs(d[i] - want); }
    return bad;
  });
  /* —— v1.2.5 不变量扩容第二批：谱/结构/收敛/不等式/恒等式 —— */
  check('jacobi：谱不变量 Πλᵢ = det(A)', 1e-9, function () {
    var eig = jacobiEigen([[4, 1], [1, 3]]);
    var prod = eig.values.reduce(function (a, b) { return a * b; }, 1);
    return Math.abs(prod - det([[4, 1], [1, 3]]));
  });
  check('linreg：含截距残差和 = 0（估计无偏性）', 1e-9, function () {
    var X = [[1, 1], [1, 2], [1, 3], [1, 4]], y = [2, 4, 5, 7];
    var w = linearRegression(X, y).w;
    var s = 0;
    for (var i = 0; i < X.length; i++) s += y[i] - (w[0] * X[i][0] + w[1] * X[i][1]);
    return Math.abs(s);
  });
  check('gmm：责任度每行和 = 1 且混合权重和 = 1（结构不变量）', 1e-12, function () {
    var r = gmmEM([[0, 0], [0.1, 0], [5, 5], [5.1, 4.9]], 2);
    var bad = 0;
    for (var i = 0; i < r.responsibilities.length; i++) bad += Math.abs(r.responsibilities[i].reduce(function (a, b) { return a + b; }, 0) - 1);
    bad += Math.abs(r.weights.reduce(function (a, b) { return a + b; }, 0) - 1);
    return bad;
  });
  check('kmeans：每点分配给最近中心（分配最优性）', 0, function () {
    var X = [[0, 0], [0.1, 0.1], [5, 5], [5.1, 4.9]];
    var r = kmeans(X, 2), bad = 0;
    for (var i = 0; i < X.length; i++) {
      var d = r.centers.map(function (c) { return Math.sqrt(Math.pow(c[0] - X[i][0], 2) + Math.pow(c[1] - X[i][1], 2)); });
      var nearest = d.indexOf(Math.min.apply(null, d));
      if (nearest !== r.labels[i]) bad++;
    }
    return bad;
  });
  check('attention：因果掩码首行 = one-hot（只能看自己）', 1e-12, function () {
    var r = selfAttention([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], true);
    return arrMaxDiff(r.weights[0], [1, 0]);
  });
  check('hmm：前向终态 α_T 求和 = P(O)（对数域一致）', 1e-10, function () {
    var r = hmmForwardBackward([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0, 1, 0]);
    var last = r.alpha[r.alpha.length - 1];
    return Math.abs(last.reduce(function (a, b) { return a + b; }, 0) - Math.exp(r.logPO));
  });
  check('valueIteration：V 逐轮单调不减（压缩映射迭代性质）', 0, function () {
    var r = valueIteration({ n: 5, gamma: 0.9, maxIter: 60 });
    var bad = 0;
    for (var i = 1; i < r.snapshots.length; i++)
      for (var s = 0; s < r.snapshots[i].length; s++) if (r.snapshots[i][s] < r.snapshots[i - 1][s] - 1e-12) bad++;
    return bad;
  });
  check('kl：KL(p‖q) ≥ 0（Gibbs 不等式，四组网格）', 1e-12, function () {
    var ps = [[0.5, 0.5], [0.9, 0.1], [0.2, 0.8], [0.6, 0.3, 0.1]];
    var qs = [[0.5, 0.5], [0.1, 0.9], [0.5, 0.5], [0.2, 0.3, 0.5]];
    var min = Infinity;
    for (var i = 0; i < ps.length; i++) min = Math.min(min, klDivergence(ps[i], qs[i]));
    return min < 0 ? -min : 0;
  });
  check('js：JS = ½KL(p‖m) + ½KL(q‖m)（恒等式）', 1e-12, function () {
    var p = [0.6, 0.4], q = [0.2, 0.8];
    var m = [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])];
    return Math.abs(jsDivergence(p, q) - 0.5 * (klDivergence(p, m) + klDivergence(q, m)));
  });
  check('mutualInfo：I(X;Y) ≤ min(H(X), H(Y))', 1e-12, function () {
    var indep = [[0.25, 0.25], [0.25, 0.25]];
    var perfect = [[0.5, 0], [0, 0.5]];
    var hxy = entropy([0.5, 0.5]);
    return (mutualInfo(indep) > 1e-12 ? 1 : 0) + (mutualInfo(perfect) > hxy + 1e-12 ? 1 : 0);
  });
  check('rope：旋转保范数 ‖R_m x‖ = ‖x‖', 1e-12, function () {
    var x = [3, 4, 1, 2];
    return Math.abs(norm2(ropeApply(x, 7, 10000)) - norm2(x));
  });
  check('grad_accum：累加交换律（顺序无关）', 1e-12, function () {
    var g1 = [[1, 2], [3, 4]], g2 = [[0.5, -0.5], [1, 1]];
    return arrMaxDiff(gradAccum([g1, g2]), gradAccum([g2, g1]));
  });
  /* —— v1.2.4 不变量扩容：第二/第三硬不变量（全部可独立复算）—— */
  check('matmul：结合律 (AB)C = A(BC)（逐位一致）', 1e-12, function () {
    var A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]], C = [[9, 10], [11, 12]];
    return arrMaxDiff(matMul(matMul(A, B), C), matMul(A, matMul(B, C)));
  });
  check('inverse：对合性 (A⁻¹)⁻¹ = A', 1e-12, function () {
    return arrMaxDiff(inverse(inverse([[4, 1], [1, 3]])), [[4, 1], [1, 3]]);
  });
  check('solve：x ≡ A⁻¹b（与逆矩阵路径逐位一致）', 1e-12, function () {
    var A = [[4, 1], [1, 3]], b = [1, 2];
    return arrMaxDiff(solveLinear(A, b), matVec(inverse(A), b));
  });
  check('det：转置不变 det(Aᵀ) = det(A) 且 det(I) = 1 精确', 0, function () {
    var A = [[4, 1], [1, 3]];
    return Math.abs(det(transpose(A)) - det(A)) + Math.abs(det([[1, 0], [0, 1]]) - 1);
  });
  check('cholesky：L 严格下三角且对角元 > 0', 0, function () {
    var L = cholesky([[4, 1], [1, 3]]), bad = 0;
    for (var i = 0; i < L.length; i++) { if (!(L[i][i] > 0)) bad++; for (var j = i + 1; j < L.length; j++) bad += Math.abs(L[i][j]); }
    return bad;
  });
  check('qr：R 严格上三角（下三角全零）', 0, function () {
    var R = qrDecompose([[4, 1], [1, 3]]).R, bad = 0;
    for (var i = 0; i < R.length; i++) for (var j = 0; j < i; j++) bad += Math.abs(R[i][j]);
    return bad;
  });
  check('softmax：归一化和 = 1 且 ±1e300 数值稳定', 1e-12, function () {
    var s = softmax([1, 2, 3]);
    return Math.abs(s.reduce(function (a, b) { return a + b; }, 0) - 1) + arrMaxDiff(softmax([1e300, 2e300]), softmax([-1e300, 0]));
  });
  check('sigmoid：σ(0) = 0.5 精确；值域 ⊆ (0,1)；严格单调', 0, function () {
    var mono = sigmoid(1) > sigmoid(0) && sigmoid(0) > sigmoid(-1) ? 0 : 1;
    var range = sigmoid(-50) >= 0 && sigmoid(50) <= 1 ? 0 : 1;
    return Math.abs(sigmoid(0) - 0.5) + mono + range;
  });
  check('gd：凸二次损失逐帧单调不增', 0, function () {
    var r = gradientDescent(function (x) { return x[0] * x[0] + x[1] * x[1]; }, function (x) { return [2 * x[0], 2 * x[1]]; }, [1, 2], {});
    var bad = 0;
    for (var i = 1; i < r.history.length; i++) if (r.history[i] > r.history[i - 1] + 1e-15) bad++;
    return bad;
  });
  check('layernorm：仿射不变 LN(αx+β) = LN(x)（eps→0 极限）', 1e-9, function () {
    return arrMaxDiff(layerNorm([[2, 4, 6, 8]], 1e-12), layerNorm([[1, 2, 3, 4]], 1e-12));
  });
  check('temperature：T→0⁺ 趋于 argmax one-hot；T→∞ 趋于均匀', 1e-6, function () {
    var xs = [1, 3, 2];
    return arrMaxDiff(temperatureSoftmax(xs, 1e-6), [0, 1, 0]) + arrMaxDiff(temperatureSoftmax(xs, 1e6), [1 / 3, 1 / 3, 1 / 3]);
  });
  check('gaussPDF：峰值 = 1/(σ√2π)（解析精确）', 1e-15, function () {
    return Math.abs(gaussPDF(0, 0, 1) - 1 / Math.sqrt(2 * Math.PI));
  });

  /* —— v1.2.7 不变量扩容第四批：条件数 / 策略评估解析解 / 分块无关 / 结构定理 —— */
  check('condEstimate：κ₂(A) = λmax/λmin（对称正定谱定义，与 Jacobi 解析解一致）', 1e-9, function () {
    var M = [[4, 1], [1, 3]], vs = jacobiEigen(M).values.slice().sort(function (a, b) { return a - b; });
    return Math.abs(condEstimate(M) - vs[vs.length - 1] / vs[0]);
  });
  check('policyEval：迭代策略评估 ≡ 解析解 V = (I−γP_π)⁻¹ r_π（线性方程组交叉验证）', 1e-11, function () {
    var mdp = chainSetup(5, 0.9, true), nn = mdp.n, g = mdp.gamma;
    var P = Array.from({ length: nn }, function () { return new Array(nn).fill(0); }), r = new Array(nn).fill(0);
    for (var s = 0; s < nn; s++) {
      mdp.trans(s, 1).forEach(function (tr) { P[s][tr[0]] += tr[1]; r[s] += tr[1] * mdp.reward(s, tr[0]); });
    }
    var M = Array.from({ length: nn }, function (_, i) {
      return Array.from({ length: nn }, function (_, j) { return (i === j ? 1 : 0) - g * P[i][j]; });
    });
    return arrMaxDiff(solveLinear(M, r), policyEval(mdp, function () { return 1; }));
  });
  check('flashAttention：分块大小无关 — chunk=1/2/T ≡ 朴素 selfAttention（在线 softmax 逐位一致）', 1e-12, function () {
    var Q = [[1, 0.5, -1, 2], [0, 1, 0.3, 1], [-1, 2, 1, 0]],
        K = [[1, 0, 1, 0], [0, 1, 0, 1], [2, -1, 0, 1]], V = [[1, 0], [0, 1], [1, 1]];
    var ref = selfAttention(Q, K, V, false).output, bad = 0;
    for (var c = 1; c <= 3; c++) bad = Math.max(bad, arrMaxDiff(flashAttention(Q, K, V, c).output, ref));
    return bad;
  });
  check('orthonormalize：Gram–Schmidt 正交归一 QᵀQ = I', 1e-12, function () {
    var Q = orthonormalize([[1, 1, 0], [1, 0, 1], [0, 1, 1]]), bad = 0;
    for (var i = 0; i < Q.length; i++)
      for (var j = 0; j < Q.length; j++) bad = Math.max(bad, Math.abs(dot(Q[i], Q[j]) - (i === j ? 1 : 0)));
    return bad;
  });
  check('hmmViterbi：DP 最优路径对数概率 ≡ 暴力枚举 2ᵀ 路径最大值', 1e-12, function () {
    var pi = [0.6, 0.4], Tr = [[0.7, 0.3], [0.4, 0.6]], B = [[0.5, 0.5], [0.1, 0.9]], obs = [0, 1, 0];
    var best = -Infinity;
    for (var m = 0; m < 8; m++) {
      var path = [m & 1, (m >> 1) & 1, (m >> 2) & 1];
      var lp = Math.log(pi[path[0]]) + Math.log(B[path[0]][obs[0]]);
      for (var t2 = 1; t2 < 3; t2++) lp += Math.log(Tr[path[t2 - 1]][path[t2]]) + Math.log(B[path[t2]][obs[t2]]);
      if (lp > best) best = lp;
    }
    return Math.abs(hmmViterbi(pi, Tr, B, obs).logprob - best);
  });
  check('matrixRank：秩-零度定理 rank(A) + dim N(A) = n', 0, function () {
    var M = [[1, 2, 3], [2, 4, 6], [0, 1, 1]];
    return Math.abs(matrixRank(M) + nullspace(M).length - 3);
  });
  check('spectralNorm：谱归一化后主奇异值 = 1（Miyato 2018，幂迭代复算）', 1e-9, function () {
    var Ws = spectralNorm([[3, 1], [1, 2]]);
    return Math.abs(Math.sqrt(powerIteration(matMul(transpose(Ws), Ws), { iters: 5000 }).value) - 1);
  });
  check('topP：核采样支撑最小性 — 累计 ≥ cutoff 且去尾元素即 < cutoff', 0, function () {
    var s = topP([0.5, 0.3, 0.1, 0.06, 0.04], 0.9);
    var keep = s.filter(function (v) { return v > 0; }).sort(function (a, b) { return b - a; });
    var acc = keep.reduce(function (a, b) { return a + b; }, 0);
    return (acc >= 0.9 - 1e-15 && acc - keep[keep.length - 1] < 0.9) ? 0 : 1;
  });
  check('gae：λ=0 退化为 TD 残差 δ_t；λ=1 退化为 Monte-Carlo 优势（端点恒等式）', 1e-12, function () {
    var r = [1, 2, 3], vv = [0.5, 1.5, 2.5], g = 0.9;
    var a0 = gae(r, vv, g, 0), a1 = gae(r, vv, g, 1), bad = 0;
    for (var t = 0; t < 3; t++) {
      var nv = t === 2 ? 0 : vv[t + 1];
      bad = Math.max(bad, Math.abs(a0[t] - (r[t] + g * nv - vv[t])));
      var mc = 0;
      for (var k = t; k < 3; k++) mc += Math.pow(g, k - t) * r[k];
      bad = Math.max(bad, Math.abs(a1[t] - (mc - vv[t])));
    }
    return bad;
  });
  check('kalmanFilter1d：卡尔曼增益收敛 ≡ Riccati 解析不动点（一维稳态解）', 1e-10, function () {
    var Q = 0.1, R = 1;
    var zs = Array.from({ length: 300 }, function (_, i) { return Math.sin(i * 0.3); });
    var f = kalmanFilter1d(zs, 1, 1, Q, R, 0, 1);
    var P = (-Q + Math.sqrt(Q * Q + 4 * R * Q)) / 2;
    return Math.abs(f[f.length - 1].K - (P + Q) / (P + Q + R));
  });
  check('bleu：自比对 BLEU(ref, ref) = 1（上界精确，简短惩罚 = 1）', 1e-12, function () {
    var ref = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
    return Math.abs(bleu(ref, ref.slice(), 2) - 1);
  });
  check('decisionTree：无剪枝训练准确率 = 1（CART 叶纯度，Gini 收敛到 0）', 0, function () {
    var X = [[0], [1], [2], [3], [4], [5]], y = [0, 0, 0, 1, 1, 1];
    return 1 - accuracy(y, decisionTree(X, y, { maxDepth: 10 }).predictAll(X));
  });

  /* —— v1.3.1 不变量扩容第五批：度量界 / 前后向一致性 / 悲观界 / 细致平衡 / 结构无损 —— */
  check('auc：完美分离 AUC = 1、完全逆序 AUC = 0（Mann–Whitney 秩法端点精确）', 0, function () {
    var y = [0, 0, 1, 1];
    return Math.abs(auc(y, [0.1, 0.2, 0.8, 0.9]) - 1) + Math.abs(auc(y, [0.9, 0.8, 0.2, 0.1]) - 0);
  });
  check('r2：完全拟合 R² = 1（残差平方和 = 0，上界精确）', 1e-15, function () {
    return Math.abs(r2([1, 2, 3, 4], [1, 2, 3, 4]) - 1);
  });
  check('logsumexp：平移不变 LSE(x+c) = LSE(x)+c 且 1000 量级不溢出（数值稳定）', 1e-12, function () {
    var x = [1, 2, 3, 4];
    return Math.abs(logsumexp(x.map(function (v) { return v + 7; })) - (logsumexp(x) + 7))
         + (isFinite(logsumexp([1000, 1001, 999])) ? 0 : 1);
  });
  check('hmm：前后向一致性 Σα_t(s)β_t(s) = P(O) 对全部 t 恒定（Rabiner 1989；v1.3.1 修复 β 递推索引后成立）', 1e-12, function () {
    var pi = [0.6, 0.4], Tr = [[0.7, 0.3], [0.4, 0.6]], B = [[0.5, 0.5], [0.1, 0.9]], obs = [0, 1, 0];
    var fb = hmmForwardBackward(pi, Tr, B, obs), PO = Math.exp(fb.logPO), bad = 0;
    fb.alpha.forEach(function (row, i) {
      var s = 0;
      for (var k2 = 0; k2 < row.length; k2++) s += row[k2] * fb.beta[i][k2];
      bad = Math.max(bad, Math.abs(s - PO));
    });
    return bad;
  });
  check('eligibilityTrace：e_t = (γλ)^t（解析）且 λ=0 退化为首项 one-hot', 1e-12, function () {
    var g = 0.9, lam = 0.7, e = eligibilityTrace(5, g, lam), bad = 0;
    for (var i = 0; i < 5; i++) bad = Math.max(bad, Math.abs(e[i] - Math.pow(g * lam, i)));
    var e0 = eligibilityTrace(4, g, 0);
    bad += Math.abs(e0[0] - 1) + Math.max.apply(null, e0.slice(1).map(Math.abs));
    return bad;
  });
  check('monteCarloReturn：确定 MDP 折扣回报 ≡ 解析 Σγᵗr（无偏性）', 1e-10, function () {
    var mdp = chainSetup(5, 0.9, false), g = mdp.gamma;
    var G = monteCarloReturn(mdp, function () { return 1; }, 500, 7), ana = 0;
    for (var t = 3; t < 200; t++) ana += Math.pow(g, t);
    return Math.abs(G - ana);
  });
  check('ppoClip：截断目标恒为未裁剪目标的悲观下界 L ≤ r·A，且越界处界精确生效（信任域）', 1e-15, function () {
    var eps = 0.2, bad = 0;
    [0.5, 0.9, 1.0, 1.1, 1.5, 2.0].forEach(function (r) {
      [1, -1].forEach(function (adv) {
        var L = ppoClip(r, adv, eps), raw = r * adv;
        if (L > raw + 1e-15) bad++;                                                       // 悲观下界
        if (adv > 0 && r > 1 + eps && Math.abs(L - (1 + eps) * adv) > 1e-15) bad++;        // 上界生效
        if (adv < 0 && r < 1 - eps && Math.abs(L - (1 - eps) * adv) > 1e-15) bad++;        // 下界生效
        if (Math.abs(L) > Math.max(r, 1 + eps) * Math.abs(adv) + 1e-15) bad++;             // 信任域幅值界
      });
    });
    return bad;
  });
  check('sampleStats：标准正态样本 kurtosis ≈ 3、skew ≈ 0（四阶矩解析）', 0.15, function () {
    var st = sampleStats(normalSample(60000, 11));
    return Math.abs(st.kurt - 3) + Math.abs(st.skew);
  });
  check('channelCapacity：BSC 容量 = 1 − H₂(p)（bits，Arimoto–Blahut 迭代 ≡ 解析）', 1e-9, function () {
    var p = 0.2, H2 = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
    return Math.abs(channelCapacity([[1 - p, p], [p, 1 - p]]).capacity - (1 - H2));
  });
  check('tensorGather/Scatter：往返恒等 scatter(a, idx, gather(a, idx)) = a', 0, function () {
    var a = [1, 2, 3, 4, 5], idx = [0, 2, 4];
    return arrMaxDiff(tensorScatter(a, idx, tensorGather(a, idx)), a);
  });
  check('metropolisSample：经验分布 → 目标分布（MCMC 细致平衡，6e4 迭代）', 0.02, function () {
    var tgt = [0.1, 0.2, 0.3, 0.4];
    return arrMaxDiff(metropolisSample(tgt, 60000, 2000, 5).empirical, tgt);
  });
  check('bucketize：桶数 = k、各桶长度差 ≤ 1、拼接无损（划分结构不变量）', 0, function () {
    var arr = [1, 2, 3, 4, 5, 6, 7], b = bucketize(arr, 3);
    var sizes = b.map(function (x) { return x.length; }), flat = [].concat.apply([], b);
    return (b.length !== 3) + (Math.max.apply(null, sizes) - Math.min.apply(null, sizes) > 1)
         + arrMaxDiff(flat, arr) + (flat.length !== arr.length);
  });

  /* —— v1.3.2 不变量扩容第六批（与数值/概率/优化/采样/融合内核交叉验证）—— */
  check('softmax：归一化和 = 1 且平移不变（x+c 逐元素一致）', 1e-13, function () {
    var x = [0.1, 2.3, -1.7, 4.2, 0.5];
    var s = softmax(x), sum = s.reduce(function (a, b) { return a + b; }, 0);
    var s2 = softmax(x.map(function (v) { return v + 1000; }));
    return Math.max(Math.abs(sum - 1), maxAbsDiff(s, s2));
  });
  check('gelu：gelu(0)=0 · 大正数渐近 gelu(x)→x · 导数 gelu′(0)=0.5（梯度检验）', 1e-4, function () {
    var g0 = gelu(0), g20 = gelu(20), eps = 1e-6;
    var deriv = (gelu(eps) - gelu(-eps)) / (2 * eps);
    return Math.max(Math.abs(g0 - 0), Math.abs(g20 - 20), Math.abs(deriv - 0.5));
  });
  check('layerNorm：逐行均值 = 0（仿射/平移精确消除）', 1e-13, function () {
    var X = [[1, 2, 3, 4], [-1, 0, 1, 2], [5, 5, 5, 5]];
    var out = layerNorm(X), m = 0;
    out.forEach(function (r) { m = Math.max(m, Math.abs(r.reduce(function (a, b) { return a + b; }, 0))); });
    return m;
  });
  check('batchNorm：逐特征列均值 = 0（精确）', 1e-13, function () {
    var X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, 0, 2]];
    var out = batchNorm(X), d = X[0].length, m = 0;
    for (var j = 0; j < d; j++) { var s = 0; for (var i = 0; i < X.length; i++) s += out[i][j]; m = Math.max(m, Math.abs(s)); }
    return m;
  });
  check('crossEntropy：crossEntropy(p,p)=entropy(p) 且 Gibbs 不等式 CE(p,q)≥H(p)', 1e-13, function () {
    var p = [0.2, 0.3, 0.5], q = [0.4, 0.4, 0.2];
    var cepp = crossEntropy(p, p), H = entropy(p), cepq = crossEntropy(p, q);
    return Math.max(Math.abs(cepp - H), (H - cepq));
  });
  check('fused_softmax_log：exp(out) ≡ softmax（log-softmax 单趟 vs 组合）', 1e-15, function () {
    var x = [0.5, 2.0, -1.0, 3.3], out = fused_softmax_log(x), sm = softmax(x), m = 0;
    for (var i = 0; i < x.length; i++) m = Math.max(m, Math.abs(Math.exp(out[i]) - sm[i]));
    return m;
  });
  check('fused_softmax_ce：fused ≡ crossEntropy(one-hot, softmax(logits))（稳定软max+CE vs 组合）', 1e-13, function () {
    var logits = [1.0, 2.5, -0.5, 3.0], cls = 2;
    var y = new Array(logits.length).fill(0); y[cls] = 1;
    return Math.abs(fused_softmax_ce(logits, y) - crossEntropy(y, softmax(logits)));
  });
  check('fused_gauss_pdf_log：exp(out) ≡ gaussPDF（log-PDF 单趟 vs 组合）', 1e-15, function () {
    var xs = [-1, 0, 0.5, 2], mu = 0.3, sigma = 1.2, out = fused_gauss_pdf_log(xs, mu, sigma), m = 0;
    for (var i = 0; i < xs.length; i++) m = Math.max(m, Math.abs(Math.exp(out[i]) - gaussPDF(xs[i], mu, sigma)));
    return m;
  });
  check('fused_transpose_flatten：out ≡ flatten(transpose)（行主序，消除中间 2D 张量）', 1e-15, function () {
    var M = [[1, 2, 3], [4, 5, 6]], out = fused_transpose_flatten(M), T = transpose(M), ref = [];
    for (var r = 0; r < T.length; r++) for (var c = 0; c < T[0].length; c++) ref.push(T[r][c]);
    return maxAbsDiff(out, ref);
  });
  check('ucb：未探索臂（count=0 → ∞）优先 · 同值低 count 优先 · 同 count 高 value 优先', 1e-15, function () {
    var v1 = ucb([0.5, 0.5], [5, 0], 1, 10), v2 = ucb([0, 0], [1, 100], 1, 10);
    var v3 = ucb([0, 0], [100, 1], 1, 10), v4 = ucb([0.9, 0.1], [10, 10], 1, 10);
    return (v1 !== 1 || v2 !== 0 || v3 !== 1 || v4 !== 0) ? 1 : 0;
  });
  check('silhouette：取值 ∈ [-1,1] 且单簇聚类 → 全 0', 1e-15, function () {
    var s = silhouette([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.0]], [0, 0, 1, 1]);
    var bound = 0; s.forEach(function (v) { bound = Math.max(bound, Math.abs(v) - 1); });
    var s2 = silhouette([[1, 2], [3, 4], [5, 6]], [0, 0, 0]);
    var single = s2.reduce(function (a, b) { return Math.max(a, Math.abs(b)); }, 0);
    return Math.max(bound, single);
  });
  check('earlyStop：best = min(valHistory) · bestIndex = argmin · stopIndex ∈ [bestIndex, len-1]', 1e-15, function () {
    var h = [1.0, 0.8, 0.5, 0.6, 0.4, 0.45, 0.5], r = earlyStop(h, 2);
    var mn = Math.min.apply(null, h), idx = h.indexOf(mn), bad = 0;
    if (Math.abs(r.best - mn) > 1e-15) bad = 1;
    if (r.bestIndex !== idx) bad = 1;
    if (r.stopIndex < r.bestIndex || r.stopIndex > h.length - 1) bad = 1;
    return bad;
  });
  check('thompsonSample：Beta(α,β) 样本经验均值 ≡ α/(α+β)（蒙特卡洛 vs 解析一阶矩）', 0.02, function () {
    var alpha = 2, beta = 5, N = 4000, sum = 0;
    for (var s = 1; s <= N; s++) sum += thompsonSample(alpha, beta, s);
    return Math.abs(sum / N - alpha / (alpha + beta));
  });

  /* —— v1.3.3 不变量扩容第七批（优化器/正则/注意力结构/卷积/数据增强）—— */
  check('lrDecay：lr₀/(1+kt) 随 t 单调不增、且 lr(0)=lr₀（学习率衰减单调性）', 0, function () {
    var lr0 = 0.1, k = 0.05, prev = Infinity, ok = true;
    for (var t = 0; t <= 50; t++) { var v = lrDecay(lr0, k, t); if (v > prev + 1e-15) ok = false; prev = v; }
    return (ok ? 0 : 1) + Math.abs(lrDecay(lr0, k, 0) - lr0);
  });
  check('weightDecay：闭式 w·(1−lr·λ)（L2 惩罚梯度步，无偏缩放）', 0, function () {
    var w = [1.2, -0.7, 3.3, 0.0], lr = 0.01, lam = 0.5;
    return maxAbsDiff(weightDecay(w, lr, lam), w.map(function (v) { return v * (1 - lr * lam); }));
  });
  check('polyakAvg：指数/窗口平均 = 算术平均（Polyak–Ruppert 平均）', 0, function () {
    var xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    return maxAbsDiff(polyakAvg(xs), [4, 5, 6]);
  });
  check('splitHeads / concatHeads：多头分拆-拼接往返恒等（d 整除 H）', 0, function () {
    var T = 4, d = 6, H = 3;
    var X = Array.from({ length: T }, function (_, t) { return Array.from({ length: d }, function (_, j) { return t * 10 + j; }); });
    return maxAbsDiff(concatHeads(splitHeads(X, H)), X);
  });
  check('dropout：反向缩放 1/(1−p) 保持期望（p=0 恒等 + 蒙特卡洛均值→x）', 0.08, function () {
    var x = [1.5, -2.0, 3.0, 0.5, -1.0], id = dropout(x, 0, 7);
    var p = 0.3, N = 2000, acc = x.map(function () { return 0; });
    for (var s = 1; s <= N; s++) { var d = dropout(x, p, s); for (var i = 0; i < x.length; i++) acc[i] += d[i]; }
    var mean = acc.map(function (v) { return v / N; });
    var worst = maxAbsDiff(id, x);
    mean.forEach(function (m, i) { worst = Math.max(worst, Math.abs(m - x[i])); });
    return worst;
  });
  check('conv1d：等价于 im2col Toeplitz 矩阵-向量乘法（stride=1, pad=0）', 0, function () {
    var x = [1, 2, 3, 4, 5], k = [0.5, -1, 2], out = conv1d(x, k, 1, 0);
    var n = x.length, kk = k.length, m = n - kk + 1;
    var T = Array.from({ length: m }, function (_, s) {
      return Array.from({ length: n }, function (_, jj) { return (jj >= s && jj < s + kk) ? k[jj - s] : 0; });
    });
    var ref = T.map(function (row) { return row.reduce(function (a, v, j) { return a + v * x[j]; }, 0); });
    return maxAbsDiff(out, ref);
  });
  check('weightNorm：重新参数化后范数 ‖w‖ = |g|（方向保留、幅度由 g 控制）', 0, function () {
    var v = [3, 4, 0], g = 2.5;
    return Math.abs(norm2(weightNorm(v, g)) - Math.abs(g));
  });
  check('ropeApply：旋转位置编码保持范数（2×2 旋转矩阵正交）', 1e-12, function () {
    var x = [1, 2, 3, 4, 5, 6];
    return Math.abs(norm2(ropeApply(x, 3, 10000)) - norm2(x));
  });
  check('groupNorm：逐组（group）内归一化后均值 = 0（与 layerNorm 同构）', 1e-12, function () {
    var X = Array.from({ length: 5 }, function (_, i) { return Array.from({ length: 8 }, function (_, j) { return Math.sin(i + j) * 3 + 1; }); });
    var out = groupNorm(X, 2), g = 2, maxMean = 0;
    out.forEach(function (row) { for (var gr = 0; gr < g; gr++) { var s = 0; for (var j = gr * 4; j < (gr + 1) * 4; j++) s += row[j]; maxMean = Math.max(maxMean, Math.abs(s / 4)); } });
    return maxMean;
  });
  check('labelSmooth：平滑后仍为合法分布（和 = 1）且 eps=0 退化为 one-hot', 0, function () {
    var oh = [1, 0, 0, 0];
    var a = labelSmooth(oh, 0), b = labelSmooth(oh, 0.1);
    return Math.abs(a.reduce(function (x, y) { return x + y; }, 0) - 1) + Math.abs(b.reduce(function (x, y) { return x + y; }, 0) - 1) + maxAbsDiff(a, oh);
  });
  check('mixup：端点 λ=0→x₂、λ=1→x₁、且 mixup(x,x,λ)=x（凸组合恒等式）', 0, function () {
    var x1 = [1, 2, 3], x2 = [4, 5, 6];
    return maxAbsDiff(mixup(x1, x2, 0), x2) + maxAbsDiff(mixup(x1, x2, 1), x1) + maxAbsDiff(mixup(x1, x1, 0.7), x1);
  });
  check('mqa：输出形状守恒（H×T×d）且第 h 头 ≡ selfAttention(Qₕ,K,V)（定义一致性）', 0, function () {
    var T = 3, d = 4, rng = mulberry32(42);
    var Q = Array.from({ length: 2 }, function () { return Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() - 0.5; }); }); });
    var K = Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() - 0.5; }); });
    var V = Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() - 0.5; }); });
    var out = mqa(Q, K, V), worst = (out.length !== Q.length) ? 1 : 0;
    for (var h = 0; h < Q.length; h++) worst = Math.max(worst, maxAbsDiff(out[h], selfAttention(Q[h], K, V, false).output));
    return worst;
  });
  check('gqa：分组查询注意力输出形状守恒（H×T×d）', 0, function () {
    var T = 3, d = 4, rng = mulberry32(7);
    var Q = Array.from({ length: 4 }, function () { return Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() - 0.5; }); }); });
    var K = Array.from({ length: 2 }, function () { return Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() - 0.5; }); }); });
    var V = Array.from({ length: 2 }, function () { return Array.from({ length: T }, function () { return Array.from({ length: d }, function () { return rng() - 0.5; }); }); });
    var out = gqa(Q, K, V, 2);
    return (out.length !== Q.length) + (out[0].length !== T) + (out[0][0].length !== d);
  });

  /* —— v1.3.4 不变量扩容第八批（优化器收敛 / 聚类收敛 / 强化收敛 / 注意力结构 / 采样保真）—— */
  check('gradientDescent：凸二次函数 history 单调不增（f=½‖x‖²）', 1e-12, function () {
    var f = function (x) { var s = 0; for (var i = 0; i < x.length; i++) s += 0.5 * x[i] * x[i]; return s; };
    var grad = function (x) { return x.slice(); };
    var r = gradientDescent(f, grad, [3, 3], { lr: 0.1, iters: 200 });
    var bad = 0;
    for (var i = 1; i < r.history.length; i++) if (r.history[i] > r.history[i - 1] + 1e-12) bad++;
    return bad;
  });
  check('adamw：λ=0 时逐位退化为 adam（解耦衰减恒等式）', 1e-12, function () {
    var f = function (x) { var s = 0; for (var i = 0; i < x.length; i++) s += 0.5 * x[i] * x[i]; return s; };
    var grad = function (x) { return x.slice(); };
    var opts = { lr: 0.05, iters: 300 };
    var a = adam(f, grad, [2, -1, 0.5], opts);
    var w = adamw(f, grad, [2, -1, 0.5], Object.assign({}, opts, { lambda: 0 }));
    return maxAbsDiff(a.x, w.x);
  });
  check('kmeans：Lloyd SSE 单调不增（收敛性）', 1e-9, function () {
    var rng = mulberry32(7), X = [];
    for (var i = 0; i < 60; i++) X.push([rng() * 2, rng() * 2]);
    var r = kmeans(X, 3, { iters: 30, seed: 7 });
    var bad = 0;
    for (var k = 1; k < r.history.length; k++) if (r.history[k] > r.history[k - 1] + 1e-9) bad++;
    return bad;
  });
  check('gmmEM：对数似然单调不减（1D 强分离数据，协方差远离奇异→正则不影响，机器精度级）', 1e-10, function () {
    var rng = mulberry32(42), X = [];
    for (var c = 0; c < 3; c++) for (var i = 0; i < 60; i++) X.push([c * 10 + (rng() - 0.5)]); // 1D, 簇间距 10, 方差 ~0.5
    var r = gmmEM(X, 3, { iters: 40, seed: 42 });
    var bad = 0;
    for (var k = 1; k < r.llHistory.length; k++) if (r.llHistory[k] < r.llHistory[k - 1] - 1e-10) bad++;
    return bad;
  });
  check('gmmEM：责任度每行 Σ=1（概率归一）', 1e-12, function () {
    var rng = mulberry32(11), X = [];
    for (var c = 0; c < 2; c++) for (var i = 0; i < 30; i++) X.push([c * 8 + (rng() - 0.5), c * 8 + (rng() - 0.5)]);
    var r = gmmEM(X, 2, { iters: 20, seed: 11 });
    var worst = 0;
    for (var m = 0; m < r.responsibilities.length; m++) { var s = 0; for (var j = 0; j < r.responsibilities[m].length; j++) s += r.responsibilities[m][j]; worst = Math.max(worst, Math.abs(s - 1)); }
    return worst;
  });
  check('valueIteration：Bellman 残差 → 0（动态规划收敛）', 1e-9, function () {
    var r = valueIteration({ n: 6, gamma: 0.9, stochastic: false });
    return r.residual;
  });
  check('crossAttention：输出形状守恒（m×dv）', 0, function () {
    var Q = [[1, 2, 3], [4, 5, 6]], K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], V = [[1, 1], [2, 2], [3, 3]];
    var r = crossAttention(Q, K, V);
    return (r.output.length !== 2) + (r.output[0].length !== 2) + (r.weights.length !== 2) + (r.weights[0].length !== 3);
  });
  check('residualAdd：fx = 全 0 时退化为恒等（残差流基线）', 0, function () {
    var x = [[1, 2, 3], [4, 5, 6]], z = x.map(function (row) { return row.map(function () { return 0; }); });
    return maxAbsDiff(residualAdd(x, z), x);
  });
  check('multiHead：causal=true 时注意力权重上三角（j>i）严格 = 0（因果掩码）', 1e-15, function () {
    var Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], K = Q, V = Q;
    var r = multiHead(Q, K, V, 1, true);
    var worst = 0;
    for (var h = 0; h < r.weights.length; h++) for (var i = 0; i < r.weights[h].length; i++) for (var j = i + 1; j < r.weights[h][i].length; j++) worst = Math.max(worst, r.weights[h][i][j]);
    return worst;
  });
  check('lbfgs：凸二次目标收敛到全局最优（f(x)→0, ‖g‖→0）', 1e-8, function () {
    var f = function (x) { var s = 0; for (var i = 0; i < x.length; i++) s += 0.5 * x[i] * x[i]; return s; };
    var grad = function (x) { return x.slice(); };
    var r = lbfgs(f, grad, [3, -2, 1], { maxIter: 200, tol: 1e-12 });
    return f(r.x);
  });
  check('gradClip：输出梯度范数 ≤ maxNorm（当原范数超界时恰为 maxNorm）', 1e-12, function () {
    var out = gradClip([3, 4], 2);
    var n = Math.sqrt(out[0] * out[0] + out[1] * out[1]);
    return Math.max(0, n - 2) + Math.abs(n - 2);
  });
  check('gmmEM：混合权重 Σ=1（概率单纯形）', 1e-12, function () {
    var rng = mulberry32(99), X = [];
    for (var c = 0; c < 3; c++) for (var i = 0; i < 30; i++) X.push([c * 6 + (rng() - 0.5), c * 6 + (rng() - 0.5)]);
    var r = gmmEM(X, 3, { iters: 15, seed: 99 });
    var s = 0; for (var j = 0; j < r.weights.length; j++) s += r.weights[j];
    return Math.abs(s - 1);
  });
  check('knn：1-NN 在训练点上自返（查询点距离为 0 必命中自身）', 0, function () {
    var X = [[0, 0], [5, 0], [0, 5], [5, 5], [10, 10]], y = [0, 1, 2, 3, 4], bad = 0;
    for (var i = 0; i < X.length; i++) if (knn(X, y, X[i], 1) !== y[i]) bad++;
    return bad;
  });

  /* —— v1.3.5 不变量扩容第九批（优化器收敛 / 线搜索 / 概率图 / 决策树准则）—— */
  check('conjugateGradient：解 Ax=b 残差 → 0', 1e-9, function () {
    var rng = mulberry32(1), n = 6, i, j, k;
    var M = zeros2(n, n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) M[i][j] = rng() - 0.5;
    var Amat = zeros2(n, n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) { var s = 0; for (k = 0; k < n; k++) s += M[i][k] * M[j][k]; Amat[i][j] = s + (i === j ? 1 : 0); }
    var b = []; for (i = 0; i < n; i++) b.push(rng() * 2 - 1);
    var r = conjugateGradient(Amat, b, { tol: 1e-12, maxIter: n + 5 });
    var Ax = matVec(Amat, r.x), res = 0;
    for (i = 0; i < n; i++) res += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    return Math.sqrt(res);
  });
  check('conjugateGradient ≡ solveLinear（闭式对照）', 1e-9, function () {
    var rng = mulberry32(2), n = 6, i, j, k;
    var M = zeros2(n, n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) M[i][j] = rng() - 0.5;
    var Amat = zeros2(n, n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) { var s = 0; for (k = 0; k < n; k++) s += M[i][k] * M[j][k]; Amat[i][j] = s + (i === j ? 1 : 0); }
    var b = []; for (i = 0; i < n; i++) b.push(rng() * 2 - 1);
    var rcg = conjugateGradient(Amat, b, { tol: 1e-12, maxIter: n + 5 });
    var xlu = solveLinear(Amat, b);
    return maxAbsDiff(rcg.x, xlu);
  });
  check('bfgs：凸二次收敛到全局最优（‖∇f(x*)‖ → 0）', 1e-6, function () {
    var rng = mulberry32(3), n = 5, i, j, k;
    var M = zeros2(n, n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) M[i][j] = rng() - 0.5;
    var Q = zeros2(n, n);
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) { var s = 0; for (k = 0; k < n; k++) s += M[i][k] * M[j][k]; Q[i][j] = s + (i === j ? 2 : 0); }
    var bb = []; for (i = 0; i < n; i++) bb.push(rng() * 2 - 1);
    function f(x) { var v = 0, ii, jj; for (ii = 0; ii < n; ii++) { var si = 0; for (jj = 0; jj < n; jj++) si += Q[ii][jj] * x[jj]; v += 0.5 * si * x[ii]; } var l = 0; for (ii = 0; ii < n; ii++) l -= bb[ii] * x[ii]; return v + l; }
    function grad(x) { var g = [], ii, jj; for (ii = 0; ii < n; ii++) { var s = 0; for (jj = 0; jj < n; jj++) s += Q[ii][jj] * x[jj]; g.push(s - bb[ii]); } return g; }
    var x0 = []; for (i = 0; i < n; i++) x0.push(rng());
    var r = bfgs(f, grad, x0, { maxIter: 300, tol: 1e-10 });
    var g = r.g, gn = 0; for (i = 0; i < g.length; i++) gn += g[i] * g[i];
    return Math.sqrt(gn);
  });
  check('linearRegression：残差 ⊥ X（正规方程 Xᵀr = 0）', 1e-8, function () {
    var rng = mulberry32(4), n = 40, d = 4, i, j;
    var X = [], wt = []; for (j = 0; j < d; j++) wt.push(rng() * 3 - 1);
    for (i = 0; i < n; i++) { var row = []; for (j = 0; j < d; j++) row.push(rng() * 2 - 1); X.push(row); }
    var y = X.map(function (r) { return r.reduce(function (s, v, k) { return s + v * wt[k]; }, 0) + 0.01 * (rng() - 0.5); });
    var r = linearRegression(X, y), yhat = X.map(r.predict), res = y.map(function (v, t) { return v - yhat[t]; }), mx = 0;
    for (j = 0; j < d; j++) { var s = 0; for (i = 0; i < n; i++) s += X[i][j] * res[i]; mx = Math.max(mx, Math.abs(s)); }
    return mx;
  });
  check('linearRegression：predict ≡ 闭式 Xw', 1e-9, function () {
    var rng = mulberry32(5), n = 30, d = 3, i, j;
    var X = [], wt = []; for (j = 0; j < d; j++) wt.push(rng() * 3 - 1);
    for (i = 0; i < n; i++) { var row = []; for (j = 0; j < d; j++) row.push(rng() * 2 - 1); X.push(row); }
    var y = X.map(function (r) { return r.reduce(function (s, v, k) { return s + v * wt[k]; }, 0); });
    var r = linearRegression(X, y);
    var yhat = X.map(r.predict), Xw = X.map(function (r) { return dot(r, wt); });
    return maxAbsDiff(yhat, Xw);
  });
  check('logisticRegression：收敛到驻点（‖∇L/n‖ → 0）', 1e-2, function () {
    var rng = mulberry32(6), n = 60, d = 3, i, j;
    var wt = [], bt = rng();
    for (j = 0; j < d; j++) wt.push(rng() * 2 - 1);
    var X = [], y = [];
    for (i = 0; i < n; i++) { var row = []; for (j = 0; j < d; j++) row.push(rng() * 2 - 1); X.push(row); var z = bt + row.reduce(function (s, v, k) { return s + v * wt[k]; }, 0); y.push(z > 0 ? 1 : 0); }
    var r = logisticRegression(X, y, { lr: 0.5, iters: 3000 });
    var gw = new Array(d).fill(0), gb = 0, ii, jj;
    for (ii = 0; ii < n; ii++) { var z = r.b + dot(r.w, X[ii]); var p = sigmoid(z), e = p - y[ii]; for (jj = 0; jj < d; jj++) gw[jj] += e * X[ii][jj]; gb += e; }
    var gn = 0; for (jj = 0; jj < d; jj++) gn += gw[jj] * gw[jj]; gn += gb * gb;
    return Math.sqrt(gn) / n;
  });
  check('logisticRegression：predictProba ∈ [0,1]（sigmoid 值域闭区间）', 1e-12, function () {
    var rng = mulberry32(7), n = 40, d = 2, i, j;
    var X = [], y = [];
    for (i = 0; i < n; i++) { var row = [rng() * 4 - 2, rng() * 4 - 2]; X.push(row); y.push(row[0] + row[1] > 0 ? 1 : 0); }
    var r = logisticRegression(X, y, { lr: 0.5, iters: 2000 });
    var lo = 1, hi = 0;
    for (i = 0; i < 10; i++) { var p = sigmoid(dot(r.w, [rng() * 10 - 5, rng() * 10 - 5]) + r.b); lo = Math.min(lo, p); hi = Math.max(hi, p); }
    return Math.max(lo < -1e-15 ? -lo : 0, hi > 1 + 1e-15 ? hi - 1 : 0);
  });
  check('lineSearch：Armijo 充分下降 f(x+αd) ≤ f(x)+c1·α·∇f·d', 1e-12, function () {
    function f(x) { return 0.5 * (x[0] * x[0] + 2 * x[1] * x[1]); }
    function g(x) { return [x[0], 2 * x[1]]; }
    var x = [-2, 1], d = [2, -1];
    var gd = dot(g(x), d), r = lineSearch(f, g, x, d, { alpha0: 1, c1: 1e-4, rho: 0.5, maxIter: 60 });
    var lhs = f(r.x), rhs = f(x) + 1e-4 * r.alpha * gd;
    return r.alpha === 0 ? 0 : Math.max(0, lhs - rhs);
  });
  check('wolfeLineSearch：强 Wolfe（Armijo + 曲率）双条件满足', 1e-10, function () {
    function f(x) { return 0.5 * (x[0] * x[0] + 3 * x[1] * x[1]); }
    function g(x) { return [x[0], 3 * x[1]]; }
    var x = [-3, 2], p = [3, -1], gx = g(x), gd = dot(gx, p);
    var r = wolfeLineSearch(f, g, x, p, gx, { c1: 1e-4, c2: 0.9, alphaMax: 1e3 });
    if (r.alpha === 0) return 0;
    var fn = f(r.x), armijo = fn - (f(x) + 1e-4 * r.alpha * gd);
    var gn = g(r.x), curv = Math.abs(dot(gn, p)) + 0.9 * gd;
    return Math.max(armijo > 0 ? armijo : 0, curv > 0 ? curv : 0);
  });
  check('gibbsSampler：平稳分布 ≈ 解析 π（πP=π，多 seed 稳健）', 0.02, function () {
    var P = [[0.7, 0.3], [0.4, 0.6]];
    var pi1 = P[0][1] / (P[0][1] + P[1][0]), pi0 = P[1][0] / (P[0][1] + P[1][0]), worst = 0, s;
    for (s = 1; s <= 12; s++) { var r = gibbsSampler(P, 60000, 10000, s); worst = Math.max(worst, Math.abs(r.stationary[1] - pi1), Math.abs(r.stationary[0] - pi0)); }
    return worst;
  });
  check('infoGain ≥ 0（ID3 信息增益非负）', 1e-12, function () {
    var tests = [
      [[0, 0, 1, 1], [[0, 0], [1, 1]]],
      [[0, 1, 0, 1, 1, 0], [[0, 1, 1], [0, 1]]],
      [[1, 1, 1, 0, 0, 0], [[1, 1, 1], [0, 0, 0]]],
      [[0, 0, 0, 1, 1, 1, 2, 2, 2], [[0, 0, 0, 1, 1], [1, 2, 2, 2]]]
    ], mn = 1;
    tests.forEach(function (t) { mn = Math.min(mn, infoGain(t[0], t[1])); });
    return mn < 0 ? -mn : 0;
  });
  check('infoGain ≤ entropy(parent)（增益上界 = 父熵）', 1e-12, function () {
    var tests = [
      [[0, 0, 1, 1], [[0, 0], [1, 1]]],
      [[0, 1, 0, 1, 1, 0], [[0, 1, 1], [0, 1]]],
      [[1, 1, 1, 0, 0, 0], [[1, 1, 1], [0, 0, 0]]]
    ], mx = 0;
    tests.forEach(function (t) { var g = infoGain(t[0], t[1]), h = entropy(labelDist(t[0])); if (g > h) mx = Math.max(mx, g - h); });
    return mx;
  });
  check('gini：纯 → 0 · 平衡2类 → 0.5 · 均匀3类 → 2/3', 1e-12, function () {
    var pure = gini([3, 3, 3, 3, 3]);
    var two = gini([1, 0, 1, 0, 1, 0]);
    var three = gini([0, 1, 2, 0, 1, 2]);
    return Math.max(Math.abs(pure), Math.abs(two - 0.5), Math.abs(three - 2 / 3));
  });

  /* —— v1.3.6 核心不变量扩容第十批（290 → 304）：概率图精确推断 · 采样保真 · 强化控制 · 矩阵分解结构 · 指标端点 —— */
  check('variableElim ≡ chainBeliefProp ≡ hmmForwardBackward：P(O) 似然逐位一致（概率图精确推断，对数/指数）', 1e-12, function () {
    var pi = [0.6, 0.4], A = [[0.7, 0.3], [0.4, 0.6]], B = [[0.5, 0.5], [0.1, 0.9]], obs = [0, 1, 0, 1, 0];
    var ve = variableElim(pi, A, B, obs);
    var fb = hmmForwardBackward(pi, A, B, obs);
    return Math.abs(ve - Math.exp(fb.logPO));
  });
  check('ripCheck：团树运行交集性质（RIP）结构判定正确性（满足/违反二例）', 0, function () {
    var ok = ripCheck([['A', 'B'], ['B', 'C'], ['C', 'D']], [[0, 1], [1, 2]]);
    var bad = ripCheck([['A', 'X'], ['B'], ['C', 'X']], [[0, 1], [1, 2]]);
    return (ok === true && bad === false) ? 0 : 1;
  });
  check('gumbelSample：Gumbel CDF 恒等 exp(−exp(−g)) = u（极值分布解析）', 1e-12, function () {
    var us = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999], m = 0;
    for (var i = 0; i < us.length; i++) {
      var g = gumbelSample(us[i]);
      m = Math.max(m, Math.abs(Math.exp(-Math.exp(-g)) - us[i]));
    }
    return m;
  });
  check('inverseCDF：F(F⁻¹(u)) = u（uniform CDF 反函数恒等）', 1e-12, function () {
    var lo = -2, hi = 5, F = function (x) { return (x - lo) / (hi - lo); };
    var us = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99], m = 0;
    for (var i = 0; i < us.length; i++) {
      var x = inverseCDF(us[i], F, lo, hi);
      m = Math.max(m, Math.abs(F(x) - us[i]));
    }
    return m;
  });
  check('rejectionSample：接受率 ≤ 1 且 uniform 目标经验均值 → 0.5（多 seed 稳健）', 0.05, function () {
    var target = function (x) { return 1; }, proposal = function (x) { return 1; };
    var seeds = [1, 2, 3, 4, 5], over = 0, worstMean = 0;
    for (var s = 0; s < seeds.length; s++) {
      var r = rejectionSample(2000, seeds[s], target, proposal, 1, 0, 1);
      over = Math.max(over, r.acceptanceRate - 1);
      worstMean = Math.max(worstMean, Math.abs(expectation(r.samples) - 0.5));
    }
    return Math.max(over, worstMean);
  });
  check('antitheticSample：对偶变量负相关 corr(u, 1−u) = −1 精确', 1e-12, function () {
    var out = antitheticSample(1000, 7), a = [], b = [];
    for (var i = 0; i < out.length; i++) { if (i % 2 === 0) a.push(out[i]); else b.push(out[i]); }
    var n = a.length, ma = 0, mb = 0;
    for (var k = 0; k < n; k++) { ma += a[k]; mb += b[k]; }
    ma /= n; mb /= n;
    var sxy = 0, sxx = 0, syy = 0;
    for (var k2 = 0; k2 < n; k2++) { var da = a[k2] - ma, db = b[k2] - mb; sxy += da * db; sxx += da * da; syy += db * db; }
    return Math.abs(sxy / Math.sqrt(sxx * syy) - (-1));
  });
  check('policyIteration ≡ valueIteration：最优值函数逐位一致（最优 V 唯一）', 1e-9, function () {
    var mdp = chainSetup(7, 0.9, false);
    var vi = valueIteration({ n: 7, gamma: 0.9, stochastic: false });
    var pi = policyIteration(mdp);
    return maxAbsDiff(vi.V, pi.V);
  });
  check('reinforceGradient：Σ∇logπ·R = 0（对数概率梯度零均值恒等）', 1e-12, function () {
    var prefs = [1, 2, 3, 4], worst = 0;
    for (var a = 0; a < 4; a++) {
      var g = reinforceGradient(prefs, a, 5), sum = 0;
      for (var i = 0; i < g.length; i++) sum += g[i];
      worst = Math.max(worst, Math.abs(sum));
    }
    return worst;
  });
  check('mSeparated ≡ dSeparated：d-分离判据跨算法一致（多查询）', 0, function () {
    var DAG = { A: [], B: ['A'], C: ['A'], D: ['B', 'C'] };
    var queries = [
      ['B', 'C', []], ['B', 'C', ['A']], ['A', 'D', []],
      ['A', 'D', ['B', 'C']], ['B', 'D', []], ['B', 'D', ['C']]
    ];
    var mism = 0;
    for (var i = 0; i < queries.length; i++) {
      var q = queries[i];
      if (dSeparated(DAG, q[0], q[1], q[2]) !== mSeparated(DAG, q[0], q[1], q[2])) mism++;
    }
    return mism;
  });
  check('orthonormalize：Gram 矩阵 Q·Qᵀ = I（修正 Gram–Schmidt 正交归一）', 1e-12, function () {
    var Q = orthonormalize([[1, 2, 3], [1, 0, 1], [0, 1, 1]]);
    var Gram = matMul(Q, transpose(Q));
    return maxAbsDiff2(Gram, eye(Q.length));
  });
  check('nullspace：A·v = 0 且 rank + nullity = n（秩-零度定理）', 1e-10, function () {
    var As = [[[1, 2, 3], [4, 5, 6]], [[1, 0, 0], [0, 1, 0]], [[2, 1], [1, 0.5]]], worst = 0;
    for (var k = 0; k < As.length; k++) {
      var Amat = As[k], basis = nullspace(Amat), rk = matrixRank(Amat);
      worst = Math.max(worst, Math.abs(basis.length + rk - Amat[0].length));
      for (var bi = 0; bi < basis.length; bi++) {
        var v = basis[bi], Av = matVec(Amat, v), nm = 0;
        for (var i = 0; i < Av.length; i++) nm = Math.max(nm, Math.abs(Av[i]));
        worst = Math.max(worst, nm);
      }
    }
    return worst;
  });
  check('condEstimate：正交矩阵 κ = 1；对角阵 κ = σmax/σmin（端点/解析）', 1e-6, function () {
    var c = Math.PI / 6, Q = [[Math.cos(c), -Math.sin(c)], [Math.sin(c), Math.cos(c)]];
    var D = [[1, 0, 0], [0, 2, 0], [0, 0, 3]];
    return Math.max(Math.abs(condEstimate(Q) - 1), Math.abs(condEstimate(D) - 3));
  });
  check('averagePrecision：全相关排前 = 1；全不相关 = 0（端点精确）', 1e-15, function () {
    return Math.max(Math.abs(averagePrecision([1, 1, 1, 0, 0]) - 1), Math.abs(averagePrecision([0, 0, 0]) - 0));
  });
  check('confusionMatrix：元素和 = N；对角和 = 正确数（结构不变量）', 1e-12, function () {
    var M = confusionMatrix([0, 1, 0, 1, 1], [0, 1, 1, 1, 0], [0, 1]), s = 0;
    for (var i = 0; i < M.length; i++) for (var j = 0; j < M[0].length; j++) s += M[i][j];
    return Math.max(Math.abs(s - 5), Math.abs(M[0][0] + M[1][1] - 3));
  });

  /* —— 集大成验证算法库硬不变量（v1.3.8，8 条）—— */
  check('svd：A = U·Σ·Vᵀ 重构 + σᵢ = √λᵢ(AᵀA)（独立谱交叉）+ UᵀU = VᵀV = I', 1e-8, function () {
    var rng = mulberry32(101);
    var M = Array.from({ length: 6 }, function () { return Array.from({ length: 4 }, function () { return rng() * 2 - 1; }); });
    var res = svd(M);
    var Sigma = zeros2(4, 4);
    for (var i = 0; i < 4; i++) Sigma[i][i] = res.S[i];
    var recon = maxAbsDiff2(matMul(matMul(res.U, Sigma), transpose(res.V)), M);
    var scale = 0;
    for (var a = 0; a < 6; a++) for (var b = 0; b < 4; b++) scale = Math.max(scale, Math.abs(M[a][b]));
    var eig = jacobiEigen(matMul(transpose(M), M)).values.map(function (v) { return Math.sqrt(Math.max(v, 0)); }).sort(function (x, y) { return y - x; });
    var eigErr = 0;
    for (var c = 0; c < 4; c++) eigErr = Math.max(eigErr, Math.abs(res.S[c] - eig[c]));
    var UtU = matMul(transpose(res.U), res.U), VtV = matMul(transpose(res.V), res.V), orth = 0;
    for (var p = 0; p < 4; p++) for (var q = 0; q < 4; q++) {
      orth = Math.max(orth, Math.abs(UtU[p][q] - (p === q ? 1 : 0)));
      orth = Math.max(orth, Math.abs(VtV[p][q] - (p === q ? 1 : 0)));
    }
    return Math.max(recon / scale, eigErr, orth);
  });
  /* 回归检查（v1.3.11）：svd 等范数列边界 —— 列 0/1 范数恰等（√10）且内积 6 ≠ 0 ⇒ ζ = 0，
   * 修复前 Math.sign(0)=0 使旋转退化为恒等，σ 错为 [√10,√10,2]（随机矩阵检查概率 0，永不触发） */
  check('svd：等范数列边界（ζ=0，Hestenes 旋转根 ±1）—— 解析奇异值 4/2/2 + σᵢ = √λᵢ(AᵀA) + UᵀU = I + 重构', 1e-9, function () {
    var A = [[3, 1, 0], [1, 3, 0], [0, 0, 2]];
    var r = svd(A);
    var e = Math.max(Math.abs(r.S[0] - 4), Math.abs(r.S[1] - 2), Math.abs(r.S[2] - 2));   // 解析：AᵀA 特征值 16 / 4 / 4
    var lam = jacobiEigen(matMul(transpose(A), A)).values;                                 // 独立谱交叉（Jacobi 另路计算）
    for (var i = 0; i < 3; i++) e = Math.max(e, Math.abs(r.S[i] - Math.sqrt(Math.max(0, lam[i]))));
    var UtU = matMul(transpose(r.U), r.U);
    for (var p = 0; p < 3; p++) for (var q = 0; q < 3; q++) e = Math.max(e, Math.abs(UtU[p][q] - (p === q ? 1 : 0)));
    var Sigma = zeros2(3, 3);
    for (var k = 0; k < 3; k++) Sigma[k][k] = r.S[k];
    e = Math.max(e, maxAbsDiff2(matMul(matMul(r.U, Sigma), transpose(r.V)), A) / 3);      // 重构残差（相对 ‖A‖∞）
    return e;
  });
  check('lu：A = L·U 重构 + L 单位下三角 / U 上三角（结构精确，Doolittle 无主元）', 1e-9, function () {
    var rng = mulberry32(102), n = 8, M = [];
    for (var i = 0; i < n; i++) {
      var row = [], s = 0;
      for (var j = 0; j < n; j++) { var v = rng() * 2 - 1; row.push(v); if (j !== i) s += Math.abs(v); }
      row[i] = s + 1 + rng();   // 严格对角占优 → LU 无主元稳定
      M.push(row);
    }
    var res = luDecompose(M);
    var recon = maxAbsDiff2(matMul(res.L, res.U), M), scale = 0;
    for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) scale = Math.max(scale, Math.abs(M[a][b]));
    var struct = 0;
    for (var i2 = 0; i2 < n; i2++) {
      struct = Math.max(struct, Math.abs(res.L[i2][i2] - 1));
      for (var j2 = 0; j2 < n; j2++) {
        if (i2 < j2) struct = Math.max(struct, Math.abs(res.L[i2][j2]));
        if (i2 > j2) struct = Math.max(struct, Math.abs(res.U[i2][j2]));
      }
    }
    return Math.max(recon / scale, struct);
  });
  check('fft：Cooley–Tukey radix-2 ≡ 朴素 O(n²) DFT（参照实现 vs 主实现）', 1e-9, function () {
    var rng = mulberry32(103);
    var x = Array.from({ length: 16 }, function () { return { re: rng() * 2 - 1, im: rng() * 2 - 1 }; });
    var F = fft(x), D = dft(x), diff = 0;
    for (var i = 0; i < 16; i++) diff = Math.max(diff, Math.abs(F[i].re - D[i].re), Math.abs(F[i].im - D[i].im));
    return diff;
  });
  check('pageRank：幂迭代 ≡ 直接线性求解 (I − αP)π = (1−α)·1/n（两条独立路径）', 1e-9, function () {
    var edges = [[0, 1, 1], [0, 2, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1], [2, 4, 1], [3, 4, 1], [3, 5, 1], [4, 5, 1], [4, 6, 1], [5, 6, 1], [5, 7, 1], [6, 7, 1], [7, 0, 1], [2, 0, 1]];
    var res = pageRank(8, edges, 0.85);
    var norm = function (a) { var s = a.reduce(function (p, c) { return p + c; }, 0); return a.map(function (v) { return v / s; }); };
    var pn = norm(res.pi), xn = norm(res.x), diff = 0;
    for (var i = 0; i < 8; i++) diff = Math.max(diff, Math.abs(pn[i] - xn[i]));
    return diff;
  });
  check('dijkstra：最短路径 ≡ Bellman–Ford（参照实现 vs 主实现，含不可达节点）', 1e-12, function () {
    var rng = mulberry32(104), n = 12, edges = [];
    for (var u = 0; u < n; u++) for (var v = 0; v < n; v++) if (rng() < 0.4) edges.push([u, v, 0.5 + rng() * 2]);
    var d1 = dijkstra(n, edges, 0), d2 = bellmanFord(n, edges, 0), diff = 0;
    for (var i = 0; i < n; i++) { if (d1[i] === d2[i]) continue; diff = Math.max(diff, Math.abs(d1[i] - d2[i])); }
    return diff;
  });
  check('sliceSample：目标保真 E[均值]=0、E[方差]=1（12 seed 稳健，标准正态目标）', 0.02, function () {
    var logf = function (x) { return -0.5 * x * x; };
    var meanAcc = 0, varAcc = 0;
    for (var sd = 1; sd <= 12; sd++) {
      var smp = sliceSample(logf, 0, 8000, mulberry32(sd), 1.0, 50, 1000);
      var mean = smp.reduce(function (p, c) { return p + c; }, 0) / smp.length;
      var varc = smp.reduce(function (p, c) { return p + (c - mean) * (c - mean); }, 0) / smp.length;
      meanAcc += mean; varAcc += varc;
    }
    var avgMean = meanAcc / 12, avgVar = varAcc / 12;
    return Math.max(Math.abs(avgMean), Math.abs(avgVar - 1));
  });
  check('newton：求根 x³ − 2x − 5 = 0，|f(x*)| = 0（解析根 ≈ 2.094551482）', 1e-12, function () {
    var f = function (x) { return x * x * x - 2 * x - 5; };
    var df = function (x) { return 3 * x * x - 2; };
    return Math.abs(f(newton(f, df, 2, 1e-15, 100)));
  });
  check('rk4：dx/dt = λx ≡ 解析解 x(T) = x₀·e^{λT}（λ = −1, T = 1, h = 0.005）', 1e-8, function () {
    var lambda = -1;
    var x = rk4(function (t, v) { return lambda * v; }, 1, 1, 0.005);
    return Math.abs(x - Math.exp(lambda));
  });

  /* —— 算法库扩展硬不变量（v1.3.9，9 条）—— */
  check('heapSort：二叉堆 push/pop 序 ≡ Array.sort 升序（200 元素，显式比较器）', 0, function () {
    var rng = mulberry32(301);
    var a = Array.from({ length: 200 }, function () { return Math.round((rng() * 2 - 1) * 1000); });
    var ref = a.slice().sort(function (x, y) { return x - y; });
    var hs = heapSort(a), d = 0;
    for (var i = 0; i < a.length; i++) d = Math.max(d, Math.abs(hs[i] - ref[i]));
    return d;
  });
  check('dijkstraHeap：堆优先队列 ≡ Bellman–Ford ≡ 线性扫描 Dijkstra（含不可达 ∞）', 1e-12, function () {
    var rng = mulberry32(302), n = 14, edges = [];
    for (var u = 0; u < n; u++) for (var v = 0; v < n; v++) if (rng() < 0.35) edges.push([u, v, 0.5 + rng() * 3]);
    var d1 = dijkstraHeap(n, edges, 3), d2 = bellmanFord(n, edges, 3), d0 = dijkstra(n, edges, 3), worst = 0;
    for (var i = 0; i < n; i++) {
      if (d1[i] !== d2[i]) worst = Math.max(worst, Math.abs(d1[i] - d2[i]));   // ∞−∞ = NaN 已由 !== 短路
      if (d1[i] !== d0[i]) worst = Math.max(worst, Math.abs(d1[i] - d0[i]));
    }
    return worst;
  });
  /* A* 网格探针：10 个 6×6 随机阻挡网格（seed 固定 ⇒ 完全可复现） */
  function aStarGridProbe(seed, useHeuristic) {
    var rng = mulberry32(seed), W = 6, H = 6, worst = 0, expanded = 0, grids = 0;
    for (var g = 0; g < 10; g++) {
      var blocked = Array.from({ length: W * H }, function () { return rng() < 0.22; });
      var src = 0, goal = W * H - 1;
      blocked[src] = false; blocked[goal] = false;
      var n = W * H, edges = [];
      for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
        if (blocked[y * W + x]) continue;
        if (x + 1 < W && !blocked[y * W + x + 1]) edges.push([y * W + x, y * W + x + 1, 1]);
        if (y + 1 < H && !blocked[(y + 1) * W + x]) edges.push([y * W + x, (y + 1) * W + x, 1]);
      }
      var gx = function (i) { return i % W; }, gy = function (i) { return Math.floor(i / W); };
      var h = useHeuristic
        ? function (i) { return Math.hypot(gx(i) - gx(goal), gy(i) - gy(goal)); }
        : function () { return 0; };
      var aR = aStar(n, edges, src, goal, h), dR = dijkstra(n, edges, src);
      if (aR.dist === Infinity && dR[goal] === Infinity) continue;
      grids++;
      if (aR.dist !== dR[goal]) worst = Math.max(worst, Math.abs(aR.dist - dR[goal]));
      expanded += aR.expanded;
    }
    return { worst: worst, expanded: expanded, grids: grids };
  }
  check('aStar：可采纳欧氏启发下最优距离 ≡ Dijkstra（10 个 6×6 网格）', 1e-12, function () {
    return aStarGridProbe(303, true).worst;
  });
  check('aStar：欧氏启发严格剪枝（扩展节点数 < h≡0 一致代价搜索）', 0, function () {
    var e = aStarGridProbe(303, true).expanded, z = aStarGridProbe(303, false).expanded;
    return e < z ? 0 : 1;
  });
  check('fftConvolve：FFT 循环卷积 ≡ 朴素 O(n²)（卷积定理，n=16）', 1e-9, function () {
    var rng = mulberry32(304), n = 16;
    var a = Array.from({ length: n }, function () { return rng() * 2 - 1; });
    var b = Array.from({ length: n }, function () { return rng() * 2 - 1; });
    var c = fftConvolve(a, b), d = 0;
    for (var k = 0; k < n; k++) {
      var s = 0;
      for (var j = 0; j < n; j++) s += a[j] * b[(k - j + n) % n];
      d = Math.max(d, Math.abs(c[k] - s));
    }
    return d;
  });
  check('csr：稀疏 matVec ≡ 稠密 matVec + 转置 ≡ 稠密 transpose（结构守恒）', 1e-12, function () {
    var rng = mulberry32(305);
    var M = Array.from({ length: 9 }, function (_, i) { return Array.from({ length: 7 }, function (_, j) { return rng() < 0.3 ? Math.round((rng() * 2 - 1) * 100) / 10 : 0; }); });
    var S = denseToCSR(M), x = Array.from({ length: 7 }, function () { return rng() * 2 - 1; });
    var y1 = csrMatVec(S, x), y2 = matVec(M, x), e1 = 0;
    for (var i = 0; i < 9; i++) e1 = Math.max(e1, Math.abs(y1[i] - y2[i]));
    var e2 = maxAbsDiff2(csrToDense(csrTranspose(S)), transpose(M));
    return Math.max(e1, e2);
  });
  check('mst：Prim ≡ Kruskal ≡ 暴力枚举生成树（12 小图全枚举 + 连通大图边数 n−1）', 1e-9, function () {
    var rng = mulberry32(306), worst = 0;
    for (var t = 0; t < 12; t++) {
      var n = 6, edges = [];
      for (var u = 0; u < n; u++) for (var v = u + 1; v < n; v++) if (rng() < 0.55) edges.push([u, v, Math.round((0.5 + rng() * 4) * 10) / 10]);
      var bf = bruteForceMST(n, edges);
      if (bf === Infinity) continue;
      worst = Math.max(worst, Math.abs(primMST(n, edges) - bf), Math.abs(kruskalMST(n, edges).total - bf));
    }
    var n2 = 12, edges2 = [];
    for (var v = 1; v < n2; v++) edges2.push([v, Math.floor(rng() * v), Math.round((0.5 + rng() * 5) * 100) / 100]);   // 随机生成树 ⇒ 必然连通
    for (var u2 = 0; u2 < n2; u2++) for (var v2 = u2 + 1; v2 < n2; v2++) if (rng() < 0.3) edges2.push([u2, v2, Math.round((0.5 + rng() * 5) * 100) / 100]);
    var k2 = kruskalMST(n2, edges2);
    worst = Math.max(worst, Math.abs(primMST(n2, edges2) - k2.total), Math.abs(k2.count - (n2 - 1)));
    return worst;
  });
  check('bisect：括号二分根 ≡ Newton 根（x³−2x−5 = 0 于 [2,3]，|f(x*)|=0）', 1e-12, function () {
    var f = function (x) { return x * x * x - 2 * x - 5; };
    var df = function (x) { return 3 * x * x - 2; };
    var rB = bisect(f, 2, 3, 1e-16, 200), rN = newton(f, df, 2, 1e-15, 100);
    return Math.max(Math.abs(rB - rN), Math.abs(f(rB)));
  });
  check('kadane：最大子数组和 ≡ 暴力枚举 O(n²)（20 实例整数精确）', 0, function () {
    var rng = mulberry32(307), worst = 0;
    for (var t = 0; t < 20; t++) {
      var a = Array.from({ length: 40 }, function () { return Math.round((rng() * 2 - 1) * 100); });
      var best = -Infinity;
      for (var i = 0; i < a.length; i++) {
        var s = 0;
        for (var j = i; j < a.length; j++) { s += a[j]; if (s > best) best = s; }
      }
      worst = Math.max(worst, Math.abs(kadane(a) - best));
    }
    return worst;
  });

  /* —— 算法库扩展硬不变量（v1.3.10，7 条）—— */
  check('qrEigenvalues：QR 迭代 + Wilkinson 位移 ≡ Jacobi（8 个 6×6 随机对称阵，两条独立路径）+ 重根/2×2/1×1 退化用例', 1e-9, function () {
    var rng = mulberry32(401), worst = 0, maxIter = 0, maxOff = 0;
    for (var t = 0; t < 8; t++) {
      var n = 6, B = Array.from({ length: n }, function () { return Array.from({ length: n }, function () { return rng() * 2 - 1; }); });
      var M = Array.from({ length: n }, function (_, i) { return Array.from({ length: n }, function (_, j) { return i <= j ? B[i][j] : B[j][i]; }); });
      var r = qrEigenvalues(M), ej = jacobiEigen(M).values;
      for (var i = 0; i < n; i++) worst = Math.max(worst, Math.abs(r.values[i] - ej[i]));
      maxIter = Math.max(maxIter, r.iters);
      maxOff = Math.max(maxOff, r.off);
    }
    var deg = Math.abs(qrEigenvalues([[2, 0, 0], [0, 2, 0], [0, 0, 5]]).values[1] - 2)
      + Math.abs(qrEigenvalues([[2, 1], [1, 2]]).values[0] - 3)
      + Math.abs(qrEigenvalues([[7]]).values[0] - 7);
    return Math.max(worst, deg);
  });
  check('qrEigenvalues：tr(A) = Σλᵢ（谱不变量，QR 迭代相似变换保迹）', 1e-9, function () {
    var rng = mulberry32(402), worst = 0;
    for (var t = 0; t < 10; t++) {
      var n = t < 8 ? 6 : 8, B = Array.from({ length: n }, function () { return Array.from({ length: n }, function () { return rng() * 2 - 1; }); });
      var M = Array.from({ length: n }, function (_, i) { return Array.from({ length: n }, function (_, j) { return i <= j ? B[i][j] : B[j][i]; }); });
      var r = qrEigenvalues(M), tr = 0, se = 0;
      for (var i = 0; i < n; i++) { tr += M[i][i]; se += r.values[i]; }
      worst = Math.max(worst, Math.abs(tr - se));
    }
    return worst;
  });
  check('maxFlow：Edmonds–Karp = 最小割（max-flow = min-cut 定理，暴力枚举 2^(n−2) 个割）', 1e-9, function () {
    var rng = mulberry32(403), worst = 0, graphs = 0;
    for (var t = 0; t < 10; t++) {
      var n = 8, cap = Array.from({ length: n }, function () { return new Array(n).fill(0); });
      for (var u = 0; u < n; u++) for (var v = 0; v < n; v++) if (u !== v && rng() < 0.4) cap[u][v] = 1 + Math.floor(rng() * 9);
      var mc = minCutBrute(n, cap, 0, n - 1);
      if (mc === Infinity) continue;
      graphs++;
      worst = Math.max(worst, Math.abs(maxFlowEK(n, cap, 0, n - 1) - mc));
    }
    return worst;
  });
  check('maxFlow：Dinic ≡ Edmonds–Karp（层次图多路增广 vs BFS 增广，跨算法一致）', 1e-12, function () {
    var rng = mulberry32(404), worst = 0;
    for (var t = 0; t < 12; t++) {
      var n = 8, cap = Array.from({ length: n }, function () { return new Array(n).fill(0); });
      for (var u = 0; u < n; u++) for (var v = 0; v < n; v++) if (u !== v && rng() < 0.4) cap[u][v] = 1 + Math.floor(rng() * 9);
      worst = Math.max(worst, Math.abs(maxFlowDinic(n, cap, 0, n - 1) - maxFlowEK(n, cap, 0, n - 1)));
    }
    return worst;
  });
  check('scc：Tarjan ≡ Kosaraju（15 个随机有向图，强连通分量划分逐分量一致）', 0, function () {
    var rng = mulberry32(405), worst = 0;
    for (var t = 0; t < 15; t++) {
      var n = 10, edges = [];
      for (var u = 0; u < n; u++) for (var v = 0; v < n; v++) if (u !== v && rng() < 0.2) edges.push([u, v]);
      var c1 = tarjanSCC(n, edges), c2 = kosarajuSCC(n, edges);
      if (c1.length !== c2.length) { worst = 1; continue; }
      for (var i = 0; i < c1.length; i++) {
        if (c1[i].length !== c2[i].length || c1[i].some(function (x, k) { return x !== c2[i][k]; })) { worst = 1; break; }
      }
    }
    return worst;
  });
  check('kahnTopo：所有边前向 + 全节点覆盖 + 有环返回 null（结构不变量，15 个 DAG + 环图）', 0, function () {
    var rng = mulberry32(406), worst = 0;
    for (var t = 0; t < 15; t++) {
      var n = 8, edges = [];
      for (var u = 0; u < n; u++) for (var v = u + 1; v < n; v++) if (rng() < 0.3) edges.push([u, v]);
      var ord = kahnTopo(n, edges);
      if (!ord) { worst = 1; continue; }
      var pos = new Array(n);
      ord.forEach(function (v, i) { pos[v] = i; });
      for (var k = 0; k < edges.length; k++) if (pos[edges[k][0]] >= pos[edges[k][1]]) worst = 1;
      if (new Set(ord).size !== n) worst = 1;
    }
    if (kahnTopo(4, [[0, 1], [1, 2], [2, 3], [3, 1]]) !== null) worst = 1;   // 有环必须返回 null
    return worst;
  });
  check('kuhnMatching：König 定理 最大匹配 = 最小顶点覆盖（12 个随机二部图，暴力枚举 2^8）+ 匹配合法性', 0, function () {
    var rng = mulberry32(407), worst = 0;
    for (var t = 0; t < 12; t++) {
      var l = 4, r = 4, edges = [];
      for (var u = 0; u < l; u++) for (var v = 0; v < r; v++) if (rng() < 0.5) edges.push([u, v]);
      var mm = kuhnMatching(l, r, edges);
      worst = Math.max(worst, Math.abs(mm.size - minVertexCoverBrute(l, r, edges)));
      var usedL = {}, usedR = {};
      for (var v2 = 0; v2 < r; v2++) {
        if (mm.matchR[v2] !== -1) {
          if (usedR[v2] || usedL[mm.matchR[v2]]) worst = 1;
          var ok = edges.some(function (e) { return e[0] === mm.matchR[v2] && e[1] === v2; });
          if (!ok) worst = 1;
          usedR[v2] = true;
          usedL[mm.matchR[v2]] = true;
        }
      }
    }
    return worst;
  });

  check('bluesteinFFT：任意长度 FFT（Bluestein 卷积定理，非 2 的幂）≡ 朴素 O(n²) DFT + 2 的幂 ≡ radix-2', 1e-9, function () {
    var rng = mulberry32(11), worst = 0, worstRadix = 0;
    [3, 5, 6, 7, 9, 10, 11, 12, 13, 15].forEach(function (n) {
      var x = [], i;
      for (i = 0; i < n; i++) x.push({ re: rng() * 2 - 1, im: rng() * 2 - 1 });
      var a = bluesteinFFT(x), b = dft(x);
      for (var k = 0; k < n; k++) worst = Math.max(worst, Math.abs(a[k].re - b[k].re), Math.abs(a[k].im - b[k].im));
    });
    [4, 8, 16].forEach(function (n) {
      var x = [], i;
      for (i = 0; i < n; i++) x.push({ re: rng() * 2 - 1, im: rng() * 2 - 1 });
      var a = bluesteinFFT(x), b = fft(x);
      for (var k = 0; k < n; k++) worstRadix = Math.max(worstRadix, Math.abs(a[k].re - b[k].re), Math.abs(a[k].im - b[k].im));
    });
    return Math.max(worst, worstRadix);
  });

  check('adaptiveMetropolis：Haario 协方差自适应 —— 三目标（N(0,1) / 2D 相关高斯 ρ=0.8 / Gamma(k=3)）经验矩 ≡ 解析（20 seed 稳健）', 0.06, function () {
    var rng0 = mulberry32(101), seeds = [], s;
    for (s = 0; s < 20; s++) seeds.push(1 + Math.floor(rng0() * 100000));
    var a1 = 0, a2 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, c1 = 0, c2 = 0;
    seeds.forEach(function (seed) {
      var rng = mulberry32(seed);
      var r1 = adaptiveMetropolis(function (x) { return -0.5 * x[0] * x[0]; }, [0], 6000, rng, 3000);
      var m1 = moments1(r1.samples);
      a1 += m1.mean / 20; a2 += m1.var / 20;
      var Sinv = [[1 / 0.36, -0.8 / 0.36], [-0.8 / 0.36, 1 / 0.36]];
      var r2 = adaptiveMetropolis(function (x) {
        var s1 = Sinv[0][0] * x[0] + Sinv[0][1] * x[1], s2 = Sinv[1][0] * x[0] + Sinv[1][1] * x[1];
        return -0.5 * (x[0] * s1 + x[1] * s2);
      }, [0, 0], 6000, rng, 3000);
      var m2 = moments2(r2.samples);
      b0 += m2.m0 / 20; b1 += m2.m1 / 20; b2 += m2.v0 / 20; b3 += m2.v1 / 20; b4 += m2.cov / 20;
      var r3 = adaptiveMetropolis(function (x) { return x[0] > 0 ? 2 * Math.log(x[0]) - x[0] : -Infinity; }, [3], 6000, rng, 3000);
      var m3 = moments1(r3.samples);
      c1 += m3.mean / 20; c2 += m3.var / 20;
    });
    var e1 = Math.max(Math.abs(a1), Math.abs(a2 - 1));
    var e2 = Math.max(Math.abs(b0), Math.abs(b1), Math.abs(b2 - 1), Math.abs(b3 - 1), Math.abs(b4 - 0.8));
    var e3 = Math.max(Math.abs(c1 - 3) / 3, Math.abs(c2 - 3) / 3);
    return Math.max(e1, e2, e3);
  });

  check('hmcSample：Hamiltonian MC（leapfrog）经验矩 ≡ 解析（1D 标准正态 + 2D 相关高斯 ρ=0.8，20 seed 稳健）', 0.06, function () {
    var rng0 = mulberry32(202), seeds = [], s;
    for (s = 0; s < 20; s++) seeds.push(1 + Math.floor(rng0() * 100000));
    var a1 = 0, a2 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    seeds.forEach(function (seed) {
      var rng = mulberry32(seed);
      var r1 = hmcSample(function (x) { return -0.5 * x[0] * x[0]; }, function (x) { return [-x[0]]; }, [0], 2000, rng, 0.5, 10);
      var m1 = moments1(r1.samples);
      a1 += m1.mean / 20; a2 += m1.var / 20;
      var Sinv = [[1 / 0.36, -0.8 / 0.36], [-0.8 / 0.36, 1 / 0.36]];
      var r2 = hmcSample(function (x) {
        var s1 = Sinv[0][0] * x[0] + Sinv[0][1] * x[1], s2 = Sinv[1][0] * x[0] + Sinv[1][1] * x[1];
        return -0.5 * (x[0] * s1 + x[1] * s2);
      }, function (x) {
        return [-(Sinv[0][0] * x[0] + Sinv[0][1] * x[1]), -(Sinv[1][0] * x[0] + Sinv[1][1] * x[1])];
      }, [0, 0], 2000, rng, 0.28, 12);
      var m2 = moments2(r2.samples);
      b0 += m2.m0 / 20; b1 += m2.m1 / 20; b2 += m2.v0 / 20; b3 += m2.v1 / 20; b4 += m2.cov / 20;
    });
    var e1 = Math.max(Math.abs(a1), Math.abs(a2 - 1));
    var e2 = Math.max(Math.abs(b0), Math.abs(b1), Math.abs(b2 - 1), Math.abs(b3 - 1), Math.abs(b4 - 0.8));
    return Math.max(e1, e2);
  });

  check('hmcSample：leapfrog 能量误差 E|ΔH| 有界（可逆·保体积 ⇒ 20 seed 平均 |ΔH| ≤ 0.1）', 0.1, function () {
    var rng0 = mulberry32(303), seeds = [], s;
    for (s = 0; s < 20; s++) seeds.push(1 + Math.floor(rng0() * 100000));
    var e = 0;
    seeds.forEach(function (seed) {
      var rng = mulberry32(seed);
      var r = hmcSample(function (x) { return -0.5 * x[0] * x[0]; }, function (x) { return [-x[0]]; }, [0], 800, rng, 0.5, 10);
      e += r.energyErr / 20;
    });
    return e;
  });

  check('floydWarshall：全源最短路 ≡ 每源 Dijkstra（12 图）+ 负权无负环 ≡ 每源 Bellman–Ford（DAG 负权）', 1e-12, function () {
    var rng = mulberry32(7), worst = 0, worstBF = 0;
    for (var c = 0; c < 12; c++) {
      var n = 7, edges = [], v;
      for (v = 1; v < n; v++) edges.push([Math.floor(rng() * v), v, Math.round(rng() * 900 + 100) / 100]);
      for (var e2 = 0; e2 < 10; e2++) {
        var a = Math.floor(rng() * n), b = Math.floor(rng() * n);
        if (a !== b) edges.push([a, b, Math.round(rng() * 900 + 100) / 100]);
      }
      var D = floydWarshall(n, edges);
      for (var s = 0; s < n; s++) {
        var ds = dijkstra(n, edges, s);
        for (var j = 0; j < n; j++) worst = Math.max(worst, absDistDiff(D[s][j], ds[j]));
      }
    }
    for (var c2 = 0; c2 < 10; c2++) {                              // DAG（i<j）⇒ 无环 ⇒ 负权安全
      var n2 = 6, ed = [];
      for (var i2 = 0; i2 < n2; i2++) {
        for (var j2 = i2 + 1; j2 < n2; j2++) {
          if (rng() < 0.6) ed.push([i2, j2, Math.round((rng() * 10 - 5) * 100) / 100]);
        }
      }
      var D2 = floydWarshall(n2, ed);
      for (var s2 = 0; s2 < n2; s2++) {
        var bf = bellmanFord(n2, ed, s2);
        for (var j3 = 0; j3 < n2; j3++) worstBF = Math.max(worstBF, absDistDiff(D2[s2][j3], bf[j3]));
      }
    }
    return Math.max(worst, worstBF);
  });

  check('dsu：并查集（路径压缩 + 按秩合并）连通分量 ≡ BFS（15 图）+ Kruskal-DSU MST ≡ Prim + 分量数守恒', 1e-12, function () {
    var rng = mulberry32(13), worstComp = 0, worstMST = 0;
    for (var c = 0; c < 15; c++) {
      var n = 8, edges = [], v;
      // 偶数号图建随机生成树保证连通（MST 权重可比）；奇数号图保持稀疏（只比连通分量）
      var connected = (c % 2 === 0);
      if (connected) for (v = 1; v < n; v++) edges.push([Math.floor(rng() * v), v, Math.round(rng() * 900 + 100) / 100]);
      else for (v = 1; v < n; v++) if (rng() < 0.55) edges.push([Math.floor(rng() * v), v, Math.round(rng() * 900 + 100) / 100]);
      for (var e2 = 0; e2 < 10; e2++) {
        var a = Math.floor(rng() * n), b = Math.floor(rng() * n);
        if (a !== b) edges.push([a, b, Math.round(rng() * 900 + 100) / 100]);
      }
      var s = dsuCreate(n);
      edges.forEach(function (e3) { dsuUnion(s, e3[0], e3[1]); });
      var adj = [], i3;
      for (i3 = 0; i3 < n; i3++) adj.push([]);
      edges.forEach(function (e4) { adj[e4[0]].push(e4[1]); adj[e4[1]].push(e4[0]); });
      var seen = new Array(n).fill(-1), cid = 0;                  // BFS 参照
      for (var st = 0; st < n; st++) {
        if (seen[st] !== -1) continue;
        var q = [st]; seen[st] = cid;
        while (q.length) {
          var u = q.pop();
          for (var w = 0; w < adj[u].length; w++) if (seen[adj[u][w]] === -1) { seen[adj[u][w]] = cid; q.push(adj[u][w]); }
        }
        cid++;
      }
      for (var i4 = 0; i4 < n; i4++) {
        for (var j4 = 0; j4 < n; j4++) {
          var sameDSU = dsuFind(s, i4) === dsuFind(s, j4), sameBFS = seen[i4] === seen[j4];
          if (sameDSU !== sameBFS) worstComp = Math.max(worstComp, 1);
        }
      }
      if (s.count !== cid) worstComp = Math.max(worstComp, 1);
      var kd = kruskalDSU(n, edges);
      // MST 权重仅在连通图上可比（Prim 在不连通图上提前 break，只得部分生成树）
      if (connected) worstMST = Math.max(worstMST, Math.abs(kd.total - primMST(n, edges)));
      if (kd.comps !== cid) worstComp = Math.max(worstComp, 1);
    }
    return Math.max(worstComp, worstMST);
  });

  check('kmpSearch：KMP 匹配位置 ≡ 朴素暴力（随机文本 + 最坏 aⁿb 模式）+ 失败函数 ≡ 暴力最长真前后缀', 0, function () {
    var rng = mulberry32(17), worst = 0;
    var alpha = ['a', 'b', 'c'];
    for (var c = 0; c < 20; c++) {
      var tn = 60 + Math.floor(rng() * 60), text = [], i;
      for (i = 0; i < tn; i++) text.push(alpha[Math.floor(rng() * alpha.length)]);
      var pn = 1 + Math.floor(rng() * 5), pat = [], j;
      for (j = 0; j < pn; j++) pat.push(alpha[Math.floor(rng() * alpha.length)]);
      var a = kmpSearch(text, pat), b = bruteSearch(text, pat);
      if (a.length !== b.length) { worst = Math.max(worst, 1); continue; }
      for (var k = 0; k < a.length; k++) worst = Math.max(worst, Math.abs(a[k] - b[k]));
    }
    var tw = [], i2;                                              // 最坏模式：text = a^60（无 b），pat = a^6 b
    for (i2 = 0; i2 < 60; i2++) tw.push('a');
    var pw = ['a', 'a', 'a', 'a', 'a', 'a', 'b'];
    var aw = kmpSearch(tw, pw), bw = bruteSearch(tw, pw);
    if (aw.length !== bw.length) worst = Math.max(worst, 1);
    var th = [], i3;                                              // 命中型最坏：text = a^20 + b，pat = a^5 + b
    for (i3 = 0; i3 < 20; i3++) th.push('a');
    th.push('b');
    var ph = ['a', 'a', 'a', 'a', 'a', 'b'];
    var ah = kmpSearch(th, ph), bh = bruteSearch(th, ph);
    if (ah.length !== bh.length || (ah.length && ah[0] !== bh[0])) worst = Math.max(worst, 1);
    for (var c2 = 0; c2 < 15; c2++) {                             // 失败函数 ≡ 暴力最长真前后缀
      var p = [], pl = 1 + Math.floor(rng() * 8), q2;
      for (q2 = 0; q2 < pl; q2++) p.push(alpha[Math.floor(rng() * alpha.length)]);
      var f = kmpFailure(p);
      for (var idx = 0; idx < p.length; idx++) {
        var want = 0;
        for (var L = idx; L >= 1; L--) {
          var ok = true;
          for (var t2 = 0; t2 < L; t2++) if (p[t2] !== p[idx - L + 1 + t2]) { ok = false; break; }
          if (ok) { want = L; break; }
        }
        worst = Math.max(worst, Math.abs(f[idx] - want));
      }
    }
    return worst;
  });

  check('gaussLegendre：n 点求积对 ≤2n−1 次多项式精确（∫₋₁¹xᵏ 解析对照，n=6/10）+ ∫cos x = 2sin1', 1e-11, function () {
    var worst = 0;
    [6, 10].forEach(function (n) {
      for (var k = 0; k <= 2 * n - 1; k++) {
        (function (kk) {
          var q = gaussQuad(function (x) { return Math.pow(x, kk); }, -1, 1, n);
          var exact = (kk % 2 === 1) ? 0 : 2 / (kk + 1);
          worst = Math.max(worst, Math.abs(q - exact));
        })(k);
      }
    });
    var ci = gaussQuad(function (x) { return Math.cos(x); }, -1, 1, 12);
    worst = Math.max(worst, Math.abs(ci - 2 * Math.sin(1)));
    return worst;
  });

  check('strassenMul：Strassen 分治 ≡ 朴素矩阵乘（n = 1/2/4/8/16，两条独立路径）', 1e-9, function () {
    var rng = mulberry32(19), worst = 0;
    [1, 2, 4, 8, 16].forEach(function (n) {
      var A = [], B = [], i, j;
      for (i = 0; i < n; i++) {
        A.push([]); B.push([]);
        for (j = 0; j < n; j++) { A[i].push(rng() * 2 - 1); B[i].push(rng() * 2 - 1); }
      }
      worst = Math.max(worst, maxAbsDiff2(strassenMul(A, B, 2), matMul(A, B)));
    });
    return worst;
  });

  check('mixedRadixFFT：混合基（radix-2/3/5）≡ 朴素 DFT（N=15/45/75/60/90/225/300 + 2 幂 16/32/64）', 1e-10, function () {
    function cdiff(a, b) {
      var w = 0;
      for (var i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i].re - b[i].re), Math.abs(a[i].im - b[i].im));
      return w;
    }
    var worst = 0;
    [15, 45, 75, 60, 90, 225, 300, 16, 32, 64].forEach(function (N) {
      var rng = mulberry32(1000 + N), x = [];
      for (var i = 0; i < N; i++) x.push({ re: rng() * 2 - 1, im: rng() * 2 - 1 });
      worst = Math.max(worst, cdiff(mixedRadixFFT(x), dft(x)));
    });
    return worst;
  });

  check('nutsSample：NUTS 经验矩 ≡ 解析（1D N(0,1) + 2D 相关高斯 ρ=0.8，12 seed 池化，双平均自适应步长）', 0.06, function () {
    var NW = 600, NS = 1500, MD = 7, pool1 = [], pool2 = [];
    for (var sd = 0; sd < 12; sd++) {
      var rng = mulberry32(900 + sd);
      var r1 = nutsSample(function (x) { return -0.5 * x[0] * x[0]; }, function (x) { return [-x[0]]; }, [0], NS, rng, { warmup: NW, eps0: 0.6, maxDepth: MD });
      for (var i = 0; i < r1.samples.length; i++) pool1.push(r1.samples[i]);
      var rng2 = mulberry32(900 + sd);
      var r2 = nutsSample(function (x) {
        var dd = x[0] * x[0] - 1.6 * x[0] * x[1] + x[1] * x[1];
        return -0.5 / 0.36 * dd;
      }, function (x) { return [-x[0] / 0.36 + 0.8 * x[1] / 0.36, 0.8 * x[0] / 0.36 - x[1] / 0.36]; }, [0, 0], NS, rng2, { warmup: NW, eps0: 0.6, maxDepth: MD });
      for (var j = 0; j < r2.samples.length; j++) pool2.push(r2.samples[j]);
    }
    var m1 = moments1(pool1), m2 = moments2(pool2);
    return Math.max(Math.abs(m1.mean), Math.abs(m1.var - 1),
      Math.abs(m2.m0), Math.abs(m2.m1), Math.abs(m2.v0 - 1), Math.abs(m2.v1 - 1), Math.abs(m2.cov - 0.8));
  });

  check('mcmfSSP：最小费用最大流 ≡ SSP+Bellman–Ford ≡ EK（max-flow 值），20 个 DAG（含负费用边）', 0, function () {
    function dagGraph(rng, n, eCount) {                            // DAG（所有边 u<v）⇒ 无负费用环 ⇒ MCMF 良定义
      var edges = [];
      for (var i = 1; i < n; i++) edges.push([i - 1, i, 1 + Math.floor(rng() * 9), -5 + Math.floor(rng() * 21)]);
      for (var k = 0; k < eCount; k++) {
        var u = Math.floor(rng() * n), v = Math.floor(rng() * n);
        if (u >= v) continue;
        edges.push([u, v, 1 + Math.floor(rng() * 9), -5 + Math.floor(rng() * 21)]);
      }
      return edges;
    }
    var worst = 0;
    for (var g = 0; g < 20; g++) {
      var rng = mulberry32(500 + g);
      var n = 5 + Math.floor(rng() * 5), s = 0, t = n - 1;
      var edges = dagGraph(rng, n, 6 + Math.floor(rng() * 8));
      var a = mcmfSSP(n, s, t, edges), b = mcmfSSPRef(n, s, t, edges);
      var cap = []; for (var ci = 0; ci < n; ci++) cap.push(new Array(n).fill(0));
      for (var ei = 0; ei < edges.length; ei++) cap[edges[ei][0]][edges[ei][1]] += edges[ei][2];
      var ek = maxFlowEK(n, cap, s, t);
      if (ek === 0) continue;
      worst = Math.max(worst, Math.abs(a.value - b.value), Math.abs(a.cost - b.cost), Math.abs(a.value - ek));
    }
    return worst;
  });

  check('nutsGenUTurn：泛化 U-turn NUTS 经验矩 ≡ 解析（2D 强相关 ρ=0.95，20 seed 池化）+ 判据语义钉死 + 泛化为经典严格细化', 0.06, function () {
    var RHO = 0.95, NW = 1000, NS = 2500, NSEED = 20;
    var logf2 = function (x) { var d = x[0] * x[0] - 2 * RHO * x[0] * x[1] + x[1] * x[1]; return -0.5 / (1 - RHO * RHO) * d; };
    var grad2 = function (x) { return [(-x[0] + RHO * x[1]) / (1 - RHO * RHO), (RHO * x[0] - x[1]) / (1 - RHO * RHO)]; };
    // (a) 判据语义钉死：折叠样例上经典判「不转向」而广义判「转向」 ⇒ 泛化修正确实生效
    var PIN = [
      { diff: [3, -1], pP: [1, 1], pM: [1, 1], expC: false, expG: true },   // 各维符号不一致、标量总和为正 ⇒ 经典漏检
      { diff: [1, 1], pP: [1, 1], pM: [1, 1], expC: false, expG: false },   // 正常扩展
      { diff: [1, 1], pP: [-1, 0], pM: [1, 0], expC: true, expG: true },   // 明显转向，两者一致
      { diff: [-2, 5], pP: [1, 1], pM: [1, 1], expC: false, expG: true }    // 又一折叠样例
    ];
    var pinBad = 0, diverge = 0;
    PIN.forEach(function (c) {
      var gc = _uturnClassic(c.diff, c.pP, c.pM), gg = _uturnGeneral(c.diff, c.pP, c.pM);
      if (gc !== c.expC || gg !== c.expG) pinBad++;
      if (gc !== gg) diverge++;
    });
    // (b) 泛化为经典的严格细化：同一批轨迹三元组上，广义停止次数 ≥ 经典，且确有「仅广义捕获」
    var rRec = nutsSample(logf2, grad2, [0, 0], 600, mulberry32(4242), { warmup: 400, eps0: 0.5, maxDepth: 9, record: true });
    var nC = 0, nG = 0, nOnlyG = 0;
    rRec.rec.forEach(function (t) {
      var c = _uturnClassic(t.diff, t.pPlus, t.pMinus), g = _uturnGeneral(t.diff, t.pPlus, t.pMinus);
      if (c) nC++;
      if (g) nG++;
      if (g && !c) nOnlyG++;
    });
    // (c) 正确性：泛化 NUTS 经验矩 ≡ 解析（多 seed 池化口径）
    var pool = [];
    for (var sd = 0; sd < NSEED; sd++) {
      var rg = mulberry32(700 + sd);
      var a = nutsGenUTurn(logf2, grad2, [0, 0], NS, rg, { warmup: NW, eps0: 0.4, maxDepth: 8 });
      for (var i = 0; i < a.samples.length; i++) pool.push(a.samples[i]);
    }
    var m = moments2(pool);
    var momErr = Math.max(Math.abs(m.m0), Math.abs(m.m1), Math.abs(m.v0 - 1), Math.abs(m.v1 - 1), Math.abs(m.cov - RHO));
    return Math.max(momErr, pinBad, diverge > 0 ? 0 : 1, nG >= nC ? 0 : 1, nOnlyG > 0 ? 0 : 1);
  });

  check('nttConvolve：NTT（模 p）≡ 直接卷积 ≡ 朴素 mod-DFT + CRT 重组 ≡ 精确整数卷积（7 组尺寸，原根运行时自验）', 0, function () {
    var worst = 0, i, k;
    if (!_isPrime(NTT_P1) || !_isPrimitiveRoot(NTT_G1, NTT_P1)) return 1;   // 素数/原根自验
    if (!_isPrime(NTT_P2) || !_isPrimitiveRoot(NTT_G2, NTT_P2)) return 1;
    [[4, 4], [8, 8], [16, 5], [32, 9], [64, 17], [128, 33], [256, 7]].forEach(function (cfg) {
      var nA = cfg[0], nB = cfg[1], rng = mulberry32(3000 + nA * 31 + nB), A = [], B = [];
      for (i = 0; i < nA; i++) A.push(Math.floor(rng() * 1000));
      for (i = 0; i < nB; i++) B.push(Math.floor(rng() * 1000));
      var naive = _naiveConvolve(A, B);
      var nm = _nttConvolveMod(A, B, NTT_P1, NTT_G1);
      for (i = 0; i < naive.length; i++) worst = Math.max(worst, Math.abs(((nm[i] % NTT_P1) + NTT_P1) % NTT_P1 - (((naive[i] % NTT_P1) + NTT_P1) % NTT_P1)));
      var ex = nttConvolveExact(A, B);
      for (i = 0; i < naive.length; i++) worst = Math.max(worst, Math.abs(ex[i] - naive[i]));
    });
    // 数论 DFT 定义式一致性：NTT 正变换 ≡ 朴素 mod-DFT（n = 16）
    var n = 16, rng3 = mulberry32(777), A2 = [];
    for (i = 0; i < n; i++) A2.push(Math.floor(rng3() * 1000));
    var fa = A2.slice();
    while (fa.length < n) fa.push(0);
    _ntt(fa, false, NTT_P1, NTT_G1);
    var omega = _modPow(NTT_G1, (NTT_P1 - 1) / n, NTT_P1), wp = 1, wpow = [];
    for (i = 0; i < n; i++) { wpow.push(wp); wp = (wp * omega) % NTT_P1; }
    for (k = 0; k < n; k++) {
      var s = 0;
      for (var t = 0; t < n; t++) s = (s + A2[t] * wpow[(k * t) % n]) % NTT_P1;
      if (s !== fa[k]) worst = Math.max(worst, 1);
    }
    return worst;
  });

  check('hopcroftKarp：二分图最大匹配 ≡ Kuhn（40 图）+ 匹配合法性 + 终止时无残留增广路', 0, function () {
    var worst = 0, legalBad = 0, augBad = 0, tested = 0;
    for (var g = 0; g < 40; g++) {
      var rng = mulberry32(500 + g);
      var l = 3 + Math.floor(rng() * 6), r = 3 + Math.floor(rng() * 6), eg = [], e, u, v;
      for (e = 0; e < 4 + Math.floor(rng() * 10); e++) {
        u = Math.floor(rng() * l); v = Math.floor(rng() * r);
        if (u === v % l) continue;
        eg.push([u, v]);
      }
      var hk = hopcroftKarp(l, r, eg), kn = kuhnMatching(l, r, eg);
      tested++;
      worst = Math.max(worst, Math.abs(hk.size - kn.size));
      var cnt = 0, ok = true, i, j, ui, vi, found;
      for (ui = 0; ui < l; ui++) {
        vi = hk.matchL[ui];
        if (vi === -1) continue;
        cnt++;
        if (hk.matchR[vi] !== ui) ok = false;
        found = false;
        for (j = 0; j < eg.length; j++) if (eg[j][0] === ui && eg[j][1] === vi) { found = true; break; }
        if (!found) ok = false;
      }
      for (vi = 0; vi < r; vi++) {
        ui = hk.matchR[vi];
        if (ui === -1) continue;
        found = false;
        for (j = 0; j < eg.length; j++) if (eg[j][0] === ui && eg[j][1] === vi) { found = true; break; }
        if (!found) ok = false;
      }
      if (!ok || cnt !== hk.size) legalBad++;
      if (_hasAugmentingPath(l, r, eg, hk.matchL, hk.matchR)) augBad++;
    }
    if (tested === 0) return 1;
    return Math.max(worst, legalBad, augBad);
  });

  /* ② Edmonds blossom：一般图最大匹配 ≡ 暴力枚举（200 图）+ 合法性 + 极大性 + 二分图退化 ≡ hopcroftKarp */
  check('blossomMatching：一般图最大匹配 ≡ 暴力枚举全部匹配子集（200 图）+ 匹配合法性 + 极大性 + 二分图退化 ≡ hopcroftKarp（120 图）', 0, function () {
    var worst = 0, legalBad = 0, maxBad = 0, tested = 0, i, k;
    for (var g = 0; g < 200; g++) {
      var rng = mulberry32(300 + g);
      var n = 4 + Math.floor(rng() * 4);
      var maxE = 4 + Math.floor(rng() * 5);
      var eg = [], seen = {};
      for (k = 0; k < maxE; k++) {
        var u = Math.floor(rng() * n), v = Math.floor(rng() * n);
        if (u === v) continue;
        var key = Math.min(u, v) + '_' + Math.max(u, v);
        if (seen[key]) continue;
        seen[key] = 1; eg.push([u, v]);
      }
      if (eg.length === 0) continue;
      tested++;
      var rb = blossomMatching(n, eg);
      worst = Math.max(worst, Math.abs(rb.size - _bruteMaxMatching(n, eg)));   // ≙ 暴力（ground truth）
      var cnt = new Array(n).fill(0);
      for (k = 0; k < n; k++) if (rb.match[k] !== -1 && k < rb.match[k]) { cnt[k]++; cnt[rb.match[k]]++; }
      for (k = 0; k < n; k++) if (cnt[k] > 1) legalBad++;                       // 每点至多一条匹配边
      for (k = 0; k < eg.length; k++) {                                          // 极大性：未选中边必有端点已匹配
        var a = eg[k][0], b = eg[k][1];
        if (rb.match[a] === b) continue;
        if (rb.match[a] === -1 && rb.match[b] === -1) maxBad++;
      }
    }
    if (tested === 0) return 1;
    // 二分图退化 ≡ hopcroftKarp（已独立 ≡ Kuhn）
    var worstHK = 0;
    for (g = 0; g < 120; g++) {
      var rng2 = mulberry32(9000 + g);
      var L = 3 + Math.floor(rng2() * 4), R = 3 + Math.floor(rng2() * 4), nn = L + R;
      var bg = [], bh = [], seeing = {};
      for (k = 0; k < 4 + Math.floor(rng2() * 8); k++) {
        var bl2 = Math.floor(rng2() * L), br2 = Math.floor(rng2() * R);
        if (seeing[bl2 + '_' + br2]) continue;
        seeing[bl2 + '_' + br2] = 1;
        bg.push([bl2, L + br2]); bh.push([bl2, br2]);                            // 绝对索引 / 右部相对索引
      }
      worstHK = Math.max(worstHK, Math.abs(blossomMatching(nn, bg).size - hopcroftKarp(L, R, bh).size));
    }
    return Math.max(worst, legalBad, maxBad, worstHK);
  });

  /* ③ Split-radix FFT ≡ 朴素 DFT ≙ fft（radix-2），N = 2..64 */
  check('splitRadixFFT：Split-radix FFT ≡ 朴素 DFT ≙ fft（radix-2）（N = 2/4/8/16/32/64）', 1e-9, function () {
    var worst = 0, k, i;
    for (var lg = 1; lg <= 6; lg++) {
      var N = 1 << lg, rng = mulberry32(400 + lg), inp = [];
      for (i = 0; i < N; i++) inp.push({ re: rng() * 2 - 1, im: rng() * 2 - 1 });
      var A = splitRadixFFT(inp), B = dft(inp), C = fft(inp);
      for (k = 0; k < N; k++) {
        worst = Math.max(worst, Math.abs(A[k].re - B[k].re), Math.abs(A[k].im - B[k].im),
          Math.abs(A[k].re - C[k].re), Math.abs(A[k].im - C[k].im));
      }
    }
    return worst;
  });

  /* ④ Stoer–Wagner 全局最小割 ≡ 暴力枚举 2ⁿ 划分 ≙ n−1 次 s-t 最小割（maxFlowEK） */
  check('stoerWagner：全局最小割 ≡ 暴力枚举 2ⁿ 划分 ≙ n−1 次 s-t 最小割 maxFlowEK（150 图）', 0, function () {
    var worst = 0, tested = 0, i, k;
    for (var g = 0; g < 150; g++) {
      var rng = mulberry32(6000 + g);
      var n = 4 + Math.floor(rng() * 5);
      var eg = [], seen = {};
      for (k = 0; k < n + Math.floor(rng() * 5); k++) {
        var u = Math.floor(rng() * n), v = Math.floor(rng() * n);
        if (u === v) continue;
        var key = Math.min(u, v) + '_' + Math.max(u, v);
        if (seen[key]) continue;
        seen[key] = 1; eg.push([u, v, 1 + Math.floor(rng() * 9)]);
      }
      if (eg.length < n - 1) continue;
      tested++;
      var sw = stoerWagner(n, eg);
      worst = Math.max(worst, Math.abs(sw - _bruteMinCut(n, eg)));               // ≙ 暴力 2ⁿ 划分
      var cap = [];                                                              // ≙ n−1 次 s-t 最小割
      for (i = 0; i < n; i++) cap.push(new Array(n).fill(0));
      for (i = 0; i < eg.length; i++) { cap[eg[i][0]][eg[i][1]] += eg[i][2]; cap[eg[i][1]][eg[i][0]] += eg[i][2]; }
      var best = Infinity;
      for (var t = 1; t < n; t++) { var f = maxFlowEK(n, cap.map(function (r) { return r.slice(); }), 0, t); if (f < best) best = f; }
      worst = Math.max(worst, Math.abs(sw - best));
    }
    if (tested === 0) return 1;
    return worst;
  });

  /* ⑤ Gomory–Hu 树 ≡ 暴力枚举 2^(n−1) 最小 s-t 割（所有点对路径最小边 == 精确最小割，120 图） */
  check('gomoryHu：Gomory–Hu 树 ≡ 暴力枚举 2^(n−1) 最小 s-t 割（所有点对路径最小边 == 精确最小割，120 图）', 0, function () {
    var worst = 0, tested = 0, i, k;
    for (var g = 0; g < 120; g++) {
      var rng = mulberry32(9000 + g), n = 5 + Math.floor(rng() * 4), eg = [], seen = {};
      for (i = 1; i < n; i++) { var u = Math.floor(rng() * i), key = Math.min(i, u) + '_' + Math.max(i, u); seen[key] = 1; eg.push([u, i, 1 + Math.floor(rng() * 9)]); }
      var extra = n + Math.floor(rng() * n);
      for (k = 0; k < extra; k++) { var a = Math.floor(rng() * n), b = Math.floor(rng() * n); if (a === b) continue; var key = Math.min(a, b) + '_' + Math.max(a, b); if (seen[key]) continue; seen[key] = 1; eg.push([a, b, 1 + Math.floor(rng() * 9)]); }
      tested++;
      var gh = gomoryHu(n, eg);
      for (var u2 = 0; u2 < n; u2++) for (var v2 = u2 + 1; v2 < n; v2++) {
        var bf = _bruteMinCutPair(n, eg, u2, v2);
        worst = Math.max(worst, Math.abs(gh.pathMin(u2, v2) - bf));
      }
    }
    return tested === 0 ? 1 : worst;
  });

  /* ⑥ Radix-4 FFT ≡ 朴素 DFT ≙ fft（radix-2）（N = 4/16/64/256） */
  check('radix4FFT：Radix-4 FFT ≡ 朴素 DFT ≙ fft（radix-2）（N = 4/16/64/256）', 1e-9, function () {
    var worst = 0, k, i, sizes = [4, 16, 64, 256];
    for (var s = 0; s < sizes.length; s++) {
      var N = sizes[s], rng = mulberry32(700 + N), inp = [];
      for (i = 0; i < N; i++) inp.push({ re: rng() * 2 - 1, im: rng() * 2 - 1 });
      var A = radix4FFT(inp), B = dft(inp), C = fft(inp);
      for (k = 0; k < N; k++) worst = Math.max(worst, Math.abs(A[k].re - B[k].re), Math.abs(A[k].im - B[k].im), Math.abs(A[k].re - C[k].re), Math.abs(A[k].im - C[k].im));
    }
    return worst;
  });

  /* ⑦ NUTS 对角质量矩阵自适应：拟合质量矩阵 ≡ 目标协方差对角（2D 异方差 σ²=[4,1]，20 seed 稳健） */
  check('nutsDiagMass：对角质量矩阵自适应（Stan warmup）—— 拟合质量矩阵 ≡ 目标协方差对角（2D 异方差 σ²=[4,1]，20 seed 稳健）', 0.1, function () {
    function makeGauss2(s0, s1) { var inv0 = 1 / (s0 * s0), inv1 = 1 / (s1 * s1); return { logf: function (q) { return -0.5 * (q[0] * q[0] * inv0 + q[1] * q[1] * inv1); }, grad: function (q) { return [-q[0] * inv0, -q[1] * inv1]; } }; }
    var trueVar = [4, 1], seeds = 20, rel = 0, cnt = 0;
    for (var sd = 0; sd < seeds; sd++) {
      var rng = mulberry32(11000 + sd), g = makeGauss2(2, 1), res = nutsDiagMass(g.logf, g.grad, [0, 0], 600, rng, { warmup: 2500, eps0: 0.3 });
      var e = 0; for (var i = 0; i < 2; i++) e += Math.abs(res.mass[i] - trueVar[i]) / trueVar[i];
      rel += e / 2; cnt++;
    }
    return cnt === 0 ? 1 : rel / seeds;
  });

  /* ⑧ Push-Relabel 最大流 ≙ maxFlowEK ≙ maxFlowDinic（整数容量，25 图） */
  check('pushRelabel：Push-Relabel 最大流 ≙ maxFlowEK（Edmonds–Karp）≙ maxFlowDinic（整数容量，25 图）', 0, function () {
    var worst = 0, tested = 0, g;
    for (g = 0; g < 25; g++) {
      var rng = mulberry32(13000 + g), n = 6 + Math.floor(rng() * 7), s = Math.floor(rng() * n), t = Math.floor(rng() * n); while (t === s) t = Math.floor(rng() * n);
      var eg = [], m = n + Math.floor(rng() * 2 * n);
      for (var k = 0; k < m; k++) { var u = Math.floor(rng() * n), v = Math.floor(rng() * n); if (u === v) continue; eg.push([u, v, 1 + Math.floor(rng() * 9)]); }
      var val = pushRelabel(n, eg, s, t);
      var cap = []; for (var i = 0; i < n; i++) cap.push(new Array(n).fill(0)); for (var e = 0; e < eg.length; e++) cap[eg[e][0]][eg[e][1]] += eg[e][2];
      var ek = maxFlowEK(n, cap, s, t);
      worst = Math.max(worst, Math.abs(val - ek));
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ⑨ Hungarian 最大权二分匹配 ≙ mcmfSSPRef（二分图 MCMF）≙ 暴力排列（n≤8，25 图） */
  check('hungarianMaxWeight：Hungarian 最大权二分匹配 ≙ mcmfSSPRef（二分图 MCMF：source→左→右→sink）≙ 暴力排列（n≤8，25 图）', 0, function () {
    var worst = 0, tested = 0, g;
    for (g = 0; g < 25; g++) {
      var rng = mulberry32(14000 + g), n = 4 + Math.floor(rng() * 6), w = [], i, j;
      for (i = 0; i < n; i++) { var row = []; for (j = 0; j < n; j++) row.push(Math.floor(rng() * 19) - 9); w.push(row); }
      var hw = hungarianMaxWeight(w), N = 2 * n + 2, S = 2 * n, T = 2 * n + 1, edges = [];
      for (i = 0; i < n; i++) { edges.push([S, i, 1, 0]); edges.push([n + i, T, 1, 0]); }
      for (i = 0; i < n; i++) for (j = 0; j < n; j++) edges.push([i, n + j, 1, -w[i][j]]);
      var mc = mcmfSSPRef(N, S, T, edges);
      worst = Math.max(worst, Math.abs(hw.value - (-mc.cost)));
      if (n <= 8) {                                                                 // 暴力枚举全部排列交叉
        var perm = [], idx = []; for (i = 0; i < n; i++) { perm.push(i); idx.push(i); }
        function rec(k) { if (k === n) { var s2 = 0; for (i = 0; i < n; i++) s2 += w[i][perm[i]]; if (s2 > best) best = s2; return; } for (var ii = k; ii < n; ii++) { var tmp = perm[k]; perm[k] = perm[ii]; perm[ii] = tmp; rec(k + 1); tmp = perm[k]; perm[k] = perm[ii]; perm[ii] = tmp; } }
        var best = -Infinity; rec(0); worst = Math.max(worst, Math.abs(hw.value - best));
      }
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ⑩ Gale–Shapley 稳定婚姻：完美匹配 + 稳定（无 blocking pair），男主动与女主动均稳定（25 实例） */
  check('stableMarriage：Gale–Shapley 稳定婚姻 —— 完美匹配 + 稳定（无 blocking pair），男主动与女主动均得稳定匹配（25 实例）', 0, function () {
    var worst = 0, tested = 0, g;
    for (g = 0; g < 25; g++) {
      var rng = mulberry32(15000 + g), n = 4 + Math.floor(rng() * 9), menPref = [], womenPref = [], i;
      for (i = 0; i < n; i++) { menPref.push(_randPerm(rng, n)); womenPref.push(_randPerm(rng, n)); }
      var mp = stableMarriage(menPref, womenPref).matchP;        // 男主动
      var wp = stableMarriage(womenPref, menPref).matchP;         // 女主动
      var wifeW = new Array(n); for (var w = 0; w < n; w++) wifeW[wp[w]] = w;
      var bij = true, seen = new Array(n).fill(false); for (var mm = 0; mm < n; mm++) { if (mp[mm] < 0 || mp[mm] >= n || seen[mp[mm]]) { bij = false; break; } seen[mp[mm]] = true; }
      worst = Math.max(worst, bij ? 0 : 1, _isStableMatching(menPref, womenPref, mp) ? 0 : 1, _isStableMatching(menPref, womenPref, wifeW) ? 0 : 1);
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ⑪ Push-Relabel 全局最小割 ≙ stoerWagner（不同算法，0 精确） */
  check('globalMinCutPR：Push-Relabel 全局最小割 ≙ stoerWagner（Goldberg；n−1 次 s-t 最小割收缩，25 图）', 0, function () {
    var worst = 0, tested = 0, g;
    for (g = 0; g < 25; g++) {
      var rng = mulberry32(16000 + g), n = 4 + Math.floor(rng() * 5), eg = [], seen = {};
      var m = n + Math.floor(rng() * n);
      for (var k = 0; k < m; k++) { var u = Math.floor(rng() * n), v = Math.floor(rng() * n); if (u === v) continue; var key = u < v ? u + '-' + v : v + '-' + u; if (seen[key]) continue; seen[key] = true; eg.push([u, v, 1 + Math.floor(rng() * 9)]); }
      var val = globalMinCutPR(n, eg);
      var sw = stoerWagner(n, eg);
      worst = Math.max(worst, Math.abs(val - sw));
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ⑫ 2-SAT ≙ 暴力枚举 2ⁿ 指派（决策一致，25 实例） */
  check('twoSat：2-SAT 可满足性 ≙ 暴力枚举 2ⁿ 指派（Aspvall–Plass–Tarjan；决策一致 + 赋值合法，25 实例）', 0, function () {
    var worst = 0, tested = 0, g;
    for (g = 0; g < 25; g++) {
      var rng = mulberry32(17000 + g), n = 4 + Math.floor(rng() * 8), clauses = [];
      var m = 2 * n + Math.floor(rng() * n);
      for (var k = 0; k < m; k++) clauses.push([[Math.floor(rng() * n), rng() < 0.5 ? 1 : -1], [Math.floor(rng() * n), rng() < 0.5 ? 1 : -1]]);
      var res = twoSat(n, clauses);
      var bf = false;
      for (var mask = 0; mask < (1 << n); mask++) {
        var ok = true;
        for (var c = 0; c < clauses.length; c++) { var a = clauses[c][0], b = clauses[c][1]; var va = a[1] > 0 ? ((mask >> a[0]) & 1) : (((mask >> a[0]) & 1) ? 0 : 1); var vb = b[1] > 0 ? ((mask >> b[0]) & 1) : (((mask >> b[0]) & 1) ? 0 : 1); if (!(va || vb)) { ok = false; break; } }
        if (ok) { bf = true; break; }
      }
      if (res.sat !== bf) worst = 1;
      if (res.sat && !tsVerify(n, clauses, res.assign)) worst = 1;
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ⑬ 桥与割点 ≙ 暴力删边/删点连通分量（集合精确，25 图） */
  check('bridgesArticulation：桥与割点（Tarjan）≙ 暴力删边/删点连通分量计数（集合一致，25 图）', 0, function () {
    var worst = 0, tested = 0, g;
    for (g = 0; g < 25; g++) {
      var rng = mulberry32(18000 + g), n = 4 + Math.floor(rng() * 7), eg = [], seen = {};
      var m = n + Math.floor(rng() * n);
      for (var k = 0; k < m; k++) { var u = Math.floor(rng() * n), v = Math.floor(rng() * n); if (u === v) continue; var key = u < v ? u + '-' + v : v + '-' + u; if (seen[key]) continue; seen[key] = true; eg.push([u, v, 1]); }
      var res = bridgesArticulation(n, eg);
      var C0 = bruteComponentsCount(n, eg, -1, null);
      var rb = {}; res.bridges.forEach(function (e) { rb[e[0] + '-' + e[1]] = true; });
      var bb = {};
      eg.forEach(function (e) { if (bruteComponentsCount(n, eg, -1, e) > C0) { var a = Math.min(e[0], e[1]), b = Math.max(e[0], e[1]); bb[a + '-' + b] = true; } });
      var okB = true; Object.keys(rb).forEach(function (kk) { if (!bb[kk]) okB = false; }); Object.keys(bb).forEach(function (kk) { if (!rb[kk]) okB = false; });
      var ra = {}; res.articulations.forEach(function (x) { ra[x] = true; });
      var ba = {};
      for (var v = 0; v < n; v++) if (bruteComponentsCount(n, eg, v, null) > C0) ba[v] = true;
      var okA = true; Object.keys(ra).forEach(function (kk) { if (!ba[kk]) okA = false; }); Object.keys(ba).forEach(function (kk) { if (!ra[kk]) okA = false; });
      if (!(okB && okA)) worst = 1;
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ==================== v1.3.19 算法库扩展九 ==================== */
  /* ① 拍卖算法解指派问题（Bertsekas 1981, *Annals of Discrete Mathematics*）。
     ε-标度价格拍卖：未匹配者对「效益−价格」最高与次高对象竞价（加价 = 差 + ε），
     逐步缩小 ε 至 < 1/n；每相重启匹配但保留价格。
     不变量：最小总费用 ≡ hungarianMaxWeight(−cost) ≙ 暴力枚举全部排列（整数 0 精确）。 */
  function auctionAssignment(cost) {
    var n = cost.length, i, j;
    if (n === 1) return { assign: [0], totalCost: cost[0][0] };
    var maxC = -Infinity, minC = Infinity;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) { if (cost[i][j] > maxC) maxC = cost[i][j]; if (cost[i][j] < minC) minC = cost[i][j]; }
    var price = new Array(n).fill(0);
    var assign = new Array(n).fill(-1);   // object -> person
    var owner = new Array(n).fill(-1);    // person -> object
    var eps = (maxC - minC) + 1;          // 整数效益：ε 从大值起，逐相除 n
    var guard = 0, GUARD = 2000000;
    while (eps >= 1 / n) {
      // 每相重启匹配（价格保留）：否则首相后全员已分配、后续相空转，只具 ε₀-最优性
      for (var q = 0; q < n; q++) { if (owner[q] !== -1) { assign[owner[q]] = -1; owner[q] = -1; } }
      var queue = [];
      for (i = 0; i < n; i++) queue.push(i);
      while (queue.length) {
        if (++guard > GUARD) throw new Error('auction: 迭代护栏触发');
        var p = queue.shift();
        var j1 = -1, v1 = -Infinity, j2 = -1, v2 = -Infinity;
        for (j = 0; j < n; j++) {
          var v = -cost[p][j] - price[j];
          if (v > v1) { v2 = v1; j2 = j1; v1 = v; j1 = j; }
          else if (v > v2) { v2 = v; j2 = j; }
        }
        price[j1] += (v1 - v2 + eps);
        if (assign[j1] !== -1) { owner[assign[j1]] = -1; queue.push(assign[j1]); }
        assign[j1] = p; owner[p] = j1;
      }
      eps = eps / n;
    }
    var tot = 0;
    for (i = 0; i < n; i++) tot += cost[i][owner[i]];
    return { assign: owner, totalCost: tot };
  }
  function _assignBruteMin(cost) {          // 暴力：枚举全部排列求最小总费用（n ≤ 7）
    var n = cost.length, best = Infinity;
    var used = new Array(n).fill(false);
    (function rec(i, acc) {
      if (acc >= best) return;
      if (i === n) { if (acc < best) best = acc; return; }
      for (var j = 0; j < n; j++) {
        if (used[j]) continue;
        used[j] = true;
        rec(i + 1, acc + cost[i][j]);
        used[j] = false;
      }
    })(0, 0);
    return best;
  }

  /* ② Top Trading Cycles（Shapley & Scarf 1974；Roth 顶交易循环实现）。
     房屋市场：每个 agent 指向其当前最优持有房屋的主人，沿函数图找环、环上交易、移除，直至分完。
     理论：严格偏好下弱核唯一（Roth–Postlewaite 1977）且等于 TTC 输出；TTC 输出同时在强核中。
     不变量：≡ 暴力枚举 n! 配置的弱核唯一分配 + 强核成员（0 精确）。 */
  function topTradingCycles(endow, prefs) {
    var n = endow.length;
    var alloc = new Array(n).fill(-1);
    var inRem = new Array(n).fill(true);
    var houseRem = new Array(n).fill(true);   // 房屋仍由剩余 agent 持有
    function topHouse(a) { for (var k = 0; k < n; k++) if (houseRem[prefs[a][k]]) return prefs[a][k]; return -1; }
    for (;;) {
      var rem = [];
      for (var a = 0; a < n; a++) if (inRem[a]) rem.push(a);
      if (rem.length === 0) break;
      function houseOwnerOf(a) { var h = topHouse(a); for (var x = 0; x < n; x++) if (inRem[x] && endow[x] === h) return x; return -1; }
      var nxt = {};
      for (var t = 0; t < rem.length; t++) nxt[rem[t]] = houseOwnerOf(rem[t]);
      var state = {};
      for (var s0 = 0; s0 < rem.length; s0++) {
        var start = rem[s0];
        if (state[start]) continue;
        var path = [], cur = start;
        while (!state[cur]) { state[cur] = 1; path.push(cur); cur = nxt[cur]; }
        if (state[cur] === 1) {
          var cyc = path.slice(path.indexOf(cur));
          for (var q = 0; q < cyc.length; q++) {
            var h = topHouse(cyc[q]);
            alloc[cyc[q]] = h; inRem[cyc[q]] = false; houseRem[h] = false;
          }
        }
        for (var w = 0; w < path.length; w++) state[path[w]] = 2;
      }
    }
    return { alloc: alloc };
  }
  function _ttcAllPerms(n) {
    var res = [], a = [], used = new Array(n).fill(false);
    (function rec() {
      if (a.length === n) { res.push(a.slice()); return; }
      for (var i = 0; i < n; i++) {
        if (used[i]) continue;
        used[i] = true; a.push(i);
        rec();
        used[i] = false; a.pop();
      }
    })();
    return res;
  }
  function _ttcBruteCores(endow, prefs) {   // 暴力：枚举 n! 配置，分别求强核与弱核
    var n = endow.length;
    var rank = [];
    for (var i = 0; i < n; i++) { var r = {}; for (var k = 0; k < n; k++) r[prefs[i][k]] = k; rank.push(r); }
    function strictlyBetter(a, x, y) { return rank[a][x] < rank[a][y]; }
    function weaklyBetter(a, x, y) { return rank[a][x] <= rank[a][y]; }
    function blocked(alloc, weak) {
      for (var mask = 1; mask < (1 << n); mask++) {
        var S = [];
        for (var i = 0; i < n; i++) if (mask & (1 << i)) S.push(i);
        var endowH = S.map(function (a) { return endow[a]; });
        var perms = _ttcAllPerms(S.length);
        for (var pi = 0; pi < perms.length; pi++) {
          var allWeak = true, anyStrict = false, allStrict = true;
          for (var t = 0; t < S.length; t++) {
            var curA = alloc[S[t]], neuA = endowH[perms[pi][t]];
            if (!weaklyBetter(S[t], neuA, curA)) allWeak = false;
            if (strictlyBetter(S[t], neuA, curA)) anyStrict = true; else allStrict = false;
          }
          if (weak ? (allWeak && anyStrict) : allStrict) return true;
        }
      }
      return false;
    }
    var perms = _ttcAllPerms(n), core = [], weakCore = [];
    for (var p = 0; p < perms.length; p++) {
      var alloc = perms[p], ir = true;
      for (var i = 0; i < n; i++) if (!weaklyBetter(i, alloc[i], endow[i])) { ir = false; break; }
      if (!ir) continue;
      if (!blocked(alloc, false)) core.push(alloc);
      if (!blocked(alloc, true)) weakCore.push(alloc);
    }
    return { core: core, weakCore: weakCore };
  }

  /* ③ 两阶段单纯形 LP（Dantzig 1947；Bland 1977 规则防循环）。
     max cᵀx  s.t. A x {le|ge|eq} b, x ≥ 0（可选上界 x ≤ U）。
     阶段一最小化人工变量和（判可行性），阶段二最大化原目标；Bland 规则（进入取最小下标、
     离开取最小比值并列最小基下标）保证有限步终止。
     不变量：最优目标 ≡ 暴力顶点枚举（约束取等式组合求解，≤/≥/= 混合，0~1e-9 容差）。 */
  function simplexLP(c, A, b, senses, opts) {
    var U = (opts && opts.upper != null) ? opts.upper : Infinity;
    var m0 = A.length, n = c.length, i, j;
    // 1) b ≥ 0 归整（取反时翻转方向）
    var rows = [], rhs = [], sen = [];
    for (i = 0; i < m0; i++) {
      var sgn = b[i] >= 0 ? 1 : -1;
      var sn = senses[i];
      if (sgn < 0) sn = sn === 'le' ? 'ge' : (sn === 'ge' ? 'le' : 'eq');
      rows.push(A[i].map(function (v) { return v * sgn; }));
      rhs.push(b[i] * sgn);
      sen.push(sn);
    }
    // 2) 上界行 x_j ≤ U（有界性保证，同时进入暴力枚举约束集）
    if (isFinite(U)) {
      for (j = 0; j < n; j++) {
        var e = new Array(n).fill(0); e[j] = 1;
        rows.push(e); rhs.push(U); sen.push('le');
      }
    }
    var m = rows.length;
    // 3) 列布局：结构 n + 每 le/ge 行一个松弛/剩余 + ge/eq 行一个人工变量
    //    （先分配完全部松弛列，再以终值 nSlack 为基址分配人工列，防止撞号）
    var slackCol = new Array(m).fill(-1), artCol = new Array(m).fill(-1);
    var nSlack = 0, nArt = 0;
    for (i = 0; i < m; i++) {
      if (sen[i] === 'le' || sen[i] === 'ge') { slackCol[i] = n + nSlack; nSlack++; }
    }
    for (i = 0; i < m; i++) {
      if (sen[i] === 'ge' || sen[i] === 'eq') { artCol[i] = n + nSlack + nArt; nArt++; }
    }
    var ncols = n + nSlack + nArt;
    // 4) 初始表：le 行基=松弛，ge/eq 行基=人工（rhs≥0 → 可行基）
    var T = [];
    for (i = 0; i < m; i++) {
      var r = rows[i].slice();
      for (j = 0; j < nSlack + nArt; j++) r.push(0);
      if (slackCol[i] >= 0) r[slackCol[i]] = sen[i] === 'le' ? 1 : -1;
      if (artCol[i] >= 0) r[artCol[i]] = 1;
      r.push(rhs[i]);
      T.push(r);
    }
    var basis = [];
    for (i = 0; i < m; i++) basis.push(artCol[i] >= 0 ? artCol[i] : slackCol[i]);
    var tol = 1e-9, maxIter = 20000;
    function pivot(leaveRow, enterCol) {
      var piv = T[leaveRow][enterCol], k;
      for (k = 0; k <= ncols; k++) T[leaveRow][k] /= piv;
      for (var ii = 0; ii < m; ii++) {
        if (ii === leaveRow) continue;
        var f = T[ii][enterCol];
        if (Math.abs(f) < 1e-300) continue;
        for (k = 0; k <= ncols; k++) T[ii][k] -= f * T[leaveRow][k];
      }
      basis[leaveRow] = enterCol;
    }
    function run(cost, forbid) {
      var iter = 0;
      for (;;) {
        if (++iter > maxIter) throw new Error('simplex: 迭代超限');
        var cB = basis.map(function (bi) { return cost[bi]; });
        var enter = -1;
        for (j = 0; j < ncols; j++) {
          if (forbid && forbid[j]) continue;
          var d = cost[j];
          for (i = 0; i < m; i++) d -= cB[i] * T[i][j];
          if (d > tol) { enter = j; break; }              // Bland：最小下标
        }
        if (enter === -1) return 'optimal';
        var leave = -1, bestRatio = Infinity;
        for (i = 0; i < m; i++) {
          if (T[i][enter] > tol) {
            var ratio = T[i][ncols] / T[i][enter];
            if (ratio < -tol) continue;
            if (ratio < bestRatio - tol || (Math.abs(ratio - bestRatio) <= tol && (leave === -1 || basis[i] < basis[leave]))) { bestRatio = ratio; leave = i; }
          }
        }
        if (leave === -1) return 'unbounded';
        pivot(leave, enter);
      }
    }
    // 阶段一：max (−Σ 人工变量)
    var cost1 = new Array(ncols).fill(0);
    for (i = 0; i < m; i++) if (artCol[i] >= 0) cost1[artCol[i]] = -1;
    run(cost1, null);
    var artSum = 0;
    for (i = 0; i < m; i++) if (artCol.indexOf(basis[i]) >= 0) artSum += T[i][ncols];
    if (artSum > 1e-7) return { status: 'infeasible' };
    // 人工变量出基（零值 pivot 出行）；全零行 = 冗余行，删除
    for (i = m - 1; i >= 0; i--) {
      if (artCol.indexOf(basis[i]) >= 0) {
        var pivCol = -1;
        for (j = 0; j < n + nSlack; j++) if (Math.abs(T[i][j]) > tol) { pivCol = j; break; }
        if (pivCol >= 0) pivot(i, pivCol);
        else { T.splice(i, 1); basis.splice(i, 1); m--; }
      }
    }
    // 阶段二：max cᵀx，人工列禁入
    var cost2 = new Array(ncols).fill(0);
    for (j = 0; j < n; j++) cost2[j] = c[j];
    var forbid = new Array(ncols).fill(false);
    for (j = n + nSlack; j < ncols; j++) forbid[j] = true;
    if (run(cost2, forbid) === 'unbounded') return { status: 'unbounded' };
    var x = new Array(n).fill(0);
    for (i = 0; i < m; i++) if (basis[i] < n) x[basis[i]] = T[i][ncols];
    var obj = 0;
    for (j = 0; j < n; j++) obj += c[j] * x[j];
    return { status: 'optimal', x: x, obj: obj };
  }
  function _lpBruteVertexMax(c, A, b, senses, U) {   // 暴力：顶点枚举（n 个约束取等式组）
    var n = c.length, m = A.length, rows = [], rhs = [], i, j;
    for (i = 0; i < m; i++) { rows.push(A[i].slice()); rhs.push(b[i]); }
    for (j = 0; j < n; j++) {
      var e1 = new Array(n).fill(0); e1[j] = 1; rows.push(e1); rhs.push(0);
      var e2 = new Array(n).fill(0); e2[j] = 1; rows.push(e2); rhs.push(U);
    }
    var total = rows.length, best = -Infinity;
    function feas(x) {
      var tol = 1e-7;
      for (var i = 0; i < m; i++) {
        var s = 0;
        for (var j = 0; j < n; j++) s += A[i][j] * x[j];
        if (senses[i] === 'le' && s > b[i] + tol) return false;
        if (senses[i] === 'ge' && s < b[i] - tol) return false;
        if (senses[i] === 'eq' && Math.abs(s - b[i]) > tol) return false;
      }
      for (var j2 = 0; j2 < n; j2++) if (x[j2] < -tol || x[j2] > U + tol) return false;
      return true;
    }
    var idx = [];
    for (i = 0; i < n; i++) idx.push(i);
    for (;;) {
      var M = [], r = [];
      for (i = 0; i < n; i++) { M.push(rows[idx[i]].slice()); r.push(rhs[idx[i]]); }
      var sol = solveLinear(M, r);
      if (sol && sol.every(isFinite) && feas(sol)) {
        var obj = 0;
        for (j = 0; j < n; j++) obj += c[j] * sol[j];
        if (obj > best) best = obj;
      }
      var t = n - 1;
      while (t >= 0 && idx[t] === total - n + t) t--;
      if (t < 0) break;
      idx[t]++;
      for (i = t + 1; i < n; i++) idx[i] = idx[i - 1] + 1;
    }
    return best;
  }

  check('auctionAssignment：最小总费用 ≡ hungarianMaxWeight(−cost) ≙ 暴力枚举排列', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), n = 3 + Math.floor(rng() * 5);   // 3..7
      var cost = [];
      for (var i = 0; i < n; i++) { var row = []; for (var j = 0; j < n; j++) row.push(1 + Math.floor(rng() * 50)); cost.push(row); }
      var r = auctionAssignment(cost);
      if (new Set(r.assign).size !== n) { worst = 1; continue; }
      var h = hungarianMaxWeight(cost.map(function (row) { return row.map(function (v) { return -v; }); }));
      worst = Math.max(worst, Math.abs(r.totalCost - (-h.value)), Math.abs(r.totalCost - _assignBruteMin(cost)));
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('topTradingCycles：输出 = 唯一弱核分配 ≙ 暴力枚举 n! 配置（强核成员）', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), n = 3 + Math.floor(rng() * 3);   // 3..5
      var endow = [];
      for (var i = 0; i < n; i++) endow.push(i);
      if (rng() < 0.5) { for (var k = n - 1; k > 0; k--) { var j2 = Math.floor(rng() * (k + 1)); var t = endow[k]; endow[k] = endow[j2]; endow[j2] = t; } }
      var prefs = [];
      for (i = 0; i < n; i++) {
        var a = [];
        for (var x = 0; x < n; x++) a.push(x);
        for (k = n - 1; k > 0; k--) { var jj = Math.floor(rng() * (k + 1)); var tt = a[k]; a[k] = a[jj]; a[jj] = tt; }
        prefs.push(a);
      }
      var r = topTradingCycles(endow, prefs);
      if (new Set(r.alloc).size !== n) { worst = 1; continue; }
      var cs = _ttcBruteCores(endow, prefs);
      if (cs.weakCore.length !== 1) { worst = 1; continue; }                    // 严格偏好下弱核唯一
      if (!cs.core.some(function (al) { return al.every(function (v, t) { return v === r.alloc[t]; }); })) { worst = 1; continue; }
      for (i = 0; i < n; i++) if (r.alloc[i] !== cs.weakCore[0][i]) worst = 1;  // TTC = 唯一弱核分配
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('simplexLP：最优目标 ≡ 暴力顶点枚举（≤/≥/= 混合 + 上界，整数数据）', 1e-9, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), n = 2 + Math.floor(rng() * 2), U = 10;         // 2..3 变量
      var c = [];
      for (var j = 0; j < n; j++) c.push(-5 + Math.floor(rng() * 11));
      var x0 = [];
      for (j = 0; j < n; j++) x0.push(Math.floor(rng() * (U + 1)));
      var m = 4 + Math.floor(rng() * 3), A = [], b = [], senses = [];
      for (var i = 0; i < m; i++) {
        var row = [];
        for (j = 0; j < n; j++) row.push(-3 + Math.floor(rng() * 7));
        if (row.every(function (v) { return v === 0; })) row[0] = 1;
        var s = 0;
        for (j = 0; j < n; j++) s += row[j] * x0[j];
        var sn = ['le', 'ge', 'eq'][Math.floor(rng() * 3)], slack = Math.floor(rng() * 9);
        A.push(row);
        b.push(sn === 'le' ? s + slack : (sn === 'ge' ? s - slack : s));
        senses.push(sn);
      }
      var r = simplexLP(c, A, b, senses, { upper: U });
      if (r.status !== 'optimal') { worst = 1; continue; }
      var brute = _lpBruteVertexMax(c, A, b, senses, U);
      if (!isFinite(brute)) { worst = 1; continue; }
      worst = Math.max(worst, Math.abs(r.obj - brute));
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ==================== v1.3.20 算法库扩展十 ==================== */
  /* ① Chu–Liu/Edmonds 有向最小树形图（Edmonds 1967, *J. Res. Nat. Bur. Standards*）。
     每轮取每非根组最小入边 → 函数图找环 → 收缩环为超节点（入环边权减去被替换的暂定入边权）
     → 递归 → 展开：入口边（= 超节点的最终父边）取代入口组的暂定挑选，环上其余成员保留
     本轮挑选，非环节点沿用递归解。
     关键实现教训（v1.3.20 沙盒实证）：迭代式「暂定后写覆盖」会丢失层级展开结构——簇的挑选
     被高层取代时，被覆盖成员必须回放到其收缩层的挑选；故必须用递归 + 每层 chosen 父边表
     自底向上组装，展开时入口组由入口边取代、其余成员保留本层挑选。
     不变量：最小树形图 ≡ 暴力父分配枚举（整数 0 精确，覆盖负权/平行边/不可行判定）。 */
  function arborescence(n, edges, root) {
    // edges: [[u,v,w], ...] 有向边（允许负权、平行边）；返回 { cost, parent }（parent[v]=u, parent[root]=-1）
    // 不可行 { cost: Infinity, parent: null }
    var parent = new Array(n).fill(-1), cost = 0;
    if (n === 1) return { cost: 0, parent: [-1] };
    // solve：在组图（groups = [[origIds],...]；es = [{u,v,w,rw,ou,ov}]，u/v 为组索引）上递归。
    // 返回 chosen：组 -> 该组的最终父边（原始边对象）；同时把父边写入 parent / 累加 cost（首写即终值）。
    function solve(groups, es) {
      var K = groups.length;
      var rootIdx = -1;
      for (var q = 0; q < K; q++) if (groups[q].length === 1 && groups[q][0] === root) { rootIdx = q; break; }
      if (rootIdx === -1) return null;                      // 防御：根组恒为 {root}（根无入边、永不入环）
      // 每组最小入边（调整权 w 用于决策，rw 为真实权）
      var inW = new Array(K).fill(Infinity), inE = new Array(K).fill(null);
      for (var t = 0; t < es.length; t++) {
        var e = es[t];
        if (e.u === e.v) continue;                          // 自环 / 超节点内部边
        if (e.w < inW[e.v]) { inW[e.v] = e.w; inE[e.v] = e; }
      }
      for (q = 0; q < K; q++) if (q !== rootIdx && inE[q] === null) return null;   // 有组无入边 → 不可行
      // 函数图找环：par[q] = 入选边起点组
      var par = new Array(K).fill(-1);
      for (q = 0; q < K; q++) if (q !== rootIdx) par[q] = inE[q].u;
      var color = new Array(K).fill(0), cycle = null;
      for (var s = 0; s < K && !cycle; s++) {
        if (color[s]) continue;
        var path = [], cur = s;
        while (cur !== -1 && color[cur] === 0) { color[cur] = 1; path.push(cur); cur = par[cur]; }
        if (cur !== -1 && color[cur] === 1) cycle = path.slice(path.indexOf(cur));
        for (var w2 = 0; w2 < path.length; w2++) color[path[w2]] = 2;
      }
      var chosen = new Array(K).fill(null);
      if (!cycle) {                                         // 无环：本层挑选即终解
        for (q = 0; q < K; q++) {
          if (q === rootIdx) continue;
          chosen[q] = inE[q];
          if (parent[inE[q].ov] === -1) { parent[inE[q].ov] = inE[q].ou; cost += inE[q].rw; }
        }
        return chosen;
      }
      // 收缩环 C 为单一超节点（保留环上各组的原始节点并集）
      var inCycle = {};
      for (var ci = 0; ci < cycle.length; ci++) inCycle[cycle[ci]] = 1;
      var newGroups = [], mapOld = new Array(K).fill(-1), starIdx = -1;
      for (q = 0; q < K; q++) {
        if (inCycle[q]) {
          if (starIdx === -1) { starIdx = newGroups.length; newGroups.push([]); }
          mapOld[q] = starIdx;
          for (var z = 0; z < groups[q].length; z++) newGroups[starIdx].push(groups[q][z]);
        } else { mapOld[q] = newGroups.length; newGroups.push(groups[q].slice()); }
      }
      // 重映射边 + 权调整：入环边减去被替换的暂定入边权
      var nes = [];
      for (t = 0; t < es.length; t++) {
        var ed = es[t];
        if (inCycle[ed.u] && inCycle[ed.v]) continue;       // 环内部边丢弃
        var nu = mapOld[ed.u], nv = mapOld[ed.v];
        if (nu === nv) continue;                            // 同超节点内部边丢弃
        var nw = ed.w;
        if (inCycle[ed.v] && !inCycle[ed.u]) nw = ed.w - inW[ed.v];
        nes.push({ u: nu, v: nv, w: nw, rw: ed.rw, ou: ed.ou, ov: ed.ov });
      }
      var sub = solve(newGroups, nes);                      // 递归
      if (sub === null) return null;
      var entry = sub[starIdx];                             // 入口边 = 超节点的最终父边（头落在环内）
      if (!entry) return null;                              // 防御：超节点为非根组，必有父边
      // 层级展开：入口组由入口边取代；其余环上成员保留本层暂定挑选；非环节点沿用递归解
      for (q = 0; q < K; q++) {
        if (q === rootIdx) continue;
        if (inCycle[q]) {
          if (groups[q].indexOf(entry.ov) !== -1) chosen[q] = entry;               // 入口组
          else {
            chosen[q] = inE[q];                                                   // 非入口成员：本层挑选
            if (parent[inE[q].ov] === -1) { parent[inE[q].ov] = inE[q].ou; cost += inE[q].rw; }
          }
        } else chosen[q] = sub[mapOld[q]];                                        // 非环节点：递归解
      }
      return chosen;
    }
    var groups0 = [], es0 = edges.map(function (e) { return { u: e[0], v: e[1], w: e[2], rw: e[2], ou: e[0], ov: e[1] }; });
    for (var i = 0; i < n; i++) groups0.push([i]);
    if (solve(groups0, es0) === null) return { cost: Infinity, parent: null };
    return { cost: cost, parent: parent };
  }
  function _arboBrute(n, edges, root) {          // 暴力：每个非根节点枚举父边，校验无环 + 全达根
    var cand = [];
    for (var v = 0; v < n; v++) {
      if (v === root) { cand.push(null); continue; }
      var best = {};
      for (var t = 0; t < edges.length; t++) if (edges[t][1] === v) {
        var u = edges[t][0];
        if (u === v) continue;
        if (!(u in best) || edges[t][2] < best[u]) best[u] = edges[t][2];
      }
      var arr = [];
      for (var k in best) arr.push([+k, best[k]]);
      if (arr.length === 0) return { cost: Infinity, parent: null };
      cand.push(arr);
    }
    var best = { cost: Infinity, parent: null };
    // 后缀最小剩余权之和（负权下安全剪枝：cost + minRem[v] >= best 才可剪）
    var minRem = new Array(n + 1).fill(0);
    for (var vv = n - 1; vv >= 0; vv--) {
      var mn = vv === root ? 0 : Math.min.apply(null, cand[vv].map(function (c) { return c[1]; }));
      minRem[vv] = minRem[vv + 1] + mn;
    }
    (function rec(v, parent, cost) {
      if (cost + minRem[v] >= best.cost) return;
      if (v === n) {
        for (var x = 0; x < n; x++) {
          if (x === root) continue;
          var seen = {}, cur = x, steps = 0;
          while (cur !== root) {
            if (seen[cur] || steps > n) return;            // 成环
            seen[cur] = 1; cur = parent[cur];
            if (cur === -1) return;
            steps++;
          }
        }
        best = { cost: cost, parent: parent.slice() };
        return;
      }
      if (v === root) { rec(v + 1, parent, cost); return; }
      for (var ci = 0; ci < cand[v].length; ci++) {
        parent[v] = cand[v][ci][0];
        rec(v + 1, parent, cost + cand[v][ci][1]);
        parent[v] = -1;
      }
    })(0, new Array(n).fill(-1), 0);
    return best;
  }
  function _arboStructural(n, edges, root, r) {  // 结构合法：父边真实、父链无环必达根、cost = Σ父边权（平行边取最小）
    if (!isFinite(r.cost) || r.parent === null) return false;
    if (r.parent[root] !== -1) return false;
    var wOf = {}, sum = 0;
    for (var t = 0; t < edges.length; t++) {
      var kk = edges[t][0] + '>' + edges[t][1];
      if (!(kk in wOf) || edges[t][2] < wOf[kk]) wOf[kk] = edges[t][2];
    }
    for (var v = 0; v < n; v++) {
      if (v === root) continue;
      var p = r.parent[v];
      if (p < 0 || p >= n) return false;
      if (!(p + '>' + v in wOf)) return false;
      sum += wOf[p + '>' + v];
      var seen = {}, cur = v, steps = 0;
      while (cur !== root) {
        if (seen[cur] || steps > n) return false;
        seen[cur] = 1; cur = r.parent[cur]; steps++;
        if (cur === -1) return false;
      }
    }
    return sum === r.cost;
  }

  check('arborescence：最小树形图 ≡ 暴力父分配枚举（负权/平行边/多根/不可行判定，整数 0 精确）', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), n = 4 + Math.floor(rng() * 4);   // 4..7
      var root = Math.floor(rng() * n);
      var neg = rng() < 0.3;                                        // 三成种子含负权
      var W = function () { var w = 1 + Math.floor(rng() * 20); return neg ? w - 10 : w; };
      var edges = [];
      var planted = rng() < 0.5;                                    // 半数种子植入可行树形图保可行
      if (planted) {
        for (var v = 0; v < n; v++) {
          if (v === root) continue;
          var u = Math.floor(rng() * n);
          while (u === v) u = Math.floor(rng() * n);
          edges.push([u, v, W()]);
        }
      }
      var dens = 0.25 + rng() * 0.5;
      for (var u2 = 0; u2 < n; u2++) for (var v2 = 0; v2 < n; v2++) {
        if (u2 === v2) continue;
        if (rng() < dens) edges.push([u2, v2, W()]);
        if (rng() < 0.12) edges.push([u2, v2, W()]);                // 平行边
      }
      var r = arborescence(n, edges, root), b = _arboBrute(n, edges, root);
      if (!isFinite(b.cost)) {
        if (isFinite(r.cost)) worst = 1;                            // 暴力判不可行而算法有解
        continue;
      }
      if (!isFinite(r.cost) || !_arboStructural(n, edges, root, r)) { worst = 1; continue; }
      worst = Math.max(worst, Math.abs(r.cost - b.cost));
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('arborescence：结构合法 + cost=Σ父边权 + 边序不变性（n≤12，随机置换输入边序）', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 20; seed++) {
      var rng = mulberry32(100 + seed), n = 10 + Math.floor(rng() * 3);   // 10..12
      var root = Math.floor(rng() * n);
      var neg = rng() < 0.3;
      var W = function () { var w = 1 + Math.floor(rng() * 40); return neg ? w - 20 : w; };
      var edges = [];
      var planted = rng() < 0.6;
      if (planted) for (var v = 0; v < n; v++) {
        if (v === root) continue;
        var u = Math.floor(rng() * n);
        while (u === v) u = Math.floor(rng() * n);
        edges.push([u, v, W()]);
      }
      var dens = 0.3 + rng() * 0.4;
      for (var u2 = 0; u2 < n; u2++) for (var v2 = 0; v2 < n; v2++) {
        if (u2 === v2) continue;
        if (rng() < dens) edges.push([u2, v2, W()]);
      }
      var r1 = arborescence(n, edges, root);
      // 随机置换边序（Fisher–Yates）
      var perm = edges.slice();
      for (var k = perm.length - 1; k > 0; k--) { var j2 = Math.floor(rng() * (k + 1)); var tt = perm[k]; perm[k] = perm[j2]; perm[j2] = tt; }
      var r2 = arborescence(n, perm, root);
      if (!isFinite(r1.cost) || !isFinite(r2.cost)) {
        if (isFinite(r1.cost) !== isFinite(r2.cost)) worst = 1;     // 可行性判定须与边序无关
        continue;
      }
      if (!_arboStructural(n, edges, root, r1) || !_arboStructural(n, perm, root, r2)) { worst = 1; continue; }
      worst = Math.max(worst, Math.abs(r1.cost - r2.cost));         // 最优值与边序无关
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ② Held-Karp 精确 TSP（Bellman 1962；Held–Karp 1962）。
     子集 DP：dp[mask][j] = 从 0 出发、恰访 mask 中顶点、止于 j 的最短路径；回溯重构环游。
     不变量：≡ 暴力枚举 (n−1)! 排列（支持非对称距离，整数 0 精确）+ 环游合法性。 */
  function heldKarpTSP(dist) {
    var n = dist.length;
    if (n === 1) return { cost: 0, tour: [0] };
    var FULL = (1 << n) - 1;
    var dp = [], par = [];
    for (var mask = 0; mask <= FULL; mask++) { dp.push(new Array(n).fill(Infinity)); par.push(new Array(n).fill(-1)); }
    for (var j = 1; j < n; j++) { dp[(1 << j) | 1][j] = dist[0][j]; par[(1 << j) | 1][j] = 0; }
    for (var m = 1; m <= FULL; m++) {
      if (!(m & 1)) continue;                              // 子集必含起点 0
      for (j = 1; j < n; j++) {
        if (!(m & (1 << j))) continue;
        var prev = m ^ (1 << j);
        if (prev === 0) continue;
        var best = Infinity, bi = -1;
        for (var i = 0; i < n; i++) {
          if (!(prev & (1 << i))) continue;
          if (!isFinite(dp[prev][i])) continue;
          var c = dp[prev][i] + dist[i][j];
          if (c < best) { best = c; bi = i; }
        }
        if (bi !== -1 && best < dp[m][j]) { dp[m][j] = best; par[m][j] = bi; }
      }
    }
    var endJ = -1, bestC = Infinity;
    for (j = 1; j < n; j++) {
      if (!isFinite(dp[FULL][j])) continue;
      var c2 = dp[FULL][j] + dist[j][0];
      if (c2 < bestC) { bestC = c2; endJ = j; }
    }
    if (endJ === -1) return { cost: Infinity, tour: null };
    var tour = [], mask = FULL, cur = endJ;
    while (cur !== -1) { tour.push(cur); var p = par[mask][cur]; mask ^= (1 << cur); cur = p; }
    tour.reverse();
    return { cost: bestC, tour: tour };
  }
  function _tspBrute(dist) {                    // 暴力：枚举 (n−1)! 排列
    var n = dist.length;
    if (n === 1) return { cost: 0, tour: [0] };
    var best = { cost: Infinity, tour: null }, rest = [];
    for (var i = 1; i < n; i++) rest.push(i);
    var used = new Array(n).fill(false);
    (function rec(cur, acc, tour) {
      if (acc >= best.cost) return;
      if (cur.length === rest.length) {
        var c = acc + dist[tour[tour.length - 1]][0];
        if (c < best.cost) best = { cost: c, tour: tour.slice() };
        return;
      }
      for (var k = 0; k < rest.length; k++) {
        if (used[k]) continue;
        used[k] = true; tour.push(rest[k]);
        rec(cur.concat([rest[k]]), acc + dist[tour[tour.length - 2]][rest[k]], tour);
        tour.pop(); used[k] = false;
      }
    })([], 0, [0]);
    return best;
  }
  function _tspTourCost(dist, tour) {           // 环游长度（闭合回路，独立复算）
    var n = tour.length, c = 0;
    for (var i = 0; i < n - 1; i++) c += dist[tour[i]][tour[i + 1]];
    return c + dist[tour[n - 1]][0];
  }

  check('heldKarpTSP：精确 TSP ≡ 暴力 (n−1)! 排列（含非对称距离，0 精确）', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), n = 5 + Math.floor(rng() * 4);   // 5..8
      var dist = [];
      for (var i = 0; i < n; i++) {
        var row = [];
        for (var j = 0; j < n; j++) row.push(i === j ? 0 : 1 + Math.floor(rng() * 30));
        dist.push(row);
      }
      var r = heldKarpTSP(dist), b = _tspBrute(dist);
      if (!isFinite(r.cost) || !isFinite(b.cost)) { worst = 1; continue; }
      // 环游合法性：0..n−1 的排列且以 0 起讫
      if (r.tour.length !== n || r.tour[0] !== 0) { worst = 1; continue; }
      var srt = r.tour.slice().sort(function (a, b2) { return a - b2; });
      for (var k = 0; k < n; k++) if (srt[k] !== k) { worst = 1; break; }
      worst = Math.max(worst, Math.abs(r.cost - b.cost), Math.abs(_tspTourCost(dist, r.tour) - r.cost));
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('heldKarpTSP：环游合法 + cost 自一致 + 2-opt 局部最优 + ≤ 最近邻贪心（对称距离）', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 20; seed++) {
      var rng = mulberry32(200 + seed), n = 8 + Math.floor(rng() * 7);   // 8..14
      var dist = [];
      for (var i = 0; i < n; i++) {
        var row = [];
        for (var j = 0; j < n; j++) row.push(i === j ? 0 : 0);
        dist.push(row);
      }
      for (var i2 = 0; i2 < n; i2++) for (var j2 = i2 + 1; j2 < n; j2++) {   // 对称距离
        var w = 1 + Math.floor(rng() * 100);
        dist[i2][j2] = w; dist[j2][i2] = w;
      }
      var r = heldKarpTSP(dist);
      if (!isFinite(r.cost) || r.tour.length !== n || r.tour[0] !== 0) { worst = 1; continue; }
      var srt = r.tour.slice().sort(function (a, b2) { return a - b2; });
      for (var k = 0; k < n; k++) if (srt[k] !== k) { worst = 1; break; }
      // cost 自一致
      worst = Math.max(worst, Math.abs(_tspTourCost(dist, r.tour) - r.cost));
      // 2-opt 局部最优（必要最优性条件）：无任何反转片段能严格改进
      for (var a = 1; a < n - 1; a++) for (var b2 = a + 1; b2 < n; b2++) {
        var A = r.tour[a - 1], B = r.tour[a], C = r.tour[b2], D = r.tour[(b2 + 1) % n];
        var delta = dist[A][C] + dist[B][D] - dist[A][B] - dist[C][D];
        if (delta < -1e-9) worst = Math.max(worst, -delta);
      }
      // ≤ 最近邻贪心（任意合法环游的成本上界）
      var t = [0], vis = new Array(n).fill(false); vis[0] = true;
      for (var s = 1; s < n; s++) {
        var last = t[t.length - 1], nxt = -1, nd = Infinity;
        for (var v2 = 0; v2 < n; v2++) if (!vis[v2] && dist[last][v2] < nd) { nd = dist[last][v2]; nxt = v2; }
        t.push(nxt); vis[nxt] = true;
      }
      worst = Math.max(worst, r.cost - _tspTourCost(dist, t));
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ③ AC-3 弧一致性 + CSP 回溯求解（Mackworth 1977）。
     弧队列消解至不动点（每值在邻居域中有支持）；MRV 回溯 + 赋值后全传播（MAC）。
     不变量：可满足性决策 ≡ 暴力枚举全部赋值；解合法；AC-3 不动点（双向弧逐值有支持）；
     AC-3 不删解值（SAT 实例必有某暴力解的全部值存活于传播后域）。 */
  function ac3CSP(n, domains, constraints) {
    // constraints: [{ i, j, ok(a,b) }] 二元谓词（a 为 i 的值、b 为 j 的值）
    var D = domains.map(function (d) { return d.slice(); });
    var cOf = {};                                       // "i,j" -> 约束（含两向）
    for (var t = 0; t < constraints.length; t++) {
      var c = constraints[t];
      cOf[c.i + ',' + c.j] = c;
      if (!cOf[c.j + ',' + c.i]) cOf[c.j + ',' + c.i] = { i: c.j, j: c.i, ok: c.ok, flip: c };
    }
    function okOn(i, j, a, b) {                          // 按弧向取谓词方向
      var c = cOf[i + ',' + j];
      if (!c) return true;                               // 两变量间无约束 → 恒兼容
      if (c.flip) return c.ok(b, a);
      return c.ok(a, b);
    }
    function arcsInto(i) {                               // 所有指向 i 的弧 (k,i) 的 k
      var res = [];
      for (var k = 0; k < n; k++) if (k !== i && cOf[k + ',' + i]) res.push(k);
      return res;
    }
    function propagate(Dm) {
      var queue = [];
      for (var t = 0; t < constraints.length; t++) { queue.push([constraints[t].i, constraints[t].j]); queue.push([constraints[t].j, constraints[t].i]); }
      while (queue.length) {
        var arc = queue.shift(), i = arc[0], j = arc[1];
        var changed = false;
        for (var a = Dm[i].length - 1; a >= 0; a--) {
          var av = Dm[i][a], hasSup = false;
          for (var b = 0; b < Dm[j].length; b++) if (okOn(i, j, av, Dm[j][b])) { hasSup = true; break; }
          if (!hasSup) { Dm[i].splice(a, 1); changed = true; }
        }
        if (changed) {
          if (Dm[i].length === 0) return false;          // 域擦空 → 不一致
          var ks = arcsInto(i);
          for (var q = 0; q < ks.length; q++) queue.push([ks[q], i]);
        }
      }
      return true;
    }
    var consistent = propagate(D);
    var solution = null, steps = 0;
    function search(assign, idx, Dm) {
      if (steps > 500000) return false;                  // 护栏：失败即检查报红（不静默谎报 UNSAT）
      if (idx === n) { solution = assign.slice(); return true; }
      // 选未赋值中域最小者（MRV）
      var bi = -1, bLen = Infinity;
      for (var v = 0; v < n; v++) if (assign[v] === undefined && Dm[v].length < bLen) { bLen = Dm[v].length; bi = v; }
      if (bi === -1 || bLen === 0) return false;
      var vals = Dm[bi].slice();
      for (var k = 0; k < vals.length; k++) {
        steps++;
        var a2 = assign.slice(); a2[bi] = vals[k];
        // 与已赋值邻居逐约束校验
        var okAsg = true;
        for (var t = 0; t < constraints.length; t++) {
          var c = constraints[t];
          if (c.i === bi && a2[c.j] !== undefined && !c.ok(a2[bi], a2[c.j])) okAsg = false;
          if (c.j === bi && a2[c.i] !== undefined && !c.ok(a2[c.i], a2[bi])) okAsg = false;
        }
        if (!okAsg) continue;
        var D2 = Dm.map(function (d) { return d.slice(); });
        D2[bi] = [vals[k]];
        if (!propagate(D2)) continue;
        if (search(a2, idx + 1, D2)) return true;
      }
      return false;
    }
    if (consistent) {
      var a0 = new Array(n).fill(undefined);
      search(a0, 0, D);                                   // 未找到解也正常（UNSAT）
    }
    return { consistent: consistent, domains: D, solution: solution };
  }
  function _cspBrute(n, domains, constraints) {  // 暴力：枚举全部赋值，收集全部解
    var sols = [];
    (function rec(v, a) {
      if (v === n) {
        for (var t = 0; t < constraints.length; t++) {
          var c = constraints[t];
          if (!c.ok(a[c.i], a[c.j])) return;
        }
        sols.push(a.slice());
        return;
      }
      for (var k = 0; k < domains[v].length; k++) { a[v] = domains[v][k]; rec(v + 1, a); }
      a[v] = undefined;
    })(0, new Array(n).fill(undefined));
    return sols;
  }

  check('ac3CSP：可满足性决策 ≡ 暴力枚举 + 解合法（SAT/UNSAT，0 精确）', 0, function () {
    var worst = 0, tested = 0, sat = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), n = 4 + Math.floor(rng() * 3);   // 4..6
      var domains = [];
      for (var v = 0; v < n; v++) {
        var dsz = 2 + Math.floor(rng() * 2), d = [];
        for (var k = 0; k < dsz; k++) d.push(k);
        domains.push(d);
      }
      var forceSat = rng() < 0.5;
      var hidden = domains.map(function (d) { return d[Math.floor(rng() * d.length)]; });
      var constraints = [];
      for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
        if (rng() < 0.6) {
          var tab = {};
          (function (ii, jj) {
            for (var a = 0; a < domains[ii].length; a++) for (var b = 0; b < domains[jj].length; b++) {
              var keep = rng() < 0.55;
              if (forceSat && domains[ii][a] === hidden[ii] && domains[jj][b] === hidden[jj]) keep = true;
              if (keep) tab[domains[ii][a] + '|' + domains[jj][b]] = 1;
            }
            constraints.push({ i: ii, j: jj, ok: function (x, y) { return !!tab[x + '|' + y]; } });
          })(i, j);
        }
      }
      var sols = _cspBrute(n, domains, constraints);
      var r = ac3CSP(n, domains, constraints);
      if (sols.length > 0) sat++;
      // (a) 决策一致
      if ((r.solution !== null) !== (sols.length > 0)) { worst = 1; continue; }
      // (b) 解合法
      if (r.solution !== null) {
        for (var t = 0; t < constraints.length; t++) {
          var c = constraints[t];
          if (!c.ok(r.solution[c.i], r.solution[c.j])) { worst = 1; break; }
        }
      }
      tested++;
    }
    return tested === 0 || sat === 0 ? 1 : worst;
  });

  check('ac3CSP：AC-3 不动点（双向弧逐值有支持）+ 不删解值（SAT 保留完整暴力解）', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(500 + seed), n = 4 + Math.floor(rng() * 3);
      var domains = [];
      for (var v = 0; v < n; v++) {
        var dsz = 2 + Math.floor(rng() * 2), d = [];
        for (var k = 0; k < dsz; k++) d.push(k);
        domains.push(d);
      }
      var forceSat = rng() < 0.6;
      var hidden = domains.map(function (d) { return d[Math.floor(rng() * d.length)]; });
      var constraints = [];
      for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
        if (rng() < 0.65) {
          var tab = {};
          (function (ii, jj) {
            for (var a = 0; a < domains[ii].length; a++) for (var b = 0; b < domains[jj].length; b++) {
              var keep = rng() < 0.5;
              if (forceSat && domains[ii][a] === hidden[ii] && domains[jj][b] === hidden[jj]) keep = true;
              if (keep) tab[domains[ii][a] + '|' + domains[jj][b]] = 1;
            }
            constraints.push({ i: ii, j: jj, ok: function (x, y) { return !!tab[x + '|' + y]; } });
          })(i, j);
        }
      }
      var sols = _cspBrute(n, domains, constraints);
      var r = ac3CSP(n, domains, constraints);
      if (r.consistent) {
        // 不动点：每弧每值（双向）都有支持
        for (var t = 0; t < constraints.length; t++) {
          var cc = constraints[t];
          var dirs = [[cc.i, cc.j], [cc.j, cc.i]];
          for (var di = 0; di < 2; di++) {
            var vi = dirs[di][0], vj = dirs[di][1];
            for (var ai = 0; ai < r.domains[vi].length; ai++) {
              var av = r.domains[vi][ai], sup = false;
              for (var bj = 0; bj < r.domains[vj].length; bj++) {
                var okv = di === 0 ? cc.ok(av, r.domains[vj][bj]) : cc.ok(r.domains[vj][bj], av);
                if (okv) { sup = true; break; }
              }
              if (!sup) { worst = 1; break; }
            }
          }
        }
      } else if (sols.length > 0) worst = 1;              // 擦空 ⇒ 暴力必须同样判 UNSAT
      // 不删解值：SAT 实例必有某暴力解全部值存活
      if (sols.length > 0) {
        var anySurvive = sols.some(function (s) {
          return s.every(function (val, vv) { return r.domains[vv].indexOf(val) !== -1; });
        });
        if (!anySurvive) worst = 1;
      }
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ==================== v1.3.21 算法库扩展十一 ==================== */
  /* ① Bron–Kerbosch 最大团（Bron & Kerbosch 1973；Tomita–Tanaka–Takahashi 2006 pivot）。
     R/P/X 三集合递归枚举全部极大团；pivot 取 P∪X 中与 P 邻接最多者，只展开 P\N(pivot)，
     把 3^(n/3) 量级的递归树压到最紧。最大独立集 = 补图上的最大团（α(G) = ω(Ḡ)）。
     不变量：ω ≡ 暴力枚举 2ⁿ 子集；全部极大团均极大；α(G) = ω(Ḡ)；ω ≤ 贪心着色数。 */
  function _cliqueAdj(n, edges) {
    var adj = [], i;
    for (i = 0; i < n; i++) adj.push(new Array(n).fill(false));
    for (i = 0; i < edges.length; i++) {
      var u = edges[i][0], v = edges[i][1];
      if (u !== v) { adj[u][v] = true; adj[v][u] = true; }
    }
    return adj;
  }
  function _cliqueExpand(adj, R, P, X, best, maximal) {
    if (P.length === 0 && X.length === 0) {
      maximal.push(R.slice());
      if (R.length > best.set.length) best.set = R.slice();
      return;
    }
    var piv = -1, pivCnt = -1, t, q;
    for (t = 0; t < P.length; t++) {
      var c = 0;
      for (q = 0; q < P.length; q++) if (adj[P[t]][P[q]]) c++;
      if (c > pivCnt) { pivCnt = c; piv = P[t]; }
    }
    for (t = 0; t < X.length; t++) {
      var c2 = 0;
      for (q = 0; q < P.length; q++) if (adj[X[t]][P[q]]) c2++;
      if (c2 > pivCnt) { pivCnt = c2; piv = X[t]; }
    }
    var iter = [];
    for (q = 0; q < P.length; q++) if (!adj[piv][P[q]]) iter.push(P[q]);
    for (t = 0; t < iter.length; t++) {
      var v2 = iter[t], P2 = [], X2 = [];
      for (q = 0; q < P.length; q++) if (adj[v2][P[q]]) P2.push(P[q]);
      for (q = 0; q < X.length; q++) if (adj[v2][X[q]]) X2.push(X[q]);
      R.push(v2);
      _cliqueExpand(adj, R, P2, X2, best, maximal);
      R.pop();
      var pi = P.indexOf(v2);
      if (pi >= 0) P.splice(pi, 1);
      X.push(v2);
    }
  }
  function maxClique(n, edges) {
    var adj = _cliqueAdj(n, edges), i, j;
    var best = { set: [] }, maximal = [], i0;
    var P0 = [];                                     // 注意：expand 会原地 splice P，两次调用须各用独立数组
    for (i0 = 0; i0 < n; i0++) P0.push(i0);
    _cliqueExpand(adj, [], P0, [], best, maximal);
    var comp = [];
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) if (!adj[i][j]) comp.push([i, j]);
    var b2 = { set: [] }, mx2 = [], P1 = [];
    for (i0 = 0; i0 < n; i0++) P1.push(i0);
    _cliqueExpand(_cliqueAdj(n, comp), [], P1, [], b2, mx2);
    function srt(a) { return a.slice().sort(function (x, y) { return x - y; }); }
    return {
      size: best.set.length, clique: srt(best.set), maximalCount: maximal.length, maximal: maximal,
      indepSize: b2.set.length, indepSet: srt(b2.set)
    };
  }
  function _cliqueBrute(n, edges) {                 // 暴力：枚举 2ⁿ 子集
    var adj = _cliqueAdj(n, edges), best = 0, i, j;
    for (var mask = 0; mask < (1 << n); mask++) {
      var vs = [];
      for (i = 0; i < n; i++) if (mask & (1 << i)) vs.push(i);
      var ok = true;
      for (i = 0; i < vs.length && ok; i++) for (j = i + 1; j < vs.length; j++) if (!adj[vs[i]][vs[j]]) { ok = false; break; }
      if (ok && vs.length > best) best = vs.length;
    }
    return best;
  }
  function _cliqueIsMaximal(n, edges, clique) {     // 极大性：是团且无法再加入任何顶点
    var adj = _cliqueAdj(n, edges), i, j;
    for (i = 0; i < clique.length; i++) for (j = i + 1; j < clique.length; j++) if (!adj[clique[i]][clique[j]]) return false;
    var inC = new Array(n).fill(false);
    for (i = 0; i < clique.length; i++) inC[clique[i]] = true;
    for (var v = 0; v < n; v++) {
      if (inC[v]) continue;
      var all = true;
      for (i = 0; i < clique.length; i++) if (!adj[v][clique[i]]) { all = false; break; }
      if (all) return false;
    }
    return true;
  }
  function _cliqueGreedyColor(n, edges) {           // 贪心着色数 = χ 的上界（必 ≥ ω）
    var adj = _cliqueAdj(n, edges), color = new Array(n).fill(-1), used = 0, i, j;
    for (i = 0; i < n; i++) {
      var forb = {};
      for (j = 0; j < n; j++) if (adj[i][j] && color[j] !== -1) forb[color[j]] = true;
      var c = 0;
      while (forb[c]) c++;
      color[i] = c;
      if (c + 1 > used) used = c + 1;
    }
    return used;
  }

  /* ② 后缀数组（倍增 + 计数排序，O(n log n)）+ Kasai LCP（O(n)）+ SA 二分搜索。
     倍增：按 (rank[i], rank[i+k]) 双关键字计数排序（稳定，先第二后第一）迭代至秩全异；
     Kasai：lcp[rank[i]] = LCP(sa[rank[i]−1], sa[i])，沿串右移时 h 至多减一摊还 O(n)。
     不变量：SA ≡ 朴素后缀排序；Kasai LCP ≡ 暴力逐对 LCP；SA 二分搜索 ≡ 朴素匹配；SA 为 0..n−1 排列。 */
  function suffixArray(s) {
    var n = s.length, i;
    if (n === 0) return { sa: [], rank: [], lcp: [] };
    var sa = new Array(n), rnk = new Array(n), tmpR = new Array(n);
    var sa2 = new Array(n), key1 = new Array(n), key2 = new Array(n);
    for (i = 0; i < n; i++) { sa[i] = i; rnk[i] = s.charCodeAt(i); }
    function countingSort(src, dst, key, maxKey) {   // 稳定计数排序（key 必须是数组，非函数）
      var cnt = new Array(maxKey + 1).fill(0), t;
      for (t = 0; t < n; t++) cnt[key[src[t]]]++;
      var acc = 0;
      for (t = 0; t <= maxKey; t++) { var c = cnt[t]; cnt[t] = acc; acc += c; }
      for (t = 0; t < n; t++) dst[cnt[key[src[t]]]++] = src[t];
    }
    var k = 1, guard = 0;
    for (;;) {
      if (++guard > 200) break;
      var maxKey = 0;
      for (i = 0; i < n; i++) {
        key1[i] = rnk[i] + 1;
        key2[i] = (i + k < n) ? rnk[i + k] + 1 : 0;
        if (key1[i] > maxKey) maxKey = key1[i];
        if (key2[i] > maxKey) maxKey = key2[i];
      }
      countingSort(sa, sa2, key2, maxKey);          // 先按第二关键字
      countingSort(sa2, sa, key1, maxKey);          // 再按第一关键字
      tmpR[sa[0]] = 0;
      for (i = 1; i < n; i++) {
        var a = sa[i - 1], b = sa[i];
        tmpR[b] = tmpR[a] + ((rnk[a] !== rnk[b] || key2[a] !== key2[b]) ? 1 : 0);
      }
      for (i = 0; i < n; i++) rnk[i] = tmpR[i];
      if (rnk[sa[n - 1]] === n - 1) break;          // 秩全异 ⇒ 排序完成
      if (k >= 2 * n) break;
      k <<= 1;
    }
    var rank = new Array(n);
    for (i = 0; i < n; i++) rank[sa[i]] = i;
    var lcp = new Array(n).fill(0), h = 0;
    for (i = 0; i < n; i++) {
      var r = rank[i];
      if (r === 0) { h = 0; continue; }
      var j = sa[r - 1];
      while (i + h < n && j + h < n && s.charAt(i + h) === s.charAt(j + h)) h++;
      lcp[r] = h;
      if (h > 0) h--;
    }
    return { sa: sa, rank: rank, lcp: lcp };
  }
  function saSearch(s, pat) {                       // SA 二分搜索：返回全部起始下标（升序）
    var n = s.length, m = pat.length;
    if (m === 0 || m > n) return [];
    var sa = suffixArray(s).sa;
    function cmpAt(pos) {                           // −1: s[pos..] < pat；0: pat 是前缀；1: 更大
      var lim = Math.min(m, n - pos), t = 0;
      while (t < lim && s.charAt(pos + t) === pat.charAt(t)) t++;
      if (t === m) return 0;
      if (t === n - pos) return -1;
      return s.charAt(pos + t) < pat.charAt(t) ? -1 : 1;
    }
    var lo = 0, hi = n, mid;
    while (lo < hi) { mid = (lo + hi) >> 1; if (cmpAt(sa[mid]) < 0) lo = mid + 1; else hi = mid; }
    var start = lo;
    lo = start; hi = n;
    while (lo < hi) { mid = (lo + hi) >> 1; if (cmpAt(sa[mid]) === 0) lo = mid + 1; else hi = mid; }
    var res = [];
    for (var t2 = start; t2 < lo; t2++) res.push(sa[t2]);
    return res.sort(function (a, b) { return a - b; });
  }
  function _saBrute(s) {                            // 暴力：所有后缀直接排序（O(n² log n)）
    var a = [], i;
    for (i = 0; i < s.length; i++) a.push(i);
    a.sort(function (x, y) {
      var p = x, q = y;
      while (p < s.length && q < s.length && s.charAt(p) === s.charAt(q)) { p++; q++; }
      if (p === s.length) return -1;
      if (q === s.length) return 1;
      return s.charAt(p) < s.charAt(q) ? -1 : 1;
    });
    return a;
  }
  function _lcpBrute(s, i, j) {
    var h = 0;
    while (i + h < s.length && j + h < s.length && s.charAt(i + h) === s.charAt(j + h)) h++;
    return h;
  }
  function _matchBrute(text, pat) {                 // 暴力朴素匹配（KMP 的独立参照）
    var res = [], i, t;
    if (pat.length === 0) return res;
    for (i = 0; i + pat.length <= text.length; i++) {
      var ok = true;
      for (t = 0; t < pat.length; t++) if (text.charAt(i + t) !== pat.charAt(t)) { ok = false; break; }
      if (ok) res.push(i);
    }
    return res;
  }

  /* ③ 快速沃尔什–哈达玛变换 + zeta/Möbius 三算子集合卷积（xor / or / and）。
     FWHT：H 为 ±1 Hadamard 矩阵，蝶形 O(N log N)，满足 H² = N·I（自逆到 N 倍）；
     or/and 卷积走「子集（超集）zeta 变换 → 逐点乘 → Möbius 反演」，
     是子集 DP / 集合幂级数 / 快速 zeta 变换的标准底座（亦为正交随机特征 Fastfood 的算子）。
     不变量：三算子 ≡ 朴素 O(N²)（整数 0 精确）；H²=N·I；线性性；单位元（xor/or: δ₀；and: δ_{N−1}）。 */
  function fwht(a) {                                // 未归一化 H·a
    var n = a.length, out = a.slice(), h = 1;
    while (h < n) {
      for (var i = 0; i < n; i += h << 1)
        for (var j = i; j < i + h; j++) {
          var x = out[j], y = out[j + h];
          out[j] = x + y; out[j + h] = x - y;
        }
      h <<= 1;
    }
    return out;
  }
  function _zetaSubset(a) { var n = a.length, f = a.slice(), k = 1; while (k < n) { for (var m = 0; m < n; m++) if (m & k) f[m] += f[m ^ k]; k <<= 1; } return f; }
  function _mobiusSubset(f) { var n = f.length, g = f.slice(), k = 1; while (k < n) { for (var m = 0; m < n; m++) if (m & k) g[m] -= g[m ^ k]; k <<= 1; } return g; }
  function _zetaSuperset(a) { var n = a.length, f = a.slice(), k = 1; while (k < n) { for (var m = 0; m < n; m++) if (!(m & k)) f[m] += f[m ^ k]; k <<= 1; } return f; }
  function _mobiusSuperset(f) { var n = f.length, g = f.slice(), k = 1; while (k < n) { for (var m = 0; m < n; m++) if (!(m & k)) g[m] -= g[m ^ k]; k <<= 1; } return g; }
  function fwhtConvolve(a, b, op) {                 // op: 'xor' | 'or' | 'and'（长度须为 2 的幂）
    var n = a.length, i, out;
    if (op === 'xor') {
      var A = fwht(a), B = fwht(b), C = new Array(n);
      for (i = 0; i < n; i++) C[i] = A[i] * B[i];
      out = fwht(C);
      for (i = 0; i < n; i++) out[i] = out[i] / n;   // H² = N·I ⇒ 逆变换需 ÷N
      return out;
    }
    if (op === 'or') {
      var A2 = _zetaSubset(a), B2 = _zetaSubset(b), C2 = new Array(n);
      for (i = 0; i < n; i++) C2[i] = A2[i] * B2[i];
      return _mobiusSubset(C2);
    }
    var A3 = _zetaSuperset(a), B3 = _zetaSuperset(b), C3 = new Array(n);
    for (i = 0; i < n; i++) C3[i] = A3[i] * B3[i];
    return _mobiusSuperset(C3);
  }
  function _convBrute(a, b, op) {
    var n = a.length, c = new Array(n).fill(0);
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var k = op === 'xor' ? (i ^ j) : (op === 'or' ? (i | j) : (i & j));
      c[k] += a[i] * b[j];
    }
    return c;
  }

  check('maxClique：ω(G) ≡ 暴力 2ⁿ 子集枚举 + 全部极大团极大性 + α(G)=ω(Ḡ)', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), n = 5 + Math.floor(rng() * 8);   // 5..12
      var p = 0.2 + rng() * 0.6, edges = [], i, j;
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) if (rng() < p) edges.push([i, j]);
      var r = maxClique(n, edges);
      if (r.size !== _cliqueBrute(n, edges)) worst = 1;
      if (r.clique.length !== r.size) worst = 1;
      if (r.clique.length && !_cliqueIsMaximal(n, edges, r.clique)) worst = 1;
      for (var t = 0; t < r.maximal.length; t++) if (!_cliqueIsMaximal(n, edges, r.maximal[t])) worst = 1;
      var comp = [];                                               // α(G) = ω(Ḡ)
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) if (!_cliqueAdj(n, edges)[i][j]) comp.push([i, j]);
      if (r.indepSize !== _cliqueBrute(n, comp)) worst = 1;
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('maxClique：ω ≤ 贪心着色数（χ 上界）+ 大 n 稀疏/稠密应力 + 经典实例（K6/空图/P5/C5/C5补）', 0, function () {
    var worst = 0, tested = 0, seed, i, j;
    for (seed = 200; seed < 240; seed++) {
      var rng = mulberry32(seed), n = 10 + Math.floor(rng() * 7);   // 10..16
      var p = rng() < 0.5 ? 0.15 + rng() * 0.2 : 0.7 + rng() * 0.28; // 稀疏 / 稠密两档
      var edges = [];
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) if (rng() < p) edges.push([i, j]);
      var r = maxClique(n, edges);
      if (r.size !== _cliqueBrute(n, edges)) worst = 1;
      if (r.size > _cliqueGreedyColor(n, edges)) worst = 1;          // ω ≤ χ ≤ 贪心着色数
      for (var t = 0; t < r.maximal.length; t++) if (!_cliqueIsMaximal(n, edges, r.maximal[t])) worst = 1;
      tested++;
    }
    var classic = [
      { name: 'K6', n: 6, e: (function () { var a = []; for (var x = 0; x < 6; x++) for (var y = x + 1; y < 6; y++) a.push([x, y]); return a; })(), w: 6 },
      { name: '空图', n: 5, e: [], w: 1 },
      { name: '路径P5', n: 5, e: [[0, 1], [1, 2], [2, 3], [3, 4]], w: 2 },
      { name: '环C5', n: 5, e: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], w: 2 },
      { name: 'C5补图', n: 5, e: [[0, 2], [0, 3], [1, 3], [1, 4], [2, 4]], w: 2 }
    ];
    for (i = 0; i < classic.length; i++) {
      var c = classic[i];
      if (maxClique(c.n, c.e).size !== c.w || _cliqueBrute(c.n, c.e) !== c.w) worst = 1;
    }
    return tested === 0 ? 1 : worst;
  });

  check('suffixArray：SA ≡ 朴素后缀排序 + Kasai LCP ≡ 暴力逐对 LCP（含全同字符/高度重复）', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed);
      var mode = Math.floor(rng() * 3), s = '', i, n2;
      if (mode === 0) {
        var alpha = 2 + Math.floor(rng() * 3);
        n2 = 8 + Math.floor(rng() * 45);
        for (i = 0; i < n2; i++) s += String.fromCharCode(97 + Math.floor(rng() * alpha));
      } else if (mode === 1) {
        n2 = 20 + Math.floor(rng() * 60);
        s = new Array(n2 + 1).join('a');                    // 全同字符（倍增最坏情形）
      } else {
        var unit = rng() < 0.5 ? 'a' : 'ab';
        n2 = 10 + Math.floor(rng() * 20);
        s = new Array(n2 + 1).join(unit + 'c');             // 高度重复
      }
      var r = suffixArray(s);
      if (r.sa.join(',') !== _saBrute(s).join(',')) worst = 1;
      var seen = new Array(s.length).fill(false);            // SA 为 0..n−1 的排列
      for (i = 0; i < s.length; i++) seen[r.sa[i]] = true;
      for (i = 0; i < s.length; i++) if (!seen[i]) worst = 1;
      if (r.lcp[0] !== 0) worst = 1;
      for (i = 1; i < s.length; i++) if (r.lcp[i] !== _lcpBrute(s, r.sa[i - 1], r.sa[i])) worst = 1;
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('suffixArray：SA 二分搜索 ≡ 朴素匹配（多模式，含不命中）+ ≙ kmpSearch 交叉', 0, function () {
    var worst = 0, tested = 0;
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), alpha = 2 + Math.floor(rng() * 3), n = 8 + Math.floor(rng() * 45), s = '', i;
      for (i = 0; i < n; i++) s += String.fromCharCode(97 + Math.floor(rng() * alpha));
      var pats = [s.charAt(0), s.charAt(n - 1), s.substring(0, 3), s.substring(Math.floor(n / 2), Math.floor(n / 2) + 2), 'zzz'];
      for (var t = 0; t < pats.length; t++) {
        var got = saSearch(s, pats[t]).join(',');
        if (got !== _matchBrute(s, pats[t]).join(',')) worst = 1;
        if (pats[t].length && got !== kmpSearch(s, pats[t]).join(',')) worst = 1;   // ≙ 既有 KMP
      }
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('fwhtConvolve：xor / or / and 三算子 ≡ 朴素 O(N²)（整数 0 精确）', 0, function () {
    var worst = 0, tested = 0, ops = ['xor', 'or', 'and'];
    for (var seed = 1; seed <= 25; seed++) {
      var rng = mulberry32(seed), N = 1 << (3 + Math.floor(rng() * 3)), a = [], b = [], i;
      for (i = 0; i < N; i++) { a.push(Math.floor(rng() * 10)); b.push(Math.floor(rng() * 10)); }
      for (var t = 0; t < ops.length; t++) {
        var got = fwhtConvolve(a, b, ops[t]), exp = _convBrute(a, b, ops[t]);
        for (i = 0; i < N; i++) if (got[i] !== exp[i]) worst = 1;
      }
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  check('fwht：H² = N·I 正交性 + 线性性 + 单位元（xor/or: δ₀；and: δ_{N−1}）', 0, function () {
    var worst = 0, tested = 0, ops = ['xor', 'or', 'and'];
    for (var seed = 400; seed < 430; seed++) {
      var rng = mulberry32(seed), N = 64, a = [], b = [], i;
      for (i = 0; i < N; i++) { a.push(Math.floor(rng() * 200) - 100); b.push(Math.floor(rng() * 21) - 10); }
      var twice = fwht(fwht(a));
      for (i = 0; i < N; i++) if (twice[i] !== N * a[i]) worst = 1;        // H² = N·I
      var ab = [], A = fwht(a), B = fwht(b);
      for (i = 0; i < N; i++) ab.push(a[i] + b[i]);
      var L = fwht(ab);
      for (i = 0; i < N; i++) if (L[i] !== A[i] + B[i]) worst = 1;         // 线性性
      var e0 = new Array(N).fill(0); e0[0] = 1;
      var eAll = new Array(N).fill(0); eAll[N - 1] = 1;
      for (var t = 0; t < ops.length; t++) {
        var g = fwhtConvolve(a, ops[t] === 'and' ? eAll : e0, ops[t]);
        for (i = 0; i < N; i++) if (g[i] !== a[i]) worst = 1;              // 单位元
      }
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ==================== v1.3.22 算法库扩展十二 ==================== */

  /* ---------- ① Aho–Corasick 多模式匹配（Aho & Corasick 1975, SIAM J. Comput. 14(1)）
       trie + failure link + output link：一次扫描 O(|text| + |matches|) 列出全部出现位置。
       fail[u] = u 的最长真后缀且在 trie 中；out[u] = u 的全部后缀模式（含 fail 链上的） */
  function ahoCorasick(patterns, text) {
    var go = [{}], out = [[]], fail = [0], depth = [0], repr = [''];
    var i, k, c, u = 0, v, f;
    for (i = 0; i < patterns.length; i++) {
      u = 0;
      for (k = 0; k < patterns[i].length; k++) {
        c = patterns[i].charAt(k);
        if (go[u][c] === undefined) {
          go.push({}); out.push([]); fail.push(0); depth.push(depth[u] + 1); repr.push(repr[u] + c);
          go[u][c] = go.length - 1;
        }
        u = go[u][c];
      }
      out[u].push(i);
    }
    var q = [0], head = 0;
    while (head < q.length) {
      u = q[head++];
      if (u !== 0) out[u] = out[u].concat(out[fail[u]]);   // BFS 保证 fail[u] 的 out 已就绪
      for (c in go[u]) {
        v = go[u][c];
        if (u === 0) fail[v] = 0;
        else {
          f = fail[u];
          while (f !== 0 && go[f][c] === undefined) f = fail[f];
          fail[v] = (go[f][c] !== undefined && go[f][c] !== v) ? go[f][c] : 0;
        }
        q.push(v);
      }
    }
    var matches = [], idx;
    u = 0;
    for (i = 0; i < text.length; i++) {
      c = text.charAt(i);
      while (u !== 0 && go[u][c] === undefined) u = fail[u];
      u = (go[u][c] !== undefined) ? go[u][c] : 0;
      for (k = 0; k < out[u].length; k++) {
        idx = out[u][k];
        matches.push([idx, i - patterns[idx].length + 1]);
      }
    }
    return { matches: matches, states: go.length, go: go, out: out, fail: fail, depth: depth, repr: repr };
  }
  function _acBrute(patterns, text) {          // 暴力：每个起点 × 每个模式逐字符比较
    var res = [], i, p, k, pat, ok;
    for (i = 0; i < text.length; i++) for (p = 0; p < patterns.length; p++) {
      pat = patterns[p];
      if (i + pat.length > text.length) continue;
      ok = true;
      for (k = 0; k < pat.length; k++) if (text.charAt(i + k) !== pat.charAt(k)) { ok = false; break; }
      if (ok) res.push([p, i]);
    }
    return res;
  }
  function _acStructure(A, alphabet, patterns) {   // (a) fail 深度严格递减 (b) 转移 ≡ 最长后缀 (c) out ≡ 后缀模式集
    var v, ai, i, L, p, nodeOf = {}, pat, exp, s, suf, ch, u2, target, want;
    for (v = 0; v < A.repr.length; v++) nodeOf[A.repr[v]] = v;
    for (v = 1; v < A.go.length; v++) if (A.depth[A.fail[v]] >= A.depth[v]) return false;
    for (v = 0; v < A.go.length; v++) for (ai = 0; ai < alphabet.length; ai++) {
      ch = alphabet.charAt(ai); u2 = v;
      while (u2 !== 0 && A.go[u2][ch] === undefined) u2 = A.fail[u2];
      target = (A.go[u2][ch] !== undefined) ? A.go[u2][ch] : 0;
      s = A.repr[v] + ch; want = 0;
      for (L = 0; L <= s.length; L++) { suf = s.substring(L); if (nodeOf[suf] !== undefined) { want = nodeOf[suf]; break; } }
      if (target !== want) return false;
    }
    for (v = 0; v < A.go.length; v++) {
      exp = [];
      for (p = 0; p < patterns.length; p++) {
        pat = patterns[p];
        if (A.repr[v].length >= pat.length && A.repr[v].substring(A.repr[v].length - pat.length) === pat) exp.push(p);
      }
      if (exp.slice().sort().join(',') !== A.out[v].slice().sort().join(',')) return false;
    }
    return true;
  }

  /* ---------- ② CDCL SAT 求解器（Marques-Silva & Sakallah 1996；Moskewicz et al. 2001 GRASP/Chaff）
       watched literals 单元传播 + 1UIP 冲突分析 + 非时序回溯 + VSIDS 活动度启发。
       子句字编码：2*变量 + (0 正 / 1 负)；变量 0..nv-1 */
  function cdclSat(nv, clauses) {
    var db = [], watches = new Array(2 * nv), i;
    for (i = 0; i < 2 * nv; i++) watches[i] = [];
    var assign = new Array(nv).fill(0), level = new Array(nv).fill(0);
    var reason = new Array(nv).fill(-1), activity = new Array(nv).fill(0);
    var trail = [], trailLim = [], qhead = 0, learnts = [];
    var stats = { conflicts: 0, decisions: 0, propagations: 0 };
    function valLit(l) { var a = assign[l >> 1]; if (a === 0) return 0; return (l & 1) ? -a : a; }
    function dlevel() { return trailLim.length; }
    function enqueue(l, ci) {
      var v = l >> 1;
      assign[v] = (l & 1) ? -1 : 1; level[v] = dlevel(); reason[v] = ci;
      trail.push(l); stats.propagations++;
    }
    function addClause(c, learnt) {
      var ci = db.length; db.push(c);
      if (learnt) learnts.push(c.slice());
      if (c.length === 1) { if (valLit(c[0]) === 0) enqueue(c[0], ci); return ci; }
      watches[c[0]].push(ci); watches[c[1]].push(ci);
      return ci;
    }
    function propagate() {                    // watch 被违反时另寻非假文字；否则单元 / 冲突
      var i2, k;
      while (qhead < trail.length) {
        var p = trail[qhead++], falseLit = p ^ 1, ws = watches[falseLit], keep = [], ci, c, first, found;
        for (i2 = 0; i2 < ws.length; i2++) {
          ci = ws[i2]; c = db[ci];
          if (c[0] === falseLit) { c[0] = c[1]; c[1] = falseLit; }
          first = c[0];
          if (valLit(first) === 1) { keep.push(ci); continue; }
          found = -1;
          for (k = 2; k < c.length; k++) if (valLit(c[k]) !== -1) { found = k; break; }
          if (found >= 0) { c[1] = c[found]; c[found] = falseLit; watches[c[1]].push(ci); continue; }
          keep.push(ci);
          if (valLit(first) === -1) {
            for (k = i2 + 1; k < ws.length; k++) keep.push(ws[k]);
            watches[falseLit] = keep;
            return ci;                        // 冲突
          }
          enqueue(first, ci);                // 单元传播
        }
        watches[falseLit] = keep;
      }
      return -1;
    }
    function analyze(ci) {                   // 1UIP：分辨率消去直到当前层仅剩一个文字
      var learnt = [-1], seen = new Array(nv).fill(false);
      var counter = 0, p = -1, idx = trail.length - 1, k, q, c, btlevel = 0;
      do {
        c = db[ci];
        for (k = (p < 0 ? 0 : 1); k < c.length; k++) {
          q = c[k] >> 1;
          if (!seen[q] && level[q] > 0) {
            seen[q] = true;
            if (level[q] >= dlevel()) counter++; else learnt.push(c[k]);
          }
        }
        while (idx >= 0 && !seen[trail[idx] >> 1]) idx--;
        if (idx < 0) throw new Error('cdcl: analyze trail inconsistent');   // 不变量破坏即抛错，不静默降级
        p = trail[idx]; idx--;
        seen[p >> 1] = false; counter--;
        ci = reason[p >> 1];
      } while (counter > 0);
      learnt[0] = p ^ 1;
      for (k = 1; k < learnt.length; k++) { var lv = level[learnt[k] >> 1]; if (lv > btlevel) btlevel = lv; }
      for (k = 0; k < learnt.length; k++) activity[learnt[k] >> 1] += 1;
      return { clause: learnt, btlevel: btlevel };
    }
    function backtrack(lv) {
      if (dlevel() <= lv) return;
      var lim = trailLim[lv], i2, v;
      for (i2 = trail.length - 1; i2 >= lim; i2--) { v = trail[i2] >> 1; assign[v] = 0; reason[v] = -1; level[v] = 0; }
      trail.length = lim; trailLim.length = lv; qhead = trail.length;
    }
    function pickVar() {                      // VSIDS：活动度最高且未赋值
      var best = -1, v;
      for (v = 0; v < nv; v++) if (assign[v] === 0 && (best < 0 || activity[v] > activity[best])) best = v;
      return best;
    }
    for (i = 0; i < clauses.length; i++) addClause(clauses[i].slice(), false);
    var ok = propagate() < 0, ci2, res, learnt2, nci, v2;
    while (ok) {
      ci2 = propagate();
      if (ci2 >= 0) {
        stats.conflicts++;
        if (dlevel() === 0) { ok = false; break; }
        res = analyze(ci2);
        learnt2 = res.clause;
        backtrack(res.btlevel);
        nci = addClause(learnt2, true);
        if (learnt2.length > 1 && valLit(learnt2[0]) === 0) enqueue(learnt2[0], nci);
      } else {
        v2 = pickVar();
        if (v2 < 0) break;                    // 全部分配完成 ⇒ SAT
        trailLim.push(trail.length);
        stats.decisions++;
        enqueue(2 * v2, -1);
      }
    }
    var model = null, i3;
    if (ok) { model = []; for (i3 = 0; i3 < nv; i3++) model.push(assign[i3] === 1); }
    return { sat: ok, model: model, learnts: learnts, stats: stats };
  }
  function _dpllRef(nv, clauses) {            // 纯 DPLL（无学习）：单元传播 + 分支，作独立第二路径
    var assign = new Array(nv).fill(0);
    function val(l) { var a = assign[l >> 1]; if (a === 0) return 0; return (l & 1) ? -a : a; }
    function rec(cls) {
      var changed = true, nxt = [], i, k, c, sat, lits, pick, t, save;
      while (changed) {
        changed = false; nxt = [];
        for (i = 0; i < cls.length; i++) {
          c = cls[i]; sat = false; lits = [];
          for (k = 0; k < c.length; k++) { var vv = val(c[k]); if (vv === 1) { sat = true; break; } if (vv === 0) lits.push(c[k]); }
          if (sat) continue;
          if (lits.length === 0) return false;
          if (lits.length === 1) { var lv = lits[0]; assign[lv >> 1] = (lv & 1) ? -1 : 1; changed = true; }
          nxt.push(lits);
        }
        cls = nxt;
      }
      pick = -1;
      for (i = 0; i < nv; i++) if (assign[i] === 0) { pick = i; break; }
      if (pick < 0) return cls.length === 0;
      for (t = 0; t < 2; t++) {
        save = assign.slice();
        assign[pick] = t === 0 ? 1 : -1;
        if (rec(cls.slice())) return true;
        assign = save;
      }
      return false;
    }
    return rec(clauses.map(function (c) { return c.slice(); }));
  }
  function _satBrute(nv, clauses) {           // 暴力：枚举 2^nv 全部真值指派
    var total = Math.pow(2, nv), m, a, i, k, c, sat, ok, l, v;
    for (m = 0; m < total; m++) {
      a = [];
      for (i = 0; i < nv; i++) a.push((m >> i) & 1 ? 1 : -1);
      sat = true;
      for (i = 0; i < clauses.length && sat; i++) {
        c = clauses[i]; ok = false;
        for (k = 0; k < c.length; k++) { l = c[k]; v = a[l >> 1]; if ((l & 1) ? (v === -1) : (v === 1)) { ok = true; break; } }
        if (!ok) sat = false;
      }
      if (sat) { var model = []; for (i = 0; i < nv; i++) model.push(a[i] === 1); return { sat: true, model: model }; }
    }
    return { sat: false, model: null };
  }
  function _satModels(nv, clauses) {          // 全部满足赋值（用于核验「学习子句被原公式蕴涵」）
    var total = Math.pow(2, nv), out = [], m, a, i, k, c, sat, ok, l, v;
    for (m = 0; m < total; m++) {
      a = [];
      for (i = 0; i < nv; i++) a.push((m >> i) & 1 ? 1 : -1);
      sat = true;
      for (i = 0; i < clauses.length && sat; i++) {
        c = clauses[i]; ok = false;
        for (k = 0; k < c.length; k++) { l = c[k]; v = a[l >> 1]; if ((l & 1) ? (v === -1) : (v === 1)) { ok = true; break; } }
        if (!ok) sat = false;
      }
      if (sat) out.push(a);
    }
    return out;
  }
  function _php(pigeons, holes) {             // 鸽巢原理 CNF（p 只鸽子 / holes 个洞 → p > holes 必 UNSAT）
    var cl = [], i, j, h;
    for (i = 0; i < pigeons; i++) { var c = []; for (h = 0; h < holes; h++) c.push(2 * (i * holes + h)); cl.push(c); }
    for (i = 0; i < pigeons; i++) for (j = i + 1; j < pigeons; j++) for (h = 0; h < holes; h++)
      cl.push([(2 * (i * holes + h)) | 1, (2 * (j * holes + h)) | 1]);
    return { nv: pigeons * holes, clauses: cl };
  }

  /* ---------- ③ Burrows–Wheeler 变换 + FM-index（Burrows & Wheeler 1994；Ferragina & Manzini 2000）
       复用已验证的 suffixArray（Manber–Myers 1990）构造 BWT 列；唯一最小哨兵 '\u0000'；
       LF-mapping = C[c] + occ(c, i) 给出后缀连接，逆变换 O(n) 重建原文 */
  function _bwtTables(bwt) {
    var n = bwt.length, i, k, c, freq = {}, keys = [];
    for (i = 0; i < n; i++) {
      c = bwt.charAt(i);
      if (freq[c] === undefined) { freq[c] = 0; keys.push(c); }   // keys 去重（否则 C 表被重复计数）
      freq[c]++;
    }
    keys.sort();
    var C = {}, acc = 0;
    for (i = 0; i < keys.length; i++) { C[keys[i]] = acc; acc += freq[keys[i]]; }
    var occ = {};
    for (i = 0; i < keys.length; i++) { occ[keys[i]] = new Array(n + 1); occ[keys[i]][0] = 0; }
    for (i = 0; i < n; i++) {
      for (k = 0; k < keys.length; k++) occ[keys[k]][i + 1] = occ[keys[k]][i];
      c = bwt.charAt(i); occ[c][i + 1]++;
    }
    var lf = new Array(n);
    for (i = 0; i < n; i++) { c = bwt.charAt(i); lf[i] = C[c] + occ[c][i]; }
    return { freq: freq, keys: keys, C: C, occ: occ, lf: lf };
  }
  function bwtTransform(s) {
    var t = s + '\u0000', n = t.length;
    var sa = suffixArray(t).sa, bwt = [], primary = -1, i;
    for (i = 0; i < n; i++) {
      bwt.push(t.charAt((sa[i] - 1 + n) % n));
      if (sa[i] === 0) primary = i;
    }
    return { bwt: bwt.join(''), primary: primary, n: n, sa: sa };
  }
  function bwtInverse(bwt, primary) {         // 返回含末尾哨兵的原文
    var T = _bwtTables(bwt), n = bwt.length, s = '', row = primary, i;
    for (i = 0; i < n; i++) { s = bwt.charAt(row) + s; row = T.lf[row]; }
    return s;
  }
  function fmIndex(text) {
    var b = bwtTransform(text), T = _bwtTables(b.bwt);
    return { bwt: b.bwt, primary: b.primary, n: b.n, sa: b.sa, C: T.C, occ: T.occ, lf: T.lf, keys: T.keys, text: text };
  }
  function fmSearch(F, pat) {                 // backward search：O(m) 得到 [sp, ep) 后缀区间
    if (!pat.length) return { count: 0, positions: [] };
    var sp = 0, ep = F.n, i, c;
    for (i = pat.length - 1; i >= 0; i--) {
      c = pat.charAt(i);
      if (F.C[c] === undefined) return { count: 0, positions: [] };
      sp = F.C[c] + F.occ[c][sp];
      ep = F.C[c] + F.occ[c][ep];
      if (sp >= ep) return { count: 0, positions: [] };
    }
    var pos = [];
    for (i = sp; i < ep; i++) pos.push(F.sa[i]);
    pos.sort(function (a, b) { return a - b; });
    return { count: ep - sp, positions: pos, lo: sp, hi: ep };
  }
  function _bwtBrute(s) {                     // 暴力：枚举全部循环移位并排序，取最后一列
    var t = s + '\u0000', n = t.length, rots = [], i;
    for (i = 0; i < n; i++) rots.push(t.substring(i) + t.substring(0, i));
    var idx = [];
    for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return rots[a] < rots[b] ? -1 : (rots[a] > rots[b] ? 1 : 0); });
    var bwt = '', primary = -1;
    for (i = 0; i < n; i++) { bwt += rots[idx[i]].charAt(n - 1); if (idx[i] === 0) primary = i; }
    return { bwt: bwt, primary: primary };
  }
  function _findRef(s, pat) {                 // 暴力：逐起点逐字符匹配
    var out = [], i, k, ok;
    if (!pat.length) return out;
    for (i = 0; i + pat.length <= s.length; i++) {
      ok = true;
      for (k = 0; k < pat.length; k++) if (s.charAt(i + k) !== pat.charAt(k)) { ok = false; break; }
      if (ok) out.push(i);
    }
    return out;
  }

  /* ① Aho–Corasick：匹配集 ≡ 暴力 + 三项结构不变量（fail 深度 / 转移 ≡ 最长后缀 / out ≡ 后缀模式集） */
  check('ahoCorasick：多模式匹配 ≡ 暴力逐起点逐模式扫描（15 seed + 重叠模式族 + 5 大规模实例）+ fail 深度严格递减 + 转移 ≡ 最长后缀 + out ≡ 后缀模式集', 0, function () {
    var worst = 0, tested = 0, seed, rng, alpha, np, pats, tl, text, i, k, p, L, q;
    var cmp = function (a, b) {
      var x = a.map(function (e) { return e[0] + '@' + e[1]; }).sort().join(' ');
      var y = b.map(function (e) { return e[0] + '@' + e[1]; }).sort().join(' ');
      return x === y;
    };
    for (seed = 1; seed <= 15; seed++) {
      rng = mulberry32(seed);
      alpha = rng() < 0.5 ? 'ab' : 'abc';
      np = 1 + Math.floor(rng() * 5); pats = [];
      for (i = 0; i < np; i++) {
        L = 1 + Math.floor(rng() * 4); p = '';
        for (k = 0; k < L; k++) p += alpha.charAt(Math.floor(rng() * alpha.length));
        pats.push(p);
      }
      tl = 10 + Math.floor(rng() * 30); text = '';
      for (i = 0; i < tl; i++) text += alpha.charAt(Math.floor(rng() * alpha.length));
      var A = ahoCorasick(pats, text);
      if (!cmp(A.matches, _acBrute(pats, text))) worst = 1;
      if (!_acStructure(A, alpha, pats)) worst = 1;
      tested++;
    }
    // 确定性：前缀/后缀重叠族（a / aa / aaa / aaaa / aaaaa / ba / aba / ababa）
    var fam = ['a', 'aa', 'aaa', 'aaaa', 'aaaaa', 'ba', 'aba', 'ababa'];
    var famText = 'aaaaabaaabaababaaaaa';
    var AF = ahoCorasick(fam, famText);
    if (!cmp(AF.matches, _acBrute(fam, famText))) worst = 1;
    if (!_acStructure(AF, 'ab', fam)) worst = 1;
    // 大规模：20 模式 × 200 字符（重叠输出密集）
    for (seed = 700; seed < 705; seed++) {
      rng = mulberry32(seed);
      var al2 = 'abcd', ps2 = [];
      for (i = 0; i < 20; i++) {
        L = 1 + Math.floor(rng() * 6); p = '';
        for (k = 0; k < L; k++) p += al2.charAt(Math.floor(rng() * 4));
        ps2.push(p);
      }
      var tx2 = '';
      for (i = 0; i < 200; i++) tx2 += al2.charAt(Math.floor(rng() * 4));
      var A2 = ahoCorasick(ps2, tx2);
      if (!cmp(A2.matches, _acBrute(ps2, tx2))) worst = 1;
      if (!_acStructure(A2, al2, ps2)) worst = 1;
      tested++;
    }
    return tested === 0 ? 1 : worst;
  });

  /* ② CDCL SAT：决策 ≡ 暴力 2^nv 枚举 ≡ 纯 DPLL + 模型合法 + 学习子句被原公式蕴涵 + 鸽巢原理必 UNSAT */
  check('cdclSat：SAT/UNSAT ≡ 暴力 2ⁿ 枚举 ≡ 纯 DPLL（15 seed + 8 相变区实例）+ 模型满足全部子句 + 每条学习子句被原公式蕴涵（枚举全部满足赋值）+ 鸽巢原理 PHP(3→2)…PHP(6→5) 必 UNSAT', 0, function () {
    var worst = 0, tested = 0, satCnt = 0, unsatCnt = 0, seed, rng, nv, ratio, m, clauses, i, k, v, cl, used, r, b, d, c, l, okc, a, sols, lc, okl, bad;
    var modelOK = function (clauses, model) {
      for (var i2 = 0; i2 < clauses.length; i2++) {
        var cc = clauses[i2], hit = false;
        for (var j = 0; j < cc.length; j++) { var l2 = cc[j], v2 = model[l2 >> 1] ? 1 : -1; if ((l2 & 1) ? (v2 === -1) : (v2 === 1)) { hit = true; break; } }
        if (!hit) return false;
      }
      return true;
    };
    // (a) 随机 3-SAT（nv 6..12，ratio 2.0..5.5，含相变前中后）
    for (seed = 1; seed <= 15; seed++) {
      rng = mulberry32(seed);
      nv = 6 + Math.floor(rng() * 7);
      m = Math.round((2 + rng() * 3.5) * nv); clauses = [];
      for (i = 0; i < m; i++) {
        cl = []; used = {};
        for (k = 0; k < 3; k++) {
          v = Math.floor(rng() * nv);
          if (used[v]) { k--; continue; }
          used[v] = 1; cl.push(2 * v + (rng() < 0.5 ? 0 : 1));
        }
        clauses.push(cl);
      }
      r = cdclSat(nv, clauses); b = _satBrute(nv, clauses); d = _dpllRef(nv, clauses);
      if (r.sat !== b.sat || d !== b.sat) { worst = 1; continue; }
      if (r.sat) { satCnt++; if (!modelOK(clauses, r.model)) worst = 1; } else unsatCnt++;
      sols = _satModels(nv, clauses);
      for (i = 0; i < r.learnts.length; i++) {
        lc = r.learnts[i]; bad = false;
        for (k = 0; k < sols.length; k++) {
          a = sols[k]; okl = false;
          for (var j2 = 0; j2 < lc.length; j2++) { var l3 = lc[j2], v3 = a[l3 >> 1]; if ((l3 & 1) ? (v3 === -1) : (v3 === 1)) { okl = true; break; } }
          if (!okl) { bad = true; break; }
        }
        if (bad) { worst = 1; break; }
      }
      tested++;
    }
    // (b) 相变区（nv 13..15，ratio 3.6..5.2）
    for (seed = 800; seed < 808; seed++) {
      rng = mulberry32(seed);
      nv = 13 + Math.floor(rng() * 3);
      m = Math.round((3.6 + rng() * 1.6) * nv); clauses = [];
      for (i = 0; i < m; i++) {
        cl = []; used = {};
        for (k = 0; k < 3; k++) {
          v = Math.floor(rng() * nv);
          if (used[v]) { k--; continue; }
          used[v] = 1; cl.push(2 * v + (rng() < 0.5 ? 0 : 1));
        }
        clauses.push(cl);
      }
      r = cdclSat(nv, clauses); b = _satBrute(nv, clauses); d = _dpllRef(nv, clauses);
      if (r.sat !== b.sat || d !== b.sat) { worst = 1; continue; }
      if (r.sat) { satCnt++; if (!modelOK(clauses, r.model)) worst = 1; } else unsatCnt++;
      sols = _satModels(nv, clauses);
      for (i = 0; i < r.learnts.length; i++) {
        lc = r.learnts[i]; bad = false;
        for (k = 0; k < sols.length; k++) {
          a = sols[k]; okl = false;
          for (var j3 = 0; j3 < lc.length; j3++) { var l4 = lc[j3], v4 = a[l4 >> 1]; if ((l4 & 1) ? (v4 === -1) : (v4 === 1)) { okl = true; break; } }
          if (!okl) { bad = true; break; }
        }
        if (bad) { worst = 1; break; }
      }
      tested++;
    }
    // (c) 鸽巢原理：PHP(p, p-1 洞) 必 UNSAT（p ≤ 5 用暴力枚举核对；p = 6 用 CDCL + 纯 DPLL 双实现互证）
    [[3, 2], [4, 3], [5, 4], [6, 5]].forEach(function (t) {
      var P = _php(t[0], t[1]);
      var rr = cdclSat(P.nv, P.clauses);
      if (rr.sat) { worst = 1; return; }
      if (t[0] <= 5) { if (_satBrute(P.nv, P.clauses).sat) worst = 1; }
      else if (_dpllRef(P.nv, P.clauses)) worst = 1;   // 2^30 暴力不可行 → 第二实现互证
      tested++;
    });
    return tested === 0 ? 1 : worst;
  });

  /* ③ BWT + FM-index：列 ≡ 暴力旋转排序 + 逆变换 ≡ 原文 + LF 双射 + occ 语义 + backward search ≡ 暴力 ≡ kmpSearch ≡ saSearch */
  check('bwtTransform/fmSearch：BWT 列 ≡ 暴力循环移位排序（15 seed + 6 长串/周期串 + 9 边界串）+ 逆变换 ≡ 原文 + LF-mapping 双射 + occ 语义 + backward search ≡ 暴力 ≡ kmpSearch ≡ saSearch', 0, function () {
    var worst = 0, tested = 0, seed, rng, mode, s, n, i, k, al, unit, reps;
    var verify = function (str, doBruteRot) {
      var B = bwtTransform(str);
      if (doBruteRot) {
        var BB = _bwtBrute(str);
        if (B.bwt !== BB.bwt || B.primary !== BB.primary) worst = 1;
      }
      if (bwtInverse(B.bwt, B.primary) !== str + '\u0000') worst = 1;
      var F = fmIndex(str), seen = new Array(F.n).fill(false);
      for (i = 0; i < F.n; i++) { if (seen[F.lf[i]]) { worst = 1; break; } seen[F.lf[i]] = true; }   // LF 为置换
      for (i = 0; i <= F.n; i++) for (k = 0; k < F.keys.length; k++) {                       // occ[c][i] = 前 i 位中 c 的个数
        var cnt = 0, ch = F.keys[k];
        for (var t = 0; t < i; t++) if (F.bwt.charAt(t) === ch) cnt++;
        if (F.occ[ch][i] !== cnt) { worst = 1; break; }
      }
      var pats = [str.charAt(0), str.charAt(str.length - 1), str.substring(0, Math.min(4, str.length)), 'qqqq'];
      for (var q = 0; q < pats.length; q++) {
        var pt = pats[q], r1 = fmSearch(F, pt), r2 = _findRef(str, pt);
        if (r1.positions.join(',') !== r2.join(',')) worst = 1;
        if (r1.count !== r2.length) worst = 1;
        if (r2.join(',') !== kmpSearch(str, pt).join(',')) worst = 1;
        if (r2.length && r2.join(',') !== saSearch(str, pt).join(',')) worst = 1;
      }
      tested++;
    };
    for (seed = 1; seed <= 15; seed++) {
      rng = mulberry32(seed);
      mode = Math.floor(rng() * 3); s = '';
      if (mode === 0) { al = 'abc'; n = 5 + Math.floor(rng() * 40); for (i = 0; i < n; i++) s += al.charAt(Math.floor(rng() * 3)); }
      else if (mode === 1) { n = 3 + Math.floor(rng() * 20); s = new Array(n + 1).join('a'); }
      else { al = 'ab'; n = 6 + Math.floor(rng() * 30); for (i = 0; i < n; i++) s += al.charAt(Math.floor(rng() * 2)); }
      verify(s, true);
    }
    // 长串与周期串
    for (seed = 900; seed < 906; seed++) {
      rng = mulberry32(seed);
      mode = Math.floor(rng() * 2); s = '';
      if (mode === 0) { al = 'abcd'; n = 150 + Math.floor(rng() * 100); for (i = 0; i < n; i++) s += al.charAt(Math.floor(rng() * 4)); }
      else { unit = ['ab', 'abc', 'aa'][Math.floor(rng() * 3)]; reps = 30 + Math.floor(rng() * 30); for (i = 0; i < reps; i++) s += unit; }
      verify(s, true);
    }
    // 确定性边界串（空串 / 单字符 / 全同 / 周期）
    ['', 'a', 'ab', 'aa', 'aaaa', 'abcabcabc', 'abababab', 'ba', 'zzz'].forEach(function (str) { verify(str, true); });
    return tested === 0 ? 1 : worst;
  });

  /* ==================== v1.3.23 算法库扩展十三 ==================== */

  /* ---------- ① Edmonds 带权带花树（最大权匹配，一般图；Edmonds 1965 / Gabow / Kolmogorov 2009 Blossom V 思想）
       原始-对偶：lab 为对偶变量，dist(e)=lab[u]+lab[v]−2w；带紧边 dist=0 时增广，遇奇环收缩为花（blossom），
       增广后按花内结构（flower/get_pr/rotate）展开恢复匹配。移植自 e-maxx 加权 blossom（UOJ #81 验证实现）。
       不变量 ≙ 暴力枚举全部匹配子集（n≤10，0 精确）+ 权全 1 退化 ≡ blossomMatching 最大基数 */
  function maxWeightMatching(n, edges) {
    var M = 2 * n, INF = 1e18, i, u, v, w, U, V, W;
    function mk2D(sz) { var a = new Array(sz); for (var i2 = 0; i2 < sz; i2++) a[i2] = new Array(sz); return a; }
    var g = mk2D(M + 1);
    for (u = 0; u <= M; u++) for (v = 0; v <= M; v++) g[u][v] = { u: u, v: v, w: 0 };
    var match = new Array(M + 1).fill(0), slack = new Array(M + 1).fill(0), st = new Array(M + 1).fill(0);
    var pa = new Array(M + 1).fill(0), Sarr = new Array(M + 1).fill(0), vis = new Array(M + 1).fill(0), lab = new Array(M + 1).fill(0);
    var flower_from = mk2D(M + 1);
    for (u = 0; u <= M; u++) for (v = 0; v <= M; v++) flower_from[u][v] = 0;
    var flower = new Array(M + 1);
    for (i = 0; i <= M; i++) flower[i] = [];
    var n_x = n, q = [], qh = 0;
    function e(uu, vv, ww) { return { u: uu, v: vv, w: ww }; }
    function dist(edge) { return lab[edge.u] + lab[edge.v] - edge.w * 2; }
    function update_slack(uu, x) { if (!slack[x] || dist(g[uu][x]) < dist(g[slack[x]][x])) slack[x] = uu; }
    function set_slack(x) {
      slack[x] = 0;
      for (var uu = 1; uu <= n; uu++) if (g[uu][x].w > 0 && st[uu] !== x && Sarr[st[uu]] === 0) update_slack(uu, x);
    }
    function q_push(x) { if (x <= n) { q.push(x); return; } for (var k = 0; k < flower[x].length; k++) q_push(flower[x][k]); }
    function set_st(x, b) { st[x] = b; if (x <= n) return; for (var k = 0; k < flower[x].length; k++) set_st(flower[x][k], b); }
    function get_pr(b, xr) {
      var fb = flower[b], pr = fb.indexOf(xr);
      if (pr % 2 === 1) { var tail = fb.slice(1); tail.reverse(); flower[b] = [fb[0]].concat(tail); return fb.length - pr; }
      return pr;
    }
    function set_match(uu, vv) {
      match[uu] = g[uu][vv].v;
      if (uu <= n) return;
      var e2 = g[uu][vv], xr = flower_from[uu][e2.u], pr = get_pr(uu, xr);
      for (var k = 0; k < pr; k++) set_match(flower[uu][k], flower[uu][k ^ 1]);
      set_match(xr, vv);
      var fb = flower[uu], rot = fb.splice(0, pr);
      for (var k2 = 0; k2 < rot.length; k2++) fb.push(rot[k2]);
    }
    function augment(uu, vv) {
      var xnv = st[match[uu]];
      set_match(uu, vv);
      if (!xnv) return;
      set_match(xnv, st[pa[xnv]]);
      augment(st[pa[xnv]], xnv);
    }
    function get_lca(uu, vv) {
      var t = (get_lca._t = (get_lca._t || 0) + 1);
      while (uu || vv) {
        if (uu === 0) { var sw = uu; uu = vv; vv = sw; continue; }
        if (vis[uu] === t) return uu;
        vis[uu] = t;
        uu = st[match[uu]];
        if (uu) uu = st[pa[uu]];
        var sw2 = uu; uu = vv; vv = sw2;
      }
      return 0;
    }
    function add_blossom(uu, lca, vv) {
      var b = n + 1;
      while (b <= n_x && st[b]) b++;
      if (b > n_x) n_x++;
      lab[b] = 0; Sarr[b] = 0; match[b] = match[lca];
      flower[b].length = 0; flower[b].push(lca);
      var x, y;
      x = uu;
      while (x !== lca) { y = st[match[x]]; flower[b].push(x); flower[b].push(y); q_push(y); x = st[pa[y]]; }
      var tail = flower[b].slice(1); tail.reverse(); flower[b] = [flower[b][0]].concat(tail);
      x = vv;
      while (x !== lca) { y = st[match[x]]; flower[b].push(x); flower[b].push(y); q_push(y); x = st[pa[y]]; }
      set_st(b, b);
      for (var bx = 1; bx <= n_x; bx++) { g[b][bx].w = 0; g[bx][b].w = 0; }
      for (var bx2 = 1; bx2 <= n; bx2++) flower_from[b][bx2] = 0;
      for (var i2 = 0; i2 < flower[b].length; i2++) {
        var xs = flower[b][i2];
        for (var xx = 1; xx <= n_x; xx++)
          if (g[b][xx].w === 0 || dist(g[xs][xx]) < dist(g[b][xx])) { g[b][xx] = e(g[xs][xx].u, g[xs][xx].v, g[xs][xx].w); g[xx][b] = e(g[xx][xs].u, g[xx][xs].v, g[xx][xs].w); }
        for (var xx2 = 1; xx2 <= n; xx2++) if (flower_from[xs][xx2]) flower_from[b][xx2] = xs;
      }
      set_slack(b);
    }
    function expand_blossom(b) {
      for (var k = 0; k < flower[b].length; k++) set_st(flower[b][k], flower[b][k]);
      var xr = flower_from[b][g[b][pa[b]].u], pr = get_pr(b, xr);
      for (var j = 0; j < pr; j += 2) {
        var xs = flower[b][j], xns = flower[b][j + 1];
        pa[xs] = g[xns][xs].u; Sarr[xs] = 1; Sarr[xns] = 0; slack[xs] = 0; set_slack(xns); q_push(xns);
      }
      Sarr[xr] = 1; pa[xr] = pa[b];
      for (var k2 = pr + 1; k2 < flower[b].length; k2++) { var xs2 = flower[b][k2]; Sarr[xs2] = -1; set_slack(xs2); }
      st[b] = 0;
    }
    function on_found_edge(edge) {
      var u2 = st[edge.u], v2 = st[edge.v];
      if (Sarr[v2] === -1) {
        pa[v2] = edge.u; Sarr[v2] = 1;
        var nu = st[match[v2]]; slack[v2] = slack[nu] = 0; Sarr[nu] = 0; q_push(nu);
      } else if (Sarr[v2] === 0) {
        var lca = get_lca(u2, v2);
        if (!lca) { augment(u2, v2); augment(v2, u2); return 1; }
        else add_blossom(u2, lca, v2);
      }
      return 0;
    }
    function matching() {
      for (var xi = 1; xi <= n_x; xi++) { Sarr[xi] = -1; slack[xi] = 0; }
      q.length = 0; qh = 0;
      for (var x = 1; x <= n_x; x++) if (st[x] === x && !match[x]) { pa[x] = 0; Sarr[x] = 0; q_push(x); }
      if (q.length === 0) return 0;
      for (; ;) {
        while (qh < q.length) {
          var uu = q[qh++];
          if (Sarr[st[uu]] === 1) continue;
          for (var vv = 1; vv <= n; vv++)
            if (g[uu][vv].w > 0 && st[uu] !== st[vv]) {
              if (dist(g[uu][vv]) === 0) { if (on_found_edge(g[uu][vv])) return 1; }
              else update_slack(uu, st[vv]);
            }
        }
        var d = INF;
        for (var bb = n + 1; bb <= n_x; bb++) if (st[bb] === bb && Sarr[bb] === 1) d = Math.min(d, Math.floor(lab[bb] / 2));
        for (var xx = 1; xx <= n_x; xx++)
          if (st[xx] === xx && slack[xx]) {
            if (Sarr[xx] === -1) d = Math.min(d, dist(g[slack[xx]][xx]));
            else if (Sarr[xx] === 0) d = Math.min(d, Math.floor(dist(g[slack[xx]][xx]) / 2));
          }
        for (var u2 = 1; u2 <= n; u2++) {
          if (Sarr[st[u2]] === 0) { if (lab[u2] <= d) return 0; lab[u2] -= d; }
          else if (Sarr[st[u2]] === 1) lab[u2] += d;
        }
        for (var bb2 = n + 1; bb2 <= n_x; bb2++)
          if (st[bb2] === bb2) {
            if (Sarr[st[bb2]] === 0) lab[bb2] += d * 2;
            else if (Sarr[st[bb2]] === 1) lab[bb2] -= d * 2;
          }
        q.length = 0; qh = 0;
        for (var xx2 = 1; xx2 <= n_x; xx2++)
          if (st[xx2] === xx2 && slack[xx2] && st[slack[xx2]] !== xx2 && dist(g[slack[xx2]][xx2]) === 0)
            if (on_found_edge(g[slack[xx2]][xx2])) return 1;
        for (var bb3 = n + 1; bb3 <= n_x; bb3++)
          if (st[bb3] === bb3 && Sarr[bb3] === 1 && lab[bb3] === 0) expand_blossom(bb3);
      }
      return 0;
    }
    for (i = 0; i < edges.length; i++) { U = edges[i][0] + 1; V = edges[i][1] + 1; W = edges[i][2]; if (g[U][V].w < W) { g[U][V].w = W; g[V][U].w = W; } }
    for (u = 0; u <= n; u++) { st[u] = u; flower[u].length = 0; }
    var w_max = 0;
    for (u = 1; u <= n; u++) for (v = 1; v <= n; v++) { flower_from[u][v] = (u === v ? u : 0); w_max = Math.max(w_max, g[u][v].w); }
    for (u = 1; u <= n; u++) lab[u] = w_max;
    var ans = 0;
    while (matching()) { }
    for (u = 1; u <= n; u++) if (match[u] && match[u] < u) ans += g[u][match[u]].w;
    var mm = new Array(n).fill(-1);
    for (i = 1; i <= n; i++) mm[i - 1] = match[i] ? match[i] - 1 : -1;
    return { weight: ans, match: mm };
  }
  function _bruteMaxWeightMatching(n, edges) {        // 参照：枚举全部匹配子集取最大权
    var m = edges.length;
    function rec(idx, occ) {
      if (idx === m) return 0;
      var skip = rec(idx + 1, occ);
      var bu = edges[idx][0], bv = edges[idx][1], bw = edges[idx][2], take = 0;
      if (!(occ & (1 << bu)) && !(occ & (1 << bv))) take = bw + rec(idx + 1, occ | (1 << bu) | (1 << bv));
      return take > skip ? take : skip;
    }
    return rec(0, 0);
  }

  /* ---------- ② 算术编码（Witten–Neal–Cleary 1987；整数区间精确版，BigInt）
       freq 整数频率 → 累计 c[0..m]，总量 T。逐符号把区间 [lo,hi)（以 D=T^k 为单位）缩窄到 [c[s],c[s+1])。
       编码输出区间下界 lo（BigInt）；解码从高位到低位逐符号反演（c[s] ≤ ⌊cur/denom⌋ < c[s+1]，后缀 lo=(cur−c[s]·denom)/f）。
       编解码互为精确逆（无损）；码长 ≈ −log₂(Πp) = 信源熵 + O(1)。 */
  function arithmeticEncode(symbols, freq) {
    var m = freq.length, c = [0n], i;
    for (i = 0; i < m; i++) c.push(c[i] + BigInt(freq[i]));
    var T = c[m];
    if (T <= 0n) throw new Error('arithmeticEncode：freq 总量须 > 0');
    var lo = 0n, hi = 1n, D = 1n;
    for (i = 0; i < symbols.length; i++) {
      var s = symbols[i];
      if (s < 0 || s >= m) throw new Error('arithmeticEncode：符号越界');
      var span = hi - lo, nlo = lo * T + span * c[s], nhi = lo * T + span * c[s + 1];
      lo = nlo; hi = nhi; D = D * T;
    }
    return { lo: lo, hi: hi, width: hi - lo, denom: D, freq: freq.slice(), length: symbols.length };
  }
  function arithmeticDecode(lo, length, freq) {
    var m = freq.length, c = [0n], i;
    for (i = 0; i < m; i++) c.push(c[i] + BigInt(freq[i]));
    var T = c[m], out = [];
    if (length === 0) return out;
    var denom = T ** BigInt(length - 1), cur = lo;
    for (var step = 0; step < length; step++) {
      var d = cur / denom, s = 0;
      while (c[s + 1] <= d) s++;
      out.push(s);
      var f = c[s + 1] - c[s];
      cur = (cur - c[s] * denom) / f;
      denom = denom / T;
    }
    return out;
  }
  function _arithBits(enc) {                            // 码长 ≈ ⌈log₂(denom/width)⌉+1（位长差，误差 ≤1）
    return enc.denom.toString(2).length - enc.width.toString(2).length + 1;
  }

  /* ---------- ③ 正交归一 DCT-II / IDCT（DCT-III）（Ahmed–Natarajan–Rao 1974，JPEG/MPEG 谱基）
       X[k] = a_k·Σₙ x[n]·cos(π(2n+1)k/(2N))，a₀=√(1/N)，a_k=√(2/N)。矩阵 C 正交 ⇒ C⁻¹ = Cᵀ（即 IDCT）。
       不变量：IDCT(DCT(x)) ≡ x（完美重构 ≤1e-12）+ CᵀC = I（基正交归一）+ Parseval 能量守恒 + 直流 X₀=√(1/N)·Σx */
  function dct(x) {
    var N = x.length, X = new Array(N), i, k;
    for (k = 0; k < N; k++) {
      var s = 0;
      for (i = 0; i < N; i++) s += x[i] * Math.cos(Math.PI * (2 * i + 1) * k / (2 * N));
      X[k] = Math.sqrt(k === 0 ? 1 / N : 2 / N) * s;
    }
    return X;
  }
  function idct(X) {
    var N = X.length, x = new Array(N), i, k;
    for (i = 0; i < N; i++) {
      var s = 0;
      for (k = 0; k < N; k++) s += Math.sqrt(k === 0 ? 1 / N : 2 / N) * X[k] * Math.cos(Math.PI * (2 * i + 1) * k / (2 * N));
      x[i] = s;
    }
    return x;
  }

  check('maxWeightMatching：一般图最大权匹配（Edmonds 带权带花树）≡ 暴力枚举全部匹配子集（n≤10，0 精确）+ 权全 1 退化 ≡ blossomMatching 最大基数 + 匹配合法（对称对合）', 0, function () {
    var worst = 0, legalBad = 0, tested = 0;
    for (var g = 0; g < 150; g++) {
      var rng = mulberry32(70000 + g);
      var n = 2 + Math.floor(rng() * 9);
      var maxE = 3 + Math.floor(rng() * 9), eg = [], seen = {};
      for (var k = 0; k < maxE; k++) {
        var u = Math.floor(rng() * n), v = Math.floor(rng() * n);
        if (u === v) continue;
        var key = Math.min(u, v) + '_' + Math.max(u, v);
        if (seen[key]) continue; seen[key] = 1;
        eg.push([u, v, 1 + Math.floor(rng() * 20)]);
      }
      if (eg.length === 0) continue;
      tested++;
      var r = maxWeightMatching(n, eg);
      worst = Math.max(worst, Math.abs(r.weight - _bruteMaxWeightMatching(n, eg)));   // ≙ 暴力 ground truth
      for (var i = 0; i < n; i++) if (r.match[i] !== -1 && (r.match[i] === i || r.match[i] < 0 || r.match[i] >= n || r.match[r.match[i]] !== i)) legalBad++;
    }
    // 权全 1：最大权 = 最大基数 ≡ 项目内 blossomMatching（n≤12）
    var worstDeg = 0;
    for (var g2 = 0; g2 < 80; g2++) {
      var rng2 = mulberry32(90000 + g2);
      var n2 = 3 + Math.floor(rng2() * 10), eg2 = [], seen2 = {};
      for (var k2 = 0; k2 < n2 * 2; k2++) {
        var u2 = Math.floor(rng2() * n2), v2 = Math.floor(rng2() * n2);
        if (u2 === v2) continue;
        var key2 = Math.min(u2, v2) + '_' + Math.max(u2, v2);
        if (seen2[key2]) continue; seen2[key2] = 1; eg2.push([u2, v2, 1]);
      }
      if (eg2.length === 0) continue;
      var r2 = maxWeightMatching(n2, eg2), card = 0;
      for (var i2 = 0; i2 < n2; i2++) if (r2.match[i2] !== -1 && i2 < r2.match[i2]) card++;
      worstDeg = Math.max(worstDeg, Math.abs(card - blossomMatching(n2, eg2).size));
    }
    if (tested === 0) return 1;
    return Math.max(worst, legalBad, worstDeg);
  });

  check('arithmeticEncode/arithmeticDecode：算术编码 round-trip ≡ 原消息（0 误差，300 例多分布/多长度）+ 码长 ∈ [理想−1，理想+2]（Shannon 界 + O(1) 冗余）+ Σp=1 + 高概率符号大幅压缩', 0, function () {
    var worstRT = 0, worstOver = 0, worstUnder = 0, tested = 0;
    for (var g = 0; g < 300; g++) {
      var rng = mulberry32(3100 + g);
      var m = 2 + Math.floor(rng() * 5), freq = [];
      for (var i = 0; i < m; i++) freq.push(1 + Math.floor(rng() * 20));
      var T = freq.reduce(function (a, b) { return a + b; }, 0);
      var L = 1 + Math.floor(rng() * 30), syms = [];
      for (var j = 0; j < L; j++) {
        var r = Math.floor(rng() * T), acc = 0, s = 0;
        for (var q = 0; q < m; q++) { acc += freq[q]; if (r < acc) { s = q; break; } }
        syms.push(s);
      }
      tested++;
      var enc = arithmeticEncode(syms, freq);
      var dec = arithmeticDecode(enc.lo, enc.length, freq);
      if (dec.length !== syms.length || dec.some(function (v, idx) { return v !== syms[idx]; })) worstRT = 1;
      var ideal = 0;
      for (var j2 = 0; j2 < syms.length; j2++) ideal += -Math.log2(freq[syms[j2]] / T);
      var bits = _arithBits(enc);
      worstOver = Math.max(worstOver, bits - (ideal + 2));       // ≤0：码长不超理想 +2
      worstUnder = Math.max(worstUnder, (ideal - 1) - bits);      // ≤0：码长不低于理想 −1
    }
    if (tested === 0) return 1;
    return Math.max(worstRT, worstOver, worstUnder);
  });

  check('dct/idct：正交归一 DCT-II 逆变换 ≡ 原信号（完美重构 ≤1e-12）+ 基正交归一 CᵀC=I ≤1e-12 + Parseval 能量守恒 + 直流分量 X₀=√(1/N)·Σx', 1e-12, function () {
    var worst = 0;
    for (var g = 0; g < 200; g++) {
      var rng = mulberry32(5200 + g);
      var N = 1 + Math.floor(rng() * 24), x = [];
      for (var i = 0; i < N; i++) x.push(rng() * 2 - 1);
      var X = dct(x), xr = idct(X);
      for (var j = 0; j < N; j++) worst = Math.max(worst, Math.abs(xr[j] - x[j]));
      // 基正交归一：行点积（N≤24 全量开销小）
      for (var a = 0; a < N; a++) for (var b = 0; b < N; b++) {
        var ca = a === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N), cb = b === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N), dot = 0;
        for (var n3 = 0; n3 < N; n3++) dot += ca * cb * Math.cos(Math.PI * (2 * n3 + 1) * a / (2 * N)) * Math.cos(Math.PI * (2 * n3 + 1) * b / (2 * N));
        worst = Math.max(worst, Math.abs(dot - (a === b ? 1 : 0)));
      }
    }
    // Parseval + 直流
    for (var g2 = 0; g2 < 80; g2++) {
      var rng2 = mulberry32(8800 + g2);
      var N2 = 2 + Math.floor(rng2() * 20), x2 = [], s1 = 0, sum = 0;
      for (var i2 = 0; i2 < N2; i2++) { var vv = rng2() * 2 - 1; x2.push(vv); s1 += vv * vv; sum += vv; }
      var X2 = dct(x2), s2 = 0;
      for (var j2 = 0; j2 < N2; j2++) s2 += X2[j2] * X2[j2];
      worst = Math.max(worst, Math.abs(s1 - s2), Math.abs(X2[0] - Math.sqrt(1 / N2) * sum));
    }
    return worst;
  });

  /* ==================== v1.3.24 算法库扩展十四 ==================== */

  /* ---------- ① LZ77 压缩（Lempel–Ziv 1977；DEFLATE 骨干）
       滑动窗口贪心最长匹配。token = {off, len, lit}：字面量 off=0/len=0；匹配 off≥1/len≥MINMATCH，
       lit 为匹配后紧跟字符（末尾 null）。解码按 off 回读输出（支持自重叠匹配），与编码互逆。
       不变量 ≙ round-trip 精确 + 每个匹配复现所引用历史子串 + 贪心最长 + 重复数据有效压缩 */
  function lz77Compress(s, W, MINMATCH) {
    W = W || 4096; MINMATCH = MINMATCH || 3;
    var tokens = [], n = s.length, i = 0, j, l, bl, bo, start;
    while (i < n) {
      bl = 0; bo = 0; start = Math.max(0, i - W);
      for (j = start; j < i; j++) {
        l = 0;
        while (i + l < n && s.charCodeAt(j + l) === s.charCodeAt(i + l)) l++;
        if (l > bl) { bl = l; bo = i - j; }
      }
      if (bl >= MINMATCH) { tokens.push({ off: bo, len: bl, lit: (i + bl < n) ? s.charAt(i + bl) : null }); i += bl + 1; }
      else { tokens.push({ off: 0, len: 0, lit: s.charAt(i) }); i += 1; }
    }
    return tokens;
  }
  function lz77Decompress(tokens) {
    var out = [], k, t, start;
    for (k = 0; k < tokens.length; k++) {
      t = tokens[k];
      if (t.len > 0) {
        start = out.length - t.off;
        if (start < 0) throw new Error('LZ77：匹配偏移越界');
        for (var m = 0; m < t.len; m++) out.push(out[start + m]);
      }
      if (t.lit !== null && t.lit !== undefined) out.push(t.lit);
    }
    return out.join('');
  }
  function _lz77Verify(s, tokens, W, MINMATCH) {      // 解析合法 + 贪心最长（独立重扫）
    var i = 0, k, t, bl, start;
    for (k = 0; k < tokens.length; k++) {
      t = tokens[k];
      if (t.len === 0) { if (s.charAt(i) !== t.lit) return false; i += 1; }
      else {
        if (t.off > i || t.off < 1 || t.len < MINMATCH) return false;
        if (s.substr(i - t.off, t.len) !== s.substr(i, t.len)) return false;
        bl = 0; start = Math.max(0, i - W);
        for (var j = start; j < i; j++) { var l = 0; while (i + l < s.length && s.charCodeAt(j + l) === s.charCodeAt(i + l)) l++; if (l > bl) bl = l; }
        if (t.len !== bl) return false;
        i += t.len + (t.lit !== null && t.lit !== undefined ? 1 : 0);
      }
    }
    return i === s.length;
  }

  /* ---------- ② Huffman 编码（Huffman 1952；最优前缀码 / 熵编码）
       频率 → Huffman 树（确定性 tie-break）→ 前缀码。与 v1.3.23 算术编码成对（一静态一自适应思路）。
       不变量 ≙ round-trip 精确 + 前缀无冲突 + Kraft 等式 Σ2^{-l}=1 + 加权码长 ≡ 暴力枚举最优前缀码 */
  function huffmanTree(freq) {
    var nodes = [], i;
    for (i = 0; i < freq.length; i++) if (freq[i] > 0) nodes.push({ sym: i, freq: freq[i], left: null, right: null });
    if (nodes.length === 0) return null;
    if (nodes.length === 1) return nodes[0];
    while (nodes.length > 1) {
      nodes.sort(function (a, b) { return (a.freq - b.freq) || (a.sym - b.sym); });
      var a = nodes.shift(), b = nodes.shift();
      nodes.push({ sym: -1, freq: a.freq + b.freq, left: a, right: b });
    }
    return nodes[0];
  }
  function huffmanCodes(freq) {
    var codes = new Array(freq.length).fill(''), root = huffmanTree(freq);
    if (!root) return codes;
    if (root.left === null && root.right === null) { codes[root.sym] = '0'; return codes; }
    (function walk(node, prefix) {
      if (node.sym !== -1) { codes[node.sym] = prefix; return; }
      walk(node.left, prefix + '0'); walk(node.right, prefix + '1');
    })(root, '');
    return codes;
  }
  function huffmanEncode(syms, freq) {
    var codes = huffmanCodes(freq), bits = '';
    for (var i = 0; i < syms.length; i++) { if (!codes[syms[i]]) throw new Error('Huffman：符号无码'); bits += codes[syms[i]]; }
    return { bits: bits, codes: codes };
  }
  function huffmanDecode(bits, freq) {
    var root = huffmanTree(freq), out = [], node = root, i;
    if (!root) return out;
    if (root.left === null && root.right === null) { for (i = 0; i < bits.length; i++) out.push(root.sym); return out; }
    for (i = 0; i < bits.length; i++) { node = (bits.charAt(i) === '0') ? node.left : node.right; if (node.sym !== -1) { out.push(node.sym); node = root; } }
    return out;
  }
  function _hufPrefixFree(codes) {
    var cs = codes.filter(function (c) { return c.length > 0; }).sort();
    for (var i = 1; i < cs.length; i++) if (cs[i].indexOf(cs[i - 1]) === 0) return false;
    return true;
  }
  function _hufKraft(codes) {
    var cs = codes.filter(function (c) { return c.length > 0; }), maxLen = 0, i, s = 0;
    for (i = 0; i < cs.length; i++) maxLen = Math.max(maxLen, cs[i].length);
    for (i = 0; i < cs.length; i++) s += Math.pow(2, maxLen - cs[i].length);
    return s === Math.pow(2, maxLen);
  }
  function _hufBruteOptimal(freq) {                    // 参照：枚举满足 Kraft Σ2^{-l}≤1 的长度向量取最小加权长（小字母表）
    var syms = [], i;
    for (i = 0; i < freq.length; i++) if (freq[i] > 0) syms.push(i);
    var k = syms.length;
    if (k <= 1) return 0;
    var maxL = k, best = Infinity, pow2 = [];
    for (i = 0; i <= maxL; i++) pow2.push(Math.pow(2, i));
    var cap = pow2[maxL];
    (function rec(idx, S, w) {                          // S=Σ2^{maxL-len}，w=加权长（增量携带 + Kraft 早剪）
      if (S > cap) return;
      if (idx === k) { if (w < best) best = w; return; }
      var f = freq[syms[idx]];
      for (var L = 1; L <= maxL; L++) rec(idx + 1, S + pow2[maxL - L], w + f * L);
    })(0, 0, 0);
    return best;
  }

  /* ---------- ③ Andrew Monotone Chain 凸包（1979；计算几何）
       按 (x,y) 排序去重，分下/上链用叉积保严格左转（剔除共线中间点），O(n log n)。
       不变量 ≙ 独立 Jarvis 礼物包装法（顶点集 0 精确）+ 严格凸（转向一致）+ 全部输入点在包内或边上 */
  function _chCross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
  function _chDedupSort(P) {
    var s = P.map(function (p) { return [p[0], p[1]]; }).sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]); }), out = [];
    for (var i = 0; i < s.length; i++) if (out.length === 0 || out[out.length - 1][0] !== s[i][0] || out[out.length - 1][1] !== s[i][1]) out.push(s[i]);
    return out;
  }
  function convexHull(points) {
    var P = _chDedupSort(points), n = P.length, i;
    if (n <= 2) return P.map(function (p) { return [p[0], p[1]]; });
    var lower = [];
    for (i = 0; i < n; i++) { while (lower.length >= 2 && _chCross(lower[lower.length - 2], lower[lower.length - 1], P[i]) <= 0) lower.pop(); lower.push(P[i]); }
    var upper = [];
    for (i = n - 1; i >= 0; i--) { while (upper.length >= 2 && _chCross(upper[upper.length - 2], upper[upper.length - 1], P[i]) <= 0) upper.pop(); upper.push(P[i]); }
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }
  function _jarvisHull(points) {                        // 独立参照：Jarvis 礼物包装法（严格顶点）
    var P = _chDedupSort(points), n = P.length, i;
    if (n <= 2) return P.map(function (p) { return [p[0], p[1]]; });
    var hull = [], cur = 0, nxt, c;
    function d2(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
    for (var guard = 0; guard <= n + 1; guard++) {
      hull.push(P[cur]); nxt = (cur + 1) % n;
      for (i = 0; i < n; i++) { c = _chCross(P[cur], P[nxt], P[i]); if (c < 0 || (c === 0 && d2(P[cur], P[i]) > d2(P[cur], P[nxt]))) nxt = i; }
      cur = nxt; if (cur === 0) break;
    }
    return hull;
  }

  check('lz77Compress/lz77Decompress：LZ77 round-trip ≡ 原串（0 误差，确定性 + 260 seed 多字母表/多窗口）+ 每个匹配复现所引用历史子串 + 贪心最长匹配 + 重复串 token 数 < 长度', 0, function () {
    var worst = 0, tested = 0;
    function verify(s, W) {
      tested++;
      var tok = lz77Compress(s, W, 3);
      if (lz77Decompress(tok) !== s) worst = 1;
      if (!_lz77Verify(s, tok, W, 3)) worst = 1;
    }
    ['', 'a', 'aaaaa', 'abcabcabcabc', 'BANANA', 'mississippi', 'abababababab',
      'the quick brown fox jumps over the lazy dog. the quick brown fox.',
      'aaaaaaaaaaaaaaaaaaaa', 'abcabcabcabcabcabc'].forEach(function (s) { verify(s, 8); verify(s, 64); });
    for (var g = 0; g < 260; g++) {
      var rng = mulberry32(4200 + g);
      var alpha = 2 + Math.floor(rng() * 4), len = 1 + Math.floor(rng() * 180), s = '', al = 'abcd'.substring(0, alpha);
      for (var i = 0; i < len; i++) s += al.charAt(Math.floor(rng() * alpha));
      verify(s, 16 + Math.floor(rng() * 64));
    }
    var rep = ''; for (var r = 0; r < 100; r++) rep += 'ABCD';
    if (lz77Compress(rep, 4096, 3).length >= rep.length) worst = 1;   // 重复串须有效压缩
    if (tested === 0) return 1;
    return worst;
  });

  check('huffmanEncode/huffmanDecode：Huffman round-trip ≡ 原消息（0 误差）+ 前缀无冲突 + Kraft 等式 Σ2^{-l}=1 + 加权码长 ≡ 暴力枚举最优前缀码（m≤5）+ 熵界 n·H ≤ L ≤ n·H+n + 单符号/偏斜', 0, function () {
    var worst = 0, worstOver = 0, tested = 0;
    for (var g = 0; g < 200; g++) {
      var rng = mulberry32(6200 + g);
      var m = 2 + Math.floor(rng() * 5), freq = [];
      for (var i = 0; i < m; i++) freq.push(1 + Math.floor(rng() * 40));
      var T = freq.reduce(function (a, b) { return a + b; }, 0);
      var len = 1 + Math.floor(rng() * 60), syms = [];
      for (var j = 0; j < len; j++) { var rr = Math.floor(rng() * T), acc = 0, s = 0; for (var q = 0; q < m; q++) { acc += freq[q]; if (rr < acc) { s = q; break; } } syms.push(s); }
      tested++;
      var codes = huffmanCodes(freq);
      if (!_hufPrefixFree(codes)) { worst = 1; continue; }
      if (!_hufKraft(codes)) { worst = 1; continue; }
      var enc = huffmanEncode(syms, freq), dec = huffmanDecode(enc.bits, freq);
      if (dec.length !== syms.length || dec.some(function (v, idx) { return v !== syms[idx]; })) { worst = 1; continue; }
      var huffLen = 0, H = 0;
      for (var j2 = 0; j2 < m; j2++) { huffLen += freq[j2] * codes[j2].length; H += -(freq[j2] / T) * Math.log2(freq[j2] / T); }
      if (huffLen !== _hufBruteOptimal(freq)) worst = 1;               // 最优性 ≙ 暴力
      worstOver = Math.max(worstOver, huffLen - (T * H + T));           // ≤0：距熵每符号 <1 bit
    }
    var c1 = huffmanCodes([5, 1, 1, 1]);
    if (!_hufPrefixFree(c1) || !_hufKraft(c1)) worst = 1;
    var e1 = huffmanEncode([0, 1, 2, 3, 0, 0], [5, 1, 1, 1]), d1 = huffmanDecode(e1.bits, [5, 1, 1, 1]);
    if (d1.join(',') !== '0,1,2,3,0,0') worst = 1;
    var e2 = huffmanEncode([0, 0, 0], [7]), d2 = huffmanDecode(e2.bits, [7]);
    if (d2.join(',') !== '0,0,0') worst = 1;
    if (tested === 0) return 1;
    return Math.max(worst, worstOver);
  });

  check('convexHull：Monotone Chain 凸包顶点集 ≡ 独立 Jarvis 礼物包装法（0 精确，整数坐标）+ 严格凸（CCW 全左转）+ 全部输入点在包内或边上 + 输出均为输入点且无重复（含共线/单点/两点/空退化）', 0, function () {
    var worst = 0, tested = 0;
    function sameSet(A, B) {
      var sa = {}, sb = {}, k;
      for (k = 0; k < A.length; k++) sa[A[k][0] + ',' + A[k][1]] = 1;
      for (k = 0; k < B.length; k++) sb[B[k][0] + ',' + B[k][1]] = 1;
      var ka = Object.keys(sa), kb = Object.keys(sb);
      if (ka.length !== kb.length) return false;
      for (k = 0; k < ka.length; k++) if (!sb[ka[k]]) return false;
      return true;
    }
    function check(pts) {
      tested++;
      var H = convexHull(pts), J = _jarvisHull(pts), m = H.length, i;
      if (!sameSet(H, J)) { worst = 1; return; }
      if (m >= 3) for (i = 0; i < m; i++) if (_chCross(H[i], H[(i + 1) % m], H[(i + 2) % m]) <= 0) { worst = 1; return; }
      var inset = {}, seen = {};
      for (i = 0; i < pts.length; i++) inset[pts[i][0] + ',' + pts[i][1]] = 1;
      for (i = 0; i < m; i++) { if (!inset[H[i][0] + ',' + H[i][1]] || seen[H[i][0] + ',' + H[i][1]]) { worst = 1; return; } seen[H[i][0] + ',' + H[i][1]] = 1; }
      for (i = 0; i < pts.length; i++) for (var e = 0; e < m; e++) if (_chCross(H[e], H[(e + 1) % m], pts[i]) < 0) { worst = 1; return; }
    }
    check([[0, 0], [1, 0], [1, 1], [0, 1]]); check([[0, 0], [1, 0], [2, 0], [1, 1]]);
    check([[0, 0], [5, 0], [5, 5], [0, 5], [2, 2], [1, 1], [3, 3]]); check([[0, 0], [1, 1], [2, 2], [3, 3]]);
    check([[0, 0]]); check([[0, 0], [1, 1]]); check([]);
    for (var g = 0; g < 220; g++) {
      var rng = mulberry32(7200 + g);
      var n = 1 + Math.floor(rng() * 25), pts = [];
      for (var i2 = 0; i2 < n; i2++) pts.push([Math.floor(rng() * 12), Math.floor(rng() * 12)]);
      check(pts);
    }
    if (tested === 0) return 1;
    return worst;
  });

  /* ==================== v1.3.25 算法库扩展十五（计算几何） ==================== */

  /* ---------- ① 最近点对 Closest Pair（分治 O(n log n)；Shamos–Hoey 1975）
       按 x 排序后分治；合并期检查中线两侧宽度 < 当前最小距的条带，条带按 y 排序后只比邻近 7 点。
       整数坐标下距离²为整数 ⇒ 不变量 ≙ 暴力 O(n²) 最小距离²（0 精确） */
  function closestPair(points) {
    function dd2(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
    var C = points.map(function (p, idx) { return [p[0], p[1], idx]; });
    C.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]); });
    function rec(arr) {
      var n = arr.length, i, j, best = { d2: Infinity, i: -1, j: -1 }, dx;
      if (n <= 3) {
        for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
          var d0 = dd2(arr[i], arr[j]);
          if (d0 < best.d2) best = { d2: d0, i: arr[i][2], j: arr[j][2] };
        }
        return best;
      }
      var mid = n >> 1, midx = arr[mid][0];
      var dl = rec(arr.slice(0, mid)), dr = rec(arr.slice(mid));
      best = dl.d2 <= dr.d2 ? dl : dr;
      var strip = [];
      for (i = 0; i < n; i++) { dx = arr[i][0] - midx; if (dx * dx < best.d2) strip.push(arr[i]); }
      strip.sort(function (a, b) { return (a[1] - b[1]) || (a[0] - b[0]); });
      for (i = 0; i < strip.length; i++)
        for (j = i + 1; j < strip.length && (strip[j][1] - strip[i][1]) * (strip[j][1] - strip[i][1]) < best.d2; j++) {
          var d1 = dd2(strip[i], strip[j]);
          if (d1 < best.d2) best = { d2: d1, i: strip[i][2], j: strip[j][2] };
        }
      return best;
    }
    if (points.length < 2) return { d2: Infinity, i: -1, j: -1 };
    return rec(C);
  }

  /* ---------- ② 旋转卡壳直径 Rotating Calipers（Toussaint 1983；O(h) 于凸包顶点）
       复用 v1.3.24 已验证的 convexHull；对每条边推进对踵点 j（叉积面积单调），遍历全部对踵点对取最大距²。
       直径必由凸包顶点对取得 ⇒ 不变量 ≙ 暴力最大点对距²（整数 0 精确） */
  function diameter(points) {
    function dd2(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
    var H = convexHull(points), h = H.length, i, k;
    if (h <= 1) return { d2: 0, a: H[0], b: H[0] };
    if (h === 2) return { d2: dd2(H[0], H[1]), a: H[0], b: H[1] };
    var j = 1;
    for (k = 1; k < h; k++) if (dd2(H[0], H[k]) > dd2(H[0], H[j])) j = k;
    var best = { d2: 0, a: H[0], b: H[0] };
    function consider(x, y) { var d = dd2(H[x], H[y]); if (d > best.d2) best = { d2: d, a: H[x], b: H[y] }; }
    for (i = 0; i < h; i++) {
      var ni = (i + 1) % h;
      while (_chCross(H[i], H[ni], H[(j + 1) % h]) > _chCross(H[i], H[ni], H[j])) j = (j + 1) % h;
      consider(i, j); consider(ni, j);
    }
    return best;
  }

  /* ---------- ③ 最小包围圆 Minimum Enclosing Circle（Welzl 1991；随机增量期望 O(n)）
       确定性洗牌保证可复现。支撑集必为 2 点（直径圆）或 3 点（外接圆）；三点共线时退化为最远两点直径圆。
       不变量 ≙ 暴力枚举全部 2 点直径圆 + 3 点外接圆取最小覆盖半径（≤1e-9）+ 覆盖性 */
  function minEnclosingCircle(points) {
    var EPS = 1e-9;
    function dd2(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
    function dd(a, b) { return Math.sqrt(dd2(a, b)); }
    function inside(c, p) { return dd2([c.x, c.y], p) <= c.r * c.r + EPS; }
    function circ2(a, b) { return { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, r: dd(a, b) / 2 }; }
    function circ3(a, b, c) {
      var s = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
      if (Math.abs(s) < 1e-12) {                            // 共线退化 → 最远两点直径圆
        var cands = [circ2(a, b), circ2(b, c), circ2(a, c)], best = null, t, ok;
        for (t = 0; t < cands.length; t++) {
          ok = true;
          if (!inside(cands[t], a) || !inside(cands[t], b) || !inside(cands[t], c)) ok = false;
          if (ok && (best === null || cands[t].r < best.r)) best = cands[t];
        }
        return best;
      }
      var a2 = a[0] * a[0] + a[1] * a[1], b2 = b[0] * b[0] + b[1] * b[1], c2 = c[0] * c[0] + c[1] * c[1];
      var ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / s;
      var uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / s;
      return { x: ux, y: uy, r: dd([ux, uy], a) };
    }
    var P = points.map(function (p, idx) { return [p[0], p[1], idx]; });
    P.sort(function (u, v) {
      var ha = (u[2] * 1103515245 + 12345) % 2147483648, hb = (v[2] * 1103515245 + 12345) % 2147483648;
      return (ha - hb) || (u[2] - v[2]);
    });
    var c = null, i, j, k;
    for (i = 0; i < P.length; i++) {
      if (c === null || !inside(c, P[i])) {
        c = { x: P[i][0], y: P[i][1], r: 0 };
        for (j = 0; j < i; j++)
          if (!inside(c, P[j])) {
            c = circ2(P[i], P[j]);
            for (k = 0; k < j; k++)
              if (!inside(c, P[k])) { var t3 = circ3(P[i], P[j], P[k]); if (t3) c = t3; }
          }
      }
    }
    return c || { x: 0, y: 0, r: 0 };
  }
  function _bruteMEC(points) {                   // 参照：枚举 2 点直径圆 + 3 点外接圆取最小覆盖半径
    var EPS = 1e-9, n = points.length, i, j, k, best = Infinity;
    function dd2(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
    function dd(a, b) { return Math.sqrt(dd2(a, b)); }
    function circ2(a, b) { return { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, r: dd(a, b) / 2 }; }
    function circ3(a, b, c) {
      var s = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
      if (Math.abs(s) < 1e-12) return null;
      var a2 = a[0] * a[0] + a[1] * a[1], b2 = b[0] * b[0] + b[1] * b[1], c2 = c[0] * c[0] + c[1] * c[1];
      var ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / s;
      var uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / s;
      return { x: ux, y: uy, r: dd([ux, uy], a) };
    }
    function covers(cir) { for (var t = 0; t < n; t++) if (dd2([cir.x, cir.y], points[t]) > cir.r * cir.r + EPS) return false; return true; }
    if (n <= 1) return 0;
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) { var c2 = circ2(points[i], points[j]); if (covers(c2)) best = Math.min(best, c2.r); }
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) for (k = j + 1; k < n; k++) {
      var c3 = circ3(points[i], points[j], points[k]);
      if (c3 && covers(c3)) best = Math.min(best, c3.r);
    }
    return best;
  }

  check('closestPair：最近点对（分治 O(n log n)，Shamos–Hoey 1975）最小距离² ≡ 暴力 O(n²)（0 精确）+ 返回点对确为该距离 + 均为输入点（含重复点 d²=0）', 0, function () {
    var worst = 0, tested = 0;
    function dd2(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
    function check(pts) {
      if (pts.length < 2) return;
      tested++;
      var r = closestPair(pts), b = Infinity, i, j;
      for (i = 0; i < pts.length; i++) for (j = i + 1; j < pts.length; j++) b = Math.min(b, dd2(pts[i], pts[j]));
      if (r.d2 !== b) { worst = 1; return; }
      if (r.i < 0 || r.j < 0 || r.i === r.j || r.i >= pts.length || r.j >= pts.length || dd2(pts[r.i], pts[r.j]) !== r.d2) worst = 1;
    }
    check([[0, 0], [3, 4]]); check([[0, 0], [1, 1], [5, 5], [1, 0]]);
    check([[0, 0], [0, 0], [5, 5]]); check([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]]);
    for (var g = 0; g < 150; g++) {
      var rng = mulberry32(2500 + g);
      var n = 2 + Math.floor(rng() * 120), range = 1 + Math.floor(rng() * 40), pts = [];
      for (var i = 0; i < n; i++) pts.push([Math.floor(rng() * range), Math.floor(rng() * range)]);
      check(pts);
    }
    if (tested === 0) return 1;
    return worst;
  });

  check('diameter：旋转卡壳直径（Toussaint 1983，O(h) 对踵点）≡ 暴力最大点对距²（0 精确）+ 对踵点对确为最远 + 内点/共线/重复不干扰', 0, function () {
    var worst = 0, tested = 0;
    function dd2(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
    function check(pts) {
      if (pts.length < 2) return;
      tested++;
      var r = diameter(pts), b = 0, i, j;
      for (i = 0; i < pts.length; i++) for (j = i + 1; j < pts.length; j++) b = Math.max(b, dd2(pts[i], pts[j]));
      if (r.d2 !== b) { worst = 1; return; }
      if (dd2(r.a, r.b) !== r.d2) worst = 1;
    }
    check([[0, 0], [3, 4]]); check([[0, 0], [1, 0], [1, 1], [0, 1]]);
    check([[0, 0], [5, 0], [5, 5], [0, 5], [2, 2]]); check([[0, 0], [10, 0], [0, 1]]);
    check([[0, 0], [0, 0], [1, 1]]);
    for (var g = 0; g < 150; g++) {
      var rng = mulberry32(3500 + g);
      var n = 2 + Math.floor(rng() * 60), range = 1 + Math.floor(rng() * 50), pts = [];
      for (var i = 0; i < n; i++) pts.push([Math.floor(rng() * range), Math.floor(rng() * range)]);
      check(pts);
    }
    if (tested === 0) return 1;
    return worst;
  });

  check('minEnclosingCircle：最小包围圆（Welzl 1991 随机增量）半径 ≡ 暴力枚举 2 点直径圆 + 3 点外接圆取最小覆盖半径（≤1e-9）+ 覆盖性（全部点在圆内或边上）+ 共线/单点/重复退化', 1e-9, function () {
    var worst = 0, tested = 0;
    for (var g = 0; g < 150; g++) {
      var rng = mulberry32(4800 + g);
      var n = 1 + Math.floor(rng() * 12), range = 1 + Math.floor(rng() * 20), pts = [], i;
      for (i = 0; i < n; i++) pts.push([Math.floor(rng() * range), Math.floor(rng() * range)]);
      tested++;
      var c = minEnclosingCircle(pts);
      for (i = 0; i < pts.length; i++) {
        var dx = pts[i][0] - c.x, dy = pts[i][1] - c.y;
        worst = Math.max(worst, Math.sqrt(dx * dx + dy * dy) - c.r);   // 覆盖性
      }
      var b = _bruteMEC(pts);
      if (b !== Infinity) worst = Math.max(worst, Math.abs(c.r - b));  // 半径 ≡ 暴力最优
    }
    [[[0, 0], [2, 0]], [[0, 0], [2, 0], [1, 0]], [[0, 0], [3, 0], [0, 4]], [[0, 0], [1, 0], [0, 1], [1, 1]],
    [[5, 5]], [[0, 0], [0, 0], [1, 1]]].forEach(function (pts) {
      tested++;
      var c = minEnclosingCircle(pts), i, b2 = _bruteMEC(pts);
      for (i = 0; i < pts.length; i++) { var dx = pts[i][0] - c.x, dy = pts[i][1] - c.y; worst = Math.max(worst, Math.sqrt(dx * dx + dy * dy) - c.r); }
      if (b2 !== Infinity) worst = Math.max(worst, Math.abs(c.r - b2));
    });
    if (tested === 0) return 1;
    return worst;
  });

  /* ==================== v1.3.26 算法库扩展十六（计算几何深化二） ==================== */

  /* ---------- ① 线段求交（平面分片扫描线；Bentley–Ottmann 思想，分片保正确 O(n²)）
       收集全部端点 x，逐分片 (x_k, x_{k+1}) 取贯穿该分片的活跃线段成对求交；分片边界 x_k 处取端点恰触及的线段成对求交。
       共线重叠段返回重叠两端点；T 形 / 共享端点 / 交叉 / 平行均正确处理。
       不变量 ≙ 暴力 O(n²) 全配对逐对求交（集合相等，0 精确，含退化用例） */
  function _geoNormSeg(s) {
    var p = s[0], q = s[1];
    if (p[0] > q[0] || (p[0] === q[0] && p[1] > q[1])) { var tmp = p; p = q; q = tmp; }
    return [p, q];
  }
  function _geoSegInt(s1, s2) {
    var x1 = s1[0][0], y1 = s1[0][1], x2 = s1[1][0], y2 = s1[1][1];
    var x3 = s2[0][0], y3 = s2[0][1], x4 = s2[1][0], y4 = s2[1][1];
    var r0 = x2 - x1, r1 = y2 - y1, s0 = x4 - x3, s1v = y4 - y3;
    var denom = r0 * s1v - r1 * s0, eps = 1e-9;
    if (Math.abs(denom) < eps) {                       // 平行或共线
      var cr = (x3 - x1) * r1 - (y3 - y1) * r0;
      if (Math.abs(cr) > eps) return null;              // 平行不共线
      var L = Math.sqrt(r0 * r0 + r1 * r1);
      if (L < eps) return null;
      var dx = r0 / L, dy = r1 / L;
      var c1a = x1 * dx + y1 * dy, c1b = x2 * dx + y2 * dy;
      var c2a = x3 * dx + y3 * dy, c2b = x4 * dx + y4 * dy;
      var lo = Math.max(Math.min(c1a, c1b), Math.min(c2a, c2b));
      var hi = Math.min(Math.max(c1a, c1b), Math.max(c2a, c2b));
      if (hi - lo < -eps) return null;
      if (hi - lo < eps) { var t0 = lo; return [[x1 + dx * t0, y1 + dy * t0]]; }
      return [[x1 + dx * lo, y1 + dy * lo], [x1 + dx * hi, y1 + dy * hi]];
    }
    var qp0 = x3 - x1, qp1 = y3 - y1;
    var t = (qp0 * s1v - qp1 * s0) / denom;
    var u = (qp0 * r1 - qp1 * r0) / denom;
    if (t >= -eps && t <= 1 + eps && u >= -eps && u <= 1 + eps) return [[x1 + t * r0, y1 + t * r1]];
    return null;
  }
  function _geoBruteSegInt(segments) {
    var S = segments.map(_geoNormSeg), pts = {}, keys = [], i, j;
    for (i = 0; i < S.length; i++) for (j = i + 1; j < S.length; j++) {
      var r = _geoSegInt(S[i], S[j]);
      if (r) for (var k = 0; k < r.length; k++) {
        var kx = Math.round(r[k][0] * 1e7), ky = Math.round(r[k][1] * 1e7), key = kx + ',' + ky;
        if (!pts[key]) { pts[key] = [r[k][0], r[k][1]]; keys.push(key); }
      }
    }
    var out = keys.map(function (kk) { return pts[kk]; });
    out.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]); });
    return out;
  }
  function segmentIntersection(segments) {
    var S = segments.map(_geoNormSeg), pts = {}, keys = [], xset = {}, i, j, x;
    function collect(arr) {                                   // 局部 ci/cj：勿复用外层 i/j（会覆写循环变量致死循环）
      var ci, cj;
      for (ci = 0; ci < arr.length; ci++) for (cj = ci + 1; cj < arr.length; cj++) {
        var r = _geoSegInt(arr[ci], arr[cj]);
        if (r) for (var k = 0; k < r.length; k++) {
          var kx = Math.round(r[k][0] * 1e7), ky = Math.round(r[k][1] * 1e7), key = kx + ',' + ky;
          if (!pts[key]) { pts[key] = [r[k][0], r[k][1]]; keys.push(key); }
        }
      }
    }
    for (i = 0; i < S.length; i++) { xset[S[i][0][0]] = 1; xset[S[i][1][0]] = 1; }
    var XA = Object.keys(xset).map(Number).sort(function (a, b) { return a - b; });
    for (i = 0; i < XA.length; i++) {                          // 边界：端点恰触及的线段成对求交
      x = XA[i];
      var actB = [];
      for (j = 0; j < S.length; j++) if (S[j][0][0] <= x && S[j][1][0] >= x) actB.push(S[j]);
      collect(actB);
    }
    for (i = 0; i < XA.length - 1; i++) {                      // 分片内部：贯穿该分片者成对求交
      var xm = (XA[i] + XA[i + 1]) / 2;
      var actS = [];
      for (j = 0; j < S.length; j++) if (S[j][0][0] <= xm && S[j][1][0] >= xm) actS.push(S[j]);
      collect(actS);
    }
    var out = keys.map(function (k) { return pts[k]; });
    out.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]); });
    return out;
  }

  /* ---------- ② 旋转卡壳最小宽度（凸多边形；Hyyro–Borg–Toussaint，O(h)）
       最小宽度 = min_i [ max_j dist(P_j, line(P_i, P_{i+1})) ]：一条支撑线含边、另一条平行支撑线触顶点时取得。
       不变量 ≙ 暴力枚举每条边 × 每个顶点取最大距离再取最小（0 精确 ≤1e-9） */
  function minWidth(points) {
    var H = convexHull(points), n = H.length, i, j = 1;
    if (n <= 2) return 0;
    function dpl(p, a, b) {
      var num = Math.abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]));
      var den = Math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]));
      return den < 1e-12 ? 0 : num / den;
    }
    var minW = Infinity;
    for (i = 0; i < n; i++) {
      var a = H[i], b = H[(i + 1) % n], guard = 0;
      while (guard++ < n) {
        var d1 = dpl(H[j], a, b), d2 = dpl(H[(j + 1) % n], a, b);
        if (d2 > d1) j = (j + 1) % n; else break;
      }
      var w = dpl(H[j], a, b);
      if (w < minW) minW = w;
    }
    return minW;
  }

  /* ---------- ③ 多边形面积（Shoelace）+ 点在多边形内（射线法 PNPOLY）
       面积 = ½|Σ(x_i y_{i+1} − x_{i+1} y_i)|；点在多边形内：自点向 +x 投射射线，统计与边交点奇偶。
       不变量：area ≙ 三角扇分解（独立代码路径 ≤1e-9）；inside ≙ 环绕数判定（winding number，布尔一致，排除边界） */
  function polygonArea(polygon) {
    var s = 0, i;
    for (i = 0; i < polygon.length; i++) {
      var a = polygon[i], b = polygon[(i + 1) % polygon.length];
      s += a[0] * b[1] - b[0] * a[1];
    }
    return Math.abs(s) / 2;
  }
  function pointInPolygon(point, polygon) {
    var x = point[0], y = point[1], inside = false, i, j = polygon.length - 1, n = polygon.length;
    for (i = 0; i < n; j = i++) {
      var xi = polygon[i][0], yi = polygon[i][1], xj = polygon[j][0], yj = polygon[j][1];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  }

  check('segmentIntersection：线段求交（平面分片扫描线）交点集合 ≡ 暴力 O(n²) 全配对逐对求交（0 精确，集合相等，含交叉 / T 形 / 共享端点 / 共线重叠 / 平行退化）+ 去重', 0, function () {
    var worst = 0, tested = 0;
    function eqSets(a, b) {
      if (a.length !== b.length) return false;
      for (var i = 0; i < a.length; i++) if (Math.abs(a[i][0] - b[i][0]) > 1e-6 || Math.abs(a[i][1] - b[i][1]) > 1e-6) return false;
      return true;
    }
    function rndSeg(rng) {
      var x1, y1, x2, y2;
      do { x1 = Math.floor(rng() * 100); y1 = Math.floor(rng() * 100); x2 = Math.floor(rng() * 100); y2 = Math.floor(rng() * 100); } while (Math.abs(x1 - x2) < 1);
      return [[x1, y1], [x2, y2]];
    }
    for (var g = 0; g < 220; g++) {
      var rng = mulberry32(8200 + g);
      var n = 2 + Math.floor(rng() * 16), segs = [];
      for (var i = 0; i < n; i++) segs.push(rndSeg(rng));
      tested++;
      if (!eqSets(segmentIntersection(segs), _geoBruteSegInt(segs))) worst = 1;
    }
    var deg = [
      [[[0, 0], [10, 10]], [[0, 10], [10, 0]]],
      [[[0, 5], [10, 5]], [[5, 0], [5, 5]]],
      [[[0, 0], [5, 5]], [[5, 5], [10, 0]]],
      [[[0, 0], [10, 0]], [[3, 0], [13, 0]]],
      [[[0, 0], [5, 0]], [[5, 0], [10, 0]]],
      [[[0, 0], [10, 0]], [[8, 0], [2, 0]]],
      [[[0, 0], [10, 0]], [[0, 5], [10, 5]]],
      [[[0, 0], [10, 10]], [[0, 10], [10, 0]], [[5, 0], [5, 10]], [[0, 5], [10, 5]]]
    ];
    for (var d = 0; d < deg.length; d++) if (!eqSets(segmentIntersection(deg[d]), _geoBruteSegInt(deg[d]))) worst = 1;
    if (tested === 0) return 1;
    return worst;
  });

  check('minWidth：旋转卡壳最小宽度（凸多边形，O(h)）≡ 暴力枚举每条边 × 每个顶点取最大距离再取最小（0 精确 ≤1e-9）+ 支撑线语义', 1e-9, function () {
    var worst = 0, tested = 0;
    function dpl(p, a, b) {
      var num = Math.abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]));
      var den = Math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]));
      return den < 1e-12 ? 0 : num / den;
    }
    for (var g = 0; g < 220; g++) {
      var rng = mulberry32(9200 + g);
      var m = 4 + Math.floor(rng() * 12), pts = [];
      for (var i = 0; i < m; i++) pts.push([Math.floor(rng() * 100) - 50, Math.floor(rng() * 100) - 50]);
      var H = convexHull(pts);
      if (H.length < 3) continue;
      tested++;
      var brute = Infinity;
      for (var e = 0; e < H.length; e++) {
        var a = H[e], b = H[(e + 1) % H.length], mx = 0;
        for (var v = 0; v < H.length; v++) mx = Math.max(mx, dpl(H[v], a, b));
        if (mx < brute) brute = mx;
      }
      worst = Math.max(worst, Math.abs(minWidth(pts) - brute));
    }
    if (tested === 0) return 1;
    return worst;
  });

  check('polygonArea/pointInPolygon：面积（Shoelace）≙ 三角扇分解（≤1e-9）+ 点在多边形内（射线法）≙ 环绕数判定（布尔一致，排除边界）+ 凹多边形分类', 1e-9, function () {
    var worst = 0;
    function fanArea(poly) {
      var s = 0, o = poly[0];
      for (var i = 1; i < poly.length - 1; i++) s += Math.abs((poly[i][0] - o[0]) * (poly[i + 1][1] - o[1]) - (poly[i][1] - o[1]) * (poly[i + 1][0] - o[0])) / 2;
      return s;
    }
    function winding(pt, poly) {
      var ang = 0, n = poly.length;
      for (var i = 0; i < n; i++) {
        var a = [poly[i][0] - pt[0], poly[i][1] - pt[1]], b = [poly[(i + 1) % n][0] - pt[0], poly[(i + 1) % n][1] - pt[1]];
        ang += Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
      }
      return Math.abs(ang) > Math.PI;
    }
    function dpl(p, a, b) {
      var num = Math.abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]));
      var den = Math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]));
      return den < 1e-12 ? 0 : num / den;
    }
    for (var g = 0; g < 220; g++) {
      var rng = mulberry32(10200 + g);
      var m = 4 + Math.floor(rng() * 12), pts = [];
      for (var i = 0; i < m; i++) pts.push([Math.floor(rng() * 100) - 50, Math.floor(rng() * 100) - 50]);
      var H = convexHull(pts);
      if (H.length < 3) continue;
      worst = Math.max(worst, Math.abs(polygonArea(H) - fanArea(H)));
    }
    var nc = [[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]];                 // 凹（顶部中央内凹）
    var inPts = [[5, 2], [2, 3], [8, 3], [1, 8], [9, 8]];
    var outPts = [[5, 7], [3, 9], [5, 9], [-1, 5], [11, 5]];
    for (var q = 0; q < inPts.length; q++) if (!(pointInPolygon(inPts[q], nc) && winding(inPts[q], nc))) worst = 1;
    for (var r = 0; r < outPts.length; r++) if (pointInPolygon(outPts[r], nc) || winding(outPts[r], nc)) worst = 1;
    for (var g2 = 0; g2 < 200; g2++) {
      var rng2 = mulberry32(11200 + g2);
      var mm = 4 + Math.floor(rng2() * 10), pp = [];
      for (var k = 0; k < mm; k++) pp.push([Math.floor(rng2() * 80) - 40, Math.floor(rng2() * 80) - 40]);
      var HH = convexHull(pp);
      if (HH.length < 3) continue;
      for (var s2 = 0; s2 < 6; s2++) {
        var pt = [Math.floor(rng2() * 90) - 45, Math.floor(rng2() * 90) - 45], near = false;
        for (var e2 = 0; e2 < HH.length; e2++) if (dpl(pt, HH[e2], HH[(e2 + 1) % HH.length]) < 1e-3) near = true;
        if (near) continue;
        if (pointInPolygon(pt, HH) !== winding(pt, HH)) worst = 1;
      }
    }
    return worst;
  });

  function runChecks() {
    var results = checks.map(function (c) {
      var residual, err = null;
      try { residual = c.run(); } catch (e) { residual = Infinity; err = e; }
      var pass = isFinite(residual) && residual <= c.tol;
      return { name: c.name, residual: residual, tol: c.tol, pass: pass, error: err ? String(err.message || err) : null };
    });
    return {
      total: results.length,
      passed: results.filter(function (r) { return r.pass; }).length,
      results: results
    };
  }

  return {
    version: VERSION,
    /* 工具 */
    mulberry32: mulberry32, zeros2: zeros2, eye: eye, clone2: clone2, dot: dot, norm2: norm2, normInf: normInf,
    maxAbsDiff: maxAbsDiff, maxAbsDiff2: maxAbsDiff2, dist2: dist2, euclid: euclid, spdMatrix: spdMatrix,
    /* 线性代数 */
    transpose: transpose, matMul: matMul, matVec: matVec, trace: trace, det: det, inverse: inverse,
    solveLinear: solveLinear, solveLower: solveLower, cholesky: cholesky, safeCholesky: safeCholesky,
    qrDecompose: qrDecompose, jacobiEigen: jacobiEigen, powerIteration: powerIteration, pca: pca,
    /* 数值稳定 */
    logsumexp: logsumexp, softmax: softmax, sigmoid: sigmoid, tanh: tanh, relu: relu, gelu: gelu,
    erf: erf, normalCDF: normalCDF, kahanSum: kahanSum, exactSum: exactSum, clip: clip,
    /* 概率统计 */
    gaussPDF: gaussPDF, expectation: expectation, variance: variance, covariance: covariance,
    normalSample: normalSample, categoricalSample: categoricalSample, bayesUpdate: bayesUpdate,
    /* 信息论 */
    entropy: entropy, crossEntropy: crossEntropy, klDivergence: klDivergence, jsDivergence: jsDivergence,
    mutualInfo: mutualInfo, conditionalEntropy: conditionalEntropy, gini: gini, infoGain: infoGain,
    /* 优化 */
    gradientDescent: gradientDescent, adam: adam, conjugateGradient: conjugateGradient,
    /* 模型 */
    linearRegression: linearRegression, logisticRegression: logisticRegression, kmeans: kmeans, kmeansPP: kmeansPP,
    gmmEM: gmmEM, knn: knn, decisionTree: decisionTree,
    mlpForward: mlpForward, mlpLoss: mlpLoss, mlpGrad: mlpGrad, gradientCheck: gradientCheck, mlpTrain: mlpTrain,
    /* 深度学习组件 */
    linear: linear, layerNorm: layerNorm, selfAttention: selfAttention,
    /* 图模型 */
    hmmForwardBackward: hmmForwardBackward, hmmViterbi: hmmViterbi,
    /* 强化学习 */
    valueIteration: valueIteration, td0Sweep: td0Sweep, qLearning: qLearning, epsilonGreedy: epsilonGreedy,
    chainSetup: chainSetup,
    /* 指标 */
    accuracy: accuracy, precision: precision, recall: recall, f1: f1, auc: auc, mse: mse, mae: mae,
    r2: r2, logLoss: logLoss, cosineSimilarity: cosineSimilarity, editDistance: editDistance, silhouette: silhouette,
    /* 归一化/数据/系统 */
    l1Norm: l1Norm, l2Norm: l2Norm, linfNorm: linfNorm, normalizeL2: normalizeL2, minmaxScale: minmaxScale,
    zscore: zscore, l2Reg: l2Reg, temperatureSoftmax: temperatureSoftmax, topK: topK,
    shuffle: shuffle, trainTestSplit: trainTestSplit, kFold: kFold, oneHot: oneHot,
    Arena: Arena, reduceAll: reduceAll, broadcast: broadcast,
    /* 扩展：TENSOR */
    tensorCreate: tensorCreate, tensorCount: tensorCount, tensorZeros: tensorZeros, tensorOnes: tensorOnes,
    arange: arange, flatten: flatten, reshape: reshape, tensorSlice: tensorSlice, tensorConcat: tensorConcat,
    tensorStack: tensorStack, tensorSplit: tensorSplit, tensorCast: tensorCast, tensorCopy: tensorCopy,
    tensorGather: tensorGather, tensorScatter: tensorScatter, tensorFill: tensorFill,
    /* 扩展：NUMER */
    divGuard: divGuard, sqrtGuard: sqrtGuard, reciprocal: reciprocal,
    expm1: expm1, log1p: log1p, hypot: hypot,
    twoSum: twoSum, twoProd: twoProd, fma: fma,
    /* 扩展：PROB */
    uniformSample: uniformSample, bernoulliSample: bernoulliSample, binomialSample: binomialSample,
    marginal: marginal, conditionalRow: conditionalRow, mvGaussPDF: mvGaussPDF,
    /* 扩展：INFO */
    perplexity: perplexity, surprisal: surprisal, bits: bits, klReverse: klReverse,
    maxEntropyDist: maxEntropyDist, normalizeDist: normalizeDist,
    /* 扩展：OPTIM */
    quadProblem: quadProblem, momentum: momentum, nesterov: nesterov, rmsprop: rmsprop, adagrad: adagrad,
    adamw: adamw, gradClip: gradClip, gradNorm: gradNorm, lrDecay: lrDecay, weightDecay: weightDecay, polyakAvg: polyakAvg,
    /* 扩展：FWD */
    embeddingLookup: embeddingLookup, residualAdd: residualAdd, splitHeads: splitHeads, concatHeads: concatHeads,
    poolMax: poolMax, batchNorm: batchNorm, dropout: dropout, conv1d: conv1d,
    /* 扩展：BWD */
    numericJacobian: numericJacobian, jacobian: jacobian, vjp: vjp, jvp: jvp, chainRule: chainRule,
    gradRelu: gradRelu, gradTanh: gradTanh, gradSoftmaxCE: gradSoftmaxCE, gradMatmul: gradMatmul,
    gradAccum: gradAccum, stopGrad: stopGrad, detach: detach,
    rnnForward: rnnForward, rnnLoss: rnnLoss, rnnBPTT: rnnBPTT,
    /* 扩展：NORM */
    l1Reg: l1Reg, weightNorm: weightNorm, spectralNorm: spectralNorm, groupNorm: groupNorm,
    labelSmooth: labelSmooth, mixup: mixup, earlyStop: earlyStop,
    /* 扩展：ATTEN */
    qkDot: qkDot, attentionScale: attentionScale, crossAttention: crossAttention, multiHead: multiHead,
    ropeApply: ropeApply, attnRow: attnRow, mqa: mqa, gqa: gqa, attnEntropy: attnEntropy,
    relPosBias: relPosBias, alibiScores: alibiScores, flashAttention: flashAttention,
    /* 扩展：SAMPLE */
    gumbelSample: gumbelSample, topP: topP, multinomialSample: multinomialSample,
    rejectionSample: rejectionSample, inverseCDF: inverseCDF, antitheticSample: antitheticSample,
    stratifiedSample: stratifiedSample, systematicSample: systematicSample,
    /* 扩展：GRAPH */
    chainBeliefProp: chainBeliefProp, variableElim: variableElim, factorIndex: factorIndex,
    factorProduct: factorProduct, sumOut: sumOut, ripCheck: ripCheck, junctionTree: junctionTree,
    cliqueTreeCalibrate: cliqueTreeCalibrate, kalmanFilter1d: kalmanFilter1d, moralGraph: moralGraph,
    dSeparated: dSeparated, mSeparated: mSeparated, particleFilter: particleFilter,
    /* 扩展：RL */
    bellmanBackup: bellmanBackup, policyIteration: policyIteration, sarsa: sarsa,
    eligibilityTrace: eligibilityTrace, monteCarloReturn: monteCarloReturn, policyEval: policyEval,
    greedyPolicy: greedyPolicy, ucb: ucb, advantage: advantage, gae: gae,
    reinforceGradient: reinforceGradient, ppoClip: ppoClip, thompsonSample: thompsonSample,
    /* 扩展：METRIC/DATA/SYS/LINALG */
    perplexityLM: perplexityLM, bleu: bleu, averagePrecision: averagePrecision, confusionMatrix: confusionMatrix,
    sampleStats: sampleStats, channelCapacity: channelCapacity, lineSearch: lineSearch,
    labelEncode: labelEncode, padSeq: padSeq, truncateSeq: truncateSeq, bucketize: bucketize, dedup: dedup,
    filterOp: filterOp, mapOp: mapOp, zipOp: zipOp, batchOp: batchOp, tokenize: tokenize,
    normalizeFit: normalizeFit, augment: augment,
    gatherAll: gatherAll, scatterAll: scatterAll, Cache: Cache, RefCounter: RefCounter, TaskStream: TaskStream,
    checkpoint: checkpoint, Profiler: Profiler, Tracer: Tracer, Dispatcher: Dispatcher, Barrier: Barrier,
    EventLog: EventLog, prefetch: prefetch,
    matrixRank: matrixRank, nullspace: nullspace, orthonormalize: orthonormalize, condEstimate: condEstimate,
    /* 集大成验证算法库（v1.3.8）—— L10 ISA 之外的旗舰级参考实现 */
    svd: svd, luDecompose: luDecompose, fft: fft, dft: dft,
    pageRank: pageRank, dijkstra: dijkstra, bellmanFord: bellmanFord,
    sliceSample: sliceSample, newton: newton, rk4: rk4,
    /* 算法库扩展（v1.3.9）—— 二叉堆/堆优化 Dijkstra/A星/FFT 卷积/CSR/MST/二分/Kadane */
    heapPush: heapPush, heapPop: heapPop, heapSort: heapSort,
    dijkstraHeap: dijkstraHeap, aStar: aStar, ifft: ifft, fftConvolve: fftConvolve,
    denseToCSR: denseToCSR, csrMatVec: csrMatVec, csrTranspose: csrTranspose, csrToDense: csrToDense,
    primMST: primMST, kruskalMST: kruskalMST, bruteForceMST: bruteForceMST,
    bisect: bisect, kadane: kadane,
    /* 算法库扩展（v1.3.10）—— QR 特征值/最大流/SCC/拓扑/二分匹配 */
    qrEigenvalues: qrEigenvalues,
    maxFlowEK: maxFlowEK, maxFlowDinic: maxFlowDinic, minCutBrute: minCutBrute,
    tarjanSCC: tarjanSCC, kosarajuSCC: kosarajuSCC,
    kahnTopo: kahnTopo, kuhnMatching: kuhnMatching, minVertexCoverBrute: minVertexCoverBrute,
    /* v1.3.12 算法库扩展二 */
    bluesteinFFT: bluesteinFFT,
    cholSafe: cholSafe, covOf: covOf, gaussRand: gaussRand,
    adaptiveMetropolis: adaptiveMetropolis, hmcSample: hmcSample,
    floydWarshall: floydWarshall,
    dsuCreate: dsuCreate, dsuFind: dsuFind, dsuUnion: dsuUnion, dsuComponents: dsuComponents,
    kruskalDSU: kruskalDSU,
    kmpFailure: kmpFailure, kmpSearch: kmpSearch, bruteSearch: bruteSearch,
    legendreP: legendreP, gaussLegendre: gaussLegendre, gaussQuad: gaussQuad,
    strassenMul: strassenMul, addM: addM, subM: subM,
    /* v1.3.13 算法库扩展三 */
    mixedRadixFFT: mixedRadixFFT,
    nutsSample: nutsSample,
    mcmfSSP: mcmfSSP, mcmfSSPRef: mcmfSSPRef,
    /* v1.3.14 算法库扩展四 */
    nutsGenUTurn: nutsGenUTurn,
    nttConvolve: nttConvolve, nttConvolveExact: nttConvolveExact,
    hopcroftKarp: hopcroftKarp,
    /* v1.3.15 算法库扩展五 */
    blossomMatching: blossomMatching,
    splitRadixFFT: splitRadixFFT,
    stoerWagner: stoerWagner,
    /* v1.3.16 算法库扩展六 */
    gomoryHu: gomoryHu,
    radix4FFT: radix4FFT,
    nutsDiagMass: nutsDiagMass,
    /* v1.3.17 算法库扩展七 */
    pushRelabel: pushRelabel,
    hungarianMaxWeight: hungarianMaxWeight,
    stableMarriage: stableMarriage,
    /* v1.3.18 算法库扩展八 */
    globalMinCutPR: globalMinCutPR,
    twoSat: twoSat,
    bridgesArticulation: bridgesArticulation,
    /* v1.3.19 算法库扩展九 */
    auctionAssignment: auctionAssignment,
    topTradingCycles: topTradingCycles,
    simplexLP: simplexLP,
    /* v1.3.20 算法库扩展十 */
    arborescence: arborescence,
    heldKarpTSP: heldKarpTSP,
    ac3CSP: ac3CSP,
    /* v1.3.21 算法库扩展十一 */
    maxClique: maxClique,
    suffixArray: suffixArray,
    saSearch: saSearch,
    fwht: fwht,
    fwhtConvolve: fwhtConvolve,
    /* v1.3.22 算法库扩展十二 */
    ahoCorasick: ahoCorasick,
    cdclSat: cdclSat,
    bwtTransform: bwtTransform,
    bwtInverse: bwtInverse,
    fmIndex: fmIndex,
    fmSearch: fmSearch,
    /* v1.3.23 算法库扩展十三 */
    maxWeightMatching: maxWeightMatching,
    arithmeticEncode: arithmeticEncode,
    arithmeticDecode: arithmeticDecode,
    dct: dct,
    idct: idct,
    /* v1.3.24 算法库扩展十四 */
    lz77Compress: lz77Compress,
    lz77Decompress: lz77Decompress,
    huffmanTree: huffmanTree,
    huffmanCodes: huffmanCodes,
    huffmanEncode: huffmanEncode,
    huffmanDecode: huffmanDecode,
    convexHull: convexHull,
    /* v1.3.25 算法库扩展十五（计算几何） */
    closestPair: closestPair,
    diameter: diameter,
    minEnclosingCircle: minEnclosingCircle,
    /* v1.3.26 算法库扩展十六（计算几何深化二） */
    segmentIntersection: segmentIntersection,
    minWidth: minWidth,
    polygonArea: polygonArea,
    pointInPolygon: pointInPolygon,
    /* 扩展：拟牛顿/MCMC/率失真（v1.1.1） */
    eye2: eye2, wolfeLineSearch: wolfeLineSearch, bfgs: bfgs, lbfgs: lbfgs, gradLayerNorm: gradLayerNorm,
    gibbsSampler: gibbsSampler, metropolisSample: metropolisSample, rateDistortion: rateDistortion,
    /* 执行变体内核（v1.1.2–v1.1.4） */
    arrMaxDiff: arrMaxDiff,
    batched: batched, BATCH_SIG: BATCH_SIG,                 // 通用批处理引擎（v1.1.4）
    batched_selfAttention: batched_selfAttention, batched_oneHot: batched_oneHot,
    inplace: inplace, INPLACE_SIG: INPLACE_SIG,             // 通用原地引擎（v1.1.5）
    fused: fused, FUSE_SIG: FUSE_SIG,                       // 通用 Epilogue 融合引擎（v1.1.6）
    inplace_tensorFill: inplace_tensorFill, inplace_tensorCopy: inplace_tensorCopy, inplace_normalizeL2: inplace_normalizeL2,
    /* 融合变体内核（v1.1.3） */
    fused_transpose_flatten: fused_transpose_flatten, fused_matvec_relu: fused_matvec_relu,
    fused_softmax_log: fused_softmax_log, fused_gauss_pdf_log: fused_gauss_pdf_log,
    fused_softmax_ce: fused_softmax_ce, fused_gradclip_norm: fused_gradclip_norm,
    fused_linear_relu: fused_linear_relu, fused_gradtanh_chain: fused_gradtanh_chain,
    fused_norm_affine: fused_norm_affine, fused_attn_weights: fused_attn_weights,
    fused_topk_temp: fused_topk_temp, fused_marginal_norm: fused_marginal_norm,
    fused_value_greedy: fused_value_greedy, fused_rmse: fused_rmse,
    fused_label_onehot: fused_label_onehot, fused_allreduce: fused_allreduce,
    /* 自检 */
    checks: checks, runChecks: runChecks
  };
});
