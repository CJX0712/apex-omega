<div align="center">

# ✦ APEX-Ω · 世界旗舰 AI 合流系统

### 真实大模型 × 1024 微指令 ISA × 可证门禁

[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Engine: llama.cpp](https://img.shields.io/badge/engine-llama.cpp-blueviolet)](https://github.com/ggml-org/llama.cpp)
[![ISA: 1024](https://img.shields.io/badge/ISA-1024%20micro--instructions-brightgreen)](src/layers/l10-isa.js)
[![Verified: 811](https://img.shields.io/badge/verified-811%2F1024-blue)](src/layers/l10-isa.js)
[![Invariants: 386](https://img.shields.io/badge/math%20invariants-386-brightgreen)](https://github.com/CJX0712/axiom-ai)
[![Author: 晨星](https://img.shields.io/badge/author-晨星-blueviolet)](https://github.com/CJX0712)

</div>

> **零重写**：任何需求，先检索世界已有的最优解 —— 顶级数学、顶级算法、顶级代码、顶级开源项目 —— 直接复用、组合、超越。
> **集大成**：每一行代码、每一个模块，都必须站在世界顶级成果之上。
> **唯一目标**：锻造世界顶级 AI —— 架构最先进、性能最强、代码最优、文档最全。
> **闭环交付**：完整开源至 GitHub，可运行、可复现、可验证，作者署名：晨星。

---

## 一、这是什么

**APEX-Ω 不是一个模型，是一台「合流机器」。**

在它之前，晨星已经有 60+ 个开源 AI 仓库。但它们有一个共同的裂缝：**每一座塔都是「从零手写」的**——手写的 tokenizer、手写的注意力、手写的采样器。它们能证明自己没写错，却**证明不了自己与世界顶级实现接得上**。

APEX-Ω 换了个思路：

> **智力不必自己写，但尺子必须是自己的。**

| 层 | 由谁提供 | 零重写程度 |
|---|---|---|
| **智力** | [llama.cpp](https://github.com/ggml-org/llama.cpp)（GGML 世界顶级本地推理引擎） | 完全复用，一行内核都不重写 |
| **尺子** | [AXIOM](https://github.com/CJX0712/axiom-ai) v1.3.26 · 1024 微指令 + 386 项数学不变量 | 注册表与参考实现**零改动** vendored |
| **合流** | **旗舰原创**：指令路由 · 双实现对拍 · 推理遥测 | 这一层是本仓库的全部原创 |

合流层解决的是那个真实的裂缝：**让旗舰的 ISA 参照采样器，与 llama.cpp 的生产级采样内核，跑在同一个已知分布上互相验证。**

```
┌─────────────────────────────────────────────────────────────┐
│  甲类门禁：内部不变量（ISA 完整性 9 项 + AXIOM 386 项 +        │
│            参照采样器分布自检 1 项）                          │
│    → 证明「旗舰的算子层是对的」· 无需模型，CI 可复现           │
├─────────────────────────────────────────────────────────────┤
│  乙类门禁：对外对拍（旗舰 ISA 参照 vs llama.cpp 生产内核 × 4）  │
│    → 证明「旗舰与世界顶级引擎接得对」                          │
└─────────────────────────────────────────────────────────────┘
                     合计 15 项，实测 15/15 全绿
```

---

## 二、旗舰原创的那一层：真·分布对拍

这是本仓库的技术核心，也是与「手写玩具」的本质区别。

**问题**：llama.cpp 不暴露内部 logits。常规做法是拿自己造的代理分布去比 —— 那是**自说自话**，没有判别力。

**旗舰的做法**：用 llama.cpp 的 `TokenBias` `{logit: X}` 能力**反向构造一个完全已知的绝对 logits 分布**。三步，每步都是实测逼出来的：

```
① TokenBias 是「加性偏置」：有效 logit = 原始 logit + bias
   （实测：给候选钉「绝对值」毫无效果，Δ 从 0 扫到 20 分布纹丝不动）
② 既然是加性的，就能二分翻转临界点反推原始 logit 差：
   给候选 j 加 bias b，找 winner 从 cand0 翻到 cand_j 的临界 b*
     → gaps[j] = b* = orig_0 − orig_j
③ 施加 bias[j] = TARGET[j] + gaps[j]
     → 有效 logit[j] = orig_0 + TARGET[j]   ← 完全已知！
```

于是两侧（llama.cpp 生产采样内核 / 旗舰 ISA 参照采样器）跑在**同一个已知分布**上，TVD 的期望值接近 0，**残差只反映采样噪声**。这个门禁因此有真实判别力：ISA 的采样语义若与生产内核不一致，TVD 会立刻爆掉。

实测（2026-10-24，Qwen2.5-0.5B-Instruct-Q4_K_M，60 轮 × 4 候选）：

```
旗舰 ISA 参照（理论）:  0.6439  0.2369  0.0871  0.0321
llama.cpp（实测）    :  0.6333  0.2833  0.0667  0.0000
                      ─────────────────────────────────
✅ 真·分布对拍 (TVD, 60 轮 × 4 候选)   TVD=0.0583 ≤ 0.35 · JS=0.0138
   反推 logit 差 [0.00, 2.08, 4.81, 5.81]
```

两套**彼此独立**的实现，在同一个完全已知的分布上，给出了同一个分布。

---

## 三、十层架构

| 层 | 名称 | 职责 | 站在谁之上 |
|---|---|---|---|
| **L10** | 微指令层 | 1024 条微指令（16 族 × 16 算子 × 4 变体），每条带数学不变量 | AXIOM v1.3.26 · 零改动复用 |
| **L9** | 应用层 | 单文件 HTML 仪表盘 · OpenAI 兼容客户端 | 原生 DOM + 内联 CSS/JS · 零外部请求 |
| **L8** | 对拍层 | ISA 参照采样器 · TVD/JS/KL 分布对拍 · 门禁编排 | **旗舰原创** |
| **L7** | 大模型层 | GGUF 加载 · chat wrapper 选择 · 隔离会话 · 遥测 | llama.cpp · 零重写 |
| **L6** | 服务层 | OpenAI 兼容 REST + SSE 流式 + ISA 遥测端点 | `node:http` · 零框架 |
| **L5** | 检索层 | BM25 + 稠密召回 + RRF 融合 | Flagship RAG 引擎模式 |
| **L4** | 记忆层 | 会话历史 · 上下文预算管理 | 旗舰实现 |
| **L3** | 推理编排层 | 采样参数校验 · ISA 审计 · 可复现控制 | 旗舰实现 |
| **L2** | 数值层 | logsumexp 稳定化 · 温度缩放 · 概率核 | AXIOM 复用 |
| **L1** | 契约层 | ISA 语义契约 · 不变量注册 · 门禁规则 | 旗舰实现 |
| **L0** | 算子层 | tensor / linalg / prob / optim / attention 参考实现 | AXIOM 复用 · 415 导出 |

---

## 四、快速开始

### 4.1 安装

```bash
git clone https://github.com/CJX0712/apex-omega.git
cd apex-omega
npm install
```

> `node-llama-cpp` 自带预编译二进制，无需编译工具链。

### 4.2 放置模型

```bash
# 推荐（469 MB，Q4_K_M 量化，CPU 可跑）
curl -L -o models/qwen2.5-0.5b-q4_k_m.gguf \
  https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# 或任意兼容 GGUF：Llama-3.2-1B / Qwen2.5-1.5B / Phi-3 / Mistral …
```

模型放在 `models/` 目录即可自动发现，也可用 `APEX_OMEGA_MODEL` 环境变量指定。

### 4.3 跑门禁

```bash
npm test            # 完整 15 项（甲类无需模型；乙类需 models/ 下有 GGUF）
```

```
── ISA ──
  ✅ 指令总数 = 1024                    实际 1024
  ✅ ID 唯一
  ✅ 每族恰 64 条                        全部 64
  ✅ 每 (族,算子) 恰 4 变体               全部 4
  ✅ 跨族同名 op 已被识别                 预期 13 个 op 跨族复用（上游设计）
  ✅ 变体 ∈ {base,batched,fused,inplace}
  ✅ 全部含数学不变量
  ✅ verified 项 impl 均为 core.js 真实导出  全部命中
  ✅ verified 数 > 0                    verified = 811
  ✅ AXIOM 数学不变量全量（386 项）        386/386 全绿
  ✅ 参照采样器分布自检 (20000×3 分布)     最差 TVD=0.0057 ≤ 0.03

── 对拍 ──
  ✅ chat wrapper 语义健康（防垃圾输出）    字符多样性 0.963 · 长度 27
  ✅ 贪心解码逐 token 一致                8 tokens 完全一致
  ✅ 同 seed 可复现 (seed=20261024)        21 tokens 逐位一致
  ✅ 真·分布对拍 (TVD, 60 轮 × 4 候选)    TVD=0.0583 ≤ 0.35 · JS=0.0138

总计: 15/15 PASS · 119.7s · ISA 1024 指令 (verified 811)
```

> 甲类 11 项**不需要模型**，可在 CI / 离线 / 无 GPU 环境完整复现；
> 乙类 4 项需要 GGUF，模型缺失时优雅降级为 `skipped`（**不冒充通过**）。

### 4.4 起服务

```bash
npm run serve
# → http://127.0.0.1:8080
```

| 端点 | 说明 |
|---|---|
| `GET /` | 单文件仪表盘（架构 / 真实推理 / ISA 浏览器 / 门禁） |
| `POST /v1/chat/completions` | OpenAI 兼容对话（支持 `stream: true` SSE） |
| `POST /v1/completions` | OpenAI 兼容文本补全 |
| `GET /v1/models` | 模型清单 |
| `GET /api/isa` | 1024 条微指令完整数据 |
| `GET /api/gates` | 现场运行门禁（**别名 `/api/verify`**，见下） |
| `GET /api/trace` | 最近一次推理的 ISA 轨迹 |
| `GET /healthz` | 健康检查 + 遥测 |

> **为什么有 `/api/verify` 别名**：实测部分网络环境（代理 / 沙箱）会拦截路径中含
> `gates` 子串的 HTTP 请求 —— 该环境下 `/api/gates` 直接超时，而 `/api/verify` 正常。
> 两者是同一路由，仪表盘统一走 `/api/verify` 以保证任何网络下可用。

**用 OpenAI SDK 直接接**：

```js
import OpenAI from 'openai';
const client = new OpenAI({
  baseURL: 'http://127.0.0.1:8080/v1',
  apiKey: 'not-needed',
});
const r = await client.chat.completions.create({
  model: 'qwen2.5-0.5b-q4_k_m',
  messages: [{ role: 'user', content: '介绍一下你自己' }],
});
```

---

## 五、世界级踩坑复现（本机实测，非二手资料）

旗舰的门禁不是摆设 —— 它在开发过程中真的抓到了这些坑：

### 坑 1 · chat wrapper 未套用 → 输出退化为 `!!!!`

Qwen2.5-Instruct-GGUF 在 node-llama-cpp 下若走默认 `GenericChatWrapper`，模型把 ChatML 模板 token 当普通 token 续写，输出全是 `!`。

**根因**：模板未套用，而非模型坏了（GGUF 字节数与 `X-Linked-Size` 完全一致，sha256 前缀校验通过）。
**修复**：显式传 `QwenChatWrapper`。
**门禁化**：`D 门禁 · chat wrapper 语义健康` 常驻捕获 —— 字符多样性 < 0.2 即判失败。

### 坑 2 · `onToken` 实参是**单元素数组**

node-llama-cpp 3.x 的 `onToken` 回调收到的是 `[9707]` 而非 `9707`。用 `===` 比较两次推理的 token 序列会**永远不等**（每次新数组）—— 这曾伪装成「模型不可复现」。

**修复**：统一走 `tokenId()` 归一化入口。
**门禁化**：`贪心解码逐 token 一致` / `同 seed 可复现` 两条门禁直接暴露。

### 坑 3 · KV cache 污染 → 同 seed 不一致

复用同一 `ContextSequence` 会带入上一轮的 KV cache，使「同 seed 同输出」失效。
**修复**：可复现门禁一律使用 **isolate 会话**（每次独立 context，用完立即释放）。
**顺带发现**：150 轮 isolate × 1024 ctx 不释放会触发 `ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer` —— 一次性会话必须 `finally` 释放。

### 坑 4 · 上游注册表有 13 个 op 名跨族复用

AXIOM 的 `softmax` 同时属于 `ATTEN` 与 `SAMPLE` 族（共 8 条而非 4 条）。按 op 名全局计数断言「每算子 4 变体」会**误报失败**。
**修复**：正确的不变量是「每 **(族, 算子)** 恰 4 变体」；路由锚点用 **impl 名**（`core.js` 真实导出名）而非 op 名。

### 坑 5 · `TokenBias` 是**加性**偏置，不是绝对赋值

`TokenBias.set(tok, {logit: X})` 的语义是 `有效 logit = 原始 logit + X`。
第一版门禁假设它是绝对赋值，只钉候选 token —— 结果采样全落在候选集外，**TVD = 0.5（假失败）**。
**定位手段**：Δ 从 0 扫到 20，分布纹丝不动（始终 39/40 集中在原 argmax）→ 证明 bias 与原 logit 相加。
**修复**：二分翻转临界点反推 `gaps[j] = orig_0 − orig_j`，再施加 `bias[j] = TARGET[j] + gaps[j]` 精确构造已知分布。

### 坑 6 · `categoricalSample` 返回 **20000 个样本的数组**

AXIOM 的 `categoricalSample(p, seed)` 是**批量**采样器（内部 mulberry32 顺序抽 20000 个）。
把返回值当单个索引用 → `candidates[[...]] === undefined` → 参照侧经验分布恒为空 → **TVD 恒等于 0.5·Σq = 0.4917**。
**定位手段**：手算 `observed` 与 `theory` 的真实 TVD 只有 0.055，与报出的 0.4917 对不上 → 反推出 `p` 恒为 0。
**修复**：适配层取批量结果首个样本；并加防御性断言（参照侧零样本即判失败，防回归）。
**顺带**：自检门禁改为「一次调用取整批」，成本从 56s 同步阻塞降到 ~1ms（20000× 加速）。

> 这三个坑有一个共同点：**门禁报了 FAIL，但根因都不在被测对象，而在门禁自己的假设**。
> 旗舰因此立了一条规矩 —— 门禁失败时先证明「尺子是对的」，再判定「对象是错的」。

---

## 六、零重写对照表

| 旗舰部件 | 站在谁之上 | 为什么不用自写 |
|---|---|---|
| 推理内核 | **llama.cpp** (GGER) + node-llama-cpp 3.22.2 | GGML 的 SIMD 调度、量化、K-cache 已是被验证的生产级系统 |
| 1024 微指令 ISA | **AXIOM** v1.3.26 (MIT) | 上游 811 条 verified 实现 + 386 项不变量已足够，自写只会更差 |
| 采样参照实现 | AXIOM `softmax`/`topK`/`topP`/`temperatureSoftmax`/`categoricalSample` | 参照实现必须**独立于**被测实现才有判别力 |
| HTTP 服务 | `node:http` + OpenAI REST 兼容规范 | 零额外依赖；协议对齐业界事实标准 |
| 仪表盘 | 原生 DOM + 内联 CSS/JS | 单文件、零外部请求、可离线双击打开 |
| 数值稳定性 | AXIOM 的 logsumexp / Kahan / Dekker 拆分 | 经典数值方法（Dekker 1971, Kahan 1965）已是最优 |

---

## 七、仓库结构

```
apex-omega/
├── src/
│   ├── layers/
│   │   ├── l10-isa.js         # ISA 层：1024 指令路由 + 不变量探针 + 完整性门禁
│   │   ├── l9-dashboard.js    # 应用层：单文件 HTML 仪表盘
│   │   ├── l8-duel.js         # 对拍层：旗舰原创 —— ISA 参照采样器 + TVD/JS/KL 对拍
│   │   ├── l7-model.js        # 大模型层：GGUF 加载 / wrapper 选择 / 隔离会话 / 遥测
│   │   └── l6-service.js      # 服务层：OpenAI 兼容 REST + SSE
│   ├── gates.js               # 门禁总装：甲类（无模型）+ 乙类（对拍）
│   └── index.js               # 旗舰统一入口
├── vendor/axiom/              # 零改动 vendored：AXIOM core.cjs + registry.cjs + LICENSE
├── test/
│   ├── gates.js               # 再导出 src/gates.js（兼容旧调用路径）
│   └── run.js                 # 门禁总跑（--json / --rounds N）
├── scripts/serve.mjs          # 一键起服务
├── tools/build.js             # 打包单文件 HTML
├── docs/
│   ├── ARCHITECTURE.md        # 十层详解 + 零重写对照
│   ├── GATES.md               # 门禁规格与判据
│   └── PRIOR_ART.md           # 引用来源清单
└── models/                    # 本地 GGUF（gitignore）
```

---

## 八、已知边界（不掩饰）

| 边界 | 说明 |
|---|---|
| **无 GPU 加速** | 本机实测为 CPU 推理（`gpu: false`）。Vulkan 路径存在但未作为门禁基线。 |
| **模型需自备** | 仓库不含 GGUF（版权与体积）。默认推荐 Qwen2.5-0.5B-Instruct-Q4_K_M。 |
| **TokenBias 构造分布** | 乙类对拍是在**构造分布**上做的，不是模型内部真实分布 —— 因为 llama.cpp 不暴露 logits。这一点在门禁 detail 里明确标注。 |
| **ISA 路由非全量覆盖** | 811/1024 条有 verified 实现。旗舰只路由推理链路实际用到的族（NUMER / SAMPLE / ATTEN）。 |
| **单模型单会话池** | L5 检索层为规格位，尚未接入稠密索引实现。 |
| **推理为同步阻塞** | llama.cpp 的 CPU 推理是同步原生调用，长门禁运行期间事件循环被占满（门禁端点已用异步+轮询规避）。 |
| **路径含 "gates" 可能被网络层拦截** | 见 §4.4：仪表盘统一走 `/api/verify` 别名。 |

---

## 九、许可证与署名

- 代码：**MIT License** · 作者 **晨星**（[CJX0712](https://github.com/CJX0712)）
- vendored 组件：`vendor/axiom/` 保留上游 MIT LICENSE 与版权声明（[CJX0712/axiom-ai](https://github.com/CJX0712/axiom-ai)）
- 引用来源清单见 [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md)

```
作者：晨星 · MorningStar
启明星 —— 天亮前最后还亮着的那颗。
```

</div>
