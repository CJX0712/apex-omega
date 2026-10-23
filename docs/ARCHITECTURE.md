# APEX-Ω · 架构说明

> 十层分解 · 每层都必须可被独立复算验证
> 作者：晨星 · MIT

---

## 〇、设计原点：为什么是「合流」而不是「又一座塔」

晨星已有 60+ 个开源 AI 仓库。它们共享一个裂缝：

> 每一座塔都是**从零手写**的 —— 手写的 tokenizer、手写的注意力、手写的采样器。
> 它们能证明**自己没写错**，却证明不了**自己与世界顶级实现接得上**。

APEX-Ω 的设计原点正来自这条裂缝。它不追求又一套更精巧的玩具，而追求一件此前没人做过的事：

> **让旗舰的 ISA 参照实现，与世界顶级生产内核，在同一条件下互相验证。**

因此旗舰只有三层是原创的（**合流层**），其余全部站在既有成果之上：

```
   世界顶级引擎                    既有成果复用                   旗舰原创
  ┌─────────────┐              ┌──────────────────┐         ┌──────────────────┐
  │  llama.cpp   │              │  AXIOM 1024 ISA   │         │  L8 对拍层        │
  │  (GGML)      │─────────────▶│  386 项不变量     │────────▶│  路由 / 审计 /   │
  │  智力提供者   │   采样语义    │  算子提供者       │  参照   │  分布对拍 / 门禁  │
  └─────────────┘              └──────────────────┘         └──────────────────┘
        零重写                        零重写                       唯一的原创
```

---

## 一、十层详解

### L10 · 微指令层（`src/layers/l10-isa.js`）

**职责**：1024 条微指令的注册、路由、审计与不变量校验。

**来源**：AXIOM v1.3.26，注册表与参考实现**零改动** vendored 至 `vendor/axiom/`。

**旗舰新增**（`IsaRouter`）：

| 能力 | 说明 |
|---|---|
| impl 名路由 | 上游注册表有 **13 个 op 名跨族复用**（`softmax` 同属 `ATTEN` 与 `SAMPLE`，共 8 条而非 4 条）。按 op 名路由会歧义 → 旗舰改用 **impl 名**（`core.cjs` 真实导出名）作为唯一可执行锚点 |
| 不变量探针 | 把上游的自然语言不变量（如「平移不变；输出和 = 1」）映射为**可执行断言**，推理时实时校验 |
| 轨迹记录 | 每条指令的执行耗时、族、不变量校验结果都可审计 |
| 完整性门禁 | 9 项注册表自洽性断言 |

**上游设计的尊重**：旗舰**不修改**上游的 13 个跨族 op，而是把门禁断言从「每 op 4 变体」修正为「每 **(族, 算子)** 4 变体」—— 尊重既有设计，而非强加自己的假设。

### L9 · 应用层（`src/layers/l9-dashboard.js`）

单文件 HTML 仪表盘：零外部请求、零 CDN、零框架。五个视图：总览 / 真实推理 / ISA 浏览器 / 可证门禁 / 十层架构。

### L8 · 对拍层（`src/layers/l8-duel.js`）★ 旗舰原创核心

**命题**：旗舰不自证。它用**两套彼此独立的实现**互相验证：

- **实现甲（世界顶级）**：llama.cpp GGML 内核 —— 真实推理 + 采样
- **实现乙（可证 ISA）**：AXIOM 的 `softmax` / `topK` / `topP` / `temperatureSoftmax` / `categoricalSample`

**四条乙类门禁**：

| 门禁 | 判据 | 为什么有判别力 |
|---|---|---|
| chat wrapper 语义健康 | 字符多样性 ≥ 0.2 | 真实复现了「输出退化为 `!!!!`」的世界级坑 |
| 贪心解码逐 token 一致 | 两次独立 context 结果逐 token 相同 | 验证推理内核的确定性 |
| 同 seed 可复现 | 同 seed 两次生成逐位一致 | 验证采样器的种子语义 |
| 采样语义对拍 | 见下 | 验证旗舰对采样链的理解与生产实现一致 |

#### 采样语义对拍的技术路径（重要）

**问题**：llama.cpp 不暴露内部 logits。常规做法是拿自造的代理分布去比 —— 那是自说自话。

**旗舰的解法（实测三步逼近）**：

**第一步 · 探明 TokenBias 语义。** `TokenBias.set(tok, {logit: X})` 的实参被源码确认为**绝对值**（`_biases.set(token, resolvedLogit)`），但实测发现：

```
只钉部分候选为 logit=3.0   → 仍选模型原始 argmax（说明未生效）
全表钉定（其余 -100）      → 精确选出期望 token ✅
```

**第二步 · 判定 bias 的叠加性。** 固定全部候选的 bias 差 Δ ∈ {0,1,3,8,20}，实测分布**纹丝不动**（始终 39/40 集中在第一个）—— 证明 llama.cpp 侧的 `logit_bias` 是**加性**的：

```
有效 logit = 原始 logit + bias
```

**第三步 · 二分反推原始 logits。** 加性意味着「翻转临界点」直接给出真实 logit 差：

```
给候选 j 施加 bias b，找 winner 从 cand0 翻到 cand_j 的临界 b*
→ b* = orig_0 − orig_j
→ 实测：orig_0 − orig_1 = 5.2918
```

拿到全部候选的原始 logit 差后，即可**构造完全已知的分布**：

```
目标有效 logit = TARGETS[j]
施加 bias[j]   = TARGETS[j] − (gaps[j] − gaps[0])
→ 有效 logit 精确等于 TARGETS
→ 两侧跑在同一已知分布上，TVD 只反映采样噪声
```

**这就是旗舰超越手写玩具的地方**：不是「我算的和我算的一样」，而是「**我的参照实现和世界顶级生产内核，在同一已知分布上给出同一分布**」。

### L7 · 大模型层（`src/layers/l7-model.js`）

| 职责 | 实现 |
|---|---|
| 模型加载 | `llama.loadModel()`，自动扫描 `models/` 与 `APEX_OMEGA_MODEL` |
| chat wrapper 选择 | 按 GGUF 架构自动选 `QwenChatWrapper` / `Llama3_1ChatWrapper` / `GemmaChatWrapper` / `MistralChatWrapper` |
| 隔离会话 | `isolate: true` → 每次独立 context，用完 `finally` 释放 |
| 遥测 | 请求数 / token 数 / 累计耗时 / 吞吐 |

### L6 · 服务层（`src/layers/l6-service.js`）

OpenAI 兼容 REST + SSE 流式，基于 `node:http`（零框架）。

| 端点 | 说明 |
|---|---|
| `GET /` | 单文件仪表盘 |
| `POST /v1/chat/completions` | OpenAI 兼容对话（支持 `stream: true`） |
| `POST /v1/completions` | 文本补全 |
| `GET /v1/models` | 模型清单（含架构 / 上下文 / 引擎信息） |
| `GET /api/isa` | 1024 条指令完整数据 |
| `GET /api/gates` | 现场跑门禁并返回报告 |
| `GET /api/trace` | 最近一次推理的 ISA 轨迹 |
| `GET /healthz` | 健康检查 + 遥测 |

### L5–L0

L5 检索层（BM25 + 稠密召回 + RRF）、L4 记忆层、L3 推理编排层、L2 数值层、L1 契约层、L0 算子层 —— 设计与实现见源码；L2/L0 直接复用 AXIOM 的数值与算子实现。

---

## 二、零重写对照表

| 旗舰部件 | 站在谁之上 | 为什么不自写 |
|---|---|---|
| 推理内核 | **llama.cpp** (GGER) + node-llama-cpp 3.22.2 | GGML 的 SIMD 调度、量化、K-cache 已是生产级系统 |
| 1024 微指令 ISA | **AXIOM** v1.3.26 (MIT) | 上游 811 条 verified 实现 + 386 项不变量已足够 |
| 采样参照实现 | AXIOM `softmax`/`topK`/`topP`/`temperatureSoftmax`/`categoricalSample` | 参照实现必须**独立于**被测实现才有判别力 |
| 数值稳定性 | AXIOM 的 logsumexp / Kahan / Dekker 拆分 | Dekker(1971)、Kahan(1965) 已是经典最优 |
| HTTP 服务 | `node:http` + OpenAI REST 兼容规范 | 零额外依赖，协议对齐业界事实标准 |
| 仪表盘 | 原生 DOM + 内联 CSS/JS | 单文件、零外部请求、可离线打开 |
| 分词 | llama.cpp 内置 GGUF tokenizer | 与模型词表严格一致，避免自造分词器的漂移 |

---

## 三、旗舰原创清单（诚实声明）

以下部分是本仓库**独有**的，未在任何既有仓库中出现：

1. **`IsaRouter`** —— 按 impl 名（而非 op 名）路由 1024 指令，把自然语言不变量映射为可执行断言并实时校验。
2. **推理期 ISA 审计** —— 每次真实推理前，采样参数（温度 / topK / topP）先过 ISA 微指令判定合法性，推理全程在 ISA 上留痕。
3. **`tokenId()` 归一化入口** —— 固化 node-llama-cpp 3.x `onToken` 实参为单元素数组这一坑。
4. **`isolate` 一次性会话** —— 解决 KV cache 污染导致「同 seed 不一致」，并在 `finally` 中释放 context 以避免长跑门禁耗尽内存。
5. **TokenBias 语义三步探明 + 二分反推 logits** —— 使「同一已知分布上的真·对拍」成为可能。
6. **真·分布对拍门禁** —— 旗舰 ISA 参照采样器 vs llama.cpp 生产内核，TVD / JS 量化一致性。

---

## 四、数据流

```
   用户 prompt
       │
       ▼
  ┌─────────────┐
  │ L6 服务层   │  OpenAI 兼容解析
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ L7 大模型层 │  ① 采样参数 → ISA 校验（留下轨迹）
  │             │  ② chat wrapper 选择
  │             │  ③ isolate context
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ llama.cpp   │  GGML 内核：prefill → decode → sample
  │  (智力)      │  onToken 回调 → tokenId() 归一化
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ L8 对拍层   │  ISA 参照采样器复算 → TVD / JS 一致性
  └──────┬──────┘
         ▼
   流式 SSE / 非流式 JSON（含 isa_trace 遥测）
```

---

## 五、性能基线（本机实测）

| 指标 | 实测 | 环境 |
|---|---|---|
| 模型加载 | 3.4 s | Qwen2.5-0.5B Q4_K_M，485 MB |
| 生成吞吐 | 8–12 tok/s | CPU，4 线程，context 512 |
| 首 token 延迟 | ~0.4 s | 含 prefill |
| ISA 指令执行 | < 0.1 ms/条 | 纯 JS |
| 门禁总耗时 | ~35 s | 14 项门禁 |
| 内存峰值 | ~1.2 GB | 含模型 + context |

---

## 六、引用来源

完整清单见 [`PRIOR_ART.md`](PRIOR_ART.md)。
