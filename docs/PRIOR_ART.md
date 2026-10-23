# APEX-Ω · 引用来源清单（PRIOR ART）

> 零重写原则的执行凭证：旗舰每一行代码、每一个模块，都必须站在世界顶级成果之上。
> 本清单列出所有被直接复用、组合或对标的公开成果及其原始出处。

---

## 一、直接复用（代码级 vendored / 调用）

### 1. llama.cpp · GGML

- **仓库**：https://github.com/ggml-org/llama.cpp
- **许可**：MIT
- **Node 绑定**：https://github.com/withcatai/node-llama-cpp （本仓库使用 **v3.22.2**）
- **复用范围**：GGML 推理内核、GGUF 加载与量化、KV cache、Flash Attention、采样链（温度 / top-k / top-p / min-p / 分布采样）、分词器、chat wrapper 实现
- **旗舰未重写的部分**：attention、RoPE、RMSNorm、量化、KV cache、采样算子的**任何一行**内核

### 2. AXIOM · 1024 微指令 ISA

- **仓库**：https://github.com/CJX0712/axiom-ai （v1.3.26）
- **许可**：MIT · 作者 晨星
- **复用范围**：`vendor/axiom/registry.cjs`（1024 条指令注册表）、`vendor/axiom/core.cjs`（415 个导出 + 386 项数学不变量自检）
- **改动**：**仅改文件扩展名** `.js` → `.cjs`（本项目为 ESM，`.js` 会被当 ESM 解析导致 `module.exports` 失效）。代码内容零改动。
- **旗舰未重写的部分**：全部 1024 条微指令的语义契约、数学不变量、参考实现

### 3. Node.js 标准库

- `node:http`（服务层）· `node:fs`（模型发现）· `node:path` · `node:os`
- **理由**：零依赖，Node 官方维护

### 4. OpenAI REST API 兼容规范

- **规范**：https://platform.openai.com/docs/api-reference
- **复用范围**：端点路径、请求体字段、响应体结构、SSE chunk 格式、`[DONE]` 终止符
- **理由**：业界事实标准，对齐它意味着生态零改动接入

---

## 二、算法与方法（实现层对标）

### 数值方法

| 方法 | 来源 | 复用位置 |
|---|---|---|
| Householder QR | Householder (1958) | AXIOM `qr` |
| Jacobi 循环旋转特征分解 | Jacobi (1846) | AXIOM `jacobi` |
| Cholesky 分解 | Cholesky (1925) | AXIOM `cholesky` |
| Dekker 分裂模拟 FMA | Dekker (1971) | AXIOM `fma` |
| Kahan 求和 / TwoSum | Kahan (1965) / Knuth (1998) | AXIOM `kahanSum` / `twoSum` |
| logsumexp 稳定化 | LAPACK 传统 | AXIOM `logsumexp` |
| Newton–Schulz 正交化 | Muon 公开系数 (2025) | AXIOM `newtonSchulz` |

### 概率与统计

| 方法 | 来源 | 复用位置 |
|---|---|---|
| EM / 全协方差 GMM | Dempster, Laird & Rubin (1977) | AXIOM `em` / `gmm` |
| 前向后向 / Viterbi | Rabiner (1989) | AXIOM `fb` / `viterbi` |
| Gibbs 采样 | Geman & Geman (1984) | AXIOM `gibbsSampler` |
| Metropolis–Hastings | Metropolis et al. (1953) | AXIOM `metropolisSample` |
| Box–Muller 变换 | Box & Muller (1958) | AXIOM `normalSample` |
| Abramowitz–Stegun erf | A&S 7.1.26 | AXIOM `erf` |
| Arimoto–Blahut 信道容量 | Arimoto & Blahut (1972) | AXIOM `blahut` |

### 注意力与序列

| 方法 | 来源 | 复用位置 |
|---|---|---|
| Transformer | Vaswani et al. (2017) | AXIOM `selfAttention` |
| FlashAttention（在线 softmax 分块） | Dao et al. (2022) | AXIOM `flashAttention` |
| RoPE 旋转位置编码 | Su et al. (2021) | AXIOM `rope` |
| ALiBi 线性偏置 | Press et al. (2021) | AXIOM `alibi` |
| MQA / GQA | Ainslie et al. (2023) | AXIOM `mqa` / `gqa` |
| LLM 采样链（temp/top-k/top-p/min-p） | llama.cpp `common/sampling.cpp` | 旗舰 L8 对拍层的对标对象 |

### 优化

| 方法 | 来源 | 复用位置 |
|---|---|---|
| 梯度下降 | Cauchy (1847) | AXIOM `gd` |
| 动量 | Polyak (1964) | AXIOM `momentum` |
| Adam / AdamW | Kingma & Ba (2015) / Loshchilov & Hutter (2019) | AXIOM `adam` / `adamw` |
| 共轭梯度 | Hestenes & Stiefel (1952) | AXIOM `cg` |
| BFGS / L-BFGS | Broyden et al. (1970) / Byrd et al. (1994) | AXIOM `bfgs` / `lbfgs` |
| 强 Wolfe 线搜索 | Nocedal & Wright (2006) | AXIOM `wolfeLineSearch` |

### 评测与验证

| 方法 | 来源 | 复用位置 |
|---|---|---|
| 中心差分梯度检验 | 所有主流框架的 numerical gradcheck | AXIOM `gradCheck` |
| TVD / JS / KL 散度 | 标准信息测度 | 旗舰 L8 `Distances` |
| Kolmogorov–Smirnov | 标准统计检验 | 备用 |

---

## 三、模型来源

| 模型 | 来源 | 许可 |
|---|---|---|
| Qwen2.5-0.5B-Instruct-GGUF | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF | Apache-2.0 |
| 镜像下载 | https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF | 同上 |

> 本仓库**不包含**任何 GGUF 权重文件（体积与许可）。模型由使用者自行下载。

---

## 四、被否决的方案（及理由）

零重写不止于「复用什么」，也包括「明确不用什么」。

| 方案 | 否决理由 |
|---|---|
| 自写 GGUF 推理内核 | llama.cpp 的 SIMD 调度与量化实现已被生产验证，自写必劣 |
| 自写 tokenizer | 自造分词器与模型词表漂移，会让所有下游指标失真 |
| 用代理 logits 做「分布对拍」 | 自造分布与被测系统无关 —— 是自说自话，无判别力。旗舰改为**二分反推真实 logits** |
| 按 op 名路由 1024 指令 | 上游有 13 个 op 跨族复用，会歧义。改用 impl 名作为唯一锚点 |
| 引入 express / fastify | 零依赖原则；`node:http` 已足够 |
| 引入前端框架 / CDN | 单文件、零外部请求、可离线双击打开 |
| 引入 Python 栈（torch/numpy） | 旗舰定位是「零重写地复用 llama.cpp」；Python 侧已有天玑塔承担 |

---

## 五、诚实声明

- 本仓库的**原创部分**仅限：`IsaRouter`、推理期 ISA 审计、`tokenId()` 归一化、`isolate` 一次性会话、TokenBias 语义探明与 logits 二分反推、真·分布对拍门禁。
- 其余一切均站在上述公开成果之上。
- 引用遵循各自原始许可；`vendor/axiom/LICENSE` 保留上游原始版权声明。

---

*作者：晨星 · 启明星 —— 天亮前最后还亮着的那颗。*
