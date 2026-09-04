### 1) DeepSeek-V3 → DeepSeek-V4 —— "压缩 KV + 稀疏 MoE"这条路的自我迭代
- **DeepSeek-V3**（671B 总参 / 37B 激活，128K 级上下文）：三个标志性结构——
  ① **MLA（Multi-head Latent Attention）**：把 K/V 压缩进低秩潜变量（`kv_lora_rank=512`、`q_lora_rank=1536`），推理只缓存压缩后的 latent，KV 缓存与相关计算量大幅下降——这是"**省注意力靠压缩 KV**"的开源代表；
  ② **无辅助损失的均衡 MoE**（`topk_method=noaux_tc` + 分组 top-k）：256 个路由专家 top-8 + 1 个共享专家，去掉路由负载均衡的辅助 loss，训练更稳更省；
  ③ **MTP（多 token 预测）层**：不止预测下一个 token，作为训练信号与加速手段。
- **DeepSeek-V4**（Flash 284B/13B、Pro 1.6T/49B，均 1M 上下文）：把 V3 的"压缩"推到更极致——
  ① **混合注意力 = CSA(压缩稀疏) + HCA(重度压缩)**：config 里单 KV 头、hash 索引层(`num_hash_layers`)选 top-K KV、`sliding_window=128` 滑窗并存——官方称 1M 语境下单 token FLOPs 约为 V3.2 的 **27%**、KV 缓存约 **10%**；
  ② **mHC（流形约束超连接）**改造残差、**Muon 优化器**、专家 FP4 量化、域专家先 GRPO 培养再 on-policy 蒸馏合并。

### 2) GLM-5.2（智谱）—— "稀疏注意力 + 压缩 KV"，主打扎实的 1M 上下文
- 目标长程任务；**MIT 开源**，宣传"solid 1M context"。
- 结构：沿用**压缩 KV**（config 有 `kv_lora_rank/q_lora_rank/qk_nope_head_dim/qk_rope_head_dim` 这套 MLA 式字段）；新增 **IndexShare 稀疏注意力**——每 4 个稀疏注意力层**共享同一个 indexer** 选出 top-K KV（`index_topk=2048`、`index_topk_freq=4`），官方称 1M 上下文下 per-token FLOPs 降约 **2.9×**（对应论文 `indexshare_paper.pdf`）。
- 其余：MoE 256 路由 top-8 + 1 共享；MTP 层兼做投机解码、接受长度最多提升 ~20%；RoPE 取大 base 且 `rope_interleave`。

### 3) Qwen3.6（35B-A3B 为主）—— "线性注意力 × softmax" 的周期混合
- 35B 总参 / **3B 激活**，40 层，原生多模态（带视觉 encoder）。
- 结构主干 `10 × (3×(Gated DeltaNet → MoE) → 1×(Gated Attention → MoE))`：即每 4 个单元里，**3 个用 Gated DeltaNet（线性/状态空间式注意力，带卷积核），1 个用 Gated(softmax) Attention**——注意力的"大部分层线性、间隔补全局"混合，和 Jamba/MiniMax 是同一哲学的中国版。
- MoE 256 专家 top-8 + 1 共享；MTP；上下文 262K 原生、可扩到 ~1M；位置用 **M-RoPE**（多媒体分段 + 交错，θ=1e7）。

### 4) Kimi-K3（月之暗面）—— "压缩/线性注意力 + 超大规模 MoE"，首个开源 3T 级
- 开放权重的原生多模态 agentic 模型，**2.8T 参数级**、1M 上下文（官方称首个开源 3T 级）。
- 结构：**KDA（Kimi Delta Attention）+ AttnRes（Attention Residuals）**——注意层既带压缩 KV（MLA 式 `kv_lora_rank=512`）又有线性注意力头；MoE 用 **Stable LatentMoE**，**896 专家中激活 16**（另 2 共享），相对 K2 扩展效率约 2.5×；原生 **MXFP4** 4-bit 量化、`SiTU` 激活。

### 一句话对照
| | DeepSeek V3→V4 | GLM-5.2 | Qwen3.6 | Kimi-K3 |
|---|---|---|---|---|
| 序列建模怎么省 | MLA 压缩 → V4 加 CSA+HCA 稀疏/超压缩混合 | 压缩 KV + IndexShare 稀疏 | Gated DeltaNet(线性) × Gated Attention 周期混合 | KDA = 压缩 KV + 线性注意力头 + AttnRes |
| MoE 怎么做 | 256/384 路由 top6–8 + 共享，无辅助 loss | 256 路由 top8 + 共享 | 256 专家 top8 + 共享 | 896 专家激活 16(+2 共享) LatentMoE |
| 长度/位置 | 1M；滑窗+hash 索引+RoPE | 1M；indexer 每 4 层共享 | 262K→~1M；M-RoPE(多媒体) | 1M |
| 招牌机制 | noaux-MoE、MTP；V4：mHC/Muon/FP4 | IndexShare | 线性×softmax 周期混合 | AttnRes、LatentMoE、MXFP4 |

> 结论先放这：**四家其实都在回答同一个问题——"长上下文下，attention 与 FFN 的成本怎么降"，且答案收敛到两类**：一类在 attention 内部做"压缩/稀疏/线性"替换，一类把 FFN 做成"大 MoE"。课外调研正好挑各家没选的那条路来对照。

---

## 0. 调研对象的选型逻辑（先记住这张图）

课内四家 + 本作业四家，其实是在同一张"结构演进地图"上各占一个点：

```
注意力层怎么变轻          FFN 怎么变稀疏            序列建模要不要守着 attention
├─ MLA/压缩 KV   (课内)    ├─ 细粒度稀疏 MoE (课内)    ├─ 仍用 softmax attention（可+滑窗）
├─ 稀疏选择/采样 (课内)    ├─ xMoE·top-2    → Mixtral   │
├─ 线性注意力    → MiniMax ├─ MoE+共享专家  → Llama 4   └─ SSM/线性 取代 attention → Jamba
└─ 无位置编码的层 → Llama 4 (iRoPE)
```

| 本次调研的模型 | 一句话结构标签 |
|---|---|
| **Llama 3.1 → Llama 4** | 开源生态"标准答案"的自我迭代：稠密 GQA 全注意力 → MoE + 交织注意力 + iRoPE 长上下文 |
| **Mistral-7B / Mixtral-8x7B** | 用"滑窗注意力"(7B) 与 "Top-2 xMoE"(8x7B) 做低推理成本的开源先行者 |
| **Jamba（AI21）** | 第一个生产级 **Mamba(SSM) × Transformer × MoE** 三合一混合架构 |
| **MiniMax-01（稀宇）** | 把 **线性注意力(Lightning Attention)** 规模化落地，冲 400 万 token 上下文 |

---

## 1. Llama 3.1 / Llama 4（Meta）——开源生态的"标准答案"如何自我迭代

### 1.1 定位
Meta 的开源权重旗舰。Llama 3.1 是"稠密、全注意力"时代的**事实基准**（很多论文拿它当对照组）；Llama 4 则标志着 Meta 从稠密转向 **MoE**，是"结构演进"里最直观的样本。

### 1.2 Llama 3.1：稠密时代的教科书配置

| 配置 | 8B | 70B | 405B |
|---|---|---|---|
| 层数 | 32 | 80 | 126 |
| hidden | 4,096 | 8,192 | 16,384 |
| Q 头数 / KV 头数 | 32 / **8** | 64 / **8** | 128 / **8** |
| FFN intermediate | 14,336 | 28,672 | 53,248 |
| 词表 | 128,256（三档共用） | | |
| 上下文 | 128K（先 8K 预训练再长度外推） | | |

结构特点（本课大多已覆盖，这里作为"基准坐标"一笔带过）：
- **GQA**：Q 头多、KV 头固定为 8（8B 4:1、70B 8:1、405B 16:1），KV 缓存省 30–50%；
- **RMSNorm + SwiGLU + RoPE(θ=5e5) + 无 bias**，属于 post-norm 的稠密 decoder-only 模板；
- 与课内 DeepSeek-V3 的差异点：Llama 3.1 **不做 MLA、不做 MoE**，用"原始"多头+全 attention 把 405B 推到大 scale——代价是推理成本高。

### 1.3 Llama 4：Meta 第一次拥抱 MoE 与超长上下文（2025-04，开放权重）

| | Scout | Maverick |
|---|---|---|
| 总参 / 激活参 | **109B / 17B** | **400B / 17B** |
| 专家 | 16 | 128 路由 + 1 共享 |
| 上下文 | **10M（训练 256K 外推）** | 1M |
| 多模态 | 原生（early fusion，统一主干） | 原生 |

结构上三个值得记的点：
1. **MoE + 共享专家 + Top-1 路由**：每个 token 同时走"永远在线的共享专家 + 从 N 个路由专家里 top-1 挑一个"，把激活参压到 ~17B。这跟课内 Mixtral 式 top-2、DeepSeek 式细粒度+共享专家的选择都不同——它把"共享专家兜底"做成标配。
2. **注意力层与 MoE 层交织，而非每层都 MoE**：部分层保持稠密 attention，控制推理开销（总参大 ≠ 每层都稀疏）。
3. **iRoPE（Interleaved RoPE）做 10M 上下文**：`Scout` 把层分两类——**每 4 层里前 3 层用标准 RoPE**（维持局部序关系），**第 4 层干脆"无位置编码"(NoPE)** 做全局因果注意力；再配合推理时的 attention 温度缩放。思路和课内"压缩 KV / 位置编码改造"是同一张卷子，但它选了"**让一部分层忘掉位置**"的答案。

> 坐标：Llama 4 证明了"稠密霸主"也转向 MoE；但在**注意力本身**上它仍守着标准 softmax attention（用 NoPE/温度缩放换长度），没有走 MLA 或线性注意力路线。

---

## 2. Mistral-7B / Mixtral-8x7B（Mistral AI）——滑窗注意力 与 开源 xMoE

### 2.1 定位
法国 Mistral 的两个"小而快"代表作：**Mistral-7B 用滑窗注意力做长上下文省钱**，**Mixtral-8x7B 是开源世界里第一批把 MoE 做进主流 transformer 生态的模型**（后来 DeepSeek/Qwen 的 MoE 都受它启发）。

### 2.2 Mistral-7B：滑窗注意力（SWA）

- 结构：32 层、hidden 4096、GQA(32 Q 头 / 8 KV 头)，decoder-only，SwiGLU + RoPE。
- 核心创新 **SWA**：每层只允许 token 看**前后固定窗（w=4096）**，复杂度 O(n²)→O(n·w)；配合"**滚动 KV 缓存**"，缓存大小固定不随长度涨。
- 靠**层叠**扩大感受野：堆 L 层后，顶层有效视野 ≈ L×w（32×4096 ≈ 128K 的理论覆盖），训练却只要 8K 窗口。
- 易错澄清（很多二手资料写错）：**滑窗是 Mistral-7B 的特征；Mixtral-8x7B 官方 config 里 `sliding_window` 是 `null`（全注意力）**——Mistral 官方还专门修过一次被错误引入 4096 的回归 commit。

### 2.3 Mixtral-8x7B：xMoE（Top-2 of 8）

| 配置 | 值 |
|---|---|
| 总参 / 激活参 | ~46.7B / ~12.9B（8x7B 是"8 个专家 ×7B 级"的意思） |
| 层数 / hidden / 头 | 32 / 4096 / 32 Q·8 KV（GQA） |
| MoE | 每层 **8 个专家，每 token top-2** |
| 专家 FFN | SwiGLU，intermediate 14336/个 |
| 注意力 | **全注意力**（非滑窗），与所有专家共享 |

结构上值得记的点：
1. **共享注意力 + 专家只换 FFN**：路由发生在 attention 之后，注意力参数全体共享，只有 FFN 按专家分——所以"稀疏"只买 FFN，attention 的 KV 开销没有 8 倍放大；
2. **top-2 软路由 + 负载均衡辅助损失**：路由网络按 token 独立打分、softmax 后取 top-2 并按权重归一化加权求和；加 `router_aux_loss` 防止专家塌缩到少数几个；
3. 结果：内存要装 ~47B（~90GB FP16），但**单 token 算力 ≈ 13B 稠密**——即"用显存换算力"，这正是 MoE 与稠密的本质 trade-off。

> 坐标：Mixtral 是"**稀疏 FFN + 标准 attention**"的课代表；对比看，课内 DeepSeek-V3 走的是"**稀疏 + 更狠的注意力压缩(MLA)**"，两边加起来才把 MoE 这条路讲完整。

---

## 3. Jamba（AI21）——生产级 Mamba(SSM) × Transformer × MoE 混合

### 3.1 定位
**Jamba = Joint Attention And Mamba**。它是第一个做到"生产级规模 + Apache-2.0 开源"的 **SSM×Transformer 混合**模型，目标：用线性复杂度的 SSM 处理超长文本，又保留注意力做全局交互——256K 上下文只占 ~12GB 缓存，能塞进单卡 80GB。

### 3.2 结构特点

- **层设计（1:7 注意力:SSM 混合）**：每 8 层一组，**1 层标准 attention + 7 层 Mamba(SSM)**（config：`attn_layer_period=8`）。Ablation 认为 1:7 收益/成本最佳。
- **MoE 只加在 attention 层的 FFN 上**：16 个专家、top-2 路由；Mamba 层保持稠密。所以稀疏化的是"全局交互段"，线性段不稀疏。
- **Mamba(SSM) 段的常量状态**：SSM 用固定维度的隐状态（`d_state=16` + 卷积核 4 + expand 2），**不产生随序列增长的 KV 缓存**，处理复杂度近线性——这是它比纯 transformer 在 256K 上省一个量级缓存的根本原因。
- 规模：**52B 总参 / ~12B 激活**，上下文 256K。后续 Jamba 1.5 家族把"函数调用/JSON/长文档"做成产品特性；1.5 Large 做到 398B/94B。

### 3.3 为什么值得放进"演进"作业
课内四家的注意力变体（MLA、稀疏/压缩等）**都在 softmax attention 框架内**省 KV；Jamba 代表另一条岔路——**干脆不用 attention 做大部分层**，用 SSM（Mamba 系列的状态空间对偶理论也可证其等价于结构化掩码注意力）。它的教训：**混合是工程正解**（纯 Mamba 长程检索差，纯 attention 缓存贵，1:7 混合取两者之长）。

> 局限：SSM 段的长程"指哪打哪"能力仍弱于 attention；这也是它必须保留 1/8 注意力层的直接原因。

---

## 4. MiniMax-01（稀宇科技）——把线性注意力推到前沿规模的"另类"

### 4.1 定位
MiniMax-Text-01（+视觉版 VL-01）是把 **Lightning Attention（线性注意力）** 首次做到前沿规模并开源的模型（MIT 许可，2025-01）。核心卖点：**400 万 token 上下文**下仍保持近线性计算，而不是靠窗口/压缩硬撑。

### 4.2 结构特点

| 配置 | 值 |
|---|---|
| 总参 / 激活参 | 456B / 45.9B |
| 层数 / hidden / 词表 | 80 / 6144 / 200,064 |
| 序列建模 | **每 8 层 = 7 层 Lightning Attention + 1 层 softmax attention**（7:1） |
| softmax 层 | GQA（KV group=8），RoPE 只加在**一半 head 维度**上，base=1e7 |
| MoE | 32 专家、top-2，专家 FFN 9216 |
| 上下文 | 训练 1M，推理可到 **4M**（部分 config 报 ~10M max position） |

结构上三个值得记的点：
1. **线性注意力的关键 = 丢掉 softmax，换来可分性**：Lightning Attention 用一个手工设计的衰减替代 softmax，使注意力可写成**矩阵结合律可重排**的形式，从而用 O(n) 处理序列；这是与课内"压缩 KV(MLA)"完全不同的思路——**不压缩缓存，而是直接改复杂度**。
2. **不是纯线性，而是 7:1 混合**：每隔 7 个线性块插 1 个标准 softmax 块，靠少量全局注意力补线性注意力的"检索精度"——和 Jamba 的 1:7 是同一个"混合"哲学的镜像。
3. **为极长序列重做了并行基建**：LASP+（线性注意力序列并行）、变长 ring attention、专家张量并行(ETP) 等，把 4M 上下文训练/推理的 MFU 做到 >75%——结构创新必须配系统创新才落得了地。

> 坐标：它和课内四家补全了"省注意力算力"整条谱系的另一端——**MLA 是"压缩表示"，MiniMax 是"改掉 softmax 的复杂度"**。后续 MiniMax-M1（2025-06，456B/45.9B 推理模型）在其上叠加了长推理输出（80K+ token）能力。

---

## 5. 横向对比一览

| | Llama 4 Scout | Mixtral-8x7B | Jamba | MiniMax-01 | （课内参照）DeepSeek-V3 |
|---|---|---|---|---|---|
| 开源许可 | Llama（开放权重） | Apache-2.0 | Apache-2.0 | MIT | 模型许可证 |
| 总参/激活参 | 109B/17B | 46.7B/12.9B | 52B/12B | 456B/45.9B | 671B/37B |
| 序列建模 | 全 softmax attention | 全 attention | 1:7 attention:Mamba(SSM) | 7:1 linear:softmax | MLA（KV 压缩） |
| 稀疏化 | MoE top1+共享专家 | MoE top2 | MoE(仅 attention 层) | MoE top2 | 细粒度 MoE+共享专家 |
| 位置编码 | RoPE + 交织 NoPE(iRoPE) | RoPE | RoPE(attention 层) | RoPE(半维, base1e7) | RoPE+（长文本策略） |
| 上下文（外推后） | 10M | 32K(训) | 256K | 1M 训/4M+ | 128K 级 |
| 一句话标签 | 稠密霸主转 MoE+超长 | 开源 xMoE 先行者 | SSM 混合生产化 | 线性注意力前沿化 | 稀疏+注意力压缩标杆 |

## 6. 对照课内主线：这几个模型补上了哪几块拼图

1. **"稀疏 FFN"这一课，Mixtral 是前半、DeepSeek 是后半**：都是"总参大、激活小"，区别在注意力要不要一起省——Mixtral 保留全注意力只省 FFN，DeepSeek 连 KV 都压缩。理解这个对比，才理解为什么 MoE 模型通常还要配 MLA 类设计。
2. **"省注意力"其实有两条路，课内只细讲了压缩那条**：DeepSeek/GLM/Kimi/Qwen 的 MLA 类方案是"**压缩 KV 表示**"；Jamba 与 MiniMax-01 是"**换序列建模/改复杂度**"（SSM 或线性注意力）。它们仍各留少量真 attention 层救精度——**"全线性/全 SSM"到现在都不是答案，"混合"才是**。
3. **长上下文的工程含量**：Llama 4 的 10M 靠"部分层无位置编码"这种训练期技巧 + 推理温度缩放；MiniMax 的 4M 靠线性复杂度 + 一套为极长序列重写的并行框架(LASP+/ring attention/ETP)。结构亮点 ≠ 单独存在，必须配训练/推理基建。
4. **一个方法论提醒**：调研这些模型时别只背参数表，要问三个问题——**注意力还是不是 softmax？FFN 稀疏化怎么路由？位置/长度怎么处理？** 三问能定位任何一个开源模型在演进图上的坐标。

## 7. 参考资料（检索核对 2026-09-04）

- Llama 3.1 配置/家族：https://github.com/meta-llama/llama-models ；Llama 3.1 技术报告（arXiv）
- Llama 4 Scout/Maverick 结构与 iRoPE：https://ai.meta.com/blog/llama-4-multimodal-intelligence/ ；https://github.com/meta-llama/Llama-4-Scout-17B-16E
- Mistral-7B / Mixtral 官方仓库与"Mixtral 无滑窗"回归修复 commit：https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 ；https://mistral.ai/news/mixtral-of-experts/
- Jamba 技术博客 / HF `JambaConfig`：https://www.ai21.com/blog/announcing-jamba ；https://huggingface.co/docs/transformers/en/model_doc/jamba
- MiniMax-01 论文与代码：*MiniMax-01: Scaling Foundation Models with Lightning Attention*（arXiv:2501.08313）；https://github.com/MiniMax-AI/MiniMax-01
- week16 课内参照模型报告（tech_reports/）：deepseek_v3/v4、glm5、kimi_k3 技术报告 PDF
