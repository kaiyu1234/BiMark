# BiMark 方案代码审查与短文本多 bit 改进研究笔记

## 1. 现有方案（从代码反推）

- 编码侧在 `LogitsProcessor` 中工作：每一步取 top-k 概率，再按多个固定随机划分（`partition_seeds`）对词表分组；随后依据前缀 PRF 产生的 `c_list` 与 `bit_idx`，决定当前 token 对某个 message bit 的“偏置方向”。
- 每步只选择一个 bit 位置（`bit_idx`）进行注入，属于随机时分复用（time-division multiplexing）。
- 解码侧统计每个 bit 上两类计数 `[count0, count1]`，用多数票恢复 bit；同时计算 z-score/p-value 作为整体显著性指标。

## 2. 主要痛点

### 2.1 短文本容量-可靠性矛盾严重
- 每步仅命中一个 bit 位置，短文本下每个 bit 平均样本量太少，导致 `x`（平票/不确定）多，BER 高。
- 代码中 bit 索引均匀随机，未根据剩余长度或 bit 置信度做调度，短文本时常出现“部分 bit 几乎没被覆盖”。

### 2.2 编码强度固定，不做内容自适应
- 目前 `delta/alpha` 基本为全局常数，虽然有按 `p0` 的缩放，但缺乏对 entropy、top1-top2 margin、句法关键位点的自适应。
- 在高不确定 token 上强推会影响流畅性，在极尖锐分布上又可能注入不足。

### 2.3 检测统计模型较粗糙
- 目前主要是 bit 级多数票 + z-test，没有使用对数似然比（LLR）或序贯检测，不能充分利用每个 token 的软信息。
- 多 bit 场景无纠错码，短文本尤其容易出现“个别 bit 拉垮导致整串失败”。

### 2.4 工程与复现问题
- 仓库里存在硬编码 `auth_token = "xxx"`，影响复现和开源合规。
- `detect_watermark_dump.py` 中出现同名函数 `decode_bimark_multibit_watermark` 重复定义，前者会被后者覆盖，增加维护风险。
- 生成阶段 `WatermarkBimark.__call__` 每步 `print("time cost")`，会显著拖慢推理并污染日志。
- 历史去重 `if prefix in hist` 直接用张量对象判断，可读性与稳定性较弱，建议统一转 tuple/hash。

## 3. 可发表的改进方向（重点：短文本多 bit）

### 方向 A：短文本优先的“自适应 bit 调度”（ABS）
核心：不再均匀随机选 bit，而是根据“当前各 bit 置信度缺口”优先注入最弱 bit。

- 维护每个 bit 的在线置信度分数（如 LLR 累积绝对值）。
- 每个 token 时刻选 `argmin(confidence)` 的 bit 注入，保证短文本下覆盖均匀。
- 可加入轻微随机扰动保持安全性：`bit_idx = argmin + Gumbel noise`。

**论文价值**：直接提升短文本有效负载下的最差位准确率（min-bit accuracy）。

### 方向 B：软解码 + 纠错编码联合（LLR+ECC）
核心：把“多数票”升级为“软信息 + 码字译码”。

- 对每个 bit 输出 LLR，而非只记硬计数。
- 外层加 BCH/LDPC/Polar（短码优先，如 BCH(31,16)）。
- 检测时做软译码，目标指标从 raw BER 转为 message success rate。

**论文价值**：短文本 token 数受限时，ECC 可显著降低整串失败概率。

### 方向 C：不确定性感知注入（UAW）
核心：让 `delta_t` 随 token 难度变化。

- 依据局部分布 entropy、margin、重复惩罚状态动态调节注入强度：
  - 低 entropy（模型很确定）→ 小幅注入即可；
  - 中 entropy → 主力注入区间；
  - 高 entropy（不稳定）→ 降低注入避免语义退化。
- 与 A 联合：先确定“注哪个 bit”，再确定“注多强”。

**论文价值**：在相同 PPL 约束下提高 bit recoverability。

### 方向 D：位置感知短码映射（Position-aware Mapping）
核心：短文本里前 20~60 token 信息密度高、改写概率低，可做非均匀保护。

- 将关键 bit（如头部、CRC）映射到更稳定位点或多次重复；
- 非关键 bit 用普通注入。

**论文价值**：提升“可验证性”和“可解码性”的下界，特别是截断文本。

## 4. 建议的论文实验设计

### 4.1 任务设定
- 长度分桶：16/32/64/128 新 token。
- payload：8/16/32 bit。
- 攻击：同义改写、截断、随机删除、DIPPER 改写、温度重采样。

### 4.2 对比方法
- 原始 BiMark（当前实现）
- BiMark + ABS
- BiMark + LLR
- BiMark + LLR + ECC
- 全量方案（ABS + UAW + LLR + ECC）

### 4.3 指标
- bit-level: BER / min-bit accuracy / uncertain-rate(`x`)。
- message-level: exact recovery rate / CRC pass rate。
- text quality: PPL、win-rate、人评一致性。
- 检测：AUC、TPR@FPR=1%。

## 5. 建议先做的最小可行原型（MVP）

1. 保持现有 partition 机制不变。
2. 把 `bit_idx` 从随机改为“最少覆盖优先”（coverage-aware）。
3. 检测端输出 LLR 并加入一个短 BCH 码。
4. 在 32/64 token 两档验证：
   - 同等 PPL 下，message recovery 至少提升 10~20%（目标）。

## 6. 预期论文标题（可选）
- **ShortText-BiMark: Adaptive Scheduling and Soft Decoding for Multi-bit Watermarking in LLMs**
- **Reliable Multi-bit Watermarks in Short LLM Outputs via Confidence-aware Embedding and ECC**

## 7. 更强调“新颖性”的候选方案（尽量避开常见套路）

> 目标：不只是把已有 ABS/LLR/ECC 组合起来，而是提出更有“方法学新意”的短文本多 bit 水印框架。

### 方案 E：Credit-based 预算分配水印（CBAW）
核心思想：把每个 token 的“可嵌入能力”显式建模为预算 credit，而不是固定概率地写入 bit。

- 先估计每步注入预算 `credit_t`（由 entropy、语义风险、候选词分布几何结构决定）。
- 再把预算分配给当前最缺样本/最低置信度的 bit（而非随机 bit）。
- 当文本很短时，系统会自动把预算集中到关键 bit，形成“短文本优先保护”。

**创新点**：把 watermark 从“离散触发”转为“预算优化”问题，可写成 constrained optimization（质量约束 + 恢复率最大化）。

### 方案 F：语义不变子空间水印（SISW）
核心思想：只在“语义近似不变”的候选子空间里编码 bit，减少短文本中语义破坏带来的可感知风险。

- 用 embedding / hidden-state 局部线性化，寻找语义梯度较小的方向（低敏方向）。
- 将词汇偏置限制在这些低敏方向对应的候选 token 集合中。
- 在同等可读性约束下可提升可注入步数，间接提高短文本 payload。

**创新点**：把水印注入和语义几何联系，区别于传统随机分桶/绿表机制。

### 方案 G：跨 token 联合码元（Joint Symbol Watermark, JSW）
核心思想：不再“一 token 写一 bit 决策”，而是若干 token 联合承载一个 q-ary 码元（例如 4/8 进制）。

- 使用长度为 `m` 的 token block 编码一个符号，检测时联合判决。
- 短文本里可以减少独立 bit 的统计方差，提升每个符号的可靠性。
- 再映射为 bitstream（Gray code）可减少临界错误传播。

**创新点**：从 bit 级独立检测转为符号级联合检测，统计效率更高。

### 方案 H：可逆前缀状态机映射（RPSM）
核心思想：让 bit 写入由“前缀状态机”控制，而不是独立同分布随机选 bit。

- PRF 给出状态转移，状态决定当前可写 bit 子集与符号表。
- 检测端通过状态机回溯进行 Viterbi/Beam 解码。
- 对短文本可设计“快速收敛状态”，保证前几十 token 就覆盖关键信息。

**创新点**：将 watermark 变成序列编码问题（类似通信中的 trellis code），与主流 token-level 方案区分明显。

### 方案 I：截断鲁棒前缀优先码（TRPC）
核心思想：针对短文本常见截断场景，采用 fountain-like 的不等保护策略。

- 前缀 token 重复携带头信息（消息长度、CRC、salt、索引）。
- 主体 bit 使用 rateless 冗余包，检测端随长度增长逐步恢复。
- 任意早停长度都能有“部分可验证”的结果。

**创新点**：把“长度不确定”作为一等约束，天然适配短文本和流式输出。

### 方案 J：对抗式最坏前缀优化（Adversarial Prefix Training for Watermark）
核心思想：训练时显式寻找最不利前缀分布，优化“最差样本”的 bit 恢复率。

- 内层：对前缀做离散近似扰动，寻找导致 bit 混淆最大的上下文。
- 外层：更新注入策略参数，使最坏情况下仍可恢复。
- 评价指标聚焦 tail risk（5th percentile message accuracy）。

**创新点**：从平均性能转向风险鲁棒性，适合论文讲“短文本极端场景”贡献。

## 8. 哪些方向更可能“没被做透”

更推荐优先级（创新性 × 可落地）：

1. **E + H 组合**：预算优化 + 状态机编码（方法新、理论可写、实验可做）。
2. **G + I 组合**：联合码元 + 截断鲁棒码（直接打短文本痛点，工程上可控）。
3. **F 单独成文**：语义子空间约束（创新强，但实现和论证成本最高）。

## 9. 可直接写进论文的“创新声明”模板

- 我们首次将短文本多 bit 水印表述为**预算受限的序列编码问题**，并提出 credit-based 分配与状态机映射联合框架，在固定语义质量约束下最大化消息恢复率。
- 我们提出联合码元检测与截断鲁棒前缀码，使水印在 32/64 token 场景下依旧具备可验证的渐进恢复能力。
- 我们引入最坏前缀风险评估，报告 tail-robust 指标，而非仅平均 BER。
