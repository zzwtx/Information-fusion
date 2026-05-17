# FedMGP 优化方案：文本条件化视觉提示

## 动机

FedMGP 为每个客户端维护多组文本-视觉提示：

```text
P_j = {p_t,j, p_v,j}, j = 1, ..., G
```

在当前设计中，第 `j` 组文本提示和第 `j` 组视觉提示会在计算 logits 时配对使用，但二者的参数基本是独立学习的。这意味着它们只是“编号上配对”，不一定真正表达相同的语义偏好。

我们提出 **Text-Conditioned Visual Prompts, TCVP**：根据同组文本提示生成或调整视觉提示，使视觉编码器在提取图像特征时，已经受到文本端语义方向的引导。

通用形式为：

```text
p_v,j = p_v,j^base + alpha * Delta p_v,j
Delta p_v,j = MetaNet(q_t,j)
```

其中，`q_t,j` 是第 `j` 组文本提示的摘要表示，`MetaNet` 是轻量级网络，`alpha` 控制残差视觉提示的强度。

## 方案 A：基于 Pooling 的文本条件

这是最简单、最稳定的版本。

给定：

```text
p_t,j in R^{L_t x d_t}
```

先计算：

```text
q_t,j = Pool(p_t,j)
```

默认使用 mean pooling：

```text
q_t,j = mean_token(p_t,j)
```

然后：

```text
Delta p_v,j = MetaNet(q_t,j)
p_v,j = p_v,j^base + alpha * reshape(Delta p_v,j)
```

期望形状：

```text
MetaNet: R^{d_t} -> R^{L_v * d_v}
reshape(Delta p_v,j): R^{L_v x d_v}
```

### 优点

- 非常轻量。
- 在联邦 few-shot 训练中更稳定。
- 工程改动最小。
- 保留 FedMGP 参数高效的特点。

### 缺点

- Mean pooling 会丢失 token 级别的 prompt 结构。
- 它使用的是原始 prompt 参数，没有利用 text encoder 中的上下文化表示。

### 当前代码实现

已实现方案 A 的可开关版本：

```text
TRAINER.FEDMGP.TCVP_ENABLED
```

默认关闭，不影响原生 FedMGP。开启后，每组视觉提示不再直接使用裸的 `p_v,j`，而是使用：

```text
p_v,j^tcvp = p_v,j^base + alpha * MetaNet(mean(p_t,j))
```

其中：

```text
p_t,j: FedMGP 第 j 组文本 prompt 参数
mean(p_t,j): 对 prompt token 维度做 mean pooling
LayerNorm(mean(p_t,j)): 作为 MetaNet 的实际输入
p_v,j^base: 原始 FedMGP visual prompt 参数，仍然保留、训练和聚合
MetaNet: 两层 MLP
alpha: 残差强度
```

MetaNet 当前结构：

```text
LayerNorm(d_text)
Linear(d_text, hidden_dim)
ReLU
Linear(hidden_dim, n_ctx_vision * d_vision)
```

最后一层使用零初始化，使训练开始时：

```text
MetaNet(mean(p_t,j)) = 0
p_v,j^tcvp = p_v,j^base
```

这样第一步等价于原生 FedMGP，能降低新增模块造成的不稳定性。

相关配置：

```yaml
N_CTX_TEXT: 4
TCVP_ENABLED: False
TCVP_HIDDEN_DIM: 256
TCVP_INPUT_NORM: "layernorm"
TCVP_ALPHA: 0.1
TCVP_DETACH_TEXT_CONDITION: False
TCVP_USE_ALIGN_LOSS: True
TCVP_ALIGN_LOSS_WEIGHT: 0.1
TCVP_REG_WEIGHT: 0.001
```

`TCVP_DETACH_TEXT_CONDITION=False` 表示 CE 损失可以通过 MetaNet 路径回传到文本 prompt，使文本 prompt 与视觉 prompt 更紧密耦合。如果训练不稳定，可以设置为 `True` 做消融，让 MetaNet 只读取文本 prompt，不反向改动文本 prompt。

当前还加入了两个本地训练项：

```text
L_align = mean_j mean_b [1 - cos(z_v,b^j, z_t,y_b^j)]
L_reg = mean_j ||MetaNet(mean(p_t,j))||_2^2
```

总损失为：

```text
L = L_CE
  + lambda_div L_div
  + lambda_align L_align
  + lambda_reg L_reg
```

其中 `L_align` 和 `L_reg` 只在 `TCVP_ENABLED=True` 时生效。

联邦侧，MetaNet 参数会像普通联邦参数一样在参与客户端之间做加权平均，并分发回参与本轮聚合的客户端；文本 prompt 和视觉 prompt 仍然走 FedMGP 原有的选择、聚合和分发逻辑。因此 TCVP 可以和前面的两个初步实验同时组合：

```text
TCVP + GroupBind
TCVP + SlotAgg
TCVP + GroupBind-SlotAgg
```

运行示例：

```bash
RUN_TAG=_tcvp_pool bash scripts/FedMGP/base2novel_train.sh \
  0 dtd 16 base2novel_vit_b16 \
  TRAINER.FEDMGP.TCVP_ENABLED True
```

组合 GroupBind-SlotAgg：

```bash
RUN_TAG=_tcvp_groupbind_slotagg bash scripts/FedMGP/base2novel_train.sh \
  0 dtd 16 base2novel_vit_b16 \
  TRAINER.FEDMGP.TCVP_ENABLED True \
  TRAINER.FEDMGP.GROUP_BIND_SELECTION True \
  TRAINER.FEDMGP.SLOT_AGGREGATION True
```

## 方案 B：基于 Text Encoder 隐状态的文本条件

这个版本使用 text encoder 中已经上下文化的 prompt 表示。

第 `j` 组文本输入形式为：

```text
[SOS] [p_t,j,1] ... [p_t,j,L] [class name] [EOS]
```

令 `H_l` 表示 text transformer 第 `l` 层的 hidden states。我们取最后一个 prompt token 位置的 hidden state：

```text
q_t,j = H_l[last_prompt_token]
```

然后：

```text
Delta p_v,j = MetaNet(q_t,j)
p_v,j = p_v,j^base + alpha * reshape(Delta p_v,j)
```

更稳定的版本可以平均最后 `M` 层：

```text
q_t,j = mean_{l in last M layers} H_l[last_prompt_token]
```

### 为什么取最后一个 Prompt Token？

在 CLIP 的 causal text transformer 中，最后一个 prompt token 可以 attend 到它前面的 prompt tokens：

```text
[SOS], p_t,j,1, ..., p_t,j,L
```

因此，经过多层 transformer 后，该位置的 hidden state 可以作为整组 prompt 的摘要表示。同时，它通常不会依赖后面的 class name，因此生成的是 **group-level visual prompt**，而不是 class-specific visual prompt。

这很重要，因为如果视觉提示依赖具体类别：

```text
p_v,j,k = MetaNet(z_t,k^j)
```

推理时可能需要为每个类别分别运行 image encoder，计算成本会大幅增加，不适合 FedMGP。

### Hidden State 与 Attention Value 的区别

建议使用：

```text
H_l[last_prompt_token]
```

而不是直接使用：

```text
V_l[last_prompt_token]
```

原因是 raw attention value vector 本身只是 value 投影，不一定已经融合上下文信息；而 hidden state 是经过 attention 和 MLP 后的表示，更适合作为 prompt group 的上下文摘要。

### 优点

- 能捕获 text encoder 中已经整合过的 prompt 信息。
- 比 mean pooling 更好地保留 token 间交互。
- 额外计算量很小，因为 text encoder 原本就需要前向传播。

### 缺点

- 需要修改 `TextEncoder`，让它返回中间层的 prompt-token hidden states。
- 会略微增加显存占用。
- 工程复杂度高于 pooling 方案。

## MetaNet 设计

推荐使用轻量 MLP：

```text
Linear(d_in, d_hidden)
ReLU
Linear(d_hidden, L_v * d_v)
```

建议初始设置：

```text
d_hidden = 128 或 256
alpha = 0.1
```

推荐使用残差式生成：

```text
p_v,j = p_v,j^base + alpha * Delta p_v,j
```

而不是完全替换原始视觉提示。这样可以保留 FedMGP 原有 visual prompt 的个性化能力，同时提高训练稳定性。

## 训练目标

保留 FedMGP 原始目标：

```text
L = L_CE + lambda_div L_div
```

可以额外加入文本-视觉输出对齐损失：

```text
L_align = 1/G * sum_j 1/B * sum_b [1 - cos(z_v,b^j, z_t,y_b^j)]
```

候选总目标：

```text
L = L_CE
  + lambda_div L_div
  + lambda_align L_align
  + lambda_reg ||Delta p_v||_2^2
```

建议初始值：

```text
lambda_align = 0.1
lambda_reg = 0.001
```

其中 `lambda_reg` 用于约束 MetaNet 生成的视觉提示残差不要过大，避免破坏 CLIP 原始视觉表示。

## 聚合原则：保持 FedMGP 的共享/本地解耦目标

FedMGP 当前根据本地 prompt group 与上一轮全局 prompt 的相似度进行动态选择。这个选择标准的核心作用是近似判断：

```text
该 prompt group 更接近全局共享信息，还是更接近客户端本地特异信息
```

因此，TCVP 的第一版设计不应改变 FedMGP 的动态聚合目标。也就是说：

```text
本地训练阶段：增强组内 text prompt 与 visual prompt 的跨模态一致性
服务器聚合阶段：仍使用 FedMGP 原始 similarity-guided probabilistic selection
```

这样可以避免把“本地跨模态对齐质量”误当成“全局共享性”。二者不是同一个概念：

```text
global prompt similarity: 衡量 prompt group 的共享性近似
local text-vision alignment: 衡量 prompt group 在本地数据上的跨模态功能有效性
```

第一阶段推荐保持职责分离：

```text
TCVP / L_align 负责本地组内跨模态对齐
FedMGP 原始动态选择负责全局共享信息筛选
```

## 初步实验1：组内绑定匹配的动态选择

在实现完整 TCVP 之前，可以先做一个改动极小的小实验，用来验证“文本提示和视觉提示是否应该作为绑定组一起参与全局匹配”。

### 当前 FedMGP 可能存在的问题

FedMGP 中每个 group 是：

```text
P_j = {p_t,j, p_v,j}
```

但是在动态选择时，如果文本提示和视觉提示分别与全局 prompt 计算相似度，再分别选择或分别排序，就可能出现一种不一致：

```text
text group j 被认为最接近全局共享语义
visual group k 被认为最接近全局共享语义
j != k
```

这会削弱 FedMGP 的“成对 text-visual prompt group”假设。因为 forward 时模型使用的是组内配对：

```text
logits_j = f(x, p_v,j) · g(p_t,j, class)^T
```

如果聚合选择时 text 和 vision 是分开判断的，那么训练和聚合的 group 语义可能不完全一致。

### 小实验目标

把动态选择从“文本、视觉分别匹配全局 prompt”改成“组内绑定后整体匹配全局 prompt group”。

保持其他部分不变：

```text
不引入 MetaNet
不引入 L_align
不改变本地训练损失
不改变 top-k / probabilistic sampling 框架
只改变 prompt group 的 similarity score 计算方式
```

### 原始匹配方式

如果原始实现分别计算：

```text
s_t,j = sim(p_t,j^local, p_t,j^global)
s_v,j = sim(p_v,j^local, p_v,j^global)
```

并可能分别选择 text prompt 和 visual prompt，那么改为统一的组级分数：

```text
s_group,j = beta_t * sim(p_t,j^local, p_t,j^global)
          + beta_v * sim(p_v,j^local, p_v,j^global)
```

默认：

```text
beta_t = 0.5
beta_v = 0.5
```

然后基于 `s_group,j` 选择同一个 group index：

```text
selected_groups = TopK_or_Sample(s_group)
```

最终上传和聚合：

```text
{p_t,j, p_v,j}, j in selected_groups
```

### 为什么这个实验有价值

这个实验更贴近 FedMGP 原始动机，因为它仍然使用 global similarity 作为共享性近似指标，只是把匹配单位从单模态 prompt 改成了 text-visual 绑定组：

```text
原目标：选择更接近全局共享信息的 prompt
新单位：选择更接近全局共享信息的 text-visual prompt group
```

它不会引入额外的本地性能指标，也不会混淆 sharedness 和 alignment quality。

### 预期观察

如果该实验有效，可能说明：

```text
FedMGP 的 text prompt 和 visual prompt 不应在聚合选择中被拆开
组内绑定选择能提升 text-vision prompt group 的语义一致性
```

如果效果不明显，则说明当前分别匹配策略已经足够稳定，后续应把主要精力放在 TCVP 的本地生成和对齐损失上。

## 初步实验2：按相似度槽位聚合与分发

当前 FedMGP 代码中的聚合方式是 **按 rank 聚合**：

```text
1. 每个客户端先选出 top-k 个 prompt
2. 所有客户端的第 1 名聚合成 global rank 0
3. 所有客户端的第 2 名聚合成 global rank 1
...
```

这种方式简单稳定，但它不能保证“特征最相似的客户端 prompt 被聚合到同一个全局 prompt”。原因是，当前代码在计算：

```text
score_i = max_k sim(p_i^local, p_k^global)
```

之后，只保留了最大相似度分数，而没有保留：

```text
best_slot_i = argmax_k sim(p_i^local, p_k^global)
```

因此，一个本地 prompt 即使最匹配 `global prompt 1`，只要它在本客户端排名第 1，也会被聚合到 `global rank 0`。

### 小实验目标

将聚合从 **RankAgg** 改为 **SlotAgg**：

```text
RankAgg: 按客户端内部排名聚合
SlotAgg: 按最匹配的全局 prompt 槽位聚合
```

核心思想：

```text
本地 prompt 应该被聚合到它最相似的 global prompt slot
聚合结果也应该分发回贡献到该 slot 的本地 prompt index
```

### 匹配与聚合规则

对每个参与客户端 `c` 的每个本地 prompt `i`，计算它和每个全局 prompt slot `k` 的相似度：

```text
s_{c,i,k} = sim(p_{c,i}^local, p_k^global)
```

得到最匹配槽位：

```text
best_slot_{c,i} = argmax_k s_{c,i,k}
best_score_{c,i} = max_k s_{c,i,k}
```

然后先根据 `best_score` 选出每个客户端的候选 prompt，例如仍然选择 top-k：

```text
selected_c = TopK_i(best_score_{c,i})
```

之后不再按 rank 聚合，而是按 `best_slot` 分桶：

```text
Bucket_k = {(c, i) | i in selected_c and best_slot_{c,i} = k}
```

每个全局槽位单独聚合：

```text
p_k^global,new = mean_{(c,i) in Bucket_k} p_{c,i}^local
```

### 聚合后的分发规则

聚合后的 `p_k^global,new` 只写回贡献到该 slot 的本地 prompt：

```text
for (c, i) in Bucket_k:
    p_{c,i} <- p_k^global,new
```

这和当前代码的 rank 写回不同。当前代码是：

```text
客户端第 r 个被选中的 prompt <- global rank r
```

SlotAgg 则是：

```text
本地 prompt i <- 它贡献的 matched global slot
```

这能让“聚合”和“分发”都围绕同一个相似度匹配关系进行。

### 最低容纳数量约束

SlotAgg 的风险是某些 global slot 可能无人匹配，导致该 prompt 长期不更新。为避免空槽位，可以为每个全局提示词设置最低容纳数量：

```text
min_bucket_size = m
```

目标：

```text
|Bucket_k| >= m, for every global slot k
```

一个简单实现方式是两阶段分配：

```text
阶段 1：每个 prompt 先分配给最匹配的 best_slot
阶段 2：检查空槽或样本不足的 slot
       从尚未固定或候选池中选择对该 slot 相似度最高的 prompt 补足
```

补足策略可以是：

```text
对每个不足的 slot k：
    从所有 selected prompts 中找还没有被强制分配的 prompt
    按 sim(p_i, global_k) 从高到低补入 Bucket_k
    直到 |Bucket_k| = m
```

如果所有 selected prompts 都已经被使用，可以允许少量重复贡献，或者保留上一轮 `p_k^global`：

```text
方案 A：允许重复补桶，保证每个 slot 都有更新
方案 B：不足时使用 EMA，p_k^new = rho * p_k^old + (1-rho) * mean(Bucket_k)
方案 C：不足时保持该 slot 不变
```

推荐第一版采用：

```text
min_bucket_size = 1
不足时优先从 selected prompts 中补足
仍不足则保持该 slot 不变
```

这样能保证每个全局提示词至少有机会更新一点，同时不会过度扭曲相似度匹配结果。

### 和组内绑定匹配的关系

这个实验可以先在单模态 text / vision 上分别做：

```text
Text SlotAgg
Vision SlotAgg
```

也可以进一步和组内绑定结合：

```text
GroupBind-SlotAgg
```

组内绑定版本计算：

```text
s_{c,i,k}^{group}
  = beta_t * sim(p_{t,c,i}, p_{t,k}^global)
  + beta_v * sim(p_{v,c,i}, p_{v,k}^global)
```

然后整个 group `{p_t,c,i, p_v,c,i}` 一起分配到同一个 global group slot `k`。

推荐实验顺序：

```text
1. SlotAgg: text 和 vision 分别按 slot 聚合
2. GroupBind: 只绑定选择 index，仍按 rank 聚合
3. GroupBind-SlotAgg: 组内绑定后按 global group slot 聚合
```

### 当前实现开关

已在代码中加入两个彼此独立的开关：

```text
TRAINER.FEDMGP.GROUP_BIND_SELECTION
TRAINER.FEDMGP.SLOT_AGGREGATION
```

因此可以运行四种设置：

```text
原生 FedMGP:
GROUP_BIND_SELECTION False
SLOT_AGGREGATION False

初步实验1 GroupBind:
GROUP_BIND_SELECTION True
SLOT_AGGREGATION False

初步实验2 SlotAgg:
GROUP_BIND_SELECTION False
SLOT_AGGREGATION True

组合实验 GroupBind-SlotAgg:
GROUP_BIND_SELECTION True
SLOT_AGGREGATION True
```

SlotAgg 相关参数：

```text
SLOT_ASSIGN_HIGHEST_SIM: True
SLOT_MIN_PROMPTS_PER_SLOT: 1
SLOT_KEEP_EMPTY_PROMPTS: True
```

当前实现中，第一轮没有上一轮全局 prompt，因此 SlotAgg 会先退回 RankAgg 来初始化 global prompt。第二轮开始，才按最匹配 global slot 聚合和分发。

运行示例：

```bash
RUN_TAG=_slotagg bash scripts/FedMGP/base2novel_train.sh \
  0 dtd 16 base2novel_vit_b16 \
  TRAINER.FEDMGP.SLOT_AGGREGATION True
```

组合实验：

```bash
RUN_TAG=_groupbind_slotagg bash scripts/FedMGP/base2novel_train.sh \
  0 dtd 16 base2novel_vit_b16 \
  TRAINER.FEDMGP.GROUP_BIND_SELECTION True \
  TRAINER.FEDMGP.SLOT_AGGREGATION True
```

### 预期观察

如果 SlotAgg 优于 RankAgg，说明当前按 rank 聚合确实存在语义混合问题：

```text
rank 相同不等于语义相近
```

如果 SlotAgg 没有提升，可能说明 FedMGP 中 global prompt slot 更像“共享程度层级”而不是“固定语义槽位”，此时按 rank 聚合反而更稳定。

## 联邦聚合选择

### 选择 1：聚合 Text Prompt、Base Visual Prompt 和 MetaNet

客户端上传：

```text
p_t,j
p_v,j^base
MetaNet parameters
```

这是最完整的版本，但会略微增加通信量。

### 选择 2：MetaNet 全局共享，Prompt 仍动态聚合

客户端上传：

```text
p_t,j
p_v,j^base
```

MetaNet 参数可以单独用 FedAvg 聚合，也可以作为全局共享模块。这个版本实现更简单，也更可能稳定。

### 选择 3：移除 Base Visual Prompt

直接使用：

```text
p_v,j = MetaNet(q_t,j)
```

这个版本参数更少，但风险更高，因为视觉分支完全依赖文本分支生成的提示。

推荐第一版实现：**选择 2 + 残差视觉提示**。

## 消融实验计划

推荐实验：

```text
Baseline: 原始 FedMGP
GroupBind: 组内绑定匹配，其他不变
SlotAgg: 按最匹配 global slot 聚合与分发
GroupBind-SlotAgg: 组内绑定后按 global group slot 聚合与分发
TCVP-A: Pooling condition，不加 L_align
TCVP-A+: Pooling condition + L_align
TCVP-B: H_l[last_prompt_token] condition，不加 L_align
TCVP-B+: H_l[last_prompt_token] condition + L_align
```

额外消融：

```text
alpha: 0.05, 0.1, 0.2
MetaNet hidden dim: 64, 128, 256
q_t,j 来源: mean pool, last layer 的 last prompt token, 最后 2/4 层平均
聚合策略: RankAgg vs. SlotAgg vs. GroupBind vs. GroupBind-SlotAgg
min_bucket_size: 1, 2
```

## 附录备选项：AlignScore 作为质量门控

`AlignScore_j` 不建议作为主方案直接加入动态聚合分数，因为它衡量的是本地跨模态对齐质量，而不是全局共享性：

```text
AlignScore_j = mean_b cos(z_v,b^j, z_t,y_b^j)
```

它回答的问题是：

```text
第 j 组 prompt 在本地数据上，图像特征是否靠近正确类别文本特征
```

而 FedMGP 动态聚合主要回答的是：

```text
第 j 组 prompt 是否更接近全局共享 prompt
```

因此不推荐直接使用：

```text
S_j = sim(P_j^local, P_j^global) + beta_a * AlignScore_j
```

这种线性相加会混合两个不同维度，可能把本地对齐很好但不具备全局共享性的 prompt group 聚合进全局模型。

更谨慎的备选用法是把 `AlignScore_j` 放在附录中，作为质量门控或后续探索：

```text
1. 先按 global similarity 得到候选 group
2. 仅剔除 AlignScore_j 明显过低的 group
3. 仍在剩余 group 中按原 FedMGP similarity 进行采样或 top-k 选择
```

也可以探索跨客户端统计：

```text
MeanAlign_j = mean_c AlignScore_c,j
VarAlign_j = var_c AlignScore_c,j
```

只有当某个 group 在多个客户端上都具有较高且稳定的对齐质量时，才把它视为可能有全局价值的功能性信号。

附录定位：

```text
AlignScore 不是 sharedness score
AlignScore 可以作为 functionality / quality-control score
```

## 实现备注

可能涉及的文件：

```text
trainers/fedmgp.py
federated_core/trainers/fedmgp_learner.py
federated_core/fed_utils.py
configs/trainers/FedMGP/base2novel_vit_b16.yaml
```

对于方案 A，大部分改动应该可以集中在：

```text
trainers/fedmgp.py
```

对于方案 B，需要修改：

```text
TextEncoder.forward()
```

使其可以返回所选 text transformer 层中的 prompt-token hidden states。

需要避免 class-specific visual prompt：

```text
p_v,j,k = MetaNet(z_t,k^j)
```

因为它会导致类别相关的图像编码，推理成本过高。
