# 核心代码阅读

## vit.py
代码的整体结构：
```bash

 整体结构

  PatchEmbed (来自 timm)     → 将图像切成 16×16 的 patch 并嵌入为向量
    ↓
  + cls_token + pos_embed    → 加分类 token 和位置编码
    ↓
  × N 个 Block                → Transformer 编码器层
    │  ├── Attention (Multi-Head Self-Attention)
    │  └── Mlp (两层全连接 + GELU)
    ↓
  LayerNorm                   → 最终归一化
    ↓
  输出特征矩阵 [B, N+1, 768]

```
VisionTransformer类：
1. 首先利用patch_embed，以16*16将图像切割为不同的patch，其维度变化为[B, 3, 224, 224]-->[B, 196, 768]
2. 建立一个可学习的向量cls_token，代表整张图像的汇总，然后拼接，共197个token（加在开头）
3. position embedding
```python
    x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
```
将位置编码加进x
4. transformer block，进入block。
5. 最终归一化

来自deepseek v4 pro的例子：
```bash
 forward() —— 数据如何流动：

  def forward(self, x, register_blk=-1):
      B = x.shape[0]  # batch size

      # Step 1: Patch Embedding
      x = self.patch_embed(x)
      # [B, 3, 224, 224] → [B, 196, 768]
      # 本质上是一个 stride=16 的 Conv2d，把每个 16×16 区域映射为 768 维向量

  这步发生了什么？ 图像被切分成 14×14 个不重叠的 16×16 块，每个块被一个卷积层（kernel=16, stride=16）映射为 768
  维向量。就像把图像变成了一句话，每个 patch 是一个"单词"。

      # Step 2: 拼接 cls_token
      cls_tokens = self.cls_token.expand(B, -1, -1)  # [1,1,768] → [B,1,768]
      x = torch.cat((cls_tokens, x), dim=1)
      # [B, 196, 768] → [B, 197, 768]
      # 在所有 patch 前面加一个 class token

      # Step 3: 加位置编码
      x = x + self.pos_embed[:, :x.size(1), :]
      # 每个位置 (0~196) 加一个独特的可学习向量
      # 这样 Transformer 才知道哪个 patch 在图像的哪个位置

      # Step 4: Dropout
      x = self.pos_drop(x)

  为什么需要 cls_token？ Transformer 本身没有"汇总"操作。cls_token 作为一个特殊的可学习向量，在自注意力过程中和所有
  patch 交互，最终它的输出向量（x[:, 0, :]）就代表了整张图像的信息——类似 BERT 的 [CLS]。

  为什么需要位置编码？ 自注意力机制本身是置换不变的——打乱 patch
  顺序，注意力输出不变（只是顺序也跟着变）。但图像有空间结构！位置编码给每个 patch 注入了它在图像网格中的坐标信息。

      # Step 5: 经过所有 Transformer Block
      for i, blk in enumerate(self.blocks):
          x = blk(x, register_blk==i)
      # 12 层自注意力，每层 197 个 token 互相看

      # Step 6: 最终 LayerNorm
      x = self.norm(x)

      return x   # [B, 197, 768]

  输出解释：
  - x[:, 0, :] — cls_token 的输出，代表整张图像的全局特征（BLIP 里用这个做对比学习和匹配）
  - x[:, 1:, :] — 196 个 patch 的输出，代表局部区域特征（BLIP 里用这些做 cross-attention）
  ```

## med.py
BLIP 的 MED = Mixture of Encoder-Decoder：同一个 BERT 结构既能当编码器（双向注意力），又能当解码器（因果注意力 + cross-attention 看图像）。
- BertEmbeddings：与标准BERT一致。

- BertSelfAttention：


