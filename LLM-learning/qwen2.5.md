

智界探索
1. AI agent
2. 多模态大模型qwen2.5 llava
3. 多模态大模型的评估
4. Cot
5. PPO DPO GRPO（RL for VLLM）
6. lora
7. moe
8. 药物重定位

Qwen2.5
相对于qwen2的改进
1. pre-training
   1. 数据量：从7T到18T

2. post-training：
   1. SFT: 1 million样本
   2. Alignment：offline DPO和online GRPO
3. 系列：0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
(相对于Qwen2增加了14B和32B)
付费的基于MoE的模型（Qwen2.5-turbo和Qwen2.5-plus）
1. 更大的size
2. 更好的更多的数据
3. 更好的使用：支持结构化输出，上下文长度2K变为8K，工具调用

1. 数据处理
   1. 数据过滤
   Qwen2-Instruct
   2. 增加math和code的data
   3. 使用高质量的合成数据，数学，代码等
   4. 数据配比，额外增加科学，学术研究等方面的知识
2. 超参数优化
   1. 选取lr
   2. 选取batchsize
3. 长上下文预训练
   1. 两个阶段：4096扩展到32768
   2. 扩展上下文长度策略：YARN和DCA
4. post-training
   1. SFT阶段使用更加多样化的高质量的SFT数据
   2. alignment阶段分为offline DPO和online GRPO阶段
5. SFT
   1. 从训练语料中提取数据，生成query,再进行过滤
   2. 数学数据
   3. 代码
   4. Instruction-following
   5. Structured Data Understanding
   6. Logic Reasoning
   7. 跨语言
   8. robust system instruction
   9. response filtering
6.  offline DPO
    1.  目的：提升model在数学，代码，instruction following,logical reasoning上的能力
    2.  这几个方面的评估比较复杂
    3.  得到正负样本DPO
7. online GRPO
   1. 目的：提升模型在truthfulness,helpfulness,conciseness等方面的表现
8. 