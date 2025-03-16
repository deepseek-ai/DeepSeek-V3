1. Introduction
===============

In recent years, Large Language Models (LLMs) have been undergoing rapid iteration and
evolution (Anthropic, 2024; Google, 2024; OpenAI, 2024a), progressively diminishing the gap towards
Artificial General Intelligence (AGI). Beyond closed-source models, open-source models,
including DeepSeek series (DeepSeek-AI, 2024a,b,c; Guo et al., 2024), LLaMA series (AI\@Meta,
2024a,b; Touvron et al., 2023a,b), Qwen series (Qwen, 2023, 2024a,b), and Mistral series (Jiang
et al., 2023; Mistral, 2024), are also making significant strides, endeavoring to close the gap with
their closed-source counterparts. To further push the boundaries of open-source model capabilities,
we scale up our models and introduce DeepSeek-V3, a large Mixture-of-Experts (MoE)
model with 671B parameters, of which 37B are activated for each token.

With a forward-looking perspective, we consistently strive for strong model performance
and economical costs. Therefore, in terms of architecture, DeepSeek-V3 still adopts Multi-head
Latent Attention (MLA) (DeepSeek-AI, 2024c) for efficient inference and DeepSeekMoE (Dai
et al., 2024) for cost-effective training. These two architectures have been validated in DeepSeek-
V2 (DeepSeek-AI, 2024c), demonstrating their capability to maintain robust model performance
while achieving efficient training and inference. Beyond the basic architecture, we implement
two additional strategies to further enhance the model capabilities. Firstly, DeepSeek-V3 pioneers
an auxiliary-loss-free strategy (Wang et al., 2024a) for load balancing, with the aim of
minimizing the adverse impact on model performance that arises from the effort to encourage
load balancing. Secondly, DeepSeek-V3 employs a multi-token prediction training objective,
which we have observed to enhance the overall performance on evaluation benchmarks.

In order to achieve efficient training, we support the FP8 mixed precision training and
implement comprehensive optimizations for the training framework. Low-precision training
has emerged as a promising solution for efficient training (Dettmers et al., 2022; Kalamkar et al.,
2019; Narang et al., 2017; Peng et al., 2023b), its evolution being closely tied to advancements in
hardware capabilities (Luo et al., 2024; Micikevicius et al., 2022; Rouhani et al., 2023a). In this
work, we introduce an FP8 mixed precision training framework and, for the first time, validate
its effectiveness on an extremely large-scale model. Through the support for FP8 computation
and storage, we achieve both accelerated training and reduced GPU memory usage. As for
the training framework, we design the DualPipe algorithm for efficient pipeline parallelism,
which has fewer pipeline bubbles and hides most of the communication during training through
computation-communication overlap. This overlap ensures that, as the model further scales up,
as long as we maintain a constant computation-to-communication ratio, we can still employ
fine-grained experts across nodes while achieving a near-zero all-to-all communication overhead.
In addition, we also develop efficient cross-node all-to-all communication kernels to fully utilize
InfiniBand (IB) and NVLink bandwidths. Furthermore, we meticulously optimize the memory
footprint, making it possible to train DeepSeek-V3 without using costly tensor parallelism.
Combining these efforts, we achieve high training efficiency.

During pre-training, we train DeepSeek-V3 on 14.8T high-quality and diverse tokens. The
pre-training process is remarkably stable. Throughout the entire training process, we did not
encounter any irrecoverable loss spikes or have to roll back. Next, we conduct a two-stage
context length extension for DeepSeek-V3. In the first stage, the maximum context length is
extended to 32K, and in the second stage, it is further extended to 128K. Following this, we
conduct post-training, including Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)
on the base model of DeepSeek-V3, to align it with human preferences and further unlock its
potential. During the post-training stage, we distill the reasoning capability from the DeepSeek-
R1 series of models, and meanwhile carefully maintain the balance between model accuracy
and generation length.

.. raw:: html

    <div align="center">
    <table border="1" cellpadding="0" cellspacing="0">
    <tbody>
    <tr>
    <th align="left">Training Costs</th>
    <th>Pre-Training</th>
    <th>Context Extension</th>
    <th>Post-Training</th>
    <th>Total</th>
    </tr>
    <tr>
    <td align="left">in H800 GPU Hours</td>
    <td align="center">2664K</td>
    <td align="center">119K</td>
    <td align="center">5K</td>
    <td align="center">2788K</td>
    </tr>
    <tr>
    <td align="left">in USD</td>
    <td align="center">$5.328M</td>
    <td align="center">$0.238M</td>
    <td align="center">$0.01M</td>
    <td align="center">$5.576M</td>
    </tr>
    </tbody>
    </table>
    Table 1 | Training costs of DeepSeek-V3, assuming the rental price of H800 is $2 per GPU hour.
    </div>
    <br>

We evaluate DeepSeek-V3 on a comprehensive array of benchmarks. Despite its economical
training costs, comprehensive evaluations reveal that DeepSeek-V3-Base has emerged as the
strongest open-source base model currently available, especially in code and math. Its chat
version also outperforms other open-source models and achieves performance comparable to
leading closed-source models, including GPT-4o and Claude-3.5-Sonnet, on a series of standard
and open-ended benchmarks.

Lastly, we emphasize again the economical training costs of DeepSeek-V3, summarized in
Table 1, achieved through our optimized co-design of algorithms, frameworks, and hardware.
During the pre-training stage, training DeepSeek-V3 on each trillion tokens requires only 180K
H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs. Consequently, our pre-
training stage is completed in less than two months and costs 2664K GPU hours. Combined
with 119K GPU hours for the context length extension and 5K GPU hours for post-training,
DeepSeek-V3 costs only 2.788M GPU hours for its full training. Assuming the rental price of
the H800 GPU is $2 per GPU hour, our total training costs amount to only $5.576M. Note that
the aforementioned costs include only the official training of DeepSeek-V3, excluding the costs
associated with prior research and ablation experiments on architectures, algorithms, or data.
Our main contribution includes:

**Architecture: Innovative Load Balancing Strategy and Training Objective**

* On top of the efficient architecture of DeepSeek-V2, we pioneer an auxiliary-loss-free
  strategy for load balancing, which minimizes the performance degradation that arises
  from encouraging load balancing.
* We investigate a Multi-Token Prediction (MTP) objective and prove it beneficial to model
  performance. It can also be used for speculative decoding for inference acceleration.

**Pre-Training: Towards Ultimate Training Efficiency**

* We design an FP8 mixed precision training framework and, for the first time, validate the
  feasibility and effectiveness of FP8 training on an extremely large-scale model.
* Through the co-design of algorithms, frameworks, and hardware, we overcome the
  communication bottleneck in cross-node MoE training, achieving near-full computation-
  communication overlap. This significantly enhances our training efficiency and reduces the
  training costs, enabling us to further scale up the model size without additional overhead.
* At an economical cost of only 2.664M H800 GPU hours, we complete the pre-training of
  DeepSeek-V3 on 14.8T tokens, producing the currently strongest open-source base model.
  The subsequent training stages after pre-training require only 0.1M GPU hours.

**Post-Training: Knowledge Distillation from DeepSeek-R1**

* We introduce an innovative methodology to distill reasoning capabilities from the long-
  Chain-of-Thought (CoT) model, specifically from one of the DeepSeek R1 series models,
  into standard LLMs, particularly DeepSeek-V3. Our pipeline elegantly incorporates the
  verification and reflection patterns of R1 into DeepSeek-V3 and notably improves its
  reasoning performance. Meanwhile, we also maintain control over the output style and
  length of DeepSeek-V3.

**Summary of Core Evaluation Results**

* **Knowledge:** (1) On educational benchmarks such as MMLU, MMLU-Pro, and GPQA,
  DeepSeek-V3 outperforms all other open-source models, achieving 88.5 on MMLU, 75.9
  on MMLU-Pro, and 59.1 on GPQA. Its performance is comparable to leading closed-source
  models like GPT-4o and Claude-Sonnet-3.5, narrowing the gap between open-source
  and closed-source models in this domain. (2) For factuality benchmarks, DeepSeek-V3
  demonstrates superior performance among open-source models on both SimpleQA and
  Chinese SimpleQA. While it trails behind GPT-4o and Claude-Sonnet-3.5 in English factual
  knowledge (SimpleQA), it surpasses these models in Chinese factual knowledge (Chinese
  SimpleQA), highlighting its strength in Chinese factual knowledge.
* **Code, Math, and Reasoning:** (1) DeepSeek-V3 achieves state-of-the-art performance on
  math-related benchmarks among all non-long-CoT open-source and closed-source models.
  Notably, it even outperforms o1-preview on specific benchmarks, such as MATH-500,
  demonstrating its robust mathematical reasoning capabilities. (2) On coding-related tasks,
  DeepSeek-V3 emerges as the top-performing model for coding competition benchmarks,
  such as LiveCodeBench, solidifying its position as the leading model in this domain. For
  engineering-related tasks, while DeepSeek-V3 performs slightly below Claude-Sonnet-3.5,
  it still outpaces all other models by a significant margin, demonstrating its competitiveness
  across diverse technical benchmarks.

In the remainder of this paper, we first present a detailed exposition of our DeepSeek-V3
model architecture (Section 2). Subsequently, we introduce our infrastructures, encompassing
our compute clusters, the training framework, the support for FP8 training, the inference
deployment strategy, and our suggestions on future hardware design. Next, we describe our
pre-training process, including the construction of training data, hyper-parameter settings, long-
context extension techniques, the associated evaluations, as well as some discussions (Section 4).
Thereafter, we discuss our efforts on post-training, which include Supervised Fine-Tuning (SFT),
Reinforcement Learning (RL), the corresponding evaluations, and discussions (Section 5). Lastly,
we conclude this work, discuss existing limitations of DeepSeek-V3, and propose potential
directions for future research (Section 6).
