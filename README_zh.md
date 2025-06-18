<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/"><img alt="主页"
    src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true"/></a>
  <a href="https://chat.deepseek.com/"><img alt="聊天"
    src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V3-536af5?color=536af5&logoColor=white"/></a>
  <a href="https://huggingface.co/deepseek-ai"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <br>
  <a href="https://discord.gg/Tc7c45Zzu5"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true"><img alt="微信"
    src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://twitter.com/deepseek_ai"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE"><img alt="代码许可证"
    src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53"/></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL"><img alt="模型许可证"
    src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53"/></a>
  <br>
  <a href="DeepSeek_V3.pdf"><b>论文链接</b>👁️</a>
</div>

## 目录

1. [简介](#1-简介)
2. [模型概述](#2-模型概述)
3. [模型下载](#3-模型下载)
4. [评估结果](#4-评估结果)
5. [聊天网站与 API 平台](#5-聊天网站与-api-平台)
6. [如何在本地运行](#6-如何在本地运行)
7. [许可证](#7-许可证)
8. [引用](#8-引用)
9. [联系方式](#9-联系方式)

## 1. 简介

我们推出了 DeepSeek-V3，这是一个强大的混合专家（MoE）语言模型，总参数量达 6710 亿，每个 Token 会激活 370 亿参数。
为了实现高效推理和经济高效的训练，DeepSeek-V3 采用了多头潜在注意力（MLA）和 DeepSeekMoE 架构，这些在 DeepSeek-V2 中已得到充分验证。
此外，DeepSeek-V3 开创了一种无辅助损失的负载均衡策略，并设定了多 Token 预测训练目标，以实现更强的性能。
我们在 14.8 万亿个多样且高质量的 Token 上对 DeepSeek-V3 进行了预训练，随后进行了监督微调和强化学习阶段，以充分发挥其能力。
综合评估显示，DeepSeek-V3 优于其他开源模型，并达到了与领先的闭源模型相当的性能。
尽管性能卓越，DeepSeek-V3 在完整训练过程中仅需 278.8 万个 H800 GPU 小时。
此外，其训练过程非常稳定。在整个训练过程中，我们没有遇到任何无法恢复的损失激增情况，也没有进行任何回滚操作。

<p align="center">
  <img width="80%" src="figures/benchmark.png">
</p>

## 2. 模型概述

---

**架构：创新的负载均衡策略和训练目标**

- 在 DeepSeek-V2 的高效架构基础上，我们开创了一种无辅助损失的负载均衡策略，该策略将因鼓励负载均衡而导致的性能下降降至最低。
- 我们研究了多 Token 预测（MTP）目标，并证明它对模型性能有益。
  它还可用于推测性解码，以加速推理。

---

**预训练：追求极致的训练效率**

- 我们设计了一个 FP8 混合精度训练框架，并首次验证了在超大规模模型上进行 FP8 训练的可行性和有效性。
- 通过算法、框架和硬件的协同设计，我们克服了跨节点 MoE 训练中的通信瓶颈，几乎实现了计算与通信的完全重叠。
  这显著提高了我们的训练效率，降低了训练成本，使我们能够在不增加额外开销的情况下进一步扩大模型规模。
- 仅以 266.4 万个 H800 GPU 小时的经济成本，我们就在 14.8 万亿个 Token 上完成了 DeepSeek-V3 的预训练，
  生成了目前最强的开源基础模型。预训练后的后续训练阶段仅需 10 万个 GPU 小时。

---

**训练后：从 DeepSeek-R1 进行知识蒸馏**

- 我们引入了一种创新方法，将长思维链（CoT）模型（特别是 DeepSeek R1 系列模型之一）的推理能力蒸馏到标准大语言模型中，
  尤其是 DeepSeek-V3。我们的流程巧妙地将 R1 的验证和反思模式融入到 DeepSeek-V3 中，并显著提高了其推理性能。
  同时，我们也对 DeepSeek-V3 的输出风格和长度进行了控制。

---

## 3. 模型下载

<div align="center">

| **模型** | **总参数数量** | **激活参数数量** | **上下文长度** | **下载地址** |
| :-----: | :----------: | :------------: | :-----------: | :---------: |
| DeepSeek-V3-Base | 6710 亿 | 370 亿 | 128K | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) |
| DeepSeek-V3 | 6710 亿 | 370 亿 | 128K | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) |

</div>

> [!NOTE]
> Hugging Face 上 DeepSeek-V3 模型的总大小为 6850 亿，其中包括 6710 亿的主模型权重和 140 亿的多 Token 预测（MTP）模块权重。

为了确保最佳性能和灵活性，我们与开源社区和硬件供应商合作，提供了多种在本地运行模型的方式。
有关详细步骤指南，请查看第 6 节：[如何在本地运行](#6-如何在本地运行)。

对于希望深入研究的开发人员，我们建议查阅[README_WEIGHTS.md](./README_WEIGHTS.md)，
了解主模型权重和多 Token 预测（MTP）模块的详细信息。请注意，目前社区正在积极开发 MTP 支持功能，我们欢迎你的贡献和反馈。

## 4. 评估结果

### 基础模型

#### 标准基准测试

<div align="center">

| | 基准测试（指标） | 样本数量 | DeepSeek-V2 | Qwen2.5 72B | LLaMA3.1 405B | DeepSeek-V3 |
| -- | ---------- | ------- | ----------- | ----------- | ------------- | ----------- |
| | 架构 | - | MoE | 密集型 | 密集型 | MoE |
| | 激活参数数量 | - | 210 亿 | 720 亿 | 4050 亿 | 370 亿 |
| | 总参数数量 | - | 2360 亿 | 720 亿 | 4050 亿 | 6710 亿 |
| 英语 | Pile-test（每 Token 比特数） | - | 0.606 | 0.638 | **0.542** | 0.548 |
| | BBH（精确匹配率） | 3 样本 | 78.8 | 79.8 | 82.9 | **87.5** |
| | MMLU（准确率） | 5 样本 | 78.4 | 85.0 | 84.4 | **87.1** |
| | MMLU-Redux（准确率） | 5 样本 | 75.6 | 83.2 | 81.3 | **86.2** |
| | MMLU-Pro（准确率） | 5 样本 | 51.4 | 58.3 | 52.8 | **64.4** |
| | DROP（F1 值） | 3 样本 | 80.4 | 80.6 | 86.0 | **89.0** |
| | ARC-Easy（准确率） | 25 样本 | 97.6 | 98.4 | 98.4 | **98.9** |
| | ARC-Challenge（准确率） | 25 样本 | 92.2 | 94.5 | **95.3** | **95.3** |
| | HellaSwag（准确率） | 10 样本 | 87.1 | 84.8 | **89.2** | 88.9 |
| | PIQA（准确率） | 0 样本 | 83.9 | 82.6 | **85.9** | 84.7 |
| | WinoGrande（准确率） | 5 样本 | **86.3** | 82.3 | 85.2 | 84.9 |
| | RACE-Middle（准确率） | 5 样本 | 73.1 | 68.1 | **74.2** | 67.1 |
| | RACE-High（准确率） | 5 样本 | 52.6 | 50.3 | **56.8** | 51.3 |
| | TriviaQA（精确匹配率） | 5 样本 | 80.0 | 71.9 | 82.7 | **82.9** |
| | NaturalQuestions（精确匹配率） | 5 样本 | 38.6 | 33.2 | **41.5** | 40.0 |
| | AGIEval（准确率） | 0 样本 | 57.5 | 75.8 | 60.6 | **79.6** |
| 代码 | HumanEval（首次通过率） | 0 样本 | 43.3 | 53.0 | 54.9 | **65.2** |
| | MBPP（首次通过率） | 3 样本 | 65.0 | 72.6 | 68.4 | **75.4** |
| | LiveCodeBench-Base（首次通过率） | 3 样本 | 11.6 | 12.9 | 15.5 | **19.4** |
| | CRUXEval-I（准确率） | 2 样本 | 52.5 | 59.1 | 58.5 | **67.3** |
| | CRUXEval-O（准确率） | 2 样本 | 49.8 | 59.9 | 59.9 | **69.8** |
| 数学 | GSM8K（精确匹配率） | 8 样本 | 81.6 | 88.3 | 83.5 | **89.3** |
| | MATH（精确匹配率） | 4 样本 | 43.4 | 54.4 | 49.0 | **61.6** |
| | MGSM（精确匹配率） | 8 样本 | 63.6 | 76.2 | 69.9 | **79.8** |
| | CMath（精确匹配率） | 3 样本 | 78.7 | 84.5 | 77.3 | **90.7** |
| 中文 | CLUEWSC（精确匹配率） | 5 样本 | 82.0 | 82.5 | **83.0** | 82.7 |
| | C-Eval（准确率） | 5 样本 | 81.4 | 89.2 | 72.5 | **90.1** |
| | CMMLU（准确率） | 5 样本 | 84.0 | **89.5** | 73.7 | 88.8 |
| | CMRC（精确匹配率） | 1 样本 | **77.4** | 75.8 | 76.0 | 76.3 |
| | C3（准确率） | 0 样本 | 77.4 | 76.7 | **79.7** | 78.6 |
| | CCPM（准确率） | 0 样本 | **93.0** | 88.5 | 78.6 | 92.0 |
| 多语言 | MMMLU-非英语（准确率） | 5 样本 | 64.0 | 74.8 | 73.8 | **79.4** |

</div>

> [!NOTE]
> 最佳结果以粗体显示。差距不超过 0.3 的分数被认为处于同一水平。DeepSeek-V3 在大多数基准测试中表现最佳，尤其是在数学和代码任务上。
> 有关更多评估细节，请查阅我们的论文。

#### 上下文窗口

<p align="center">
  <img width="80%" src="figures/niah.png">
</p>

在“大海捞针”（NIAH）测试中的评估结果。DeepSeek-V3 在高达 **128K** 的所有上下文窗口长度上均表现良好。

### 聊天模型

#### 标准基准测试（参数大于 670 亿的模型）

<div align="center">

| | **基准测试（指标）** | **DeepSeek V2-0506** | **DeepSeek V2.5-0905** | **Qwen2.5 72B-Inst.** | **Llama3.1 405B-Inst.** | **Claude-3.5-Sonnet-1022** | **GPT-4o 0513** | **DeepSeek V3** |
| ---- | ------ | ------- | ------------- | -------- | ----- | ------- | ----- | ------ |
| | 架构 | MoE | MoE | 密集型 | 密集型 | - | - | MoE |
| | 激活参数数量 | 210 亿 | 210 亿 | 720 亿 | 4050 亿 | - | - | 370 亿 |
| | 总参数数量 | 2360 亿 | 2360 亿 | 720 亿 | 4050 亿 | - | - | 6710 亿 |
| 英语 | MMLU（精确匹配率） | 78.2 | 80.6 | 85.3 | **88.6** | **88.3** | 87.2 | **88.5** |
| | MMLU-Redux（精确匹配率） | 77.9 | 80.3 | 85.6 | 86.2 | **88.9** | 88.0 | **89.1** |
| | MMLU-Pro（精确匹配率） | 58.5 | 66.2 | 71.6 | 73.3 | **78.0** | 72.6 | 75.9 |
| | DROP（3 样本 F1 值） | 83.0 | 87.8 | 76.7 | 88.7 | 88.3 | 83.7 | **91.6** |
| | IF-Eval（严格提示） | 57.7 | 80.6 | 84.1 | 86.0 | **86.5** | 84.3 | 86.1 |
| | GPQA-Diamond（首次通过率） | 35.3 | 41.3 | 49.0 | 51.1 | **65.0** | 49.9 | 59.1 |
| | SimpleQA（正确） | 9.0 | 10.2 | 9.1 | 17.1 | 28.4 | **38.2** | 24.9 |
| | FRAMES（准确率） | 66.9 | 65.4 | 69.8 | 70.0 | 72.5 | **80.5** | 73.3 |
| | LongBench v2（准确率） | 31.6 | 35.4 | 39.4 | 36.1 | 41.0 | 48.1 | **48.7** |
| 代码 | HumanEval-Mul（首次通过率） | 69.3 | 77.4 | 77.3 | 77.2 | 81.7 | 80.5 | **82.6** |
| | LiveCodeBench（首次通过率-思维链） | 18.8 | 29.2 | 31.1 | 28.4 | 36.3 | 33.4 | **40.5** |
| | LiveCodeBench（首次通过率） | 20.3 | 28.4 | 28.7 | 30.1 | 32.8 | 34.2 | **37.6** |
| | Codeforces（百分位数） | 17.5 | 35.6 | 24.8 | 25.3 | 20.3 | 23.6 | **51.6** |
| | SWE Verified（已解决） | - | 22.6 | 23.8 | 24.5 | **50.8** | 38.8 | 42.0 |
| | Aider-Edit（准确率） | 60.3 | 71.6 | 65.4 | 63.9 | **84.2** | 72.9 | 79.7 |
| | Aider-Polyglot（准确率） | - | 18.2 | 7.6 | 5.8 | 45.3 | 16.0 | **49.6** |
| 数学 | AIME 2024（首次通过率） | 4.6 | 16.7 | 23.3 | 23.3 | 16.0 | 9.3 | **39.2** |
| | MATH-500（精确匹配率） | 56.3 | 74.7 | 80.0 | 73.8 | 78.3 | 74.6 | **90.2** |
| | CNMO 2024（首次通过率） | 2.8 | 10.8 | 15.9 | 6.8 | 13.1 | 10.8 | **43.2** |
| 中文 | CLUEWSC（精确匹配率） | 89.9 | 90.4 | **91.4** | 84.7 | 85.4 | 87.9 | 90.9 |
| | C-Eval（精确匹配率） | 78.6 | 79.5 | 86.1 | 61.5 | 76.7 | 76.0 | **86.5** |
| | C-SimpleQA（正确） | 48.5 | 54.1 | 48.4 | 50.4 | 51.3 | 59.3 | **64.8** |

</div>

> [!NOTE]
> 所有模型均在输出长度限制为 8K 的配置下进行评估。对于样本数量少于 1000 的基准测试，
> 会使用不同的温度设置多次测试，以得出可靠的最终结果。DeepSeek-V3 是性能最佳的开源模型，
> 并且与前沿的闭源模型相比也展现出了有竞争力的性能。

#### 开放式生成评估

<div align="center">

| 模型 | Arena-Hard | AlpacaEval 2.0 |
| --- | ---------- | -------------- |
| DeepSeek-V2.5-0905 | 76.2 | 50.5 |
| Qwen2.5-72B-Instruct | 81.2 | 49.1 |
| LLaMA-3.1 405B | 69.3 | 40.5 |
| GPT-4o-0513 | 80.4 | 51.1 |
| Claude-Sonnet-3.5-1022 | 85.2 | 52.0 |
| DeepSeek-V3 | **85.5** | **70.0** |

</div>

> [!NOTE]
> 英语开放式对话评估。对于 AlpacaEval 2.0，我们使用长度控制的胜率作为指标。

## 5. 聊天网站与 API 平台

你可以在 DeepSeek 的官方网站上与 DeepSeek-V3 聊天：[chat.deepseek.com](https://chat.deepseek.com/sign_in)

我们还在 DeepSeek 平台上提供了与 OpenAI 兼容的 API：[platform.deepseek.com](https://platform.deepseek.com/)

## 6. 如何在本地运行

DeepSeek-V3 可以使用以下硬件和开源社区软件在本地部署：

1. **DeepSeek-Infer Demo**：我们提供了一个简单轻量的用于 FP8 和 BF16 推理的演示。
2. **SGLang**：完全支持 DeepSeek-V3 模型的 BF16 和 FP8 推理模式，多
   Token 预测功能[即将推出](https://github.com/sgl-project/sglang/issues/2591)。
3. **LMDeploy**：支持在本地和云端部署中进行高效的 FP8 和 BF16 推理。
4. **TensorRT-LLM**：目前支持 BF16 推理和 INT4/8 量化，即将支持 FP8。
5. **vLLM**：支持 DeepSeek-V3 模型的 FP8 和 BF16 模式，用于张量并行和流水线并行。
6. **AMD GPU**：通过 SGLang 在 AMD GPU 上以 BF16 和 FP8 模式运行 DeepSeek-V3 模型。
7. **华为昇腾 NPU**：支持在华为昇腾设备上运行 DeepSeek-V3。

由于我们的框架原生采用了 FP8 训练，我们仅提供 FP8 权重。如果你需要 BF16 权重进行实验，可以使用提供的转换脚本进行转换。

以下是将 FP8 权重转换为 BF16 的示例：

```shell
cd inference
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights
```

> [!NOTE]
> 目前尚未直接支持 Hugging Face 的 Transformers。

### 6.1 使用 DeepSeek-Infer Demo 进行推理（仅为示例）

#### 系统要求

> [!NOTE]
> 仅支持带有 Python 3.10 的 Linux 系统。不支持 Mac 和 Windows 系统。

依赖项：

```pip-requirements
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

#### 模型权重与演示代码准备

首先，克隆我们的 DeepSeek-V3 GitHub 仓库：

```shell
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
```

进入 `inference` 文件夹并安装 `requirements.txt` 中列出的依赖项。最简单的方法是使用像
`conda` 或 `uv` 这样的包管理器创建一个新的虚拟环境并安装依赖项。

```shell
cd DeepSeek-V3/inference
pip install -r requirements.txt
```

从 Hugging Face 下载模型权重，并将其放入 `/path/to/DeepSeek-V3` 文件夹中。

#### 模型权重转换

将 Hugging Face 模型权重转换为特定格式：

```shell
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```

#### 运行

然后你可以与 DeepSeek-V3 聊天：

```shell
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
```

或者对给定文件进行批量推理：

```shell
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --input-file $FILE
```

### 6.2 使用 SGLang 进行推理（推荐）

[SGLang](https://github.com/sgl-project/sglang) 目前支持
[MLA 优化](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations)、
[DP 注意力](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models)、
FP8（W8A8）、FP8 KV 缓存和 Torch Compile，在开源框架中提供了最先进的延迟和吞吐量性能。

值得注意的是，[SGLang v0.4.1](https://github.com/sgl-project/sglang/releases/tag/v0.4.1)
完全支持在 **NVIDIA 和 AMD GPU** 上运行 DeepSeek-V3，使其成为一个高度通用和强大的解决方案。

SGLang 还支持[多节点张量并行](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208)，使你能够在多台通过网络连接的机器上运行此模型。

多 Token 预测（MTP）正在开发中，进展可在[优化计划](https://github.com/sgl-project/sglang/issues/2591)中跟踪。

以下是 SGLang 团队的启动说明：
https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

### 6.3 使用 LMDeploy 进行推理（推荐）

[LMDeploy](https://github.com/InternLM/lmdeploy)
是一个为大语言模型量身定制的灵活且高性能的推理和服务框架，现在支持 DeepSeek-V3。
它提供离线流水线处理和在线部署功能，无缝集成基于 PyTorch 的工作流程。

有关使用 LMDeploy 运行 DeepSeek-V3 的详细分步说明，请参阅此处：
https://github.com/InternLM/lmdeploy/issues/2960

### 6.4 使用 TRT-LLM 进行推理（推荐）

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
现在支持 DeepSeek-V3 模型，提供 BF16 和 INT4/INT8 仅权重的精度选项。
对 FP8 的支持目前正在进行中，即将发布。你可以通过以下链接访问专门为支持 DeepSeek-V3
的 TRTLLM 自定义分支，直接体验新功能：
https://github.com/NVIDIA/TensorRT-LLM/tree/deepseek/examples/deepseek_v3

### 6.5 使用 vLLM 进行推理（推荐）

[vLLM](https://github.com/vllm-project/vllm) v0.6.6 支持在 NVIDIA 和 AMD GPU 上对 DeepSeek-V3
进行 FP8 和 BF16 模式的推理。除了标准技术外，vLLM 还提供**流水线并行**功能，允许你在多台通过网络连接的机器上运行此模型。
有关详细指南，请参阅 [vLLM 说明](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)。
也请随时关注[增强计划](https://github.com/vllm-project/vllm/issues/11539)。

### 6.6 在 AMD GPU 上推荐的推理功能

与 AMD 团队合作，我们通过 SGLang 实现了对 AMD GPU 的首日支持，完全兼容 FP8 和 BF16 精度。
有关详细指南，请参阅 [SGLang 说明](#63-使用-lmdeploy-进行推理推荐)。

### 6.7 在华为昇腾 NPU 上推荐的推理功能

华为昇腾社区的 [MindIE](https://www.hiascend.com/en/software/mindie) 框架已成功适配了 DeepSeek-V3 的 BF16 版本。
有关在昇腾 NPU 上的分步指南，请按照[此处的说明](https://modelers.cn/models/MindIE/deepseekv3)进行操作。

## 7. 许可证

此代码仓库根据 [MIT 许可证](LICENSE-CODE)进行授权。使用 DeepSeek-V3 基础/聊天模型需遵循[模型许可证](LICENSE-MODEL)。
DeepSeek-V3 系列（包括基础版和聊天版）支持商业使用。

## 8. 引用

```
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}
```

## 9. 联系方式

如果你有任何问题，请提出问题或通过 [service@deepseek.com](service@deepseek.com) 与我们联系。
