<p align="center">
  <img src="assets/logo.png" alt="Uni-CoT" width="200"/>
</p>

# Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision

[[Project]](https://github.com/SAIS-FUXI/projects) / [[Report]](technical_report.md) / [[Model]](https://huggingface.co)

[Luozheng Qin](https://scholar.google.com/citations?user=41BWCzkAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup><sup>\*</sup>,
[Jia Gong](https://scholar.google.com/citations?user=ZV-ThegAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup><sup>\*</sup>,
[Yuqing Sun]()<sup>1</sup><sup>\*</sup>,
[Tianjiao Li](https://scholar.google.com/citations?hl=zh-CN&user=so6xMg8AAAAJ)<sup>3</sup>,
[Mengping Yang](https://scholar.google.com/citations?user=yF34LtcAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>,
[Xiaomeng Yang](https://scholar.google.com/citations?hl=zh-CN&user=7evPWQYAAAAJ)<sup>1</sup>,
[Zhiyu Tan](https://scholar.google.com/citations?user=XprTQQ8AAAAJ&hl=en)<sup>1,2</sup><sup>+</sup>,
[Hao Li](https://scholar.google.com/citations?user=pHN-QIwAAAAJ&hl=zh-CN)<sup>1,2</sup><sup>#</sup>,

## Overview
Chain-of-Thought (CoT) reasoning has significantly enhanced LLM performance on complex text tasks by encouraging interpretable, step-by-step problem solving. However, extending this paradigm to multimodal tasks presents new challenges. In vision-language scenarios, human cognition depends on understanding how visual states evolve over time, inferring causality and planning based on object movements, spatial interactions, and transformations, which are critical for physical reasoning, visual planning, and story comprehension.

To bridge this gap, we introduce the Unified Chain-of-Thought (Uni-CoT) framework, designed to empower Multimodal Large Language Models (MLLMs) to perform structured and interpretable reasoning across both text and vision. Uni-CoT first decomposes a given multimodal task into simple, modular steps, and then processes each step either sequentially or in parallel, as illustrated below. Thus, it enables more systematic and scalable reasoning across modalities. 
Specifically, the Uni-CoT reasoning pipeline consists of four key components:
1. **Planning**: Decompose the complex task into a sequence of simpler, manageable subtasks.
2. **Subtask Execution**: Execute each subtask using the unified model with step-by-step reasoning.
3. **Self-Check**: After completing each subtask, perform a validation check to ensure the intermediate result aligns with the intended goal.
4. **Final Result**: Aggregate the validated subtask results to generate the final output.

With these designs, our Uni-CoT framework aims to enable unified large models to tackle a wide range of challenging multimodal applications, including:
* Highly reliable image generation/editing
* Visual planning
* Geometric and physical reasoning


<p align="center">
  <img src="assets/pipeline.png" width="800"/>
</p>

---

## Key Insight

A major challenge in Uni-CoT learning is the heightened complexity introduced by visual reasoning.
Each reasoning step involves not only generating explanatory text but also synthesizing a corresponding image. 
Producing a high-quality image via VAE consumes approximately 4,096 tokens, with an additional 1,369 tokens required for ViT-based representation, totaling nearly 6,000 tokens per step. 
This is a significant increase compared to the 512–1,024 tokens typically needed for text-only reasoning, substantially raising the cost of both training and inference. 
Consequently, when the reasoning chain grows with multiple image-text pairs, the model struggles to converge and generalize effectively, limiting its performance on multimodal tasks.

<p align="center">
  <img src="assets/motivation.png" width="800"/>
</p>

To mitigate the complexity introduced by long multimodal reasoning chains, we reformulate the Uni-CoT process as a Markov Decision Process (MDP), where each step depends solely on the current state. 
Concretely, we model each reasoning step as a discrete MDP node, which only depends on the preceding step and the task instruction. 
This formulation enables the model to focus on learning local transition dynamics between adjacent nodes, rather than capturing dependencies across the entire reasoning chain as shown below. 
Such a design choice significantly reduces computational overhead and improves training efficiency.
<p align="center">
  <img src="assets/mdp_process.png" width="800"/>
</p>

---

## Details

## Uni-CoT MDP Node Design

Each MDP node is defined by the following components:

* **State ($s_t$)**: Current context, refer to last reasoning step, including both text and images.
* **Action ($a_t$)**: A hybrid operation that involves generating editing instructions and performing corresponding image edits.
* **Next State ($s_{t+1}$)**: The updated context resulting from the applied action, including the edited image, a textual summary according to the edited image.
* **Reward ($r_{t+1}$)**: A textual conclusion or scalar score that quantifies the alignment between the outcome and the task objective.

<p align="center">
  <img src="assets/mdp_architecture.png" width="800"/>
</p>
Uni-CoT components that requires loss during training are highlighted in pink.


## Training Strategy

With above design, our training focuses on three core objectives:

* Learning to generate **hybrid actions** (text and image edits) that drive reasoning progression.
* Predicting the **next state summary** given the current state and action.
* Estimating **reward** that reflect task completion and reasoning quality.

---

## Comparison

We compare the proposed MDP-based Uni-CoT (BAGEL-MDP) against the traditional long-chain Uni-CoT reasoning baseline (BAGEL-LC). Both models are trained for 6,000 steps on a dataset of approximately 10,000 samples. Evaluation is conducted on the WISE benchmark, which is specifically designed to assess the reasoning capabilities of Multimodal Large Language Models (MLLMs). As shown below, the MDP-based formulation consistently outperforms the long-chain baseline across all metrics, demonstrating its superior learning efficiency and output quality.

|              | Culture↑ | Time↑   | Space↑  | Biology↑ | Physics↑ | Chemistry↑ | Overall↑ |
|--------------|----------|---------|---------|----------|----------|------------|----------|
| Bagel-base   | 0.76     | **0.69** | <u>0.75</u> | <u>0.65</u>   | <u>0.75</u>   | <u>0.58</u>     | <u>0.70</u>   |
| Bagel-LC     | *0.73*   | *0.67*  | <u>0.75</u> | *0.60*   | <u>0.75</u>   | *0.65*     | *0.70*   |
| **Bagel-MDP**| **0.77** | <u>0.67</u> | <u>0.75</u> | **0.69** | **0.76** | **0.70**   | **0.73** |
