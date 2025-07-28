<p align="center">
  <img src="assets/logo.png" alt="Uni-CoT" width="480"/>
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

\* equal contribution + project leader # Corresponding author 

<sup>1</sup>Shanghai Academy of AI for Science, <sup>2</sup>Fudan University, <sup>3</sup>Nanyang Technological University

## Overview
Chain-of-Thought (CoT) reasoning has shown remarkable success in boosting the performance of Large Language Models (LLMs) on complex text-based tasks by encouraging interpretable, step-by-step problem solving. However, extending this paradigm to multimodal reasoning introduces new challenges. In particular, human cognition in multimodal tasks—especially those involving vision—relies heavily on understanding how visual states evolve over time. Humans naturally infer causality and plan actions based on visual changes, such as object movements, spatial interactions, and transformations, which are crucial in scenarios like physical reasoning, visual planning, and story understanding.

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
## ✅ To-Do: Uni-CoT Roadmap

A list of planned features and enhancements for the **BATI** framework:

### 🧠 Reasoning Framework
- [✅] Release self-check mechanism  
- [ ] Rlease planning and subtask execution mechanism
- [ ] Develop more fine-grained reasoning strategies  

### 🤖 Training Framework
- [ ] Provide SFT (Supervised Fine-Tuning) framework for multimodal reasoning  
- [ ] Provide RL (Reinforcement Learning) framework for multimodal reasoning  

### 📊 Evaluation & Benchmarking
- [✅] Evaluate Uni-CoT on a reasoning-based text-to-image generation benchmark, wise.
- [ ] Evaluate Uni-CoT on a reasoning-based editing benchmark.
- [ ] Evaluate Uni-CoT on a reasoning-based understanding benchmark.

---

## Preliminary Results for Highly Reliable Image Generation
### Qualitative Results
<p align="center">
  <img src="assets/qualitative_results.png" width="800"/>

### Quantitative Results
|               | Culture↑ | Time↑   | Space↑  | Biology↑ | Physics↑ | Chemistry↑ | Overall↑ |
|---------------|----------|---------|---------|----------|----------|------------|----------|
| Janus         | 0.16     | 0.26    | 0.35    | 0.28     | 0.30     | 0.14       | 0.23     |
| MetaQuery     | 0.56     | 0.55    | 0.62    | 0.49     | 0.63     | 0.41       | 0.55     |
| Bagel         | 0.76     | **0.69** | <u>0.75</u> | 0.65     | <u>0.75</u> | <u>0.58</u>   | 0.70     |
| **Ours**      | **0.77** | <u>0.67</u> | <u>0.75</u> | **0.69** | **0.76** | **0.70**   | **0.73** |
| *GPT4O*       | *0.81*   | *0.71*  | *0.89*  | *0.83*   | *0.79*   | *0.74*     | *0.80*   |

---

## Quickstart

### Installation


### Model Download



### Self-check Reasoning


---
## Citation

```bibtex
@misc{Uni-CoT,
  author       = {SAIS-FUXI},
  title        = {Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision},
  howpublished = {\url{https://github.com/Fr0zenCrane/BagelCoT}},
  year         = {2025},
  note         = {Accessed: 2025-07-28}
}

```
---
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Fr0zenCrane/BagelCoT&type=Date)](https://star-history.com/#Fr0zenCrane/BagelCoT&Date)

---
## Acknowledgement

- This project is based on [Bagel](https://github.com/ByteDance-Seed/Bagel) proposed by ByteDance-Seed team. Bagel is a powerful and popular unified model for multimodal understanding and generation, making it an ideal foundation and startup for this project. We thank the ByteDance-Seed team for their outstanding work, which has made BATI possible.
