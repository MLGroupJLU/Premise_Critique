# Don’t Take the Premise for Granted: Evaluating the Premise Critique Ability of Large Language Models

<p align="center" width="50%">
<a ><img src="resources/img/ill_comic.png" alt="ill_comic" style="width: 40%; min-width: 300px; display: block; margin: auto;"></a>
</p>

<div align="center">
  <a href="https://arxiv.org/html/2505.23715v1">
    <strong>📃 Paper</strong>
  </a>
  •
  <a href="https://huggingface.co/datasets/ALIENS232/PCBench">
    <strong>🤗 Dataset</strong>
  </a>
  •
  <a href="https://github.com/MLGroupJLU/Premise_Critique">
    <strong>🖥️ Code</strong>
  </a>
</div>

## Updates
[2025/05] We released codes for this project.

## Contents
- [Introduction](#introduction)
- [Key Findings](#key-findings)
- [Data Construction](#data-construction)
- [Install](#install)
- [Run Code](#run-code)
- [Citation](#citation)

## Introduction

<p align="center" width="50%">
<a ><img src="resources/img/illustration.svg" alt="introduction" style="width: 40%; min-width: 200px; display: block; margin: auto;"></a>
</p>

Large language models (LLMs) have witnessed rapid advancements, demonstrating remarkable capabilities. However, a notable vulnerability persists: LLMs often uncritically accept flawed or contradictory premises, leading to inefficient reasoning and unreliable outputs. This emphasizes the significance of possessing the **Premise Critique Ability** for LLMs, defined as the capacity to proactively identify and articulate errors in input premises. Most existing studies assess LLMs' reasoning ability in ideal settings, largely ignoring their vulnerabilities when faced with flawed premises. Thus, we introduce the **Premise Critique Bench (PCBench)**, designed by incorporating four error types across three difficulty levels, paired with multi-faceted evaluation metrics. We conducted systematic evaluations of 15 representative LLMs.

## Key Findings

- Most models show limited ability to autonomously critique flawed premises, relying heavily on explicit prompts to detect errors. 
- Both question difficulty and the error type can influence models' premise critique ability: models excel at identifying simple, surface-level flaws but struggle with complex inconsistencies or procedural errors. 
- There is no consistent correlation between a model’s reasoning capability and its ability to critique premises. Some reasoning models internally catch inconsistencies but fail to articulate them outwardly. 
- Flawed premises deepen overthinking in reasoning models, leading to significantly longer responses. 

## Data Construction

We construct **PCBench** to systematically evaluate LLMs' premise critique abilities for erroneous inputs via a structured process:  
1. **Error Categories**: Define 4 types of premise errors to assess model capabilities in identifying flawed inputs.  
2. **Difficulty Levels**:  
   - Normal: From GSM8K dataset  
   - Medium: Adapted from Chinese College Entrance Examination (OlympiadBench)  
   - Difficult: From Omni-MATH (difficulty >6)  
3. **Problem Variants** for each base problem (error category + difficulty):  
   - **Original Problem**: Correct premises (baseline).  
   - **Flawed Problem**: Intentional errors in premises (to test autonomous critique).  
   - **Flawed Problem with Explicit Instruction**: Adds prompts to check for errors (comparative reference).  

**Scale**: 100 base problems per error-difficulty combination → 1,200 base problems → 3,600 problems (3 variants each).  
Designed to analyze how error type and task complexity impact premise critique ability.
<p align="center" width="50%">
<a ><img src="/resources/img/construct.png
" alt="construction" style="width: 40%; min-width: 500px; display: block; margin: auto;"></a>
</p>

## Results

<p align="center" width="50%">
<a ><img src="/resources/img/results.png
" alt="results" style="width: 40%; min-width: 550px; display: block; margin: auto;"></a>
</p>

specific response and corresponding evaluation results can be found in `evaluation/infer_result` and `evaluation/eval_result`

The statistical results of the evaluation results are acceptable at `evaluation/statistics`

## Install
```
pip install -r requirements.txt
```
## Run Code

For reference, the command - line scripts are located in the `evaluation/reference_sh` folder.

### Inference
Run following commad to get LLM's responses.

```bash
python ./evaluation/inference.py --model_name <model_name> --mode inference --save_frequency <save_frequency> --dataset_load_proc <dataset_load_proc> --infer_proc <infer_proc>
```

Validates inference results for missing samples (repeat until no omissions).

```bash
python ./evaluation/inference.py --model_name <model_name> --mode check --save_frequency <save_frequency> --dataset_load_proc <dataset_load_proc> --infer_proc <infer_proc>
```
Scripts located in `evaluation/reference_sh`.

### Evaluation
Run following commad to get o3-mini-high's evaluation result to corresponding responses.

```bash
python ./evaluation/eval.py --model_name <model_name> --mode inference --evaluator <evaluator_model> --save_frequency <save_frequency> --infer_proc <infer_proc>
```

Validates evaluation results for missing samples (repeat until no omissions).

```bash
python ./evaluation/eval.py --model_name <model_name> --mode check --evaluator <evaluator_model> --save_frequency <save_frequency> --infer_proc <infer_proc>
```

Scripts located in `evaluation/reference_sh`.

### Statistics

After evaluating all the models, you can run following command to get the statisitcs of the evaluation result.

```bash
python ./evaluation/statistics.py
```

## Citation
```
@misc{li2025dont,
    title={Don't Take the Premise for Granted: Evaluating the Premise Critique Ability of Large Language Models},
    author={Jinzhe Li and Gengxu Li and Yi Chang and Yuan Wu},
    year={2025},
    eprint={2505.23715},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
Please cite our paper if you find our research and code useful.
