# <span style="color:#4285F4">S</span><span style="color:#EA4335">L</span><span style="color:#FBBC04">E</span><span style="color:#34A853">D</span>: Self Logits Evolution Decoding for Improving Factuality in Large Language Models [NeurIPS 2024]
The official implementation for our NeurIPS 2024 paper "SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Models"

**[Jianyi Zhang<sup>1</sup>](https://jayzhang42.github.io/)** **Da-Cheng Juan<sup>2</sup>** **Cyrus Rashtchian<sup>2</sup>** **Chun-Sung Ferng<sup>2</sup>** **Heinrich Jiang<sup>2</sup>** **Yiran Chen<sup>1</sup>**

[//]: # ([<sup>1</sup>]&#40;https://cei.pratt.duke.edu/&#41; ![Duke University Logo]&#40;assets/cei_log.jpg&#41;[<sup>2</sup>]&#40;https://research.google.com/&#41; ![Google Research Logo]&#40;assets/google_log.jpg&#41;)

<sup>1</sup>[![Duke University Logo](assets/cei_log.jpg)](https://cei.pratt.duke.edu/)

<sup>2</sup>[![Google Research Logo](assets/google_log.jpg)](https://research.google.com/)


## 📌News
[2024.11.27] - We released the latest code on Github.  
[2024.11.26] - We launched the official project website launched [here](https://jayzhang42.github.io/sled_page/)!  
[2024.11.01] - The paper is available at [Arxiv](https://arxiv.org/abs/2411.02433).  
[2024.09.25] - Our SLED paper accepted for NeurIPS 2024!  


## 🧨 Why Choose SLED?

- <span style="color:#4285F4">Model Versatility:</span> Compatible with most large language model (LLM) families due to their multi-layered structures, such as LLaMA 2, LLaMA 3, Gemma, and MoE LLMs; scalable from 2B to 70B parameters.
- <span style="color:#4285F4">Task Versatility:</span> Tested with factual accuracy enhancement across various tasks and benchmarks, such as TruthfulQA, StrategyQA, FACTOR, GSM8K, HotPotQA, Natural Questions, and TriviaQA.
- <span style="color:#4285F4">High Compatibility:</span> SLED can be flexibly combined with other decoding methods, enhancing their performance.  
- <span style="color:#4285F4">High-Quality Outputs:</span> Reduces repetition, ensures fluent responses.  
- <span style="color:#4285F4">Negligible Computational Overhead:</span> Minimal additional costs, suited for real-time use.  
- <span style="color:#4285F4">Interpretability:</span> Provides new insights into inference-time computing algorithms.  


## 🔮Overview of SLED
![SLED](assets/sled_page.png)

We introduce <strong>S</strong>elf <strong>L</strong>ogits <strong>E</strong>volution <strong>D</strong>ecoding (SLED), a novel factuality decoding approach that leverages the latent knowledge within LLMs by contrasting the final layer’s logits with early layers' logits. SLED tracks the logits evolution process to unearth the latent knowledge within LLMs, and enables the self-evolution of the output distribution further to align it more closely with real-world facts.  


## 🛠Installation
- **Hardware**: We recommend using the NVIDIA A100 80GB GPU for efficient inference. While this configuration is recommended, other hardware configurations also work but could yield slightly different performance outcomes.
- **Python**: Recommended to use Python 3.10 or higher.
- **PyTorch**: We recommend using PyTorch version 2.0.1 with CUDA 11.8. You can install this specific version of PyTorch using the following command:
  ```bash
  pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  ```
- **Transformers**: Install the `transformers` library from the local directory included in the project folder.
  ```bash
  pip install -e transformers
  ```
- **Transformers-4.46.3**: Alternatively install newer version of `transformers` library from the local directory included in the project folder.
  ```bash
  pip install -e transformers-4.46.3
  ```
- **Other Dependencies**: 
  ```bash
  pip install -r requirements.txt
  ```

## 📈Evaluation
Below we provide example scripts for running `SLED` and other baseline methods such as `dola` and `Greedy Decoding`. For `SLED` and `dola`, the default setting for `--early-exit-layers` will include all the earlier layers of the LLM model before the final output layer. 

### Dataset Preparation
```bash
tar -xzvf demo_dataset.tar.gz
```

### FACTOR (Multiple Choices)
  
```bash
python run_factor.py --model-name meta-llama/Llama-2-7b-hf  --data-path Data/FACTOR/wiki_factor.csv  --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_factor.py --model-name meta-llama/Llama-2-7b-hf  --data-path Data/FACTOR/wiki_factor.csv  --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_factor.py --model-name meta-llama/Llama-2-7b-hf  --data-path Data/FACTOR/wiki_factor.csv  --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 2  --evolution_scale 10
```

### TruthfulQA (Multiple Choices)
  
```bash
python run_tfqa.py --model-name meta-llama/Llama-2-7b-hf  --data-path Data/TruthfulQA --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_tfqa.py --model-name meta-llama/Llama-2-7b-hf  --data-path Data/TruthfulQA --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_tfqa.py --model-name meta-llama/Llama-2-7b-hf  --data-path Data/TruthfulQA --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 2.5  --evolution_scale 75
```

### StrategyQA 
  
```bash
python run_strqa.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/StrategyQA --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_strqa.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/StrategyQA --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_strqa.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/StrategyQA --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 1.75 --evolution_scale 5
```

### GSM8K
  
```bash
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 2 --evolution_scale 10
```
Additional experiments involving various models can be found in the `scripts` folder.


## 💡Important Recommendations


We strongly encourage you to try `SLED` method on your own **open-ended generation** tasks and datasets. To ensure good performance and effective outcomes, consider the following recommended parameters:

- **Evolution Rate**: Set `--evolution_rate` within a range of **0.5 to 3**. 
- **Evolution Scale**: Set `--evolution_scale` values of **5, 10, or 20**. 
- **Repetition Penalty**: Adjust the `--repetition_penalty` to between **1.01 and 1.05**.

**We hope this will be a good starting point for your experiments!**



## Acknowledgement

This codebase is based on the official repo of [DoLa](https://github.com/voidism/DoLa). We also highly recommend reading their [excellent work](https://arxiv.org/abs/2309.03883).


## Citation

We would greatly appreciate it if you cite our SLED paper when you find our repository helpful for your research or projects.
```
@inproceedings{
zhang2024sled,
title={SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Models},
author={Jianyi Zhang and Da-Cheng Juan and Cyrus Rashtchian and Chun-Sung Ferng and Heinrich Jiang and Yiran Chen},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024},
year={2024},
url={https://arxiv.org/abs/2411.02433}
}
```















