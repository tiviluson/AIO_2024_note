# Tree of Thought: group of experts
![alt text](image.png)

    Imagine three different experts are independently solving the question.
    All experts will write down 1 step of their thinking, then share it with the group
    Then all experts will go on to the next step, etc.
    If any expert realises ther're wrong at any point, they leave.

# Parameter-Efficient Fine-Tuning (PEFT)
## Quantization
![alt text](image-4.png)
![alt text](image-2.png)
![alt text](image-3.png)
## Low-Rank Adaptation (LoRA)
![alt text](image-5.png)
![alt text](image-6.png)
## Quantized LoRA (QLoRA)
[QLORA:Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)


# Reinforcement Learning
## Overview
## Policy-based optimization:
```math
\pi_\theta(S_t)=\hat{A}_{t+1} \\
       \Rightarrow \theta^* \text{ (using policy gradients)} \\
        \Rightarrow \pi_{\theta^*}(S_t)
```
## Value-based optimization:
```math
\pi(S_t) = V(S_t, A_t) \\
\begin{array}{ c |  c |  c |  c }
    & A_0 & A_1 & A_2 \\
\hline
S_0 & * & * & * \\ 
\hline
S_1 & * & * & * \\
\hline  
S_2 & * & * & *\\
\hline    
\end{array}
```

## Training with GRPO
