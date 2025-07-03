# Tree of Thought: group of experts
![alt text](image.png)

    Imagine three different experts are independently solving the question.
    All experts will write down 1 step of their thinking, then share it with the group
    Then all experts will go on to the next step, etc.
    If any expert realises ther're wrong at any point, they leave.

# Overview of training LLMs
![alt text](image-14.png)
## Tokenization with Byte Pair Encoding (BPE)
![alt text](image-15.png)
![alt text](image-16.png)
![alt text](image-17.png)
## Metrics (TODO)
### Perplexity
![alt text](image-18.png)
### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- Evaluates **recall**: how much of the reference text appears in the generated output.  
- Often used in **text summarization**.

🔸 **ROUGE-N** (n-gram Recall): measures overlap of n-grams between candidate and reference.

```math
\text{ROUGE-N} = \frac{\sum_{\text{ngram} \in \text{Ref}} \text{Count}_{\text{match}}(\text{ngram})}{\sum_{\text{ngram} \in \text{Ref}} \text{Count}(\text{ngram})}
```

🔸 **ROUGE-L** (Longest Common Subsequence): measures the longest sequence of words shared between candidate and reference.
```math
P = \frac{LCS(X, Y)}{|Y|}, \quad
R = \frac{LCS(X, Y)}{|X|}, \quad
F_1 = \frac{(1 + \beta^2)PR}{R + \beta^2 P}
```
Where
- $X$: reference text  
- $Y$: candidate text  
- $\beta$: weight between precision $P$ and recall $R$ (usually $1$)
### BLEU (Bilingual Evaluation Understudy)
Measures n-gram **precision** between candidate and reference.
```math
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \log p_n \right)
```
Where:  
- **BP** (Brevity Penalty):

```math
\text{BP} = \begin{cases}
1 & \text{if } c > r \\
e^{(1 - \frac{r}{c})} & \text{if } c \leq r
\end{cases}
```
- $p_n$ = precision of n-grams  
- $w_n = \dfrac{1}{N}$ (equal weights)  
- $c$ = candidate length, $r$ = reference length
# Parameter-Efficient Fine-Tuning (PEFT)
## Prefix Tuning
Add **a set** of small, continuous, task-specific, learnable vectors (the *"prefixes"*) to the **hidden states at each layer**.
![alt text](image-25.png)
![alt text](image-26.png)
![alt text](image-19.png)

## Prompt Tuning
Prepend **a** small, continuous, task-specific, learnable vector (the *"soft prompt"*) to the **input embedding layer**.
![alt text](image-20.png)
![alt text](image-21.png)
## Instruction Tuning
May not be parameter-efficient, but is a form of fine-tuning. The LLM is trained on a dataset of **(instruction, output)** pairs. The goal is to teach the model to follow human instructions better.  
The subtle difference is the framing of the training data set. When a user asks a similar question, the output answer aligns more with the correct response.

## Quantization
![alt text](image-4.png)
![alt text](image-2.png)
![alt text](image-3.png)
## Low-Rank Adaptation (LoRA)
![alt text](image-5.png)
![alt text](image-6.png)
![alt text](image-22.png)
![alt text](image-23.png)
![alt text](image-24.png)
## Quantized LoRA (QLoRA)
[QLORA:Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- **Low-Rank Adapters (LoRA)**
- **Block-wise k-bit Quantization**: group the weights of the model into blocks and quantize each block separately.
![alt text](image-7.png)
![alt text](image-8.png)
- **Double Quantization**: to further save memory, the quantization constants themselves are also quantized
![alt text](image-11.png)
![alt text](image-12.png)
- **Storage Data Type NormalFloat4**: instead of using uniformly distributed 4-bit integers, QLoRA uses a custom data type called NormalFloat4, which is distributed normally around zero.
![alt text](image-9.png)
![alt text](image-10.png)
- **Paged Optimizers**: to manage memory spikes during training, QLoRA uses paged optimizers, which offload optimizer states to CPU RAM when GPU memory is full.
![alt text](image-13.png)

## Example notebook
[PEFT_example.ipynb](PEFT_example.ipynb)




# Reinforcement Learning (TODO)
## Overview
![alt text](image-28.png)
## Policy-based optimization:
```math
\pi_\theta(S_t)=\hat{A}_{t+1} \\
       \Rightarrow \theta^* \text{ (using policy gradients)} \\
        \Rightarrow \pi_{\theta^*}(S_t)
```
## Value-based optimization:
Learn a value function that estimates the expected return (cumulative future reward) for an agent being in a particular state $V(s)$, or for taking a specific action in a state $Q(s, a)$. The policy is then derived implicitly from this value function.
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

## Reward modeling
![alt text](image-27.png)
May use the Classification head (with [Margin Ranking Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html)) or Regression head in place of the LM head.
$$
\text{MRL}(x_1,x_2,y,\text{margin})=\max(0, -y*(x_1-x_2) + \text{margin}) \\
\text{ where } y= \begin{cases}
    1 & \text{, if } x_1 > x2 \\
    -1 & \text{, else}
\end{cases}
$$
For simplicity, we denote $r_{t+1}=\text{Reward}(s_t,a_t)$, which is the reward received after taking action $a_t$ in state $s_t$.


# Advanced Actor-Critic (A2C)
![alt text](image-1-1.png)

A2C combines the strengths of policy-based and value-based approaches.
- **Actor**: The LLM (or policy network) that generates actions (tokens) based on the current state (context). It is specified by $\pi_\theta(a_t|s_t)$ (policy-based).
- **Critic**: A separate model that estimates the expected **cummulative, total reward** (e.g., from the RM) achievable from the current generated sequence $s_t$ onwards. It is specified by $V_\phi(s_t)$ or $Q_\phi(s_t|a_t)$ (policy-based).

$$
    \underbrace{A(s_t, a_t)}_{\text{Advantage}} = \underbrace{Q(s_t, a_t)}_{\text{Reward}} - \underbrace{V_\phi(s_t)}_{\text{Value}}
$$
Example:
> - Scenario: The LLM is given a document as input (initial state) and generates the summary token by token (actions).  
> - Reward Model Example: After the LLM finishes generating the entire summary, a separate reward model takes the complete generated summary and compares it to the original document and perhaps a high-quality example summary. It outputs a single scalar score, say $+0.9$, indicating it's a very good summary, or $-0.5$, if it's irrelevant. This score is the final reward for that generated sequence.  
> - Critic Model Example: While the LLM is in the process of generating the summary, after producing, say, the first three sentences (this partial summary is the current state), the critic model takes this partial text as input. It outputs an estimated value, say $+7.3$. This value means, "Based on the current partial summary, and assuming the LLM continues generating text according to its current abilities, the expected total cumulative reward for the rest of the summary generation is estimated to be $7.3.$" This is the critic's prediction of future success from this intermediate state.  
> - The A2C algorithm uses the reward (e.g., $+0.9$ at the end) and the critic's value estimates (e.g., $+7.3$ mid-way) to calculate the advantage and update both the LLM (actor) and the critic itself via **policy gradients (aka back-propagation)**. The reward model provides the objective signal; the critic helps predict and use that signal during the step-by-step generation process.

## Proximal Policy Optimization with KL divergence and Pre-train loss (PPO)
### PPO:
$$
J_\text{PPO}(\theta) = \sum_{x \in D}\sum_{t=1}^{T} \min_\theta(A_tr_t,A_t*\text{clip}(r_t, 1-\epsilon, 1+\epsilon))
$$
where
$$
    r_t=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$
Pick the $\theta$ that attains the reward, but not too agressively. Perform this update in multiple passes (iteratively) over the same batch to update the policy and value function.

Example:
> - Initial State: The state is the input prompt: "What is the capital of France?"  
> - Actor (LLM) Generates: The LLM starts generating tokens. Let's say at one step, the current state is "What is the capital of France? Paris is the". The LLM's policy (its internal probabilities over the vocabulary) determines the next token. Let's say it generates the token "capital".
> - Old Policy Snapshot: Before this training batch started, you had a snapshot of the LLM's parameters, representing the "old policy" ($\pi_\text{old}$).  
> - New Policy Probabilities: The LLM is currently using the "new policy" (π 
new), which is being trained. Let's say, in the state "What is the capital of France? Paris is the", the probability of generating "capital" under the new policy is $P_\text{new}(\text{"capital"} | state)=0.7$. The probability under the old policy was $P_\text{old}(\text{"capital"} | state)=0.5$.
> - Probability Ratio: The ratio is $0.7/0.5=1.4$.  
> - Reward & Advantage: The environment or reward model (trained separately) evaluates the generated text (potentially after the whole sequence is finished, but let's consider the conceptual advantage signal here). Suppose, based on this being part of a good answer, the advantage A(state,"capital") is calculated to be $+2.0$ (meaning generating "capital" here was better than expected from this state).  
> - PPO Clipped Objective: The PPO objective for this step involves:
> - The unclipped term: $\text{ratio}\times\text{Advantage}=1.4\times2.0=2.8$
> - The clipped term: Assume $\epsilon=0.2$. The clip range is $[1−0.2,1+0.2]=[0.8,1.2]$. The ratio $1.4$ is greater than $1.2$. The clipped ratio is $1.2$. The clipped term is $\text{clipped\_ratio}\times \text{Advantage}=1.2\times2.0=2.4$.  
> - PPO takes the minimum of these two: min(2.8,2.4)=2.4. This value contributes to the objective that is maximized (or the negative is minimized) to update the actor's parameters.
### Update with PPO
Updating the LLM (actor) with the advantage in PPO involves using the calculated advantage to guide the policy gradient update, specifically through the clipped surrogate objective function.

Here's a breakdown:

1.  **Policy Gradient Foundation:** PPO is a policy gradient method. The general idea behind policy gradient is to adjust the policy parameters ($\theta$, which are the LLM's weights) in the direction that increases the probability of taking actions that lead to high returns (or advantages). The core update rule for policy gradient methods is proportional to:  
    $$\nabla_{\theta} J(\theta) \propto \mathbb{E} \left[ \sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) A^{\pi}(s_t, a_t) \right]$$  
    This means we increase the log-probability of actions with positive advantage and decrease it for actions with negative advantage.

2.  **PPO's Modified Objective:** PPO doesn't directly use the above simple gradient. Instead, it defines a "clipped surrogate objective" ($J_{PPO}(\theta)$) that it optimizes using gradient ascent. This objective incorporates the ratio of the new policy probability to the old policy probability ($r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$) and the advantage $A_t$:
    $$J_{PPO}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t) \right]$$
    where $\mathbb{E}_t$ denotes taking the expectation over a batch of data, $A_t$ is the advantage at timestep $t$, and $\epsilon$ is the clipping hyperparameter.

3.  **Calculating the Gradient:** During training, for a batch of collected data (state-action pairs and their computed advantages), PPO calculates the gradient of this $J_{PPO}(\theta)$ with respect to the LLM's parameters ($\theta$).

4.  **Updating LLM Parameters:**
    * The LLM acts as the policy $\pi_{\theta}$. Its parameters $\theta$ determine the probability $\pi_{\theta}(a_t | s_t)$ of generating the next token $a_t$ given the context $s_t$.
    * The advantage $A_t$ for each step in the batch is calculated using the reward received and the critic's value estimates (as described previously).
    * The probability ratio $r_t(\theta)$ is computed by comparing the probability of the action $a_t$ under the current LLM policy ($\pi_{\theta}$) and the policy before the update ($\pi_{\theta_{old}}$).
    * The PPO loss (typically minimizing $-J_{PPO}(\theta)$) is computed.
    * Standard backpropagation is used to calculate the gradients of this loss with respect to every parameter in the LLM.
    * An optimizer (like Adam or AdamW) applies these gradients to update the LLM's parameters $\theta$, taking a step that aims to increase the clipped objective function.

5.  **Iteration:** This process is repeated for **multiple epochs on the same batch of data** before a new batch is collected. The clipping ensures that even with multiple updates on the same data, the policy doesn't change too drastically at any single step, maintaining training stability.

6.  **Updating the Old Policy**: After the training phase on the current batch of data is complete (i.e., after running multiple epochs of gradient updates on the new policy), the parameters of the new policy are again copied to become the old policy for the next major iteration of data collection and training.
### With KL Penalty
$$J^{KL\_\text{PPO}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) A_t - \beta \cdot D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) || \pi_{\theta}(\cdot|s_t)) \right]$$
where:
* $r_t(\theta)$ is the probability ratio $\frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$.
* $A_t$ is the advantage estimate.
* $\beta$ is a coefficient that controls the strength of the KL penalty.
* $D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) || \pi_{\theta}(\cdot|s_t))$ is the KL divergence between the distribution of actions under the old policy and the new policy for state $s_t$.

**How it Controls Divergence:** The KL divergence term measures how much the new policy's probability distribution has shifted away from the old policy's distribution. By subtracting this term (scaled by $\beta$) from the policy objective, the algorithm discourages large changes in the policy. If the new policy diverges significantly from the old one, the KL term becomes large, reducing the overall objective and thus discouraging such updates.

### With Pre-train tasks loss:
$$J(\theta)=J^{KL\_\text{PPO}}+J^{\text{Pre-trained tasks}}\\
=J^{\text{PPO}}-\beta L^{\text{KL}}+J^{\text{Pre-trained tasks}}$$
In essence, maximize reward while:
 * Restraining excessive update of the policy with clipping and iterative update
 * Avoiding excessive divergence in the output (probabilistic distribution) from the original output
 * Maintaining objective in downstream tasks.

 ### Example notebook
 [[Colab]-PPO.ipynb]([Colab]-PPO.ipynb)

# LangChain (TODO)

# LangGraph (TODO)

# SmolAgent (TODO)

# XAI (TODO)
## LIME method
## ANCHOR method