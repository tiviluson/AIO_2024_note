- Tree of Thought: group of experts
    > Imagine three different experts are independently solving the question.<br>
    All experts will write down 1 step of their thinking, then share it with the group.<br>
    Then all experts will go on to the next step, etc.<br>
    If any expert realises ther're wrong at any point, they leave.<br>
- Training with GRPO
    ![alt text](image.png)
- Policy-based optimization:
    > $\pi_\theta(S_t)=\hat{A}_{t+1} \\
       \Rightarrow \theta^* \text{ (using policy gradients)} \\
        \Rightarrow \pi_{\theta^*}(S_t)$ <br>
- Value-based optimization:
    > $ \pi(S_t) = V(S_t, A_t) \\
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
      $