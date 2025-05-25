# Convolutional Neural Networks (CNN)
[Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
```math
\text{Output size} = \left\lfloor \frac{\text{Input size} - \text{Kernel size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
```
```math
\Rightarrow \text{padding="same"} \Leftrightarrow k=2p+1
```

# Recurrent Neural Networks (RNNs)
[RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
```math
h_t=\tanh(x_tW_{ih}^T+b_{ih}+h_{t-1}W_{hh}^T+b_{hh})
```
where 
> $x$ of $(L, H_{in})$ </br>
> $W_{ih}$ of $(H, H_{in})$ </br>
> $W_{hh}$ of $(H, H)$ </br>
> $b_{ih}, b_{hh}$ of $(H)$

# Long Short-Term Memory (LSTM)
[LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
    ![alt text](lstm.png)

# Attention and Transformers
## Attention Mechanism
```math
    \text{Attention}(Q, K, V) = \text{softmax}\left(\dfrac{QK^T}{\sqrt{d}}\right)V
```
where
> $Q$ of $(B, L_{\text{target}}, E)$ </br>
> $K$ of $(B, L_{\text{source}}, E)$ </br>
> $V$ of $(B, L_{\text{source}}, E)$ </br>
Note that $Q,K,V$ can have different $E_{i}$, and be projected to the same $E$.


## Multi-Head Attention
[MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
```
where $\text{head}_i = \text{Attention}(QW_{Q_i}, KW_{K_i}, VW_{V_i})$ and $W_{Q_i}, W_{K_i}, W_{V_i}$ are projection matrices for each head. Each head will have dimension $d_k$, where $d_k \times embedding\_dim=num\_head$.

## Positional Encoding (TODO)

## Transformer
[Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)  
[Transformer Layers](https://docs.pytorch.org/docs/stable/nn.html#transformer-layers)  

![alt text](image.png)

### Examples (TODO)

### Variations (TODO)