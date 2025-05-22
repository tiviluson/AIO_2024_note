# Style transfer
![alt text](image.png)
## Gram Matrix
$$ \text{Gram}(X)
 = \text{flatten}(X).\text{flatten}(X)^T$$
where 
* $X$ of $(B,C,H,W)$
* $\text{flatten}(X)$ of $(B,C,H*W)$
* $\text{Gram}(X)$ of $(B,C,C)$

This removes the spatial information, which aligns with the definition of "style".

## Loss
$$ L(\text{output}, \text{content}, \text{style}) 
=\lambda_\text{content}\times\underbrace{L_\text{content}}_{\text{MSE}(output,content)} + \lambda_\text{style}\times\underbrace{L_\text{style}}_{\text{MSE}(\text{Gram}(\text{Output}),\text{Gram}(\text{style})) } $$

# GAN (Generative Adversarial Network)
![alt text](image-1.png)
## Loss
$$ \begin{align*}
L_G(z) &= -\log(D(G(z)))\\
L_D(z,x) &= -y\log(D(x)) - (1-y)\log(1-D(G(z)))
\end{align*} $$

## Transposed Convolution 2D (`ConvTranspose2d`)
See [ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
$$ H_{out} = W_{out} 
= (H_{in}−1)\times stride−2\times padding+dilation\times(kernel\_size−1)+output\_padding+1 $$


# VAE (Variational AutoEncoder)
![alt text](image-3.png)
where $p_\theta(x|z)$ is the *true* posterior and $q_\phi(z|x)$ is the *approximated* prior.
![alt text](image-2.png)
## Loss function
$$ \begin{align*}
L 
&= \text{BCE}(x, \hat{x}) + \text{KL}(x, \hat{x}) \\
&= \text{BCE}(x,\hat{x}) - \dfrac{1}{2}\sum(1+\text{logvar}-\mu^2-\exp(\text{logvar}))
\end{align*} $$
### Proof of the equivalence of KL divergence (optional)
Let $P(z)=N(\mu,\Sigma^2)$ and $Q(z)=N(0,1)$ with $\Sigma$ being a diagonal covariance matrix. Therefore, the *pdf*s of $P$ and $Q$ are respectively:
$$ \begin{align*}
p(z) &= \frac{1}{\sqrt{(2\pi)^D |\Sigma|}} \exp\left(-\frac{1}{2}(z-\mu)^T \Sigma^{-1} (z-\mu)\right)\\
    &= (2\pi)^{-D/2} \left(\prod_{i=1}^D \sigma_i^2\right)^{-1/2} \exp\left(-\frac{1}{2}\sum_{i=1}^D \frac{(z_i - \mu_i)^2}{\sigma_i^2}\right)\\
q(z) &= \frac{1}{\sqrt{(2\pi)^D |I|}} \exp\left(-\frac{1}{2}z^T z\right) = (2\pi)^{-D/2} \exp\left(-\frac{1}{2}\sum_{i=1}^D z_i^2\right)
\end{align*} $$
We have
$$ \begin{align*}
\text{KL}(p||q)
&=\int{p(z)\log\dfrac{p(z)}{q(z)}}dz\\
&=\int{p(z)\left(\underbrace{\log{p(z)}}_{-\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^D \log(\sigma_i^2) - \frac{1}{2}\sum_{i=1}^D \frac{(z_i - \mu_i)^2}{\sigma_i^2}}-\underbrace{\log{q(z)}}_{-\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^D z_i^2}\right)}dz\\
&= \int{p(z)\left(-\frac{1}{2}\sum_{i=1}^D \left(\log(\sigma_i^2) + \frac{(z_i - \mu_i)^2}{\sigma_i^2} - z_i^2\right)\right)}dz\\
\end{align*} $$
Due to the diagonal covariance, the dimensions are independent. We can evaluate the expectation for each dimension $i$ separately.
$$E_{p(z)}
\left[\log(\sigma_i^2) + \frac{(z_i - \mu_i)^2}{\sigma_i^2} - z_i^2\right] = E_{p(z)}[\log(\sigma_i^2)] + E_{p(z)}\left[\frac{(z_i - \mu_i)^2}{\sigma_i^2}\right] - E_{p(z)}[z_i^2] $$
$E_{p(z)}[\log(\sigma_i^2)]$ is simply $\log(\sigma_i^2)$ since it's a constant with respect to $z$.  
$E_{p(z)}[(z_i - \mu_i)^2]$ is the variance of $z_i$ under $p(z)$, which is $\sigma_i^2$. So, $E_{p(z)}\left[\frac{(z_i - \mu_i)^2}{\sigma_i^2}\right] = \frac{\sigma_i^2}{\sigma_i^2} = 1$.  
$E_{p(z)}[z_i^2]=\sigma_i^2+(E_{p(z)}[z_i])^2 = \sigma_i^2 + \mu_i^2$, since $\text{var}(x)=E(x^2)-(E(x))^2$.

Substituting these expected values back:
$$ \begin{align*}
E_{p(z)}\left[\log(\sigma_i^2) + \frac{(z_i - \mu_i)^2}{\sigma_i^2} - z_i^2\right] &= \log(\sigma_i^2) + 1 - (\sigma_i^2 + \mu_i^2)\\
&= \log(\sigma_i^2) + 1 - \sigma_i^2 - \mu_i^2
\end{align*} $$

Summing over all dimensions:
$$ \begin{align*}
D_{KL}(P||Q)
&= -\frac{1}{2}\sum_{i=1}^D (\log(\sigma_i^2) + 1 - \sigma_i^2 - \mu_i^2)\\
&= \frac{1}{2}\sum_{i=1}^D (\sigma_i^2 + \mu_i^2 - 1 - \log(\sigma_i^2))
\end{align*} $$

# DDPM - Denoising Diffusion Probabilistic Model
![alt text](image-7.png)
![alt text](image-8.png)
## Training
### Forward
Gradually add increasing noise to a clean input.  
Generate the constants $\beta, \alpha, \bar\alpha,\sqrt{\bar\alpha},\sqrt{1-\bar\alpha}$ from $\beta_{s}, \beta_{e}, T$ and noised images $x_t$ from $x_0$. Note that the added noise (and of course the cummulative noise) in each time step $t$ increases with $t$.
![alt text](image-5.png)
![alt text](image-6.png)
$x_t$ can be sampled from $x_{t-1}$ or $x_0$.
### Reverse
* Predict $\epsilon_t(x_t,t)$ approximating $\epsilon$, the noise added to $x_0$ to sample $x_t$ via $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$.
* Predict $\hat{x}_{\theta}(x_t, t)$ or $\mu_\theta(x_t, t)$ to (indirectly) approximate $\mu_{t-1}(x_t, x_0)$
* Predict $s(x_t, t)$
### Sampling/Inference: TODO
#### From $x_t$ and $\epsilon_t$
$$ x_{t-1}
=\underbrace{\dfrac{1}{\sqrt{\alpha_t}}\left(x_t-\dfrac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t(x_t,t)\right)}_{\text{mean}}+\underbrace{\sqrt{\beta_t}}_{\text{std}}\epsilon, \text{ where } \beta_0=0, t\in[1,T] $$

#### From $x_0$ and $x_t$
$$ x_0
=\dfrac{x_t-\sqrt{1-\bar\alpha_t}*\epsilon_t}{\sqrt{\bar\alpha_t}}$$
$$\mu_{t-1}(x_0, x_t) 
= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

where:
- $x_0$ is the original data.
- $x_t$ is the data at timestep $t$.
- $\alpha_t = 1 - \beta_t$, where $\beta_t$ is the variance schedule.
- $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.
- $\bar{\alpha}_{t-1} = \prod_{i=1}^{t-1} \alpha_i$ (with $\bar{\alpha}_0 = 1$).
### Sample code
[[Colab]_Denoising_Diffusion_Probabilistic_Models.ipynb]([Colab]_Denoising_Diffusion_Probabilistic_Models.ipynb)

$$ c = \dfrac{\sqrt{\bar\alpha_{t-1}}\times(1-\alpha_{t})}{1-\bar\alpha_t}\\
lc = \dfrac{\sqrt{\alpha_t}\times(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\\
\begin{align*}
\mu_{t-1}&=oc*x_0+lc*x_t\\
&=oc*(\dfrac{x_t-\sqrt{1-\bar\alpha_t}*\epsilon}{\sqrt{\bar\alpha_t}})+lc*x_t\\
&=\dfrac{\sqrt{\bar\alpha_{t-1}}\times(1-\alpha_{t})}{1-\bar\alpha_t}*(\dfrac{x_t-\sqrt{1-\bar\alpha_t}*\epsilon}{\sqrt{\bar\alpha_t}})+\dfrac{\sqrt{\alpha_t}\times(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}*x_t\\
&=x_t\times\left(\dfrac{\sqrt{\bar\alpha_{t-1}}\times(1-\alpha_{t})}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}}+\dfrac{\sqrt{\alpha_t}\times(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\right)-\dfrac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}*\dfrac{\sqrt{\bar\alpha_{t-1}}\times(1-\alpha_{t})}{1-\bar\alpha_t}\times\epsilon
\end{align*} $$

### Classifier guidance

### Classifier-free guidance