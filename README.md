# Nonlocal Reaction-Advection-Diffusion Simulations in 2D
Code to simulate the 2D version of Painter et al. (2015)'s nonlocal reaction-advection-diffusion system. See the paper [[here]] for details of the model and theory.

[[here describe what each of the python files do...]]


## Numerical Integration
Numerical integration of the integro-PDE
$$\frac{\partial u}{\partial t} = \nabla^2 u + \rho u(1-\frac{u}{U}) -\boldsymbol{\nabla}\cdot\left(u(1-u)\frac{\mu}{\xi^{2}} \int\int\boldsymbol{\hat{s}}\tilde{\Omega}\left(\frac{s}{\xi}\right)u(\boldsymbol{x}+\boldsymbol{s},t) d s_x d s_y\right)$$
is carried out in `simulation_class.py`. Here, we use a method-of-lines, first discretising in space and then integrating the resulting ODEs with backwards differentiation formulae. The latter is implemented through SciPy's `integrate.solve_ivp` function - see [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for details. The diffusion term, $\nabla^2 u$, is discretised using the standard centred 5-point stencil. For the integral term, we use the fast Fourier transform method described below. 

## Computing the Integral Term
The nonlocal advection term in the integro-PDE is given by
$$\boldsymbol{\nabla}\cdot \left(u(\boldsymbol{x},t)(1-u(\boldsymbol{x},t))\frac{\mu}{\xi^2}\int\int \boldsymbol{\hat{s}}\tilde{\Omega}\left(\frac{s}{\xi}\right)u(\boldsymbol{x+s}, t)ds_x ds_y\right).$$
To compute this, we discretise space, then calculate the integral using a fast Fourier transform convolution method, then multiply by $\frac{\mu}{\xi^2}$ and the discretised $(u(\boldsymbol{x},t)(1-u(\boldsymbol{x},t))$, and then finally take the divergence using finite difference with the centred 3-point stencil. 

More specifically, we discretise space, $\boldsymbol{x}=(x,y)^T$, into equally spaced points along a 2D grid, such that $u(x,y) \to u_{i,j}$, where $$i,j \in \mathbb{M}\equiv \lbrace 1-\frac{N}{2}, ..., -1, 0, 1, ..., \frac{N}{2} \rbrace,$$ where the number of mesh points, $N$, is an even positive integer, and the stepsize is given by $h=\frac{L}{N}$.

The integral term in the integro-PDE is thus discretised such that
$$\boldsymbol{I}(\boldsymbol{x}) \equiv (I^{x}(\boldsymbol{x}), I^{y}(\boldsymbol{x}))^T \equiv \int\int \boldsymbol{\hat{s}}\tilde{\Omega}\left(\frac{s}{\xi}\right)u(\boldsymbol{x+s}, t)ds_x ds_y \longrightarrow \boldsymbol{I}\_{i,j} \equiv (I^{x}\_{i,j}, I^{y}\_{i,j})^T \equiv h^2  \displaystyle\sum\limits_{k=1-\frac{N}{2}}^{\frac{N}{2}-1}\displaystyle\sum\limits_{l=1-\frac{N}{2}}^{\frac{N}{2}-1} u_{i+k,j+l}\left[\boldsymbol{\hat{s}}\tilde{\Omega}\right]\_{k, l},$$
where, due to the periodic boundary conditions, the sums ' $i+k$ ' and ' $j+l$ ' are wrapped round $\mathbb{M}$ if they are less than $1-\frac{N}{2}$ or greater than $\frac{N}{2}$. That is, $i+k$ is understood to be $\left(\left[i+k -(1-\frac{N}{2})\right]\text{mod}N\right)+(1-\frac{N}{2})$, and similarly for $j+l$. 

The kernel, $\boldsymbol{\hat{s}}\tilde{\Omega}\left(\frac{s}{\xi}\right)$, is discretised on a 2D grid, such that $$\tilde{\Omega}\left(\frac{s}{\xi}\right)\to \tilde{\Omega}\left(\frac{h}{\xi}\sqrt{k^2 +l^2}\right)_\{k,l},$$ $$\boldsymbol{\hat{s}} \to (\frac{k}{\sqrt{k^2 +l^2}}, \frac{l}{\sqrt{k^2 +l^2}} )^T,$$ and so $$\boldsymbol{\hat{s}}\tilde{\Omega}\left(\frac{s}{\xi}\right) \to \left[\boldsymbol{\hat{s}}\tilde{\Omega}\right]\_{k, l}=(\frac{k}{\sqrt{k^2 +l^2}}, \frac{l}{\sqrt{k^2 +l^2}} )^T \tilde{\Omega}\left(\frac{h}{\xi}\sqrt{k^2 +l^2}\right)\_{k,l},$$ where $k,l \in$ { $1-\frac{N}{2}, ..., -1, 0, 1, ..., \frac{N}{2}-1$ }. We use sufficiently small signalling ranges, $\xi$, with sufficiently fast decaying interaction kernels so that all interaction at separations larger than $\frac{L}{2}-h$ has negligible contribution to the integral. We thus set $\left[\boldsymbol{\hat{s}}\tilde{\Omega}\right]\_{k, l}=0$ for $\sqrt{k^2 + l^2}>\frac{N}{2}-1$. This prevents any single point from interacting with another point twice (i.e. once from both directions) with the periodic boundary. 

$\boldsymbol{I}\_{i,j}$ is equivalent to a discrete periodic convolution, where $$\boldsymbol{I}\_{i,j}\equiv h^2  \displaystyle\sum\limits_{k=1-\frac{N}{2}}^{\frac{N}{2}-1}\displaystyle\sum\limits_{l=1-\frac{N}{2}}^{\frac{N}{2}-1} u_{i+k,j+l}\left[\boldsymbol{\hat{s}}\tilde{\Omega}\right]_{k, l}=h^2 [u\ast \boldsymbol{K}]\_{i,j}=h^2 \mathcal{F}^{-1}\left(\mathcal{F}(u)\mathcal{F}(\boldsymbol{K})\right),$$

where $\boldsymbol{K}\_{k,l}=\left[\boldsymbol{\hat{s}}\tilde{\Omega}\right]_{-k, -l}$. The final equality above follows from the convolution theorem,  where $\mathcal{F}$ is the 2D discrete (periodic) Fourier transform and $\mathcal{F}^{-1}$ is its inverse. We calculate these using SciPy's `scipy.fft.fft2` and `scipy.fft.ifft2` functions, respectively. See [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html) for details. We compute the integral term using this fast fourier method, which has complexity $\mathcal{O}(N^2\text{log}(N^2))$ and is thus significantly more efficient than direct summation, which has complexity $\mathcal{O}(N^4)$.

As stated above, to obtain the full nonlocal advection term, we then compute $\boldsymbol{F}\_{i,j}=\frac{\mu}{\xi^2}u_{i,j}(1-u_{ij})\boldsymbol{I}\_{i,j}$, and then take the divergence with the centred 3-point stencil, $\left[\boldsymbol{\nabla}\cdot \boldsymbol{F}\right]\_{i,j} = (F^{(x)}\_{i+1,j} - F^{(x)}\_{i-1,j})/2h  + (F^{(y)}\_{i,j+1} - F^{(y)}\_{i,j-1})/2h$.
