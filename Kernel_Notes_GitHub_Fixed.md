# Kernel Functions, Gaussian Processes, SVMs, RKHS and MMD

These notes cover kernel functions and positive semidefiniteness, Gaussian processes (GPs) and Gaussian process regression (GPR), the kernel trick and Mercer’s theorem, reproducing kernel Hilbert spaces (RKHSs), maximum mean discrepancy (MMD), mean embeddings, and the MMD witness function. Mathematical expressions are written in LaTeX so that GitHub can render them.

## 1. Basic definition of a kernel

A kernel function is a bivariate function

$$
K:X\times X\to\mathbb{R}.
$$

A **positive semidefinite (PSD) kernel** is symmetric and satisfies a positivity condition. For every finite collection of points $x_1,\ldots,x_n\in X$ and real coefficients $c_1,\ldots,c_n$,

$$
\sum_{i=1}^{n}\sum_{j=1}^{n}c_i c_j K(x_i,x_j)\geq 0.
$$

Equivalently, the Gram matrix $\mathbf K$ with entries $K_{ij}=K(x_i,x_j)$ is positive semidefinite:

$$
\mathbf c^{\mathsf T}\mathbf K\mathbf c\geq 0
\qquad\text{for every }\mathbf c\in\mathbb R^n.
$$

This is the matrix analogue of a non-negative quadratic form. Note that PSD means $\mathbf c^{\mathsf T}\mathbf K\mathbf c\geq0$; it does **not** generally mean that every entry of $\mathbf K\mathbf c$ is non-negative.

## 2. Gaussian processes

A Gaussian process is a collection of random variables indexed by a continuous domain, such as time or space. It is written

$$
f(x)\sim\mathcal{GP}\bigl(m(x),k(x,x')\bigr),
$$

where $m(x)$ is the mean function and $k(x,x')$ is the covariance function, which is a PSD kernel:

$$
m(x)=\mathbb E[f(x)],\qquad
k(x,x')=\mathrm{Cov}(f(x),f(x')).
$$

For any finite set of inputs, the corresponding vector of function values has a multivariate Gaussian distribution. If $\mathbf f=(f(x_1),\ldots,f(x_n))^{\mathsf T}$, then

$$
\mathbf f\sim\mathcal N(\boldsymbol\mu,\mathbf K),
$$

where $\boldsymbol\mu_i=m(x_i)$ and $K_{ij}=k(x_i,x_j)$. Its density, when $\mathbf K$ is nonsingular, is

$$
p(\mathbf f)=\frac{1}{\sqrt{(2\pi)^n|\mathbf K|}}
\exp\!\left[-\frac12(\mathbf f-\boldsymbol\mu)^{\mathsf T}
\mathbf K^{-1}(\mathbf f-\boldsymbol\mu)\right].
$$

The mean function is sometimes set to zero for simplicity. More generally, write $f(x)=m(x)+g(x)$ and model the centred process $g$ separately. The covariance/kernel function then determines the assumed correlation structure of the function values.

### 2.1 Common kernels

Kernel choice encodes prior assumptions about the functions that are plausible for the data. The parameters inside a kernel are called **hyperparameters**.

**Squared-exponential (SE), radial basis function (RBF), or Gaussian kernel**

$$
k_{\mathrm{SE}}(x,x')=\sigma^2\exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right).
$$

Here $\ell$ is the length scale and $\sigma^2$ is the signal variance. A smaller length scale generally permits more rapid variation across the input domain. The signal variance controls the vertical scale of variation around the mean; it is not a horizontal range parameter.

**Rational quadratic kernel**

$$
k_{\mathrm{RQ}}(x,x')=\sigma^2\left(1+\frac{(x-x')^2}{2\alpha\ell^2}\right)^{-\alpha}.
$$

The rational quadratic kernel can be interpreted as a scale mixture of squared-exponential kernels with different length scales, giving it multi-scale flexibility. The parameter $\alpha>0$ controls the mixture.

**Periodic kernel**

$$
k_{\mathrm{per}}(x,x')=\sigma^2\exp\!\left(-\frac{2\sin^2\!\left(\pi|x-x'|/p\right)}{\ell^2}\right),
$$

where $p$ is the period and $\ell$ controls smoothness within a period. This kernel is useful when the data exhibit repeating structure, such as seasonal patterns.

### 2.2 Kernel combinations

A single kernel may not capture all features of a dataset. Sums and products of valid kernels are also kernels.

**Product:** intuitively, multiplication combines constraints, so similarity is high when both component kernels indicate high similarity. A locally periodic kernel can combine periodic structure with a slowly changing envelope:

$$
k_{\mathrm{local\,per}}(x,x')=k_{\mathrm{per}}(x,x')\,k_{\mathrm{SE}}(x,x').
$$

**Sum:** intuitively, addition allows either component kernel to contribute to the similarity:

$$
k(x,x')=k_1(x,x')+k_2(x,x').
$$

Kernel selection can be guided by prior knowledge and refined by fitting hyperparameters or comparing predictive performance.

## 3. Gaussian process regression (GPR)

In GPR, the kernel specifies prior covariance structure. For training inputs $X=(x_1,\ldots,x_n)$ and observations $\mathbf y=(y_1,\ldots,y_n)^{\mathsf T}$, let $\mathbf K_\theta$ be the kernel matrix and let $\theta$ collect the kernel hyperparameters. Under a zero-mean Gaussian model with observation noise variance $\sigma_n^2$,

$$
\mathbf y\sim\mathcal N\!\left(\mathbf 0,\mathbf K_\theta+\sigma_n^2\mathbf I\right).
$$

A common objective for learning hyperparameters is the log marginal likelihood. If $\mathbf C=\mathbf K_\theta+\sigma_n^2\mathbf I$, then

$$
\log p(\mathbf y\mid X,\theta)
=-\frac12\mathbf y^{\mathsf T}\mathbf C^{-1}\mathbf y
-\frac12\log|\mathbf C|-\frac n2\log(2\pi).
$$

The three terms represent data fit, a complexity penalty through the covariance determinant, and a normalisation term. Hyperparameters such as $\sigma$, $\ell$, and the noise level can be estimated by maximising this quantity.


For a test input $x_*$, define the kernel vector

$$
\mathbf{k}_* =
\left(k(x_1,x_*),\ldots,k(x_n,x_*)\right)^{\mathsf T},
\qquad
k_{**}=k(x_*,x_*).
$$



The posterior predictive distribution is Gaussian:

$$
f_* \mid \mathbf{X},\mathbf{y},x_*
\sim \mathcal{N}(\mu_*,\sigma_*^2).
$$


The posterior mean is

$$
\mu_* = \mathbf{k}_{*}^{\mathsf T}\mathbf{C}^{-1}\mathbf{y}.
$$

The posterior variance is

$$
\sigma_*^2 = k_{**} - \mathbf{k}_{*}^{\mathsf T}\mathbf{C}^{-1}\mathbf{k}_{*}.
$$

The posterior combines the prior with the observed data to infer the function values at unobserved inputs.

## 4. Support vector machines and the kernel trick

A support vector machine (SVM) can use a feature map

$$
\phi:X\to\mathcal H
$$

to represent inputs in a feature space $\mathcal H$, where a linear decision boundary may separate data that are not linearly separable in the original input space. For example, a polynomial expression can be written as a linear function of a feature vector:

$$
\theta_3x^3+\theta_2x^2+\theta_1x+\theta_0
=\boldsymbol\theta^{\mathsf T}\phi(x),
$$

with a suitable choice of $\phi(x)$.

The feature space may be high- or infinite-dimensional, so explicitly constructing $\phi(x)$ can be expensive. The **kernel trick** uses the inner product directly:

$$
K(x,y)=\langle\phi(x),\phi(y)\rangle_{\mathcal H}.
$$

This allows algorithms that depend on feature vectors only through inner products to work with $K(x,y)$ without explicitly computing the feature map.

## 5. From symmetric matrices to kernels

For a real symmetric matrix $A$, the spectral theorem gives an orthogonal matrix $Q$ and a diagonal matrix $D$ such that

$$
A=QDQ^{\mathsf T}.
$$

If $q_i$ are the orthonormal eigenvectors and $\lambda_i$ their eigenvalues, then

$$
A=\sum_{i=1}^{n}\lambda_i q_iq_i^{\mathsf T}.
$$

A kernel $K(x,y)$ can be viewed as a continuous analogue of a matrix: its two arguments play roles analogous to row and column indices. The corresponding integral operator is

$$
(T_K f)(x)=\int_X K(x,y)f(y)\,dy.
$$

Under suitable assumptions, eigenfunctions of this operator satisfy

$$
T_K\psi_i=\lambda_i\psi_i,
\qquad
\int_X\psi_i(x)\psi_j(x)\,dx=\delta_{ij},
$$

where $\delta_{ij}$ is the Kronecker delta for an orthonormal family. This is the functional analogue of an orthogonal eigenvector basis. For symmetric kernels, the relevant operator is self-adjoint; orthogonality applies to eigenfunctions associated with distinct eigenvalues, and an orthonormal basis can be chosen within each eigenspace.

### 5.1 Mercer’s theorem

In a standard form, Mercer’s theorem states that a continuous, symmetric, positive semidefinite kernel on a compact domain has an eigenfunction expansion under the theorem’s assumptions:

$$
K(x,y)=\sum_{i=1}^{\infty}\lambda_i\psi_i(x)\psi_i(y),
\qquad \lambda_i\geq0.
$$

The eigenfunctions are orthonormal in the relevant $L^2$ space, and the series has the convergence properties guaranteed by the theorem’s hypotheses. The exact assumptions and convergence mode depend on the version of Mercer’s theorem being used.

This expansion suggests an inner-product representation. Define a feature map by

$$
\phi(x)=\bigl(\sqrt{\lambda_1}\psi_1(x),\sqrt{\lambda_2}\psi_2(x),\ldots\bigr)^{\mathsf T}.
$$

Then

$$
\langle\phi(x),\phi(y)\rangle_{\ell^2}
=\sum_{i=1}^{\infty}\lambda_i\psi_i(x)\psi_i(y)
=K(x,y),
$$

provided the expansion and inner product are well-defined. Thus a PSD kernel can be represented as an inner product in a feature space.

## 6. Reproducing kernel Hilbert spaces (RKHSs)

An RKHS $\mathcal H$ is a Hilbert space of functions in which point evaluation is a continuous linear functional. For a kernel $K$, its associated RKHS has the **reproducing property**

$$
f(x)=\langle f,K(x,\cdot)\rangle_{\mathcal H}
\qquad\text{for all }f\in\mathcal H,\ x\in X.
$$

The kernel section $K(x,\cdot)$ is therefore the representer of evaluation at $x$. The reproducing property can also be written

$$
\langle K(x,\cdot),K(y,\cdot)\rangle_{\mathcal H}=K(x,y).
$$

For a PSD kernel, one can construct a corresponding RKHS. The kernel trick follows immediately from the feature-map identity

$$
K(x,y)=\langle\phi(x),\phi(y)\rangle_{\mathcal H}.
$$

The feature map need not be explicitly known in order to evaluate the kernel. The important computational point is that inner products in the feature space can be obtained directly from the kernel function.

### 6.1 An eigenfunction-based feature representation

Using the Mercer expansion, one can represent a function through coefficients in an orthogonal basis. For example, a feature vector associated with a fixed input $x$ has the form

$$
\phi(x)=\bigl(\sqrt{\lambda_1}\psi_1(x),\sqrt{\lambda_2}\psi_2(x),\ldots\bigr)^{\mathsf T}.
$$

The kernel is the inner product of two such vectors:

$$
K(x,y)=\langle\phi(x),\phi(y)\rangle.
$$

This connects the spectral representation of a kernel to the feature-space interpretation used in SVMs.

## 7. A kernel two-sample test

The two-sample problem asks whether two samples arise from the same probability distribution. Let $p$ and $q$ be Borel probability measures on a suitable space $X$. The hypotheses are

$$
H_0:p=q,
\qquad H_1:p\ne q.
$$

Let $X_1,\ldots,X_m\overset{\mathrm{i.i.d.}}{\sim}p$ and $Y_1,\ldots,Y_n\overset{\mathrm{i.i.d.}}{\sim}q$. One approach compares expectations of functions under the two distributions. If a suitable function class is rich enough, a difference in expectations can witness that the distributions differ.

### 7.1 Maximum mean discrepancy (MMD)

For a class of functions $\mathcal F$, the maximum mean discrepancy is

$$
\mathrm{MMD}[\mathcal F,p,q]
=\sup_{f\in\mathcal F}
\left|\mathbb E_{X\sim p}[f(X)]-\mathbb E_{Y\sim q}[f(Y)]\right|.
$$

The absolute value makes the discrepancy non-negative. A large value means that some function in $\mathcal F$ has substantially different expectations under $p$ and $q$. Equality of expectations for one function alone does not prove that $p=q$; the function class must be sufficiently rich for the criterion to distinguish distributions.

For samples, the empirical estimate is

$$
\widehat{\mathrm{MMD}}[\mathcal F,X,Y]
=\sup_{f\in\mathcal F}
\left|\frac1m\sum_{i=1}^{m}f(X_i)
-\frac1n\sum_{j=1}^{n}f(Y_j)\right|.
$$

The population definition uses expectations under the true distributions, while the empirical version uses the observed samples.

A useful function class should be rich enough to distinguish distributions, while restrictive enough for estimation and computation to be manageable. An important choice is the unit ball in an RKHS:

$$
\mathcal F=\{f\in\mathcal H:\|f\|_{\mathcal H}\leq1\}.
$$

## 8. Mean embeddings

For a probability measure $p$, the kernel mean embedding is

$$
\mu_p=\mathbb E_{X\sim p}[\phi(X)],
$$

where $\phi(X)=K(X,\cdot)$ can be viewed as an RKHS-valued feature map. Under suitable integrability conditions, $\mu_p\in\mathcal H$. By the reproducing property, for any $f\in\mathcal H$,

$$
\mathbb E_{X\sim p}[f(X)]=\langle f,\mu_p\rangle_{\mathcal H}.
$$

Thus a probability distribution is represented by a vector in the RKHS. This converts differences between expectations into inner products with mean embeddings.

### 8.1 Why the mean embedding exists

The Riesz representation theorem states that every bounded linear functional $A$ on a Hilbert space $\mathcal H$ can be represented by an inner product with a fixed element $g_A\in\mathcal H$:

$$
Af=\langle f,g_A\rangle_{\mathcal H}
\qquad\text{for every }f\in\mathcal H.
$$

For the expectation functional $A_p(f)=\mathbb E_{X\sim p}[f(X)]$, one establishes boundedness under appropriate assumptions. A typical bound follows from the reproducing property and Cauchy–Schwarz:

$$
|f(x)|=|\langle f,K(x,\cdot)\rangle_{\mathcal H}|
\leq \|f\|_{\mathcal H}\sqrt{K(x,x)}.
$$

Consequently, if $\mathbb E_{X\sim p}[\sqrt{K(X,X)}]<\infty$, then

$$
|A_p(f)|\leq \|f\|_{\mathcal H}
\mathbb E_{X\sim p}[\sqrt{K(X,X)}],
$$

so the expectation functional is bounded. The Riesz representation theorem then gives its representer $\mu_p$, the mean embedding.

## 9. MMD as a distance between mean embeddings

Using the reproducing property,

$$
\begin{aligned}
\mathrm{MMD}[\mathcal H,p,q]
&=\sup_{\|f\|_{\mathcal H}\leq1}
\left|\mathbb E_p[f(X)]-\mathbb E_q[f(Y)]\right|\\
&=\sup_{\|f\|_{\mathcal H}\leq1}
\left|\langle f,\mu_p-\mu_q\rangle_{\mathcal H}\right|.
\end{aligned}
$$

By the Cauchy–Schwarz inequality, this is at most $\|\mu_p-\mu_q\|_{\mathcal H}$. If the difference is nonzero, equality is attained by choosing $f$ in the direction of $\mu_p-\mu_q$. Therefore,

$$
\mathrm{MMD}[\mathcal H,p,q]
=\|\mu_p-\mu_q\|_{\mathcal H}.
$$

The squared population MMD can be expanded as

$$
\begin{aligned}
\mathrm{MMD}^2[\mathcal H,p,q]
&=\|\mu_p-\mu_q\|_{\mathcal H}^2\\
&=\mathbb E_{X,X'\sim p}[k(X,X')]
+\mathbb E_{Y,Y'\sim q}[k(Y,Y')]\\
&\quad-2\mathbb E_{X\sim p,Y\sim q}[k(X,Y)],
\end{aligned}
$$

where $X'$ is an independent copy of $X$ and $Y'$ is an independent copy of $Y$. This is the kernel trick for MMD: the feature-space expectations can be calculated using kernel evaluations alone.

## 10. Injective mean embeddings and characteristic kernels

For MMD to distinguish all distributions in a chosen family, the mean embedding must be injective:

$$
\mu_p=\mu_q\implies p=q.
$$

This property does not hold for every kernel. If a feature representation captures too little information, different distributions may have the same embedding. A **characteristic kernel** is one for which the mean embedding is injective on the relevant class of probability measures.

A universal RKHS is one route to characteristicness under suitable assumptions. In a common compact-space setting, universality means that the RKHS is dense in $C(X)$ with respect to the uniform norm. Gaussian and Laplace kernels are standard examples of characteristic kernels under their usual domains and conditions.

A theorem in this area states, under appropriate assumptions, that for a universal kernel on a compact metric space,

$$
\mathrm{MMD}[\mathcal H,p,q]=0
\quad\Longleftrightarrow\quad p=q.
$$

## 11. Empirical MMD estimators

Given samples $X_1,\ldots,X_m$ from $p$ and $Y_1,\ldots,Y_n$ from $q$, an unbiased estimator of squared MMD (for the usual independent-sample setting) is

$$
\begin{aligned}
\widehat{\mathrm{MMD}}_{u}^{\,2}
&=\frac{1}{m(m-1)}\sum_{i\ne j}k(X_i,X_j)
+\frac{1}{n(n-1)}\sum_{i\ne j}k(Y_i,Y_j)\\
&\quad-\frac{2}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}k(X_i,Y_j).
\end{aligned}
$$

This estimator excludes diagonal terms in the within-sample sums. It is unbiased for the population squared MMD, although its realised value can be negative due to sampling variability.

A biased, non-negative V-statistic estimator includes the diagonal terms:

$$
\begin{aligned}
\widehat{\mathrm{MMD}}_{b}^{\,2}
&=\frac{1}{m^2}\sum_{i=1}^{m}\sum_{j=1}^{m}k(X_i,X_j)
+\frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}k(Y_i,Y_j)\\
&\quad-\frac{2}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}k(X_i,Y_j).
\end{aligned}
$$

When $m=n$, the unbiased estimator can also be expressed as a one-sample U-statistic using

$$
h((x_i,y_i),(x_j,y_j))
=k(x_i,x_j)+k(y_i,y_j)-k(x_i,y_j)-k(x_j,y_i).
$$

## 12. The MMD witness function

The witness function identifies the direction in the RKHS that maximises the difference between expectations. In the population setting, one maximiser is

$$
f^*=\arg\max_{\|f\|_{\mathcal H}\leq1}
\left(\mathbb E_{X\sim p}[f(X)]-\mathbb E_{Y\sim q}[f(Y)]\right).
$$

If $\mu_p\ne\mu_q$, the maximising unit-norm function is

$$
f^*(\cdot)=\frac{\mu_p-\mu_q}{\|\mu_p-\mu_q\|_{\mathcal H}},
$$

where elements of the RKHS are functions. Evaluating it at $t$ gives

$$
f^*(t)=\frac{\langle K(t,\cdot),\mu_p-\mu_q\rangle_{\mathcal H}}
{\|\mu_p-\mu_q\|_{\mathcal H}}
=\frac{\mathbb E_{X\sim p}[k(X,t)]-\mathbb E_{Y\sim q}[k(Y,t)]}
{\|\mu_p-\mu_q\|_{\mathcal H}}.
$$

Up to normalisation, the witness function is therefore the difference between the two kernel mean functions. Its empirical analogue replaces the expectations with sample averages:

$$
\widehat f(t)\propto
\frac1m\sum_{i=1}^{m}k(X_i,t)
-\frac1n\sum_{j=1}^{n}k(Y_j,t).
$$

The sign and magnitude of the witness function indicate where the two distributions differ in the RKHS representation.

## References

- Gaussian-process kernels: [Peter Roelants, Gaussian Process Kernels](https://peterroelants.github.io/posts/gaussian-process-kernels/)
- MMD: Arthur Gretton et al., *A Kernel Two-Sample Test* ([JMLR paper](https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf))

  
## Formula test

Inline formula: $x_i$

Block formula:

$$
x_i = \mathcal{N}(\mu,\sigma^2)
$$

