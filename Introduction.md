\section{Introduction}

Since the advent of ChatGPT, generative models have become the latest craze as far as Machine Learning is concerned. For the past year, \textit{"GenAI"} has flooded the news, becoming a trending topic as well in non-technical audience as in the scientific community. New models are being released nearly on a daily basis, with numerous applications ranging from text generation \cite{touvron2023llama, vaswani2017attention, devlin2018bert, scao2022bloom} to image generation \cite{ho2020denoising, shaham2019singan, goodfellow2014generative}, by way of music or video generation \cite{chan2019everybody}. The latest developments are showing impressive results in terms of multi-modal generation \cite{ramesh2021zero, copet2023simple, kreuk2023audiogen}.

In spite of the recent hype, Generative Modeling is not a new concept. Indeed, it has been around for quite some time in the Statistical and Machine Learning community \cite{ng2001discriminative} and several mathematical results that we will present can be traced back to the early 2010's. The latest advances have been however made possible as technical progresses have brought a so-far unmatched computational power, leveraging the possibilities offered by efficient GPU programming for example \cite{charlier2021kernel, paszke2017automatic, abadi2016tensorflow}.

We note $\mathcal{D}_N = \{(\mathbf{x}_n,y_n), 1\leq n \leq N \}$ a set composed of data instances and their associated labels. Formally, while discriminative models are trying to capture the conditional probability $p(y | \mathbf{x})$, generative models estimate the joint probability $p(\mathbf{x},y)$ (or solely $p(\mathbf{x})$ in the absence of labels). We consider $N$ samples $\{\mathbf{x}_n \in \mathbb{R}^d\}_{1\leq n \leq N}$ independent and identically distributed according to an unknown probability distribution $\mathbf{x}_n \sim p_{data}(\mathbf{x})$. In the classical statistical approach, we consider a family of parameterized models $\mathcal{P} = (p_{\theta})_{\theta \in \mathbb{R}^d} $ and we want to choose a model $p_{\theta}$ such that $p_{\theta}(\mathbf{x}) \approx p_{data}(\mathbf{x})$. One can do so by comparing distances between probability distributions \cite{feydy2020geometric}, using metrics such as the Kullback-Leibler divergence yielding maximum likelihood \cite{arjovsky2017wasserstein} or the Fisher-Rao distance if the geometry is propitious.
Modelling a probability distribution is however more delicate than performing a classification task: one of the main challenge has been that as the model becomes more complex, density estimation rapidly becomes impractical due to an intractable normalizing term. This issue only worsens with high-dimensional data. In such setting, the problem can no longer be directly considered and needs clever techniques to circumvent this intractable partition function.

Behind the catch-all term generative models, we can distinguish between two main different types: likelihood-based models and implicit generative models. The former category contains models such as autoregressive models \cite{van2016pixel, larochelle2011neural}, normalizing flow models \cite{rezende2015variational, dinh2014nice}, energy-based models (EBMs), and variational auto-encoders (VAEs) \cite{kingma2022autoencoding, doersch2021tutorial}. These explicitly learn the probability density function of the distribution by maximizing the likelihood of observed data (or an approximation for VAEs), and their parameters hold an interpretable probabilistic significance. However, due to the issue of intractable partition function, their application is either limited to specific architectures (for autoregressive and normalizing flow models), or rely on an adapted objective function (evidence-lower bound for VAEs and contrastive divergence in EBMs). In contrast, implicit generative models, as generative adversarial networks (GANs) \cite{goodfellow2014generative, jolicoeurmartineau2018relativistic}, represent the probability distribution only implicitly as they focus on modelling the sampling process. For instance, to go on with GANs, they utilize a game-theoretic framework where a generator and discriminator respectively learn to produce and differentiate between real and synthetic data. As a consequence, the parameters of implicit generative models lack a direct probabilistic interpretation, as the emphasis lies on generating realistic samples rather than explicitly defining the probability distribution. Although they avoid the pitfall of the normalizing term, GANs often grapple with issues related to training stability as they require adversarial training.

In this context, we introduce score-based generative models as an alternative to the aforementioned models. We will define in Section $2$ the mathematical object and place it in a generative context using Langevin dynamics. In Section $3$, we will present several techniques and results making possible the estimation of this score function (i.e., score matching). We will state in Section 4 the two main contributions discussed in \citet{song2019generative}, namely the addition of Gaussian noise on the data and annealed Langevin dynamics, an adapted sampling algorithm. Finally, we will make some comments about the presented approach in Section 5 and conclude in Section 6.

\section{Score-Based Generative Modeling}
We consider a parametrized density model $p_{\theta}(\mathbf{x})$. We want to estimate the vector of parameters $\theta$ such that $p_{\theta}(\mathbf{x}) \approx p_{data}(\mathbf{x})$. Using energy-based formulation, the density can be written as follows:
\begin{equation*}
    p_{\theta}(\mathbf{x}) = \frac{e^{q_{\theta}(\mathbf{x})}}{Z(\theta)}
\end{equation*}
where $Z(\theta) = \int_{\mathbf{x}}e^{q_{\theta}(\mathbf{x})}d\mathbf{x}$ is the (usually) intractable normalizing constant.

\subsection{Score function}
We define (\textit{Stein) score} \cite{liu2016kernelized} function as follows:
\begin{equation}
    \mathbf{s_{\theta}(x)} \triangleq \nabla_{\mathbf{x}} \log p_{\theta}(\mathbf{x})
\end{equation}
We emphasize that this score function is different from Fisher's Information Score which we more commonly come across in statistics: the latter is obtained by taking the gradient of the log-likelihood function w.r.t. the parameters $\theta$, whereas Stein's score is computed by differentiating w.r.t. the data.\\
We can then write:
\begin{equation*}
    \mathbf{s_{\theta}(x)} = \nabla_{\mathbf{x}}q_{\theta}(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}} \log Z(\theta)}_{=0} 
\end{equation*}
Therefore, the pivotal advantage of score-based models lies in their ability to circumvent the intractable normalizing term. 

\subsection{Langevin Monte-Carlo Sampling}
We now place the score function in a context of generative modeling. Since we no longer estimate distribution densities, we need to use an adapted method in order to sample from the distribution knowing solely its score function.

In the statistical community, this matter has been extensively studied as part of Monte-Carlo Markov Chain (MCMC) methods \cite{welling2011bayesian, Roberts1998OptimalSO}. We recall that these methods aim at neatly-building stochastic processes under the form of ergodic Markov chains, for which the stationary distribution is the target distribution from which we want to sample \cite{roberts2004general}. Therefore, by letting such-built chains run for enough time, we obtain samples from the target distribution. Among the numerous algorithms available, a range of methods have been developed with the objective of incorporating the structure of the target density in the process. Building on the well-known idea behind gradient ascent, these algorithms have integrated the gradient of the distribution in the MCMC procedure: this term will have the effect of pushing the Markov chain towards values having higher density. Besides the mere intuition, such processes relies on strong theoretical justifications based on Langevin's diffusions.

We define the overdamped Langevin Itô diffusion as follows:
\begin{equation}
    d\mathbf{x}_t \triangleq \frac{1}{2}\nabla_{\mathbf{x}}\log p_{\theta}(\mathbf{x}_t)dt + d\mathbf{B}_t
\end{equation}
where $\mathbf{B}_t$ is a Brownian motion. It has been proven that the stationary distribution of this diffusion is the density $p_{\theta}$ \cite{roberts1996exponential}. 

\subsubsection{Metropolis-Adjusted Langevin Algorithm}
In practice, we consider discrete-time approximations of this diffusion to approach its paths. Using Euler–Maruyama method, we obtain the following discrete-time stochastic process:
\begin{equation}\label{eq:langevin}
    \mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\epsilon}{2}\nabla_{\mathbf{x}}\log p_{\theta}(\mathbf{x}_{t-1}) + \sqrt{\epsilon}\mathbf{z}_t
\end{equation}
where $\mathbf{z}_t \sim \mathcal{N}(0,I_d)$ is a standard Gaussian noise and $\epsilon > 0$ is the step size. Because of the bias introduced by the discrete-time approximation, we note that (\ref{eq:langevin}) no longer targets $p_{\theta}$. To overcome this drift, methods have been proposed to correct the resolution by adding a Metropolis-Hasting step: the update rule becomes a proposal step followed by an acceptance step. The proposal step is accepted with probability:
\begin{equation}
    \alpha(\mathbf{x}_{t-1},\mathbf{x}_t) = \min\left(1, \frac{p_{\theta}(\mathbf{x}_t)q(\mathbf{x}_{t-1}|\mathbf{x}_t)}{p_{\theta}(\mathbf{x}_{t-1})q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right)
\end{equation}
where $q(\mathbf{x}_t|\mathbf{x}_{t-1}) \propto \exp\left(-\frac{1}{2\epsilon}||\mathbf{x}_t - \mathbf{x}_{t-1} - \frac{\epsilon}{2}\nabla_{\mathbf{x}}\log p_{\theta}(\mathbf{x}_{t-1})||_2^2\right)$ is the proposal distribution. We note here that as in classical Metropolis-Hasting, we can proceed to the acceptance step even when solely knowing the density up to a constant as it will vanish in the quotient.\\
This method is known as Metropolis-adjusted Langevin algorithm (MALA) \cite{roberts1996exponential}.\\
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\linewidth]{images/score_based_generative/gmm_traj.png}
  \includegraphics[width=0.8\linewidth]{images/score_based_generative/banana_traj.png}
  \caption{Sample Trajectory (red points) Obtained with Euler-Maruyama Method for Langevin Dynamics using the exact score functions. The original unnormalized densities are encoded with a warm colormap: warmer color implies higher density. The chains rapidly head towards high-densities regions.}
  \Description{}
\end{figure}
\begin{figure}[h] \label{fig:mala}
  \centering
  \includegraphics[width=1\linewidth]{images/score_based_generative/gmm_sampling.png}
  \includegraphics[width=1\linewidth]{images/score_based_generative/banana_sampling.png}
  \caption{\textbf{Left}: Original Unnormalized Densities; \textbf{Right}: Estimated Densities with Metropolis-Adjusted Langevin Algorithm using the exact score functions. In accordance with the original paper, we observe that despite having added a Metropolis-Hasting correction, the Langevin dynamics cannot estimate properly the mixing weights in the case of multimodal distribution.}
  \Description{}
\end{figure}
We performed numerical experiments to simulate the MALA. The toy distributions used are described in Appendix \ref{app:a}, and the parameters used are given in Appendix \ref{app:b}.

\subsubsection{Unadjusted Langevin Algorithm}
In the original paper, the Metropolis-Hasting step is discarded and the update is accepted at each iteration. As the bias fade out when $\epsilon \rightarrow 0$ and $t \rightarrow \infty$, this choice was explained by assuming the error negligible when $\epsilon$ is small and $t$ is large.
This equation is known as the Unadjusted Langevin Algorithm (ULA) and recent theoretical work has showed that it enjoyed convergence bounds without the correction step \cite{durmus2017nonasymptotic, debortoli2021efficient, dalalyan2017theoretical}.
Numerical results obtained with this version of the algorithm matched really closely the ones with the acceptance step.

%%%%% Section 2
\section{Score Matching}

To take advantage of these sampling algorithms, we now turn to the estimation of the score function. While the numerical results have been obtained using the exact score functions, there is no reason why one could obtain such closed-form formulas when facing a dataset. We therefore need to adapt to our problem the classical statistical approach detailed in the introduction. We consider a set of $N$ i.i.d. samples $\{\mathbf{x}_n\}_{1\leq n \leq N}$ drawn from the unknown data distribution $\mathbf{x} \sim p_{data}$. We want to estimate the score function $\mathbf{s_{\theta}} : \mathbb{R}^d \rightarrow \mathbb{R}^d$ such that $\mathbf{s_{\theta}(x)} \approx \nabla_{\mathbf{x}}\log p_{\theta}(\mathbf{x})$. While we could use maximum likelihood for example to estimate $\theta$ when dealing with density functions, here we no longer compare probability distributions but rather vector fields. We thus introduce the Fisher divergence as a new objective function:
\begin{equation}\label{eq:explicit_sm}
    J_{\textit{ESM}}(\theta) = \frac{1}{2}\mathbb{E}_{p_{data}(\mathbf{x})}\left[||\text{s}_{\theta}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{data}(\mathbf{x})||_2^2 \right]
\end{equation}
This approach is called \textit{Score Matching}, and we will discuss three different methods to estimate this score function.
Numerical estimations can be found in Appendix (\ref{app:3}).

\subsection{Implicit Score Matching}
The first main result when it comes to score matching is the following theorem \cite{hyvarinen2005estimation}:
\begin{theorem}[Implicit Score Matching]
    Assuming that the score function is differentiable and under some regularity conditions ensuring existence of all the terms, the Fisher divergence (\ref{eq:explicit_sm}) can be written as:
    \begin{equation}
        J_{\textit{ISM}}(\theta) = \mathbb{E}_{p_{data}(\mathbf{x})} \left[\text{tr}\left( \nabla_{\mathbf{x}} \text{s}_{\theta}(\mathbf{x}) \right) + \frac{1}{2} ||\text{s}_{\theta}(\mathbf{x})||_2^2 \right] + c
    \end{equation}
    where $c$ is a constant term that does not depend on $\theta$.
\end{theorem}
This formulation is quite spectacular as it yields an \textit{Implicit Score Matching} (ISM) objective equivalent to the \textit{Explicit Score Matching} (\ref{eq:explicit_sm}) that no longer depends on score of the unknown data distribution $p_{data}(\mathbf{x})$. In practice, the computation of the Jacobian $\nabla_{\mathbf{x}} \text{s}_{\theta}(\mathbf{x})$ becomes really expensive when facing high-dimensional data (e.g., images) thereby limiting this approach.

\subsection{Denoising Score Matching}
To overcome the issue of high-dimensional data, \citet{vincent2011connection} exhibits an unexpected connection between score matching and denoising auto-encoders. Denoising autoencoders are a particular type of auto-encoders that are trained to reconstruct the original data from a corrupted version of it. We thus consider pairs of clean and corrupted data $(\mathbf{x},\mathbf{\tilde{x}})$ where $\mathbf{\tilde{x}}$ is obtained by adding Gaussian noise $\mathbf{\tilde{x}} \sim \mathcal{N}(\mathbf{x},\sigma^2I_d)$. 
This density can be written using the multivariate isotropic Gaussian kernel with variance $\sigma^2$:
\begin{equation}\label{eq:gaussian_kernel}
    q_{\sigma}(\mathbf{\tilde{x}}|\mathbf{x}) \triangleq \frac{1}{(2\pi\sigma^2)^{d/2}}\exp\left(-\frac{1}{2\sigma^2}||\mathbf{\tilde{x}} - \mathbf{x}||_2^2\right)
\end{equation}
Using kernel density estimation, we can define the perturbed data distribution as follows:
\begin{equation}\label{eq:joint_dist}
    q_{\sigma}(\mathbf{\tilde{x}}) \triangleq \int_{\mathbf{x}} q_{\sigma}(\mathbf{\tilde{x}}|\mathbf{x})p_{data}(\mathbf{x})d\mathbf{x} = \int_{\mathbf{x}} q_{\sigma}(\mathbf{\tilde{x}},\mathbf{x}) dx
\end{equation}
The explicit score matching objective for the perturbed data distribution thus write as follows:
\begin{equation} \label{eq:explicit_sm_perturbed}
    J_{\textit{ESM}_{\sigma}}(\theta) = \frac{1}{2}\mathbb{E}_{q_{\sigma}(\mathbf{\tilde{x}})} \left[||\text{s}_{\theta}(\mathbf{\tilde{x}})-\nabla_{\mathbf{\tilde{x}}} \log q_{\sigma}(\mathbf{\tilde{x}})||_2^2 \right]
\end{equation}
\citet{vincent2011connection} then proceeds to demonstrate the following result:
\begin{theorem}[Denoising Score Matching]
    Under weak conditions of log-differentiability, the explicit score matching objective for the perturbed data distribution (\ref{eq:explicit_sm_perturbed}) is equivalent to the following \textit{Denoising Score Matching} (DSM) objective:
    \begin{equation}
        J_{\textit{DSM}_{\sigma}}(\theta) = \frac{1}{2}\mathbb{E}_{q_{\sigma}(\mathbf{\tilde{x}},\mathbf{x})} \left[||\text{s}_{\theta}(\mathbf{\tilde{x}}) - \nabla_{\mathbf{\tilde{x}}} \log q_{\sigma}(\mathbf{\tilde{x}}|\mathbf{x})||_2^2 \right]
    \end{equation}
    over the joint distribution of clean and corrupted data as previously expressed in (\ref{eq:joint_dist}).
\end{theorem}
While the score of perturbed data distribution $q_{\sigma}\nabla_{\mathbf{\tilde{x}}} \log q_{\sigma}(\mathbf{\tilde{x}})$ has no reason to be explicitly known, the considered quantity in Denoising Score Matching possesses the appreciable following property:
\begin{equation}
    \nabla_{\mathbf{\tilde{x}}} \log q_{\sigma}(\mathbf{\tilde{x}}|\mathbf{x}) = - \frac{1}{\sigma^2}(\mathbf{\tilde{x}} - \mathbf{x})
\end{equation}
We can thus write the Denoising Score Matching objective as follows:
\begin{equation}
    J_{\textit{DSM}_{\sigma}}(\theta) = \frac{1}{2}\mathbb{E}_{q_{\sigma}(\mathbf{\tilde{x}},\mathbf{x})} \left[||\text{s}_{\theta}(\mathbf{\tilde{x}}) + \frac{1}{\sigma^2}(\mathbf{\tilde{x}} - \mathbf{x})||_2^2 \right]
\end{equation}
Finally, \citet{vincent2011connection} shows that performing score matching on the Parzen density estimate defined in (\ref{eq:joint_dist}) is equivalent to training a denoising autoencoder on data with conditional density defined in (\ref{eq:gaussian_kernel}).
The main advantage of this method is that it is computationally efficient as it only requires to train a denoising auto-encoder. However, choosing $\sigma$ must be done cautiously: to ensure that the perturbed data distribution is close to the original data distribution, $\sigma$ must be sufficiently small.

\subsection{Sliced Score Matching}

The last score matching technique considered in the original article relies on \cite{song2020sliced}. In this paper, the authors circumvent the heavy computation required in implicit score matching with a technique called \textit{sliced score matching} which consists in solely computing projections of the Jacobian along random vectors. The objective then writes as follows:
\begin{equation}
    J_{\textit{SSM}}(\theta) = \mathbb{E}_{p_{\mathbf{v}}} \mathbb{E}_{p_{data}(\mathbf{x})} \left[||\mathbf{v}^T\nabla_{\mathbf{x}} \text{s}_{\theta}(\mathbf{x})\mathbf{v} + \frac{1}{2}||\text{s}_{\theta}(\mathbf{x})||_2^2 \right]
\end{equation}
where $p_{\mathbf{v}}$ is a distribution over random vectors (e.g., $\mathbf{v} \sim \mathcal{N}(0,I_d)$). This estimator has been proven to be consistent and asymptotically normal.
While this technique has the advantage of approximating the unperturbed data distribution, the authors mention that it is more computationally expensive than denoising score matching.

\begin{figure}[H] 
  \centering
  \includegraphics[width=1\linewidth]{images/score_matching/gmm_sm.png}
  \includegraphics[width=1\linewidth]{images/score_matching/banana_sm.png}
  \caption{\textbf{Left}: $\nabla_{\mathbf{x}} \log p_{data}(\mathbf{x})$; \textbf{Right}: $\mathbf{s_{\theta}(x)}$ (SSM). The scores have been normalized and are encoded using a warm colormap: warmer color implies a higher value for the norm of score. As one gets closer to the data point cloud, the scores' norm decrease and the estimation obtained by score matching improves.}
  \label{fig:score_matching}
  \Description{}
\end{figure}

%%%%% Section 3


\section{Improvement of Score-Matching Generative Modeling}






\section{Improving Score-Matching with Perturbed Data and Annealed Langevin Dynamics}

Despite clever techniques for approximating the score function of unknown data distribution, the original paper highlights several shortcomings with the generating process as it is. 

\subsection{Noise Conditional Score Network}

The first consideration made builds on a discussion element we mentioned when defining denoising score matching: the choice of $\sigma$. On one hand, the objective defined by denoising score matching needs a small value for $\sigma^2$ to ensure that the perturbed data distribution is close to the original data distribution. However, a trade-off has to be made when tuning this parameter. Indeed, as noise is added to the data, the perturbed data distribution will be more spread out, allowing to better estimations of the score function in the low-densities regions of the distribution support. Since the score matching is made through empirical mean, we would not be able otherwise to estimate the score function in these regions. 

Secondly, it has been shown that most of the high-dimensional data distributions are actually concentrated on a low-dimensional manifold \cite{bengio2013representation, mcinnes2018umap, cayton2005algorithms}, itself embedded in the high-dimensional space. This phenomenon is known as the \textit{maniafold hypothesis}. In our context, this would mean that the score function is not defined in the whole space but rather on this lower-dimensional manifold. This is problematic as the hypotheseses allowing to go from \textit{explicit} score matching to \textit{implicit} score matching are no longer verified. Indeed, the score function is no longer differentiable everywhere and the Jacobian is no longer defined.

To simultaneously overcome these issues, \citet{song2019generative} propose to perturb the data with a geometric decreasing sequence of Gaussian noise defined as follows:
\begin{equation}
    \{\sigma_i \in \mathbb{R}_+^*\}_{1\leq i \leq L} \text{ s.t. } \frac{\sigma_1}{\sigma_2} = \dots = \frac{\sigma_{L-1}}{\sigma_L} > 1
\end{equation}
The rationale behind this choice is that the first levels of noise will allow to estimate the score function in the low-densities regions of the distribution support, while the last levels of noise will become sufficiently small to allow to accurately approximate the unknown score function. 
As each noise level will require a different score function, it seems impractical to train a different model for each level of noise as we might need a large number of noise levels to properly estimate the score function. The authors then propose to train a \textit{Noise Conditional Score Network} (NCSN) to simultaneously estimate the score functions for each level of noise. The NCSN takes as input a data point and a level of noise and outputs an estimation of the score function. The Denoising Score Matching objective is then adapted to optimize over all noise levels:
\begin{equation}
    J_{\textit{NCSN}}(\theta) = \frac{1}{L} \sum_{i=1}^L \mathbb{E}_{q_{\sigma_i}(\mathbf{\tilde{x}},\mathbf{x})} \left[||\text{s}_{\theta}(\mathbf{\tilde{x}}, \sigma_i) + \frac{1}{\sigma_i^2}(\mathbf{\tilde{x}} - \mathbf{x})||_2^2 \right]
\end{equation}
In practice, the authors have introduced a term correcting the magnitude orders, yielding the following objective:
\begin{equation}
    J_{\textit{NCSN}}(\theta) = \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{q_{\sigma_i}(\mathbf{\tilde{x}},\mathbf{x})} \left[||\text{s}_{\theta}(\mathbf{\tilde{x}}, \sigma_i) + \frac{(\mathbf{\tilde{x}} - \mathbf{x})}{\sigma_i^2}||_2^2 \right]
\end{equation}
where $\lambda(\sigma_i) = \sigma_i^2$. The authors have empirically observed that this correction term allows to obtain better results and improves the convergence of the algorithm.
 




\subsection{Annealed Langevin Dynamics}

Secondly, as noticed when reproducing experiments (Figure \ref{fig:mala}), the Langevin dynamics is not able to properly estimate the mixing weights in the case of multimodal distributions. This is no news as this issue was already known in the context of MCMC methods: a larger step size $\epsilon$ will lead to a better mixing, allowing the chain to "jump" between modes, but will also lead to a less accurate approximation of the target distribution, with an acceptance rate that will tend to 0 in a high-dimensional setting. Conversely, a smaller step size will lead to a better approximation of the target distribution, but will also lead to a poor mixing as the chain won't be able to efficiently explore the support of the distribution. To bypass these issues, the idea of perturbing the data seems promising in this context as well: by adding noise to the samples, the chain will be able to explore the support of the distribution more efficiently, while still being able to approximate the target distribution. In this regard, \citet{song2019generative} propose a modified version of the Langevin dynamics, called \textit{annealed Langevin dynamics}. The algorithm will sequentially run $L$ different chains, one for each noise level $\sigma_i$, and use the samples obtained at the previous level of noise as starting points for the next level. Concomitantly, the step size used in the Langevin dynamics will be decreased as the noise level increases (we recall that a large noise level actually corresponds to a small perturbation of the data, c.f. \ref{eq:noise_levels}). This algorithm is called \textit{annealed} as a parallel can be made with a cooling process: the initial distribution is heated up by adding noise to the data, making it smoother over an extended support, and then gradually cooled down as the noise level decreases, allowing to obtain samples from the original distribution. The annealed Langevin dynamics is summarized in Algorithm \ref{alg:annealed_langevin}.
\begin{algorithm}[H]
\caption{Annealed Langevin Dynamics}
\label{alg:annealed_langevin}
\begin{algorithmic}[1]
\STATE \textbf{Input:} Data $\{\mathbf{x}_n\}_{1\leq n \leq N}$, number of noise levels $L$, noise levels $\{\sigma_i\}_{1\leq i \leq L}$, step sizes $\{\epsilon_i\}_{1\leq i \leq L}$, number of iterations $T$
\STATE \textbf{Output:} Samples $\{\mathbf{x}_n\}_{1\leq n \leq N}$
\STATE Initialize $\{\mathbf{x}_n\}_{1\leq n \leq N}$
\FOR{$i=1$ to $L$}
    \FOR{$t=1$ to $T$}
        \STATE Sample $\{\mathbf{z}_n\}_{1\leq n \leq N} \sim \mathcal{N}(0,I_d)$
        \STATE $\mathbf{x}_n \leftarrow \mathbf{x}_n + \frac{\epsilon_i}{2}\nabla_{\mathbf{x}}\log p_{\theta}(\mathbf{x}_n) + \sqrt{\epsilon_i}\mathbf{z}_n$
    \ENDFOR
    \STATE $\epsilon_i \leftarrow \frac{\epsilon_i}{2}$
\ENDFOR
\end{algorithmic}
\end{algorithm}
The authors have empirically observed that this algorithm allows to obtain better results, especially when dealing with multimodal distributions. 

\section{Discussions}

Reflecting on the proposed work, we can make several comments. First of all, the authors have proposed an innovative approach to generative modeling, by introducing score-based models. This approach has the advantage of circumventing the intractable normalizing term, while still being able to sample from the target distribution. As this approach requires to estimate the score function of the target distribution, the authors have built on several theoretical results in order to propose techniques to estimate this score function, namely implicit score matching, denoising score matching and sliced score matching. Facing issues to sample efficiently from image distributions, the authors have proposed to perturb the data with Gaussian noise of various magnitudes, and to train a Noise Conditional Score Network to simultaneously estimate the score functions for each level of noise. This approach has the advantage of being computationally efficient, while still being able to explore the distribution support. Finally, the authors have proposed to use an annealed version of Langevin dynamics, which allows to obtain better results, especially when dealing with multimodal distributions. When the paper was published, the authors showed that their approach obtained state-of-the-art results on the CIFAR-10 dataset, generated very realistic samples from the MNIST and CelebA datasets and was able to learn meaningful representations by performing image inpainting. At the time, GANs were the ones-to-beat in terms of image generation. The authors therefore stress the fact that their method is able to produce samples of comparable quality while not requiring any adversarial training, making it supposedly less prone to training instability. 

During our work, we designed experiments so as to assert the justifications and choices made by the authors. In the following, we extend on two aspects: the choice of the noise levels and parameters, and the computational aspect of the proposed approach.

\subsection{Noise levels}

In the original paper, some recommendations are made regarding the choice of the noise levels. First of all, the authors recommend to choose the noise levels such that $\frac{\sigma_1}{\sigma_2} = \dots = \frac{\sigma_{L-1}}{\sigma_L} > 1$. This choice is motivated by the fact that the noise levels should be chosen diverse enough to allow to estimate the score function in the low-densities regions of the distribution support, while still being able to accurately approximate the unknown score function. However, the authors do not provide any theoretical justification on how to appropriately select either the number of noise levels or the noise levels themselves. The experiment they describe for images with pixel values between $0$ and $1$ used $L=10$, with $\sigma_1 = 1$ and $\sigma_{10} = 0.01$. In our experiments, we have observed that this choice is not necessarily optimal. Indeed, even by adding a Gaussian noise of variance $\sigma_1 = 1$, there were still areas of the support that were poorly estimated with denoising score matching. The number of levels lacks interpretation as well and might strongly depend on the dataset considered. One could easily motivate the selection of less levels but with noise levels further apart. We could also consider the class of Interacting MCMC methods, and more specifically an adapted version of the Parallel Tempering algorithm \cite{geyer1991computing}. This algorithm is based on the idea of running several chains in parallel, each of them targeting a different temperature. The chains are then allowed to stochastically swap their temperatures according to an acceptance criterion, allowing to explore the distribution support more efficiently.
\newline A second parameter linked to the noise levels is the scaling factor introduced in the loss function. The authors have empirically observed that the loss function was more stable when using a scaling factor $\lambda(\sigma_i) = \sigma_i^2$. This choice was motivated by the fact that it allowed to observe orders of magnitude for $\lambda(\sigma_i) \mathbb{E}_{q_{\sigma_i}(\mathbf{\tilde{x}},\mathbf{x})} \left[||\text{s}_{\theta}(\mathbf{\tilde{x}}, \sigma_i) + \frac{(\mathbf{\tilde{x}} - \mathbf{x})}{\sigma_i^2}||_2^2 \right]$ that were similar for all noise levels. However, one might consider to weight the loss function differently, with the idea of penalizing more the noise levels that are closer to the original data distribution. This could be done by introducing a scaling factor $\lambda(\sigma_i) = \frac{1}{\sigma_i^2}$ for example. This would allow to better estimate the score function in the high-densities regions of the distribution support.

\subsection{Computational aspect}

The second aspect we will mention is linked to the computational power required for such models. The authors mention having trained their models on powerful GPUs (one Titan XP GPU for MNIST, two Titan XP GPUs for CIFAR-10 and CelebA). In our experiments, we were unfortunately short of such resources, and thus restrained ourselves to toy experiments. The drawback from this is that all of the aforementioned considerations and potential intuitions could not be supported. Indeed, the curse of dimensionality renders useless considerations made on toy data as unexpected phenomena arise when dealing with high-dimensional data such as images.

On another note, as the sampling process relies on MCMC method, it might be interesting to have an estimate of the number of steps required for the burn-in phase in order to assess the scalability of such sampling.

\section{Conclusion}

To conclude, we have presented the main ideas behind score-based generative modeling, as well as the main contributions of \citet{song2019generative}. We have then discussed several aspects of the proposed approach, and supported our remarks with numerical experiments.

It is important to note that, since 2019, the authors wrote two follow-up papers for this article that have known great recognition in the community. In \citet{song2020improved}, the authors have addressed several issues related to the training process. They provide a serie of recommandations for hyperparameter tuning that showed significant improvements in terms of sample quality, especially for high-quality images (up to $256 \times 256$ resolution). In \citet{song2021scorebased}, the authors have proposed a way to considerably improve the sampling process: instead of a discrete diffusion process, they propose a method reversing a stochastic differential equation (SDE) to obtain a continuous-time diffusion process. This method is equivalent to having an infinite level of noise, and thus allows to sample images from pure noise (Figure \ref{fig:cover}). 