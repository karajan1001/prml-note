# 线性模型回归

##  线性基函数模型
线性回归方程

$$y(x,w) = w_0 + \sum_{j=1}^{M-1}w_j\phi_j(x) = \vec{w}^T\vec{\phi}(x)$$

其中$\phi_0(x) = 1$ 可以用**基函数**(basis function) 突破线性束缚，拟合非线性结果。
有很多种基函数选择，多项式，高斯，傅里叶，等等。
对线性模型求最小二乘可以得到

$$W_{ML} = (\Phi^t\Phi)^{-1} \Phi^Tt$$
这个方程要对矩阵求逆，当点数量很大时计算复杂度过高，所以一般采用*时序/在线*算法比如**stochastic gradient descent**

$$w^{\tau+1} = w^\tau - \eta\triangledown E_n$$

## 偏差-方差分解
偏差-方差模型是频率学家的分解方法。在平方差损失函数下，最佳预测为条件期望。总期望可以分解为

$$expected\_loss = (bias)^2 + variance + noise$$
其中：

$$(bias)^2 = \int \{E_D[y(x;D)] - h(x)\}^2p(x) dx$$ 

$$variance= \int E_D[\{y(x;D) - E_D[y(x;D)]\}^2]p(x)dx$$ 

$$noise=\int\{h(x) - t\}^2 p(x,t)dxdt$$

其中噪音为采样带来的差异。偏差为模型和真实分布的差异，方差为选择不同数据集时得到模型不同带来的差异。灵活的模型会带来更小的偏差，但是随着数据集不同模型会有较大的方差。而不那么复杂的模型会带来更小的方差，不过更大的偏差。

## 贝叶斯线性回归
贝叶斯方法天生就能避免过拟合，而且可以只依靠训练数据就决定模型复杂性。
### 参数分布
如果参数满足先验概率：

$$p(w) = N(w|m_0, S_0)$$
则加上数据后后验概率为：

$$p(w|t) = N (w|m_N, S_N)$$
其中

$$m_N = S_N(S^{-1}_0m_0 + \beta\Phi^Tt)\\
		S^{-1}_N = S_0^{-1} + \beta \Phi^T\Phi$$
如果N=0则退化到先验概率，如果$S_0 = -\infty$也就是先验概率分布在无穷广的空间，则后验概率变为$w_{ML}$。

### 等效核
`未看懂`
## 贝叶斯模型选择
在贝叶斯理论中对于数据集$D$从模型${M_i}$中某个模型生成，$D$包含(x,t)对。则其概率为

$$p(M_i|D) = p(M_i)p(D|M_i)$$

其中$p(D|M_i)$模型证据(**model evidence**)有被称为边缘似然(**marginal likeli**)。两个模型的边缘似然相除$p(D|M_i)/p(D|M_j)$又叫贝叶斯系数(**bayes factor**)。一旦给定了后验概率就能知道在给定数据集，给定x的条件下t的概率

$$p(t|x,D) = \sum_{i=1}^Lp(t|x,M_i,D)p(M_i|D)$$
选取可能性最高的模型是这个理论的一个简单近似，也叫模型选择。
一个简单近似对单个参数

$$p(D) = \int p(D|w)p(w)dw \simeq p(D|w_{MAP})\frac{\Delta w_{posterior}}{\Delta w_{prior}}$$
取对数对于M个参数

$$p(D) \simeq lnp(D|w_{MAP}) + M ln(\frac{\Delta w_{posterior}}{\Delta w_{prior}})$$
可以看到随着参数增加第二项减少。简单模型生成的数据集集中在少数范围，复杂模型生成的数据集则分布在一个很广的范围。对于中等复杂度的数据集，简单模型无法生成这个数据集这叫欠拟合，而复杂模型相反，可能的数据集过于广泛，所以缺少足够证据证明现有数据确实是来自这个模型的。

## 证据近似
证据近似(**evidence approximation**): 一种近似，将超参设为最大边缘似然方程中的最大值，计算方式是对不同参数$w$积分。边缘超参的最大值为$$p(\alpha, \beta|t) \propto p(t|\alpha,\beta)p(\alpha,\beta)$$  如果$\alpha,\beta$的先验概率是均匀的，则等效为最大化$p(t|\alpha,\beta)$
### 证据方程的演化
$$p(t|\alpha, \beta) = \int p(t|w,\beta)p(w|\alpha)dw\\= (\frac{\beta}{2\pi})^{N/2}(\frac{\alpha}{2\pi})^{M/2}\int exp\big\{-E(w)\big\} dw \\ ln(p(t|\alpha,\beta)) = \frac M2 ln\alpha + \frac N2 ln\beta - E(m_N) - \frac12ln|A| - \frac N2 ln(2\pi)$$
### 最大化证据方程
最大化$p(t|\alpha,\beta)$可以解出$\alpha和\beta$，和最大似然方程需要验证集不同$\alpha和\beta$都可以直接从数据集中解出。
### 参数的有效个数
`未看懂`

### 固定基的局限性
固定基在数据测量之前就确定，而且会带来维度灾难。不过这些生成的线性基大多具有一定联系，所以实际空间比看起来要小。而且我们可以只使用本地坐标基，这项技术在神经网络和SVM中运用很广。

