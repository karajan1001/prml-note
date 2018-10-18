# 介绍
>模式识别是`利用计算机算法自动从数据中找出规律，并且利用这些规律进行决策比如将数据分类。`
>比如一个案例，训练集（*training set*）有N个数字$\left \{ X_{1},X_{2},\cdots ,X_{N} \right \}$，每个数字有一个手动输入的类别，用向量**t**表示。机器学习算法的作用就是一个输入$X$输出其分类$y$的函数$y\left(x\right)$。函数$y\left(x\right)$的精确表示由训练集决定。

- **泛化(generalization)**: 模型对测试集的预测能力，也就是算法忽略次要特征，抓住主要特征的能力，泛化能力越高则训练集和测试集的准确率越相似。
- **特征抽取(feature extraction)**：原始输入经过预处理变化到一个新空间，在新空间中想要模式识别会更加容易。预处理还有一个功能降维，可以加快处理速度。不过特征抽取需要小心谨慎，不然有用信息可能被抛弃。
- **监督学习(supervised learning)**：训练样本集已经标注了目标向量，监督学习包含两类。
	- **分类(classification)**：目标结果离散，并且只有有限的可能性。
	- **回归(regression)**：目标结果中包含连续并且有无限的可能性。
- **无监督学习(unsupervised learning)**：训练样本不包含目标向量，无监督学习包含三种主要应用。
	- **聚类(clustering)**：找出数据中相似的群体。
	- **密度估算(density estimation)** ：计算输入数据的分布。
	- **可视化(visualization)**：将高维数据变换到二维或者三维空间。
- **增强学习(reinforcement learning)**：选择特定的步骤使得回报最大化。和监督学习不同，样本集并不包含各种情况的最优解，而是要靠算法不断试错得出。
- **转导学习(tranductive learning)**：

>机器学习的三个重要理论工具：
1. **概率论**：
2. **决策论**：
3. **信息论**：
## 多项式拟合

- **线性模型(linear model)**：未知参数为线性的模型。比如多项式方程，其中$\omega$是未知参数。
- 
$$y\left( x, \omega \right) = \omega_{0} + \omega_{1}x + \omega_{2}x^{2} + \cdots + \omega_{M}x^{M} = \sum_{j=0}^{M}\omega_{j}x^{j}$$

- **误差方程(error function)**：表示训练集和预测值的误差的方法。比如常用的误差的平方和。

$$E\left( \omega\right) = \frac{1}{2} \sum_{n=1}^{N} \left\{y\left(x_{n},\omega\right) - t_{n}\right\}^{2}$$

- **模型比较和模型选择(model comparison and model selection)**：比较和选择合适复杂度的模型，比如上条中选择多项式的阶数。
- **过拟合(over-fitting)**：当模型过于复杂则会开始与无规律噪音的而不是规律的数据相匹配。数据量越大，选择复杂的模型时就越难过拟合。一般来说数据量应该是模型参数的5-10倍。
- **正则化(regularization)**：一种控制过拟合的方法，向误差方程中增加惩罚项。比如：

$$\widetilde{E}\left( \omega\right) = \frac{1}{2} \sum_{n=1}^{N} \left\{y\left(x_{n},\omega\right) - t_{n}\right\}^{2} + \frac{1}{2} \left \| \omega \right \|^{2}$$

>正则化方法也可以理解为：我对于我选择函数空间的函数做了一种限制，使得这个函数空间比原本的空间小，不会包含过拟合的函数，选择进入需要的函数空间。引入此先验概率可以得到正规化的最小误差方差方程。`L2正规化相当于认为参数满足高斯分布`

- **收缩(shrinkage)**：统计学术语 -减小参数值的方法。
- **岭回归(ridge regression)**：二次正则化方法。
- **权值衰减(weight decay)**：神经网络中二次正则化方法。

## 概率论
>模式识别的关键是*不确定性*，不确定性来自于测量误差和数据量的大小。为分布建模是模式识别的一个重要工作。概率论可以帮我们量化不确定性。

| y \ x| $x_{1}$| $x_{2}$|  $\cdots$ | $x_{i}$|  $\cdots$ |
|:----:| :-----:| :-----:| :------: | :------:|: --------:|
|$y_{1}$|$n_{11}$|$n_{21}$|  $\cdots$ |$n_{i1}$|
|$y_{2}$|$n_{12}$|$n_{22}$| $\cdots$  |$n_{i2}$|  
|$\cdots$| $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ |
|$y_{j}$|$n_{1j}$|$n_{2j}$| $\cdots$ |$n_{ij}$| 
|$\cdots$|
- **联合概率(joint probability)**：$p\left( X = x_{i}, Y = y_{j} \right) = \frac{n_{ij}}{N}$，xi和yj同时发生的概率
- **加法定理(sum rule)**：$p\left(X \right) = \sum_{Y}p\left( X , Y  \right)$ 其中 ：$p\left(X \right)$ 又被叫做**边际/边缘概率( marginal probability)**， 如果x，y连续则其形式变为$p(x) = \int p(x,y) dy$
- **乘法定理(product rule)**：$p\left( X, Y  \right)  = p\left(Y  | X \right) \cdotp\left(X\right)$  其中$p\left( Y |X \right)$ 又被叫做**条件概率(conditional probability)** ，如果x，y连续则其形式变为$p(x,y) = p(y|x) \cdot p(x)$
- **贝叶斯定理(Bayes's theorem)**：乘法定则的两种分解方法。

$$p\left(Y|X\right) = \frac{p\left(X|Y\right)\cdot p\left(Y\right)  }{p\left(X\right) }$$ 
- **先验概率(prior probability)**：$p\left(Y\right)$ ，测量之前知道的Y发生的概率。
- **后验概率(posterior probability)**：$p\left(Y|X\right)$，已经观测到X时Y发生的概率。
- **概率密度(probability density)**：$p\left( x \propto\left( a, b\right)\right) = \int_{a}^{b} p(x)dx$，连续变量的概率。
- **积累分部方程(cumulative distribution function)**：$P(z) = \int_{-\infty}^{z}p(x)dx$，概率密度的积分，
- **期望(expectation)**：对于$f(x)$ 的期望 $E[f]$
	- 连续变量有$E[f] = \frac{1}{N}\sum_{n=1}^{N}f(x_{n})$
	- 离散变量有$E[f] = \int_x f(x)p(x)dx$
- **条件期望(conditional expectation)**：$E_{x}[f|y] = \sum_{x}p(x|y)\cdot f(x)$ 多变量时x的期望方程。此方程为是y的函数，随着y变化而变化。
- **方差(variance)**：$var[f] = E[(f(x) - E[f(x)])^{2}] = E[f(x)^{2}] - E[f(x)]^{2}$。
- **协方差(covariance)**：协方差$cov[x, y] = E_{x,y}[\{x - E[x]\}\cdot\{y - E[y]\}] = E_{x,y}[x\cdot y] - E[x]\cdot E[y]$ 表示两个变量的变化关联，如果两者独立则协方差为0。
- **似然函数(likelihood function)**：也就是$p\left(X|Y\right)$。根据贝叶斯定理，因为此时X的值固定P(X)值也就固定，所以有关系：

$$后验概率 \propto 似然函数 \times 先验概率$$
- **误差方程(error function)**：$ - log(似然函数)$ 求似然函数的最大值与求误差方程的最小值是等价过程。
- **引导程序(bootstrap)**：从原始采样数据中随机取样，每次采样N个数据，这N个数据可以有重复，也不要求每个数据都被采样到。重复此取样动作多次。
- **independent and identically distributed(i.i.d)**：数据独立从同一个分布中采样得到。 
- **最大后验概率(maximum posterior)[MAP]**：通过采集到的数据，求后验概率最大时模型的参数值。
- **正太分布/高斯分布(normal or Gaussian distribution)**：$\mu$为平均值，而$\sigma^{2}$为方差，$\sigma$为**标准差(standard deviation)**而$\beta = \frac{1}{\sigma^{2}}$为**精密度(precision)**。

$$N(x| \mu, \sigma^{2}) = \frac{1}{(2\pi\sigma^{2})^{\frac{1}{2}}} exp\{ - \frac{1}{2}\sigma^{2}(x-\mu)^{2}\}$$ 

对于D维向量$\widetilde{x}$作为变量的高斯分布的形式变成：

$$N(\widetilde{x}|\widetilde{\mu} ,\widetilde{ \sigma}) = \frac{1}{(2\pi)^{\frac{D}{2}}\widetilde{ \sigma}^{\frac{1}{2}}} exp\{ - \frac{1}{2}\sigma^{2}(\widetilde{x}-\widetilde{\mu})^{T}\sigma^{-1}(\widetilde{x}-\widetilde{\mu})\}$$

> 高斯分布的最大似然方程：

$$lnp(x|\mu,\sigma^2) =  -\frac{1}{2\sigma^{2}}\sum_{n=1}^{N}(x_{n} - \mu)^{2} - \frac{N}{2}ln\sigma^2 - \frac{N}{2}ln(2\pi)$$ 

> 最大似然估计系统性的低估了方差，这是引起过拟合的一个原因，因为最大似然估计中的期望不是真实期望。对于多项式拟合问题，假设误差满足高斯分布，均值为$y(x)$,方差为$\beta$。对最大似然对数方程求导可以得到$\mu 和 \sigma$的值，这和最小二乘法求最小值等价。假设模型参数的先验概率为一个期望值为0精密度为$\alpha$的高斯分布，则相当于给最小二乘法增加了一个L2的正则化项。

> 频率论和贝叶斯概率论的区别：`频率论认为似然函数中的Y是固定的，每次测量的数据X遵循一定分布，所以会有所不同。贝叶斯概率论认为，每次测得的数据X是固定的，而Y存在一定分布，测量到的X值会消除Y的不确定性。频率经常使用最大似然分布，使用似然函数的最大值作为Y的估计。而贝叶斯概率则引入了先验概率的概念`

## 模型选择

- **交叉验证(crorss-validation)**：S个数据分成N组每次留下一组作为**验证集(validation set)**，循环使用。交叉验证在数据量大，*超参数(hyper parameter)*多的时候难以使用。
- **讯息准则(information criteria)**：向最大似然匹配增加一个常数项来降低参数过多时的过拟合效应。有两种比较著名的*Akaike information criterion (AIC) *和 *Bayesian information criterion (BIC)*
## 维度灾难
> 实际模式识别过程经常遇到高维度，高维度会带来很多困难。如果用分块的办法，高维度的空间需要指数增加的数据来充实。而对于多项式拟合，D维M阶多项式拥有$D^{M}$个参数。

## 决策论
>决策论可以帮助我们在分类问题上借助由概率论得到的不确定性做出最优选择。
> 解决决策问题的三种不同方法：
1. 使用**生成模型(generative model)**：计算每个分类的概率密度，计算每个分类的先验概率，然后使用贝叶斯定理计算后验概率，最后用决策论分类，生成模型计算联合概率分布。生成模型的缺点是需要大量数据，优点是可以提供边缘概率$p(x)$，检测孤立点。
2. **判别模型(discriminative model)**：直接对后验概率建模，然后用决策论分类。对于不平衡的数据训练时可能需要平衡不同类别的数据量，进行训练，最后通过先验概率重新计算后验概率。
3. **判别式方程(discriminant function)**：找到一个方程直接对输入进行分类。判别式方程最直接但是缺少了一些重要信息。如果损失函数发生变化，则判别式方程需要重新训练。而且判别式方程无法判断拒绝区域。

- **决策区域(decision region)**：输入空间中的区域，区域内的所有输入都被分为同一个类。区域不一定要连通。
- **决策平面边界(decision boundary/ decision surface)**：决策区域的分界线。如果要求是分类总错误率少，给定输入时选择后验概率最大的类别是最优选择。
- **损失函数(loss function / cost function)**：对于输入x真实类别是k但是被错分为j。引入损失矩阵$L_{kj}$。最优选择是最小化损失函数的期望值。

$$E[L] = \sum_{k}\sum_{j}\int_{R_{j}}L_{kj}p(x, C_k)dx$$
- **拒绝区域(reject region)**：对于不容易判断的区域可以设置拒绝区域，将问题交给后续判断。

> 对于回归问题损失函数的期望变为

$$E[L] = \int\int L(t,y(x))p(x, t)dxdt$$

求此选取损失函数为平方差并且求方程最小值得到**回归方程(regression function)**$y(x) = E_{t}[t|x]$ 特定X时求t的期望。

## 信息论
- **信息量**：可以被看成是意外的程度。得知一个小概率时间发生比得知一个大概率事件发生的获得的信息量要大。独立事件的信息量等于它们之和。计算信息量的公式为$h(x) = -log_{2}p(x)$。对数底为2时信息量单位是比特。
- **熵(entropy)**：对于一个变量，熵代表其混乱度，平均信息量的期望就是熵，所有可能状态出现概率都相等时熵最大。

$$H[x] = - \sum_{x}p(x)log_{2}p(x)$$

根据**无噪音编码理论(noiseless coding theorem)**可以证明熵是传输随机变量信息所需bit的最小值。
- **微分熵(differential entyroy)**：连续函数的熵$H[x] = - \int p(x)ln(px)dx$。`给定期望和方差时，正态分布的微分熵最大`
- **相对熵( relative entropy / KL divergence)**：使用模型$q(x)$去表示分布$p(x)$时的熵与实际$p(x)$的差

$$KL(p||q) = - \int p(x)ln\frac{q(x)}{p(x)}$$

 ，KL大于等于0，当且仅当两者相等时等于0。`最小化相对熵等价于最大化似然方程`
- **交叉熵( cross entropy)**：如果使用错误分布q来表示来自真实分布p的平均编码长度：就是交叉熵

$$cross = - \int p(x)ln\frac1{p(x)}$$
- **交互信息(mutual information)**：用来衡量两个变量变化是否一致的KL方程，或者给定X，求对Y的确定度。

$$I[x, y] = KL(p(x,y)||p(x)p(y)) = H[x] - H[x|y] = H[y] - H[y|x]$$

> Functional Geometrical Analysis 高维空间的 [^问题]
