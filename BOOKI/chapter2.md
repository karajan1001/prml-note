

## 概率分布
- **密度估计(density estimation)**：给定一个变量，和有限采样数据集，估计变量的概率分布。

### 二项变量
- **伯诺利分布(Bernoulli distribution)**：对一个二态变量两态$\{0,1\}$。1态的概率为$\mu$。概率分布可以表示为

$$Bern(x\mid \mu) = \mu^{x}(1-\mu)^{1-x}$$

期望为$\mu$方差为$\mu(1-\mu)$，对其使用最大似然估计可以得到

$$\mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}x_{n}$$
- **二项式分布(binomial distribution)**：伯诺利分布中x=1的次数为m总次数为N

$$Bin(m\mid N,\mu) = \begin{pmatrix} N \\ m \end{pmatrix} \mu^{m}(1-\mu)^{N-m}$$
- **贝塔分布(beta distribution)**：一个和二项式分布共轭的先验概率方程。

$$Beta(\mu\mid a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$


其中**伽马方程(gamma function)** 为：$\Gamma(x) = \int_{0}^{\infty}\mu^{x-1}e^{-\mu}d\mu$ ，ab是先验等效x=1和x=0的次数。

> 可以把每次实验的结果当做后面实验的先验概率。这样形成连续的数据流。当N趋于无穷时，贝叶斯方法和最大似然方法的结果相等。后验概率平均$\theta$的期望对所有数据集合D求期望等于$\theta$的先验概率。

$$E_{\theta}[\theta] = E_{D}[E_{\theta}[\theta\mid D]]$$

## 概率分布
- **密度估计(density estimation)**：给定一个变量，和有限采样数据集，估计变量的概率分布。

### 二项变量
- **伯诺利分布(Bernoulli distribution)**：对一个二态变量两态$\{0,1\}$。1态的概率为$\mu$。概率分布可以表示为

$$Bern(x\mid \mu) = \mu^{x}(1-\mu)^{1-x}$$

期望为$\mu$方差为$\mu(1-\mu)$，对其使用最大似然估计可以得到

$$\mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}x_{n}$$
- **二项式分布(binomial distribution)**：伯诺利分布中x=1的次数为m总次数为N

$$Bin(m\mid N,\mu) = \begin{pmatrix} N \\ m \end{pmatrix} \mu^{m}(1-\mu)^{N-m}$$
- **贝塔分布(beta distribution)**：一个和二项式分布共轭的先验概率方程。

$$Beta(\mu\mid a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$


其中**伽马方程(gamma function)** 为：$\Gamma(x) = \int_{0}^{\infty}\mu^{x-1}e^{-\mu}d\mu$ ，ab是先验等效x=1和x=0的次数。

> 可以把每次实验的结果当做后面实验的先验概率。这样形成连续的数据流。当N趋于无穷时，贝叶斯方法和最大似然方法的结果相等。后验概率平均$\theta$的期望对所有数据集合D求期望等于$\theta$的先验概率。

$$E_{\theta}[\theta] = E_{D}[E_{\theta}[\theta\mid D]]$$

### 多项变量
> 对于多项变量定义其值的方法为$(x_{1},x_{2},x_{3} \cdots x_{K})^{T}$。其期望为$(\mu_{1},\mu_{2},\mu_{3} \cdots \mu_{K})^{T}$ 对于一个N次独立观测数据集合D其概率为

$$p(D\mid\mu) = \prod_{k=1}^{K} \mu_{k}^{(\sum_{n} x_{nk})} = \prod_{k=1}^{K}\mu_{k}^{m_{k}}$$

- **多项式分布(multinomial distribution)**：上述分布变量为$m_{1},m_{2}...m_{k}$的话得到

$$Mult(m_{1},m_{2}\cdots m_{k}\mid \mu, N) = \begin{pmatrix}N \\ m_{1},m_{2}\cdots m_{k}\end{pmatrix} \prod_{k=1}^{K}\mu_{k}^{m_{k}}$$

约束为$\sum_{k=1}^{K}m_{k} = N$
- **狄利克莱分布(Dirichlet distribution)**：类似二项分布中的贝塔分布而引入先验概率

$$Dir(\mu\mid\alpha) = \frac{\Gamma(\alpha_{0})}{ \Gamma(\alpha_{1}) \cdots  \Gamma(\alpha_{k})}\prod_{k=1}^{K}\mu_{k}^{\alpha_{k}-1}$$

### 高斯分布
>多随机个变量的和，随着随机变量数量增加会趋近于高斯分布。[^问题]

- **马哈拉诺比斯距离(Mahalanobis distance)**：对于高斯分布x作用在

$$\Delta^{2} = (x-\mu)^{T}\Sigma^{-1}(x-\mu)$$

其中$\Delta$就是马哈拉诺比斯距离，在$\Sigma$是单位阵时马哈拉比斯距离就是欧几里得距离。将其变换到本征态上则。则得到一个空间中的椭球体。椭球体表面密度相等。$\Sigma$是多个x变量的协方差矩阵。高斯分布的参数随着维度增长平方增长。一个方法是对高斯分布中参数进行限制，但是这种限制会制约它的灵活性，影响匹配复杂数据的能力。还有高斯分布是单极值的，所以无法拟合多极值分布。
- **条件高斯分布**：如果两组变量联合概率分布是高斯分布，则其条件概率是高斯分布，边际概率也是高斯。
- **精度矩阵(precission matrix)**：$\Delta$的逆矩阵。
可以从公式推出条件高斯分布的方差和期望。
- **边缘高斯分布**：对于两组变量中一组变量边缘高斯分布其方差和期望就是原本分布中那组变量的期望和方差。
- **高斯变量的贝叶斯理论**：给定一组变量的边际高斯分布和条件高斯分布，可以计算出条件的那组变量的编辑高斯分布和条件高斯分布。
- **高斯分布的最大似然**：对高斯分布用最大似然估计。期望和真实期望相等，方差为真实方差的$\frac{N-1}{N}$倍。
- **回归方程(regression functions)**：对于一个联合概率分布$p(z,\theta)$，$z$对$\theta$的条件概率是$\theta$的方程$f(\theta)$。这个方程被称为回归方程。对于拟合问题$z$是参数$\theta$是那些点。
- **Robbins- Monro Algorithm**：

$$\theta^{(N)} = \theta^{(N-1)} + a_{N-1}z(\theta^{(N-1)})$$

#### 序列问题
对于高斯分布，重复N次采样的似然函数有

$$p(X|\mu) = \prod_{n=1}^Np(x_n| \mu) = \frac
{1}{(2\pi\sigma^2)^{N/2}}exp\Big \{-\frac1{2\sigma^2}\sum_{n=1}^N(x_n-\mu)^2\Big \}$$

其中$\mu$满足高斯分布，$\lambda满足Gamma方程$
计算$\mu$的先验概率如果$\mu$的先验概率和后验概率有相同的高斯函数分布，可以求得

$$p(\mu|X) = p(X|\mu)p(\mu) = N(\mu|\mu_N,\sigma^2_N)$$
其中

$$\mu_N = \frac{\sigma^2}{N\sigma_0^2 + \sigma^2}\mu_0 + \frac{N\sigma_0^2}{N\sigma_0^2 + \sigma^2}\mu_{ML}$$
$$\frac{1}{\sigma^2_N} = \frac{1}{\sigma^2_0}  + \frac{N}{\sigma^2}  $$
可以看出0等于0时$\mu$等于先验概率随着N增加，精度增加，期望向着$\mu_{ML}$靠拢。对于序列型的问题，可以把前面的测量当成先验概率，后面的测量当成后验概率。
同理，计算$\sigma$的先验概率。对于精度$\lambda = \frac1{\sigma^2}$Gamma方程表达式为：

$$Gam(\lambda|a,b) = \frac1{\Gamma(a)}b^a\lambda^{a-1}exp(-b\lambda)$$

该方程的期望为$E[\lambda] = \frac ab$，方差为$var(\lambda) = \frac a {b^2}$。带入可以求得

$$a_N = a_0 + \frac N2$$ $$b_N = b_0 + \frac 1 2 \sum_{n=1}N(x_n - \mu)^2 = b_0 + \frac N 2 \sigma_{ML}^2$$
**Gaussian-gamma**方程：同时考虑两者的先验概率方程：

$$p(\mu, \lambda) = N(\mu | \mu_0 , (\beta\lambda)^{-1}) Gamma(\lambda | a, b)$$

#### 学生t-分布
结合高斯分布$N(x | \mu_0 , \tau^{-1})$和先验概率伽马方程$Gamma(\tau|a,b)$结合，得到t-分布，t-分布可以看成无数连续高斯分布的混合，其鲁棒性比高斯分布更好。

#### 周期变量
对周期变量满足方程

$$p(\theta) \ge 0 $$
$$\int_0^{2\pi}p(\theta)d\theta = 1$$
$$ p(\theta + 2\pi) = p(\theta)$$

可以将周期变量当做x,y平面两个变量高斯分布处理

$$p(x_1,x_2) = \frac1{2\pi\sigma^2}exp \big\{ -\frac{(x_1-\mu_1)^2 + (x_2-\mu_2)^2}{2\sigma^2}\big\}$$

取极坐标得到 **Von Mises 分布**，又称为圆正太分布 

$$p(\theta|\theta_0,m) = \frac1{2\pi I_0(m)}exp\big\{ m cos\theta \big\}$$

其中$\theta_0$是分布均值，而m则是集中参数，与精度成反比。当m很大的时候这个分布可以近似为高斯分布。对VM分布求最大期望最后可以得到

$$\theta_0^{ML} = tan^{-1}\big\{\frac{\sum_nsin\theta_n}{\sum_ncos\theta_n}\big\}$$

#### 混合高斯模型
有些问题一个高斯分布解决不了，需要多个高斯分布的叠加，多个高斯分布叠加可以看成先验概率和后验概率的组合。
### 指数家族

本章中除了高斯混合分布都是蜘蛛家族的成员。指数家族具有形势 

$$p(x|\eta) = h(x) g(\eta) exp\big\{\eta^Tu(x)\big\}$$
其中$\eta$是分布的自然参数。
对于伯努利分布$p(x|\mu) = Bern(x|\mu)$有$\eta = ln(\frac{u}{1-\mu})$ 或者 $\mu=\frac1{1+e^{-\eta}}$ 求解就是**sigmoid**方程。
对于multinomial 分布。同样过程可以得到 $\mu_k = \frac{exp(\eta_k)}{1+\sum_jexp(\eta_j)}$ **softmax** 方程。
对于指数家族可以求得

$$-\triangledown lng(\eta) = E[\mu(x)] $$

，对多次采样是用最大似然估计得到

$$-\triangledown lng(\eta_{ML}) = \frac1N\sum_{n=1}^N\mu(x_n) $$ 

可以看到$\eta_{ML}$ 只取决于$\sum_{n=1}^N\mu(x_n)$ 这叫做分布的有效统计。对于伯努利分布我们只要知道总和，对高斯分布我们只要知道总和与平方和。
#### 共轭先验概率
后验概率和先验概率有一样的形势。对指数家族的话



### 非参数方法
举例直方图，直方图可以提供一个快速检视1或者2维数据的方法。但是有两个缺陷，第一不连续，第二对高维空间不适用。
#### 内核密度估计
V为空间R大小，N个点中K个在R内，假设R内密度变化不大可以得到R内密度估计为

$$p(x) = \frac{K}{NV}$$
固定K就是K近邻，固定V就是内核估计。
定义

$$k(\mu) = \begin{cases}
 & 1 ,  |u_i| \leq \frac12, i = 1,2...,D.   \\  
 & 0  , otherwise 
\end{cases}$$
则内核密度估计方程为

$$p(x) = \frac1N\sum_{n=1}^N\frac{1}{h^D}k(\frac{x-x_n}{h})$$

因为内核方程对称这个方程可以看成中心点x附近的$x_n$求和也可以看成N个$x_n$形成的方块求和。可以用高斯方程代替k函数，此方法不需要训练但是预测时需要巨大计算量。

#### N近邻
K近邻计算量要求大。这个近似在分类问题中使用可以得出某个点的分类只与附近的K个点相关。而与先验概率等都无关。


