# Attack_classification_models_with_transferability
Attack classification models with transferability, black-box attack; unrestricted adversarial attacks on imagenet, [CVPR2021 安全AI挑战者计划第六期：ImageNet无限制对抗攻击](https://tianchi.aliyun.com/competition/entrance/531853/introduction), 决赛第四名(team name: Advers)

[详细方案介绍](https://tianchi.aliyun.com/forum/postDetail?postId=208941)

论文：[Improving Adversarial Transferability with Gradient Refining](https://arxiv.org/abs/2105.04834)
## 1. Prerequisites
```
1. python >= 3.6
2. pytorch >= 1.2.0
3. torchvision >= 0.4.0 
4. numpy >= 1.19.1
5. pillow >= 7.2.0
6. scipy >= 1.5.2
```

## 2. Code Overview
* ./codes/
  - ```main.py```: 攻击原始图像，生成并保存攻击后的图像
  - ```data.py```: 加载原始图像，保存图像，图像标准化处理
  - ```model.py```: 模型集成，利用集成模型计算logits
  - ```utils.py```: Input Diversity, 高斯平滑处理等
 
* ./input_dir/
  - ./images/: 原始图像所在路径
  - ./dev.csv：图像的标记文件（images name, true label）


* demo
```
python main.py --source_model 'resnet50'
```

## 3. 思路

* 本文分享我们团队（Advers）的解决方案，欢迎大家交流讨论，一起进步。
* 本方案最终得分：**9081.6**， 线上后台模型攻击成功率：**95.48%**，决赛排名：**TOP 4**。
* 本方案初赛排名：**TOP 4**，复赛排名：**TOP 10**。

### 3.1 赛题分析

1. **无限制对抗攻击**可以用不同的方法来实现，包括范数扰动攻击、GAN、粘贴Patch等。但是由于 fid、lpips 两个指标的限制，必须保证生成的图像质量好（不改变语义、噪声尽量小），否则得分会很低。经过尝试，我们最终确定利用范数扰动来进行迁移攻击，这样可以较好地平衡攻击成功率和图像质量。

2. 由于无法获取后台模型的任何参数和输出，甚至不知道后台分类模型输入图像大小，这增加了攻击难度。原图大小是 500 * 500，而 ImageNet 分类模型输入是 224 * 224 或 299 * 299，对生成的对抗样本图像 resize会导致对抗样本的攻击性降低。

3. 由于比赛最终排名为人工打分，所以没有用损失函数去拟合 fid、lpips 两个指标。

4. 对抗样本的攻击性和图像质量可以说是两个相互矛盾的指标，一个指标的提升往往会导致另一个指标的下降，如何在对抗性和图像质量两个方面找到一个平衡点是十分重要的。在机器打分阶段，采用较小的噪声，把噪声加在图像敏感区域，在尽量不降低攻击性的前提下提升对抗样本的图像质量是得分的关键

### 3.2 解题思路

#### 3.2.1 输入模型的图像大小

本次比赛的图像被 resize 到了 500 * 500 大小，而标准的 ImageNet 预训练模型输入大小一般是 224 * 224 或 299 * 299。我们尝试将不同大小的图片（500，299，224）输入到模型中进行攻击，发现 224 大小的效果最好，计算复杂度也最低。

#### 3.2.2 L2 or Linf 

采用 L2 范数攻击生成的对抗样本的攻击性要强一些，但可能会出现比较大的噪声斑块，导致人眼看起来比较奇怪，采用 Linf 范数生成的对抗样本，人眼视觉上要稍好一些。在机器打分阶段，采用 L2 范数扰动攻击，在人工评判阶段，采用 Linf 范数扰动来生成对抗样本。


#### 3.2.3 提升对抗样本迁移性方法

**1. MI-FGSM<sup>1</sup>**：在机器打分阶段采用 MI-FGSM 算法生成噪声，但是 MI-FGSM 算法生成的噪声人眼看起来会明显，由于决赛阶段是人工打分，最终舍弃了该方法。

**2. Translation-Invariant（TI）<sup>2</sup>**：用核函数对计算得到的噪声梯度进行平滑处理，提升了噪声的泛化性。

**3. Input Diversity（DI）<sup>3</sup>** ：通过增加输入图像的多样性来提高对抗样本的迁移性，其提分效果明显。Input Diversity 本质是通过变换输入图像的多样性让噪声不完全依赖相应的像素点，减少了噪声过拟合效应，提高了泛化性和迁移性。

#### 3.2.4 改进后的DI攻击

Input Diversity 会对图像进行随机变换，导致生成的噪声梯度带有一定的随机性。虽然这种随机性可以使对抗样本的泛化性更强，但是也会引入一定比例的噪声，这种噪声也会抑制对抗样本的泛化性，因此如何消除 DI 随机性带来的噪声影响，同时保证攻击具有较强的泛化性是提升迁移性的有效手段。

![image](https://github.com/yufengzhe1/Attack_classification_models_with_transferability/blob/main/input_dir/math.jpg)

#### 3.2.5 Tricks

* 在初赛和复赛阶段，采用 L2 和 Linf 范数扰动攻击，其中 L2 范数扰动攻击得分更高一些。由于复赛阶段线上模型比较鲁棒，所以适当增加扰动范围是提升攻击成功率的关键。
* 考虑到决赛阶段是人工打分，需要考虑攻击性和图像质量，我们最终采用 Linf 范数扰动进行攻击，扰动大小设为 32/255，迭代次数设为 40，迭代步长设为 1/255。
* 攻击之前，对图像进行高斯平滑处理，可以提升攻击效果，但是也会让图像变模糊。
* Ensemble models: resnet50、densenet161、inceptionv4等。

## 4. 攻击结果

![image](https://github.com/yufengzhe1/Attack_classification_models_with_transferability/blob/main/input_dir/adv_images.jpg)

多次实验表明，采用改进的 DI+TI 攻击方法得到的噪声相对于 MI-FGSM 方法更小，泛化性和迁移性更强，同时人眼视觉效果也比较好。

## 5. 参考文献

1. Dong Y, Liao F, Pang T, et al. Boosting adversarial attacks with momentum. CVPR 2018.
2. Dong Y, Pang T, Su H, et al. Evading defenses to transferable adversarial examples by translation-invariant attacks. CVPR 2019. 
3. Xie C, Zhang Z, Zhou Y, et al. Improving transferability of adversarial examples with input diversity. CVPR 2019.
4. Wierstra D, Schaul T, Glasmachers T, et al. Natural evolution strategies. The Journal of Machine Learning Research, 2014.

## 6. Citation



**如有问题，欢迎交流：wangguoqiu@buaa.edu.cn**
