# Attack_classification_models_with_transferability
Attack classification models with transferability, black-box attack; unrestricted adversarial attacks on imagenet, [CVPR2021 安全AI挑战者计划第六期：ImageNet无限制对抗攻击](https://tianchi.aliyun.com/competition/entrance/531853/introduction), 决赛第四名(team name: Advers)

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
  - ```utils.py```: Input Diversity, 高斯平滑处理
 
* ./input_dir/
  - ./images/:原始图像所在路径
  - ./dev.csv：图像的标记文件（images name, true label）


* demo
```
python main.py --source_model 'resnet50'
```

## 3. 思路

* 我们的方案最终得分：**9081.6**， 线上后台模型攻击成功率：**95.48%**，决赛排名：**TOP4**。

### 3.1 赛题分析

1. 本题目是**无限制对抗攻击**，可以利用很多种不同的方法来进行攻击，包括范数扰动、GAN、Patch等。但是，必须保证生成的图像质量要好（不改变语义、加的噪声尽量小）才可以，不然fid、lpips两个指标得分会很低，最终的人工打分也会很低。经过尝试，我们团队最终决定利用范数扰动来进行攻击，这样可以较好的平衡攻击成功率和图像质量。

2. 由于无法获得后台模型的任何参数和输出，甚至不知道后台的分类模型输入图像大小是多少，这就大大增加了攻击难度。原图是500 * 500大小的图片，而 ImageNet 分类模型一般的输入大小是 224 * 224 或者 299 * 299，对生成的对抗样本图像 resize会导致生成对抗样本的攻击性降低。

3. 由于比赛最终的排名为人工打分，所以我们并没有用损失函数去拟合fid、lpips两个指标，这也是为什么我们最终排名较高的原因之一吧。

4. 对抗样本的攻击性和图像质量可以说是两个相互矛盾的指标，一个指标的提升大概率会导致另一个指标的下降，所以，如何在对抗性和图像质量两方面找到一个调和的点是十分重要的。我们的方案也是从这个角度出发的，即：尽量减小噪声，尽量把噪声加在图像的敏感区域，在尽量不降低攻击性的前提下提升对抗样本的图像质量。


### 3.2 解题思路

#### 3.2.1 输入模型的图像大小

本次比赛所有的图像都被resize到了500 * 500大小，而标准的 ImageNet 预训练模型输入大小一般是224 * 224或者299 * 299。我们尝试了将不同大小的图片（500，299，224）输入到模型中进行攻击，发现224大小的效果最好，计算复杂度也最低。

#### 3.2.2 L2 or Linf 

利用L2范数攻击生成的对抗样本的攻击性要强一些，但是可能会出现比较大的噪声斑块，导致人眼看起来比较奇怪，利用Linf范数生成的对抗样本，人眼视觉上要稍好一些。考虑到最终排名为人工打分，最终采用Linf范数扰动来生成对抗样本。

#### 3.2.3 MI-FGSM<sup>[1]</sup>

利用MI-FGSM算法，提高了对抗样本的迁移性，但是生成的噪声人眼看起来会稍大一些，考虑到人工打分，我们最终并没有采用这种方法，但是本方法给我们后续方法提供了一些思路和启发。

#### 3.2.4 Translation-Invariant（TI）<sup>[2]</sup>

TI方法利用kernel对生成的噪声梯度进行平滑处理，增加了噪声的迁移性,随着kernel尺寸的增大，对抗样本的迁移性会增强；同时生成的对抗样本给人的视觉效果也会变差，综合考虑迁移性和视觉效果，最终确定了一个合适的kernel大小。

#### 3.2.5  Input Diversity<sup>[3]</sup>

Input Diversity通过增加输入图像的多样性来提高对抗样本的迁移性，加入Input Diversity后涨分效果很明显。深入地思考，Input Diversity本质是通过变换输入图像的多样性（图像大小、在图像上添加pad的位置）让噪声不完全依赖相应的像素点，减少了噪声过拟合效应，提高了迁移性，这也为我们进一步思考和实验提供了思路。

#### 3.2.6 Noise Grad Average(Ours)

MI-FGSM利用动量迭代，稳定了噪声的更新方向，提高了迁移性；Input Diversity利用输入多样性来提升迁移性。以上的方法都是利用sign()函数来处理噪声梯度的，导致梯度大的像素点和梯度小的像素点最后得到的噪声大小相同，动量迭代的方法虽然能够得到稳定的更新方向，获得较好的迁移性，但是更多的是关注过去的信息g_t。

基于以上的思考，我们设计了一种**Noise Grad Average**方法来更新噪声，该方法利用噪声梯度均值，抵消了无用噪声，留下的都是攻击性强的像素点位置的有用噪声，提高了对抗样本的迁移性。最终的结果表明Noise Grad Average方法得到的噪声相对于MI-FGSM算法更小，迁移性也更好。

## 4. 生成的对抗样本

![image](https://github.com/yufengzhe1/Attack_classification_models_with_transferability/blob/main/input_dir/adv_images.jpg)
