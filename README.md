# 深度学习识别验证码
## Roadmap
深度学习，其实就是训练一大堆数据（10^4量级），本项目的使用成本已经很低了。
不过，这个项目是我成功跑起来的第一个NN项目，我还是希望进行拓展、优化的：
- model里面有多个模型，他们的效果分别怎么样？有没有办法自动跑起来，然后给我一个总的对比结果？——计划中
- 最终得到的，其实就是`model.pkl`，可以做成web_ui提供服务——计划中
- 对GPU的压榨还不够狠。我情况是：batch_size=256，8个worker, GPU`3507MiB / 16160MiB |     35%   `，不过看CPU是400%，显然瓶颈在CPU——无法优化
  

## 介绍
- 本项目致力于使用神经网络来识别各种验证码。
- 基于：https://github.com/pprp/captcha.Pytorch， pprp在这个库的基础上，进行了改动，添加了很多trick来增强识别效果，如attention机制，dual pooling, ibn模块，bnneck,center loss等。
- pprp的项目又基于: https://github.com/dee1024/pytorch-captcha-recognition 



改动
===
- 添加了更多torchvision中支持的模型
- 改了一下文件的名称
- 支持了GPU，当然cpu也可以
- 添加了以下功能：训练完成一个epoch之后进行测试，（需要保证test和train中的模型一致）
- 添加了以下功能：将每次得到的测试结果写入results.txt文件，运行torch_util.py得到results.png可视化准确率。
- RES152为基础网络进行训练，混合数字和大写字符只能达到94%，还达不到原作者的识别率，希望得到赐教

特性
===
- __端到端，不需要做更多的图片预处理（比如图片字符切割、图片尺寸归一化、图片字符标记、字符图片特征提取）__
- __验证码包括数字、大写字母、小写__
- __采用自己生成的验证码来作为神经网络的训练集合、测试集合、预测集合__
- __纯四位数字，验证码识别率高达 99.9999 %__
- __四位数字 + 大写字符，验证码识别率约 96 %__
- __深度学习框架pytorch + 验证码生成器ImageCaptcha__


原理
===

- __训练集合生成__

    使用常用的 Python 验证码生成库 ImageCaptcha，生成 10w 个验证码，并且都自动标记好;
    如果需要识别其他的验证码也同样的道理，寻找对应的验证码生成算法自动生成已经标记好的训练集合或者手动对标记，需要上万级别的数量，纯手工需要一定的时间，再或者可以借助一些网络的打码平台进行标记

- __训练卷积神经网络__
    构建一个多层的卷积网络，进行多标签分类模型的训练
    标记的每个字符都做 one-hot 编码
    批量输入图片集合和标记数据，大概15个Epoch后，准确率已经达到 96% 以上


验证码识别率展示
========
![](https://raw.githubusercontent.com/dee1024/pytorch-captcha-recognition/master/docs/number.png)

快速开始
====
- __步骤一：10分钟环境安装__

    Python3.6+ 、ImageCaptcha库(pip install captcha)、 Pytorch(参考官网http://pytorch.org)


- __步骤二：生成验证码__
    ```bash
    python captchaGenerator.py
    ```
    执行以上命令，会在目录 dataset/train/ 下生成多张验证码图片，图片已经标注好，数量可以是 1w、5w、10w，通过 captchaGenerator.py 内的 count 参数设定
    
- __步骤三：训练模型__
    ```bash
    python train.py
    ```
    使用步骤一生成的验证码图集合用CNN模型（在 models 中定义）进行训练，训练完成会生成文件保存在weights文件夹中，最好的结果保存为`cnn_best.pt`

- __步骤四：测试模型__
    ```bash
    python test.py
    ```
    可以在控制台，看到模型的准确率（如 95%） ，如果准确率较低，回到步骤一，生成更多的图片集合再次训练

- __步骤五：使用模型做预测__
    ```bash
    python predict.py
    ```
    可以在控制台，看到预测输出的结果
    

可视化
===

![results_res101](assets/results_res101.png)

贡献
===

我们期待你的 pull requests !

有问题欢迎提出issue

作者
===
* __Dee Qiu__ <coolcooldee@gmail.com>
* 补充：__pprp__ <1115957667@qq.com>


声明
===
本项目仅用于交流学习
