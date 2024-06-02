@[TOC](Ray Tracing in one Weekend But on CUDA)

# 环境说明
代码运行在Visual Studio 2019环境，显卡为NVIDIA GeForce GTX 1650，CUDA版本为11.6，cuDNN版本为8.4.0。具体配置方式见[CUDA C/C++ 从入门到入土 第一步——让你的CUDA跑起来](https://blog.csdn.net/supersyzz/article/details/125667431).

代码参考以下内容：
[Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)

[Ray Tracing in One Weekend](https://raytracing.github.io/v3/books/RayTracingInOneWeekend.html)

[可视计算讲稿16](https://vcl.pku.edu.cn/course/vci/slides/16-advanced-rendering.pdf)

# 项目内容
该项目实现了PathTracing的渲染算法，并提供了四种基本材质（朗伯表面、金属表面、玻璃介质、光源介质），一种物体球体，并提供环境光，相机系统和配套的数学运算。

并为光追渲染算法提供了GPU支持，使速度有所提高。

# 思路
先实现CPU上的PathTracing算法，然后将项目移植到GPU上。

PathTracing的大致思想为，从相机镜头向幕布处的每个像素发射光线，计算光线进入世界后在不同物体间的折射、反射（镜面、漫反射），并对光照强度进行积分，最终取多条光线的平均值即可。

CUDA编程的思路需要并行。在光追中可以并行的部分在于计算每个像素的颜色，即可以将原先计算像素颜色的循环并行计算。即每个CUDA核心的一个线程计算一个像素的颜色。在所有线程执行完毕之后，将数据从GPU转移到CPU上并进行保存。

# 实现方法
C++版本PathTracing。利用基本的C++代码，造轮子。需要说明的是，光追函数需要限制递归的次数（否则有小概率导致无限递归，虽然在这个例子里大概率不可能），同时对输出结果进行gamma校正从而保证生成图像的颜色质量。在判断光线与物体相交的过程中，需要利用法线来判断相交的位置是在物体表面外还是物体内。

CUDA版本的PathTracing做出了以下优化：
-  光追函数利用并行计算，加速效率。
- 光线路径追踪函数的递归展开，从而加快运行效率。
- 对随机数进行重写（因为CUDA并行需要每个线程一个随机数种子）

需要注意的是，在CUDA编程过程中，需要注意device和host的关系，尤其需要注意内存的位置，以及申请的内存位于哪里？否则会出现严重的内存泄露以及段错误。

还需要注意的是，在进行Render之前，需要统一进度，保证各个线程的世界包含相同的内容。在Render之后，也需要保证所有线程运行完毕后，再进行内存复制到Host。


# 效果
测试效果，其中三个球的材质分别为光源、玻璃、朗伯表面（由近到远）。迭代次数为10.光照强度为1.
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c1a7b0ee58554f31b41bf043b583cfcc.png)

最终效果
迭代次数400，光源材质光强为2.0
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4e2adec030ea47c7a74eb1b1862fa6af.png)


迭代次数50，光源材质光强为2.0
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/42c9780db6b348c098c9456f84f27ad4.png)

迭代次数10，且光源材质光强为2
![](https://img-blog.csdnimg.cn/direct/6dfc6c1077d44c6eb42fd68603c8d523.png)


