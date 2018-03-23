
## 代码运行步骤
#### 1. 初始化

  1. regularization
  2. 规整化数据
  3. max_length用法 35 数据的长度
  4. build_model 创建和初始化控制和shared的生成模型
    1. controller 初始化
      + tokens
      + embedding
      + encoder
      + decoder
    2. shared_model
      + 各种 dropout
      + 10000 -> 1000 embed
  5.


#### 2.cnn

  1. 通道固定: 1-->128-->256-->512-->1024-->avgpool
  2. 共享的卷积核:  128 3*3， 5*5， 1*3 3*1， 7*7， 1*1 (depth wise)? 先用avg pool? 1*1(3->128) 1*1(3->256, 128->256,)
    1*1(3->512? 128->512, 256->512)  1024....    avgpool
    （可以试一下 1x1 卷积要不要layer之间共享）


#### 3. step
  1. 写controller sample [doing]
  2. 写cnn init [done]
  3. 写cnn 训练
  4. 写controller 训练
  5. 整个步骤的cnn 训练
  6. pytorch cifar dataset
  7. 整个的cnn 训练

#### 4. problem
  1. 分布式怎么解决
