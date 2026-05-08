1.structure：
PureFER/
├── checkpoints/             # 预训练模型权重及评估可视化图表 (CAM/混淆矩阵)
│   ├── rafdb_acc0.8985.pth  # RAF-DB 最优权重
│   ├── FERPlus_acc0.9126.pth# FERPlus 最优权重
│   ├── affectnet7...        # AffectNet 相关权重及曲线图 (.png/.svg)
├── networks/                # 模型核心架构代码
│   ├── DDCM.py              # 双域压缩模块 (Spatial & Channel Compression)
│   └── MixedFeatureNet.py   # 混合特征主干网络
├── pretrained/              # 预训练基础权重存放目录
│   └── MFN_msceleb.pth      # Ms-Celeb-1M 上的骨干预训练权重
├── affectnet_train.py       # AffectNet 数据集训练脚本
├── affectnet_test.py        # AffectNet 数据集测试脚本
├── ferPlus_train.py         # FERPlus 数据集训练脚本
├── ferplus_confusion.py     # FERPlus 混淆矩阵生成及可视化脚本
├── rafdb_train.py           # RAF-DB 数据集训练脚本
├── rafdb_test.py            # RAF-DB 数据集测试脚本
└── sam.py                   # Sharpness-Aware Minimization (SAM) 优化器
2.Installation：
git clone https://github.com/yourusername/PureFER.git
cd PureFER

#安装必要依赖 (PyTorch 2.5.1 及对应 CUDA 版本)
pip install torch torchvision numpy opencv-python pandas
3.Data Preparation：
模型支持以下四大基准数据集。请提前下载并解压：
RAF-DB   
FERPlus   
AffectNet-7 / AffectNet-8
4.Pre-progress：
所有输入图像需使用 RetinaFace 提取面部特征，统一缩放至 112 x 112 像素大小
5.Quick Start：
5.1： 骨干网络初始化在开始训练前，请确保将预先在 Ms-Celeb-1M 数据集上训练好的主干权重文件 MFN_msceleb.pth 放入 pretrained/ 文件夹中 ，以保证模型初始化的特征提取能力。
5.2： 模型训练 (Training)项目中为不同数据集提供了独立的训练脚本。以 RAF-DB 为例：Bashpython rafdb_train.py
      超参数说明：默认采用 SAM 优化算法 ，联合优化标准交叉熵损失（$L_{cls}$）与区域聚焦损失（$L_{RF}$，权重 $\lambda=0.5$）。初始学习率设为 0.00001，Batch Size 为 64 。  
5.3： 模型测试与可视化 (Testing & Evaluation)我们已在 checkpoints/ 目录下提供了所有数据集达到 SOTA 的预训练权重。要测试 RAF-DB 测试集的准确率，请运行：Bashpython rafdb_test.py --checkpoint checkpoints/rafdb_acc0.8985.pth
如需生成特征分类的混淆矩阵（Confusion Matrix），可以运行对应的分析脚本：Bashpython ferplus_confusion.py
