# ESC-50 Environmental Sound Classification

本项目基于 **ESC-50** 公共数据集，完成一个从零开始的音频分类任务，目标是快速实现一个 **可复现、可视化、可部署** 的完整深度学习音频分类流程。适合作为简历项目或入门音频分类的学习项目。

## ✨ 项目亮点

- 📁 **数据集**：ESC-50 (2000 条音频，50 个类别，每条5秒)
- 🎵 **特征提取**：Mel Spectrogram + Log-Mel 特征
- 🏧️ **模型架构**：轻量级 CNN（从零搭建）
- 📊 **训练流程**：完整 PyTorch pipeline，训练日志可视化
- 💻 **推理部署**：支持单音频文件推理、Gradio 可视化界面
- 🧹 **模块化项目结构**：清晰分离数据、模型、训练、推理模块

---

## 📂 项目结构

```
esc50_classification/
├── config.py                  # 配置文件（路径、参数等）
├── train.py                   # 训练主程序
├── predict.py                 # 单条音频推理
├── inference_gradio.py        # Gradio 界面推理
├── train_log_visualize.py     # 训练日志可视化
├── models/
│   └── cnn_model.py           # CNN模型结构
├── data/
│   ├── download_esc50.py      # 自动下载ESC-50
│   ├── esc50_dataset.py       # PyTorch Dataset定义
│   └── esc50_files/           # 存放原始音频和元数据
└── weights/                   # 保存训练好的模型权重
```

---

## 🚀 快速开始

### 1. 安装依赖
```bash
conda create -n esc50 python=3.11
conda activate esc50
pip install torch torchaudio matplotlib pandas gradio
```

### 2. 下载数据集
```bash
cd data
python download_esc50.py
```

### 3. 训练模型
```bash
python train.py
```

### 4. 单音频推理
```bash
python predict.py --audio_path example.wav
```

### 5. 启动Gradio推理界面
```bash
python inference_gradio.py
```

---

## 📝 数据集介绍

- **来源**：Karol J. Piczak, ESC-50: Dataset for Environmental Sound Classification, 2015
- **类别**：50 类，如狗叫声、雷声、娃娃哭声、锤木声等
- **每类样本数**：40 条
- **采样率**：44.1 kHz，单声道，5秒/条

数据集主页：[https://github.com/karoldvl/ESC-50](https://github.com/karoldvl/ESC-50)

---

## 🌟 后续可拓展方向

- ☑️ 升级使用 **PANNs 预训练模型**
- ☑️ 嘗试 Transformer 架构（如 AST）
- ☑️ 引入 Mixup/CutMix 等数据增强
- ☑️ 使用 wandb/Tensorboard 完整训练监控

---

## 📢 项目目的

本项目旨在快速掌握 **音频特征提取 + 深度学习分类 + 部署** 的完整工作流程，适合：
- AI 入门音频分类学习
- 简历项目补充音频方向技能
- 后续业务音频分类项目快速参考

---


