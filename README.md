# DeepSeek 模型微调项目

本项目提供了一套完整的流程，用于微调DeepSeek系列大语言模型。通过本项目，您可以轻松地对模型进行个性化训练，使其更好地适应特定任务。

## 目录

- [环境准备](#环境准备)
  - [安装CUDA](#安装cuda)
  - [配置Conda环境](#配置conda环境)
- [项目设置](#项目设置)
  - [导入项目到IDE](#导入项目到ide)
  - [下载模型和数据集](#下载模型和数据集)
- [模型微调前准备](#模型微调前准备)
  - [检查环境](#检查环境)
  - [初始模型调用](#初始模型调用)
- [开始模型微调](#开始模型微调)
  - [数据格式转换](#数据格式转换)
  - [执行微调](#执行微调)
  - [验证微调效果](#验证微调效果)

## 环境准备

### 安装CUDA

本项目使用CUDA 11.8版本进行GPU加速。请从NVIDIA官方网站下载并安装CUDA 11.8版本。

1. 访问[NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. 选择适合您操作系统的CUDA 11.8版本
3. 按照安装向导完成安装

### 配置Conda环境

1. 创建并激活新的conda环境：

```bash
conda create -n deepseek python=3.10
conda activate deepseek
```

2. 安装项目依赖：

```bash
# 进入项目目录后，安装requirements.txt中的依赖
pip install -r requirements.txt
```

3. **特别注意**：
   - 对于requirements.txt中的triton依赖，需要修改路径为您本地triton文件的绝对路径：
     ```
     # 将requirements.txt中的这一行
     triton @ file:///"改成你triton-2.0.0-cp310-cp310-win_amd64.whl文件的绝对地址"
     # 修改为（示例）
     triton @ file:///E:/deepseek Fine tune/triton/triton-2.0.0-cp310-cp310-win_amd64.whl
     ```

4. 单独安装特定版本的PyTorch和xformers（这些不包含在requirements.txt中）：

```bash
# 安装指定版本的PyTorch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# 安装指定版本的xformers
pip install xformers==0.0.26.post1+cu118
```

## 项目设置

### 导入项目到IDE

将项目文件夹拖拽到您喜欢的IDE中（如PyCharm、VSCode等）。

### 下载模型和数据集

您可以通过以下两种方法下载所需的模型和数据集：

#### 方法1：使用项目脚本下载

项目提供了自动下载脚本：

```bash
# 下载模型
python model/model.py

# 下载数据集
python datasets/datasets.py
```

#### 方法2：手动下载

您也可以直接从以下平台手动下载：

- [魔搭社区](https://modelscope.cn/)：搜索并下载DeepSeek相关模型和数据集
- [Hugging Face](https://huggingface.co/)：需要科学上网，搜索并下载DeepSeek相关模型和数据集

## 模型微调前准备

### 检查环境

运行环境检查脚本，确认CUDA是否可用：

```bash
python 1_Check_the_library.py
```

如果输出显示CUDA可用，则可以继续下一步。

### 初始模型调用

运行初始模型调用脚本，测试模型是否正常加载：

```bash
python 2_Initial_call_model.py
```

请确保在脚本中正确设置了模型路径。

## 开始模型微调

### 数据格式转换

如果您的数据集不是JSONL格式，需要先进行格式转换：

```bash
python "json to jsonl.py"
```

### 执行微调

根据您的数据集格式选择合适的微调脚本：

- 如果数据集包含思考过程（reasoning）：
  ```bash
  python 3_main_ito.py
  ```

- 如果数据集不包含思考过程：
  ```bash
  python 3_main_io.py
  ```

请确保在脚本中正确设置了模型路径和数据集路径。

### 验证微调效果

运行验证脚本，查看微调后的模型效果：

```bash
python 4_Fine-tune_the_invocation_model.py
```

该脚本将加载微调后的模型，并使用示例问题测试模型的回答质量，以便您比较微调前后的变化。

## 注意事项

- 确保您的GPU内存足够运行选定的模型
- 微调过程可能需要较长时间，请耐心等待
- 如遇到CUDA内存不足错误，可尝试减小batch_size或使用更小的模型

---

祝您微调顺利！如有问题，请参考项目中的源代码或相关文档。