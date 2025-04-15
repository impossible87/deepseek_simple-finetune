import os
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

# 初始Hub API
api = HubApi()
api.login('YOUR_MODELSCOPE_SDK_ACCESS_TOKEN')  # 替换为你的令牌

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建缓存目录路径（脚本同级目录下的models文件夹）
cache_dir = os.path.join(script_dir, 'models')

# 下载模型到脚本同级的models目录
model_path = snapshot_download(
    model_id='unsloth/DeepSeek-R1-Distill-Llama-8B',
    cache_dir=cache_dir
)

print(f"Model downloaded to {model_path}")