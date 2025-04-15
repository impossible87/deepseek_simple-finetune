import os
from modelscope import snapshot_download


def download_mo_dataset(dataset_name, save_subdir):
    """参数说明：
    dataset_name: 魔搭数据集ID（如'damo/nlp_NuminaMath-CoT'）
    save_subdir: 相对于脚本目录的保存子目录（如'datasets'）
    """
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建完整保存路径
    full_save_path = os.path.join(script_dir, save_subdir)

    # 下载数据集到指定路径
    dataset_dir = snapshot_download(
        dataset_name,
        cache_dir=full_save_path,  # 动态生成的路径
        repo_type='dataset',
        revision='master',
        ignore_file_pattern='*.md'
    )
    return dataset_dir


# 使用示例（下载到脚本同级的 datasets 文件夹）
download_mo_dataset(
    'huangxp/hwtcm-deepseek-r1-distill-data',
    'datasets'  # 相对脚本目录的子目录
)




