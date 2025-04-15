from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 设置模型路径（替换为你的实际路径）
model_path = r"D:\porn model"  # 包含所有配置文件的文件夹

# 2. 加载模型和分词器
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)


# 3. 推理函数
def generate_response(query):
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    ).to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=5120,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 4. 测试示例
if __name__ == "__main__":
    test_queries = [
        "你自己的问题"

    ]

    for query in test_queries:
        print(f"输入：{query}")
        print(f"输出：{generate_response(query)}\n")