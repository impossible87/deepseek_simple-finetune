import json


def json_to_jsonl(input_file, output_file):
    """将JSON文件转换为JSONL格式"""
    try:
        # 读取原始JSON数据
        with open(input_file, "r", encoding="utf-8") as fin:
            data = json.load(fin)

        # 写入JSONL格式
        with open(output_file, "w", encoding="utf-8") as fout:
            for item in data:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ 转换成功！生成文件：{output_file}")
        print(f"📂 共转换 {len(data)} 条数据")

    except FileNotFoundError:
        print(f"❌ 错误：输入文件 {input_file} 不存在")
    except json.JSONDecodeError:
        print("❌ 错误：输入的JSON格式无效")


if __name__ == "__main__":
    # 配置路径（按需修改）
    json_to_jsonl("medical_r1_distill_sft_Chinese.json", "datasets_medical.jsonl")