from transformers import AutoTokenizer, AutoModelForCausalLM

model_name="替换成自己模型的绝对地址"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

print("-------加载模型成功-------")

import dateset_R1,json
samples = dateset_R1.samples
with open("datasets.jsonl","w",encoding="utf-8") as f:
    for s in samples:
        json_line = json.dumps(s,ensure_ascii=False)
        f.write(json_line+"\n")
    else:
        print("--------写入文件成功-------")


from datasets import load_dataset
dataset = load_dataset(path="json", data_files={"train": "datasets.jsonl"}, split="train")
print("数据数量：", len(dataset))

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"训练集数量：{len(train_dataset)}")
print(f"测试集数量：{len(eval_dataset)}")

print("完成准备")


def tokenize_function(many_examples):
    texts =[f"{instruction}: {output}" for instruction, output in zip(many_examples["instruction"], many_examples["output"])]
    tokens=tokenizer(texts, truncation=True, max_length=256, padding="max_length")
    tokens["labels"]=tokens["input_ids"].copy()
    return tokens

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)

print("-------完成tokenizing-------")


#量化设置
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(model_name, config=quantization_config,device_map="auto")

print("-------完成量化-------")


from peft import LoraConfig , get_peft_model , TaskType

Lora_Config = LoraConfig(
    r=8 , lora_alpha=32, lora_dropout=0.2 ,task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model , Lora_Config)
model.print_trainable_parameters()

print("-------完成lora微调-------")

#设置训练参数
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./fubetuned models",
    num_train_epochs=3,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=10,
    fp16=True,
    logging_steps=10,
    save_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    learning_rate=2e-5,
    logging_dir="./logs",
    run_name="deepseek-r1-distill-finetune",
)

print("-------完成训练参数设置-------")
#定义训练器
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
)

print("-------完成训练器定义-------")

#训练模型
trainer.train()

print("-------完成模型训练-------")

#评估模型
#trainer.evaluate()

#模型微调

#保存lora模型
save_path="./saved_models"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("-------完成模型保存-------")

#保存全量模型
final_save_path="./final_models"

from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model=PeftModel.from_pretrained(base_model,save_path)
model=model.merge_and_unload()

model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("-------完成全量模型保存-------")