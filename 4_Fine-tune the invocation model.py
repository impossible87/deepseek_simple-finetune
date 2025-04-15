from transformers import AutoTokenizer, AutoModelForCausalLM


#加载微调模型
final_save_path="./final_models"
model=AutoModelForCausalLM.from_pretrained(final_save_path)
tokenizer=AutoTokenizer.from_pretrained(final_save_path)

model.to("cuda")
#构建推理流程
from transformers import pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "你自己的问题"
generated_text= pipe(prompt,max_length=1024,num_return_sequences=1,truncation=True)

print("开始回答：--------",generated_text[0]["generated_text"])