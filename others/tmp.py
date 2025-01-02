from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "baichuan-inc/baichuan-7B"  # 这里可以替换为你想要的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="D:\\baichuan",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir="D:\\baichuan", device_map="auto", trust_remote_code=True)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
