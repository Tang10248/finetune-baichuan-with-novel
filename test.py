import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import datasets
from peft import PeftModel, prepare_model_for_kbit_training
import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from model import load_model
from peft import PeftModel
from transformers import  GenerationConfig

model, tokenizer = load_model("baichuan", "./pretrained/baichuan-7b/", quantization="16bit",local_rank='auto')
lora_model = PeftModel.from_pretrained(model, "./output/baichuan_lorasft/")
# lora_model=model
generation_config = GenerationConfig(
        temperature=0.5,
        top_p = 0.85,
        do_sample = True, 
        repetition_penalty=2.0, 
        max_new_tokens=1024,  # max_length=max_new_tokens+input_sequence

)
device = model.device



def pro(prompt):
    
    # input ="Human: " + prompt + "\n\nAssistant: "
    inpp=prompt
    inputs = tokenizer(inpp,return_tensors="pt").to(device)
    generate_ids = lora_model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.decode(generate_ids[0][len(inputs.input_ids[0]):])
    print(output)
    return output
prompt='你是我的朋友，现在你要和我聊天。'
while True:
    inp=input("输入：")
    prompt+='我：'+inp+'。'+'你：'
    # prompt+='Human:'+inp+'。'+'Assistant:'
    prompt+=pro(prompt).split('</s>')[0].strip()
