import json

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    instructions = []
    for i in range(0, len(lines), 1):
        instruction = lines[i].strip()

        output = lines[i + 1].strip() if i + 1 < len(lines) else ""
        
        if instruction=='' or output=='':
            continue
        instructions.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    with open(output_file, 'w', encoding='utf-8') as file:
        for instruction in instructions:
            file.write(json.dumps(instruction, ensure_ascii=False) + '\n')

# 使用示例
input_file = '/home/4T/tangshunye/fri/Play-with-LLMs/finetune-qlora-baichuan/data/1046-海边的卡夫卡.txt'
output_file = '/home/4T/tangshunye/fri/Play-with-LLMs/finetune-qlora-baichuan/data/1046-海边的卡夫卡.json'
process_file(input_file, output_file)
