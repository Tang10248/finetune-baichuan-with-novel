# how to run

`torchrun --nproc_per_node=8 train.py --output_dir ./output`

u can use the [.sh file](command/baichuan_lora_belle_20k.sh) to set ur own parms 

# note

if u wanna fitune the model in 4bit quantization,`pip install bitsandbytes==0.39.1`

if in 8bit quantization,`pip install bitsandbytes==0.37.2`