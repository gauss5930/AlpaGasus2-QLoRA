from datasets import load_dataset

# Loading json type dataset using load_dataset of datasets library
def loading_dataset(data_path):
  data = load_dataset('json', data_files = '/content/drive/MyDrive/Research/AlpaGasus2-QLoRA/alpaca_t45.json')
  dataset = data['train']
  return dataset

# This is the formatting function of dataset.
# We follow the data format of Stanford Alpaca
def formatting_prompts_func(example):
  output_texts = []
  for i in range(len(example)):
    if example['input'][i] == '':
      text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction'][i]}\n\n### Response: "
    else:
      text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:"
    targets = f"{example['output'][i]}"
    texts = text + targets
    output_texts.append(texts)
  return output_texts

# max_steps calculation function
def max_steps_calc(epochs, batch, data_line_count):
  return int(data_line_count / batch * epochs)
