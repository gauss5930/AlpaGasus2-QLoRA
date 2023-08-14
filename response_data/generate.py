from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import json
import jsonlines
import fire

def main(
  base_model: str = "meta-llama/Llama-2-13b-hf",
  lora_weight: str = "StudentLLM/Alpagasus-2-13B-QLoRA",
  auth_token: str = "",
  test_path: str = "AlpaGasus2-QLoRA/test_data/",
  save_path: str = "",
):
  
  tokenizer = AutoTokenizer.from_pretrained(lora_weight)

  config = PeftConfig.from_pretrained(lora_weight)
  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      use_auth_token=auth_token,
  )
  model = PeftModel.from_pretrained(model, lora_weight)

  result = []
  for i in range(len(test_data)):
    path = test_path + test_data[i]
    count = 0
    name = test_data[i].split('_')[0]
    with jsonlines.open(path) as f:
      for line in f:
        if col[i]:
          input_data = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{line[col[i]]}\n\n### Response:\n"
        else:
          if line['instances'][0]['input']:
            input_data = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{line['instruction']}\n\n### Input:\n{line['instances'][0]['input']}\n\n### Response:\n"
          else:
            input_data = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{line['instruction']}\n\n### Response:\n"
        model_inputs = tokenizer(input_data, return_tensors='pt').to(torch_device)
        model_output = model.generate(**model_inputs, max_new_tokens=256)
        model_output = tokenizer.decode(model_output[0], skip_special_tokens=True)
        count += 1
        index = name + '_' + str(count)
        ind = {"question_id": index}
        output = {"model_output": model_output}
        result.append(ind.update(output))
  
  with open(save_path, "x") as json_file:
    json.dump(result, json_file, indent=4)

if __name__ == "__main__":
  fire.Fire(main)
