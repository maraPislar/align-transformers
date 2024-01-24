import pyvene
from pyvene import IntervenableRepresentationConfig, IntervenableConfig, IntervenableModel
import random
from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from sklearn.metrics import classification_report
from pyvene import CausalModel
import numpy as np
from datasets import Dataset
import torch

def generate_sum_examples(inputs, labels):
    prompts = []
    answers = []
    full_text = []

    for sum_input, label in zip(inputs,labels):
        prompt = f"{int(sum_input[0])}+{int(sum_input[1])}+{int(sum_input[2])}="
        full_text.append(f"{prompt}{int(label.item())}")

        prompts.append(prompt)
        answers.append(str(int(label.item())))

    return prompts, answers, full_text

def input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    return {"X":A, "Y":B, "Z":C}

def generate_file(file_path, inputs, labels):

    _, _, data = generate_sum_examples(inputs, labels)

    # Open the file in write mode and write each string followed by a newline
    with open(file_path, 'w') as file:
        for string in data:
            file.write(string + '\n')

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator


def train(train_file_path,
          model,
          tokenizer,
          output_dir,
          overwrite_output_dir,
          batch_size,
          num_train_epochs,
          save_steps):
  
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    _ = model.train()

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        learning_rate=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=train_dataset
    )
        
    _ = trainer.train()
    trainer.save_model()

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def get_predicted_label(model, tokenizer, prompt, max_length):
    ids = tokenizer.encode(f'{prompt}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    print(generated_text.strip())
    return generated_text[len(prompt):].strip()

def eval_finetuned_gpt2(prompts, labels, num_examples=100):
    model_path = "/gpfs/home1/mpislar/align-transformers/result/"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    _ = model.eval()
    max_len=2
    count = 0
    for prompt, label in zip(prompts, labels):
        pred_label = get_predicted_label(model, tokenizer, prompt, max_len)
        if pred_label == label:
            count += 1
    if count > 0:
        print(f"Accuracy is {count/num_examples}")
    else:
        print("Accuracy is 0.")

def randvec(n=50, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for i in range(n)])

def randNum(lower=1, upper=10):
        return random.randint(lower, upper)

def causal_model_1():

    variables =  ["X", "Y", "Z", "P", "O"]
    number_of_entities = 20

    reps = [randNum() for _ in range(number_of_entities)]
    values = {variable:reps for variable in ["X", "Y", "Z"]}
    values["P"] = list(range(2, 21))
    values["O"] = list(range(3, 31))

    parents = {"X":[], "Y":[], "Z":[], 
            "P":["X", "Y"],
            "O":["P", "Z"]}

    def FILLER():
        return reps[0]

    functions = {"X":FILLER, "Y":FILLER, "Z":FILLER, 
                "P": lambda x,y: x+y,
                "O": lambda x,y: x+y}

    pos = {"X":(1,0.1), "Y":(2,0.2), "Z":(2.8,0), 
            "P":(1,2),
            "O":(1.5,3)}

    return CausalModel(variables, values, parents, functions, pos = pos)

def causal_model_2():
    
    variables =  ["X", "Y", "Z", "P", "O"]
    number_of_entities = 20

    reps = [randNum() for _ in range(number_of_entities)]
    values = {variable:reps for variable in ["X", "Y", "Z"]}
    values["P"] = list(range(2, 21))
    values["O"] = list(range(3, 31))

    parents = {"X":[], "Y":[], "Z":[], 
            "P":["Y", "Z"],
            "O":["X", "P"]}

    def FILLER():
        return reps[0]

    functions = {"X":FILLER, "Y":FILLER, "Z":FILLER, 
                "P": lambda x,y: x+y,
                "O": lambda x,y: x+y}

    pos = {"X":(1,0.1), "Y":(2,0.2), "Z":(2.8,0), 
            "P":(2,1),
            "O":(1.5,3)}

    return CausalModel(variables, values, parents, functions, pos = pos)

def causal_model_3():
    
    variables =  ["X", "Y", "Z", "P", "O"]
    number_of_entities = 20

    reps = [randNum() for _ in range(number_of_entities)]
    values = {variable:reps for variable in ["X", "Y", "Z"]}
    values["P"] = list(range(2, 21))
    values["O"] = list(range(3, 31))

    parents = {"X":[], "Y":[], "Z":[], 
            "P":["X", "Z"],
            "O":["P", "Y"]}

    def FILLER():
        return reps[0]

    functions = {"X":FILLER, "Y":FILLER, "Z":FILLER, 
                "P": lambda x,y: x+y,
                "O": lambda x,y: x+y}

    pos = {"X":(1,0.1), "Y":(2,0.2), "Z":(2.8,0), 
            "P":(1,2),
            "O":(1.5,3)}

    return CausalModel(variables, values, parents, functions, pos = pos)

def main():

    _, tokenizer, gpt2 = pyvene.create_gpt2_lm()
    tokenizer.pad_token = tokenizer.eos_token
    _ = gpt2.to("cuda")

    train_file_path = "/gpfs/home1/mpislar/align-transformers/my_experiments/sum_training_data/training_sums.txt"

    n_examples = 1280000
    causal_model = causal_model_2()
    inputs, labels = causal_model.generate_factual_dataset(n_examples, input_sampler)
    generate_file(train_file_path, inputs, labels)

    output_dir = "/gpfs/home1/mpislar/align-transformers/result/"
    overwrite_output_dir = False
    batch_size = 64
    num_train_epochs = 30
    save_steps = 500

    train(
        train_file_path=train_file_path,
        model=gpt2,
        tokenizer=tokenizer,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

    n_examples = 100
    test_inputs, test_labels = causal_model.generate_factual_dataset(n_examples, input_sampler)
    eval_finetuned_gpt2(test_inputs, test_labels, n_examples)

if __name__ =="__main__":
    main()