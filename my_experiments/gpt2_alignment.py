import sys, os
sys.path.append(os.path.join('..', '..'))

import pyvene
from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import classification_report
from pyvene import CausalModel
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
import random

from pyvene import (
    IntervenableModel,
    RotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)

# generate the prompts given the inputs and outputs generated with the causal model
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

# sample such numbers to be fed into the task
def input_sampler():
    A = randNum()
    B = randNum()
    C = randNum()
    # return f"{int(A)}+{int(B)}+{int(C)}="
    return {"X":A, "Y":B, "Z":C}

# save all the data in a file for easier training of gpt2
def generate_file(file_path, inputs, labels):

    _, _, data = generate_sum_examples(inputs, labels)

    # Open the file in write mode and write each string followed by a newline
    with open(file_path, 'w') as file:
        for string in data:
            file.write(string + '\n')

######## Helper functions ########

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

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

################ End ################

# training function for gpt2
def train(train_file_path,
          model,
          tokenizer,
          output_dir,
          overwrite_output_dir,
          batch_size,
          num_train_epochs):
  
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

def get_predicted_label(model, tokenizer, prompt_ids, max_length):
    final_outputs = model.generate(
        prompt_ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    print(generated_text.strip())
    # return generated_text[len(prompt_ids[0]):].strip()
    return generated_text.split('=')[1].strip()

def eval_finetuned_gpt2(model, tokenizer, prompts_ids, labels, num_examples=100):
    _ = model.eval()
    max_len=2
    count = 0
    for prompt_ids, label in zip(prompts_ids, labels):
        pred_label = get_predicted_label(model, tokenizer, prompt_ids, max_len)
        if pred_label == str(int(label.item())):
            count += 1
    if count > 0:
        print(f"Accuracy is {count/num_examples}")
    else:
        print("Accuracy is 0.")

def randNum(lower=1, upper=10):
    tokenizer = load_tokenizer("/gpfs/home1/mpislar/align-transformers/result/")
    number = random.randint(lower, upper)
    return tokenizer.encode(f'{number}', return_tensors='pt')

def causal_model_1():

    variables =  ["X", "Y", "Z", "P", "O"]
    number_of_entities = 20
    tokenizer = load_tokenizer("/gpfs/home1/mpislar/align-transformers/result/")

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
                "P": lambda x, y: int(tokenizer.decode(x[0], skip_special_tokens=True).split()[0]) + int(tokenizer.decode(y[0], skip_special_tokens=True).split()[0]),
                "O": lambda x, y: x + int(tokenizer.decode(y[0], skip_special_tokens=True).split()[0])}

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

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += eval_pred == eval_label
    accuracy = float(correct_count) / float(total_count)
    return {"accuracy": accuracy}


def compute_loss(outputs, labels):
    CE = torch.nn.CrossEntropyLoss()
    return CE(outputs, labels)


def batched_random_sampler(data, batch_size):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

def intervention_id(intervention):
    if "P" in intervention:
        return 0

def train_gpt2(causal_model, n_examples):
    _, tokenizer, gpt2 = pyvene.create_gpt2_lm()
    tokenizer.pad_token = tokenizer.eos_token
    _ = gpt2.to("cuda")

    train_file_path = "/gpfs/home1/mpislar/align-transformers/my_experiments/sum_training_data/training_sums.txt"

    # generate data for training gpt2
    inputs, labels = causal_model.generate_factual_dataset(n_examples, input_sampler, inputFunction=tokenizePrompt)
    generate_file(train_file_path, inputs, labels)

    # train gpt2 on summing three numbers
    output_dir = "/gpfs/home1/mpislar/align-transformers/result/"
    overwrite_output_dir = False
    batch_size = 64
    num_train_epochs = 70

    train(
        train_file_path=train_file_path,
        model=gpt2,
        tokenizer=tokenizer,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs
    )

def tokenizePrompt(prompt):
    tokenizer = load_tokenizer("/gpfs/home1/mpislar/align-transformers/result/")
    prompt = f"{tokenizer.decode(prompt['X'], skip_special_tokens=True)}+{tokenizer.decode(prompt['Y'], skip_special_tokens=True)}+{tokenizer.decode(prompt['Z'], skip_special_tokens=True)}="
    print(prompt)
    return tokenizer.encode(prompt, return_tensors='pt')

def main():

    n_examples = 1280000
    causal_model = causal_model_1()

    # train gpt2
    # train_gpt2(causal_model, n_examples)

    # load the trained model
    model_path = "/gpfs/home1/mpislar/align-transformers/result/"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)

    print('evaluating...')

    # generate data for testing if gpt2 has learnt the task well
    n_examples = 100
    test_causal_model = causal_model_1()
    test_inputs, test_labels = test_causal_model.generate_factual_dataset(n_examples, input_sampler, inputFunction=tokenizePrompt)
    # test_inputs, test_labels, _ = generate_sum_examples(test_inputs, test_labels) # convert back to prompt
    eval_finetuned_gpt2(model, tokenizer, test_inputs, test_labels, n_examples)

    # # define intervention model
    # intervenable_config = IntervenableConfig(
    #     model_type=type(model),
    #     representations=[
    #         RepresentationConfig(
    #             0,  # layer
    #             "block_output",  # intervention type
    #             "pos",  # intervention unit is now aligne with tokens
    #             1,  # max number of unit
    #             subspace_partition=None,  # binary partition with equal sizes
    #             intervention_link_key=0,
    #         )
    #         # RepresentationConfig(
    #         #     0,  # layer
    #         #     "block_output",  # intervention type
    #         #     "pos",  # intervention unit is now aligne with tokens
    #         #     1,  # max number of unit
    #         #     subspace_partition=None,  # binary partition with equal sizes,
    #         #     intervention_link_key=0,
    #         # ),
    #     ],
    #     intervention_types=RotatedSpaceIntervention,
    # )

    # intervenable = IntervenableModel(intervenable_config, model, use_fast=True)
    # intervenable.set_device("cuda")
    # intervenable.disable_model_gradients()

    # epochs = 10
    # gradient_accumulation_steps = 1
    # total_step = 0
    # # target_total_step = len(dataset) * epochs

    # # t_total = int(len(dataset) * epochs)
    # optimizer_params = []
    # for k, v in intervenable.interventions.items():
    #     optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
    #     break
    # optimizer = torch.optim.Adam(optimizer_params, lr=0.001)

    # print('generating data for DAS...')

    # n_examples = 12800
    # batch_size = 64

    # train_dataset = causal_model.generate_counterfactual_dataset(
    #     n_examples,
    #     intervention_id,
    #     batch_size,
    #     sampler=input_sampler,
    #     inputFunction=tokenizePrompt
    # )

    # # train DAS
    # print('training DAS...')

    # embedding_dim = 6

    # intervenable.model.train()  # train enables drop-off but no grads
    # print("intervention trainable parameters: ", intervenable.count_parameters())
    # train_iterator = trange(0, int(epochs), desc="Epoch")

    # for epoch in train_iterator:
    #     epoch_iterator = tqdm(
    #         DataLoader(
    #             train_dataset,
    #             batch_size=batch_size,
    #             sampler=batched_random_sampler(train_dataset, batch_size),
    #         ),
    #         desc=f"Epoch: {epoch}",
    #         position=0,
    #         leave=True,
    #     )
    #     for batch in epoch_iterator:
    #         batch["input_ids"] = batch["input_ids"].unsqueeze(1)
    #         batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
    #         batch_size = batch["input_ids"].shape[0]
    #         for k, v in batch.items():
    #             if v is not None and isinstance(v, torch.Tensor):
    #                 batch[k] = v.to("cuda")

    #         if batch["intervention_id"][0] == 0:
    #             _, counterfactual_outputs = intervenable(
    #                 {"inputs_embeds": batch["input_ids"]}, # base
    #                 [{"inputs_embeds": batch["source_input_ids"][:, 0]}], # source
    #                 {
    #                     "sources->base": (
    #                         [[[0]] * batch_size],
    #                         [[[0]] * batch_size],
    #                     )
    #                 }, # unit locations
    #                 subspaces=[
    #                     [[_ for _ in range(0, embedding_dim * 2)]] * batch_size
    #                 ],
    #             )

    #         eval_metrics = compute_metrics(
    #             counterfactual_outputs[0].argmax(1), batch["labels"].squeeze()
    #         )

    #         # loss and backprop
    #         loss = compute_loss(
    #             counterfactual_outputs[0], batch["labels"].squeeze().to(torch.long)
    #         )

    #         epoch_iterator.set_postfix({"loss": loss, "acc": eval_metrics["accuracy"]})

    #         if gradient_accumulation_steps > 1:
    #             loss = loss / gradient_accumulation_steps
    #         loss.backward()
    #         if total_step % gradient_accumulation_steps == 0:
    #             optimizer.step()
    #             intervenable.set_zero_grad()
    #         total_step += 1
    
    # # test DAS

    # test_dataset = test_causal_model.generate_counterfactual_dataset(
    #     10000, intervention_id, batch_size, device="cuda:0", sampler=input_sampler, inputFunction=tokenizePrompt
    # )

    # eval_labels = []
    # eval_preds = []
    # with torch.no_grad():
    #     epoch_iterator = tqdm(DataLoader(test_dataset, batch_size), desc=f"Test")
    #     for step, batch in enumerate(epoch_iterator):
    #         for k, v in batch.items():
    #             if v is not None and isinstance(v, torch.Tensor):
    #                 batch[k] = v.to("cuda")
    #         batch["input_ids"] = batch["input_ids"].unsqueeze(1)
    #         batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)

    #         if batch["intervention_id"][0] == 0:
    #             _, counterfactual_outputs = intervenable(
    #                 {"inputs_embeds": batch["input_ids"]},
    #                 [{"inputs_embeds": batch["source_input_ids"][:, 0]}, None],
    #                 {
    #                     "sources->base": (
    #                         [[[0]] * batch_size, None],
    #                         [[[0]] * batch_size, None],
    #                     )
    #                 },
    #                 subspaces=[
    #                     [[_ for _ in range(0, embedding_dim * 2)]] * batch_size
    #                 ],
    #             )
            
    #         eval_labels += [batch["labels"]]
    #         eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
    # print(classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu()))

if __name__ =="__main__":
    main()