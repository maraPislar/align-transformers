import pyvene
from pyvene import IntervenableRepresentationConfig, IntervenableConfig, IntervenableModel
import random
from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

def generate_sum_examples(num_examples=100):
    prompts = []
    answers = []
    full_text = []

    for _ in range(num_examples):
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        num3 = random.randint(1, 10)

        prompt = f"Calculate {num1}+{num2}+{num3}="
        answer = str(num1 + num2 + num3)
        full_text.append(f"{num1}+{num2}+{num3}=" + str(answer))

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers, full_text

def generate_training_file(file_path, num_examples = 10000):

    _, _, data = generate_sum_examples(num_examples)

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
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
  
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
        
    trainer.train()
    trainer.save_model()

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(prompt, label, max_length):
    model_path = "../align-transformers/result"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{prompt}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
    print(f"True labal: {label}")

def eval_finetuned_gpt2(num_examples=100):
    prompts, labels, _ = generate_sum_examples(num_examples)
    max_len=1
    for prompt, label in zip(prompts, labels):
        generate_text(prompt, label, max_len)

def main():

    _, tokenizer, gpt2 = pyvene.create_gpt2_lm()
    tokenizer.pad_token = tokenizer.eos_token
    _ = gpt2.to("cuda")
    _ = gpt2.eval()

    file_path = "training_sums.txt"

    generate_training_file(file_path)

    train_file_path = "sums.txt"
    output_dir = '../align-transformers/result'
    overwrite_output_dir = False
    per_device_train_batch_size = 8
    num_train_epochs = 5
    save_steps = 500

    train(
        train_file_path=train_file_path,
        model=gpt2,
        tokenizer=tokenizer,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

    eval_finetuned_gpt2(prompts, labels)

if __name__ =="__main__":
    main()