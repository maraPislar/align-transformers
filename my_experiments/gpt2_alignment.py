import pyvene
from pyvene import IntervenableRepresentationConfig, IntervenableConfig, IntervenableModel
import random
from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from sklearn.metrics import classification_report

def generate_sum_examples(num_examples=100):
    prompts = []
    answers = []
    full_text = []

    for _ in range(num_examples):
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        num3 = random.randint(1, 10)

        prompt = f"{num1}+{num2}+{num3}="
        answer = str(num1 + num2 + num3)
        full_text.append(f"{num1}+{num2}+{num3}=" + str(answer))

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers, full_text

def generate_file(file_path, num_examples = 10000):

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
    print(generated_text)
    return generated_text[len(prompt):].strip()

def eval_finetuned_gpt2(num_examples=100):
    model_path = "/gpfs/home1/mpislar/align-transformers/result/"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    _ = model.eval()
    prompts, labels, _ = generate_sum_examples(num_examples)
    max_len=9
    count = 0
    for prompt, label in zip(prompts, labels):
        pred_label = get_predicted_label(model, tokenizer, prompt, max_len)
        if pred_label == label:
            count += 1
    if count > 0:
        print(f"Accuracy is {count/num_examples}")
    else:
        print("Accuracy is 0.")

def main():

    _, tokenizer, gpt2 = pyvene.create_gpt2_lm()
    tokenizer.pad_token = tokenizer.eos_token
    _ = gpt2.to("cuda")

    train_file_path = "/gpfs/home1/mpislar/align-transformers/my_experiments/sum_training_data/training_sums.txt"

    generate_file(train_file_path, 128000)

    output_dir = "/gpfs/home1/mpislar/align-transformers/result/"
    overwrite_output_dir = False
    batch_size = 64
    num_train_epochs = 10
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

    eval_finetuned_gpt2(100)

if __name__ =="__main__":
    main()