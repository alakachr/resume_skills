from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

from resume_skills.dataset import DonutDataset

# hyperparameters used for multiple args
hf_repository_id = "donut-base-resume"

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")


train_dataset = DonutDataset()
model.config.pad_token_id = train_dataset.processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = train_dataset.processor.tokenizer.convert_tokens_to_ids([""])[
    0
]
# Arguments for training
training_args = Seq2SeqTrainingArguments(
    output_dir=hf_repository_id,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=hf_repository_id,
    hub_token=HfFolder.get_token(),
)

# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
train_dataset.processor.save_pretrained(hf_repository_id)
trainer.create_model_card()
trainer.push_to_hub()
