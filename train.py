import argparse
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import librosa
import torch
import wandb
from datasets import load_dataset, load_from_disk
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

def get_parser(
    parser=argparse.ArgumentParser(
        description="fine-tune a speech model on a speech recognition task"
    ),
):
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--data_path", type=pathlib.Path, help="path to the train dataset"
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda") if torch.cuda.is_available() else "cpu",
        help="path to the train file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="whisper",
        choices=("whisper", "hubert", "wav2vec2.0"),
        help="name of the model to fine-tune",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("saved_models"),
        help="where to save trained mdodel",
    )
    parser.add_argument(
        "--cache_dir",
        type=pathlib.Path,
        default=pathlib.Path("cache"),
        help="where to save cache",
    )
    return parser


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    features = feature_extractor(audio["array"], sampling_rate=int(audio["sampling_rate"]))
    batch["input_features"] = features["input_features"][0]

    transcript = batch["transcript"]
    if transcript:
        # encode target text to label ids 
        batch["labels"] = tokenizer(transcript.lower()).input_ids
    else:
        batch["labels"] = tokenizer("").input_ids
    return batch


def train(args):
    print(args.save_dir)
    print(args.cache_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    if args.model_name == "whisper":
        # load the model and processor
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir=args.cache_dir)
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe", cache_dir=args.cache_dir)
    metric = evaluate.load("wer", cache_dir=args.cache_dir)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
        
    # load the dataset
    dataset = load_from_disk(args.data_path) # the datasetdict object only contains the train split
    # prepare the dataset
    dataset = dataset.map(lambda batch: prepare_dataset(batch, processor.feature_extractor, processor.tokenizer),\
                          cache_file_names={"train": f"{args.cache_dir}/cache_map.pkl"})
    processed_dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=42)
    # instantiate the data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # instantiate the training arguments
 
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
)
    # instantiate the trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
)
    # train the model
    trainer.train()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.save_dir = "" + str(args.model_name)
    args.cache_dir = "" + str(args.model_name)
    args.data_path = ""
    train(args)