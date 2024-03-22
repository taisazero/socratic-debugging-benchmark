"""
This class is used to finetune a huggingface model on the Socratic Debugging dataset.
This class is a wrapper around the huggingface Trainer class.
"""

import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
from data_utils.data_reader import XMLDataReader
import numpy as np
from metrics.metric_computer import MetricComputer
from transformers import T5Config


def get_huggingface_path(model_name):
    path = (
        os.environ["HF_MODELS"] + f"/{model_name}"
        if "HF_MODELS" in os.environ.keys()
        else model_name
    )
    return path


class Seq2SeqFineTuner:
    def __init__(self, model_name, train_dataset, val_dataset=None, export_path=None):
        # model_config = T5Config.from_pretrained(model_name)
        # model_config.max_position_embeddings = 50_000
        # self.model = AutoModelForSeq2SeqLM.from_config(model_config)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=8_000)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.metric_computer = MetricComputer(
            export_path=export_path if export_path is not None else "results.xlsx",
            export_to_excel=True,
        )

    def preprocess_function(self, examples, input_field="goal", target_field="sol1"):
        inputs = examples[input_field]
        targets = examples[target_field]

        return self.tokenizer(
            inputs, text_target=targets, padding="longest", return_tensors="pt"
        )

    def collate_fn(self, examples):
        return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")

    def compute_metrics(self, eval_pred):
        preds, labels, input_ids = (
            eval_pred.predictions,
            eval_pred.label_ids,
            eval_pred.inputs,
        )

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print('decoded labels', decoded_labels)

        # This fixes TypeError argument `ids`: `list` object cannot be interpreted as an integer
        #
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print('decoded preds', decoded_preds)
        contexts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print('contexts', contexts)
        # create a list of lists for each pred and label split by '\n* ' and remove `* ` from the first element
        decoded_preds = [pred.split("* ") for pred in decoded_preds]
        decoded_preds = [
            pred[2:] if pred[0:2] == "* " else pred for pred in decoded_preds
        ]
        decoded_labels = [label.split("* ") for label in decoded_labels]
        decoded_labels = [
            label[2:] if label[0:2] == "* " else label for label in decoded_labels
        ]

        return self.metric_computer.compute_thoroughness(
            decoded_preds, decoded_labels, contexts=contexts
        )

    def fine_tune(
        self,
        output_dir,
        batch_size=2,
        epochs=3,
        learning_rate=1e-5,
        input_field="goal",
        target_field="sol1",
    ):
        train_data = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            fn_kwargs={"input_field": input_field, "target_field": target_field},
        )
        val_data = self.val_dataset.map(
            self.preprocess_function,
            batched=True,
            fn_kwargs={"input_field": input_field, "target_field": target_field},
        )
        # remove unnecessary columns
        self.train_dataset = train_data.remove_columns(self.train_dataset.column_names)
        self.val_dataset = val_data.remove_columns(self.val_dataset.column_names)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            save_total_limit=2,
            # save_steps=1000,
            # eval_steps=250,
            logging_dir="./trainer_logs",
            logging_steps=50,
            logging_first_step=True,
            report_to="tensorboard",
            metric_for_best_model="loss",
            gradient_accumulation_steps=32,  # total should be 32
            include_inputs_for_metrics=True,
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_max_length=1024,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # load a small subset of the dataset for testing

    dataset = load_dataset("piqa")
    train_dataset = dataset["train"].select(
        range(100)
    )  # Select a small subset for testing
    val_dataset = dataset["validation"].select(
        range(10)
    )  # Select a small subset for testing

    model_name = "google/flan-t5-small"
    model_name = get_huggingface_path(model_name)
    fine_tuner = Seq2SeqFineTuner(model_name, train_dataset, val_dataset)
    fine_tuner.fine_tune(output_dir="test-model")
