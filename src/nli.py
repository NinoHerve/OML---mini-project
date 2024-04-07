import evaluate
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

class MNLITask:
  def __init__(self, model_name):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    self.data_collator = DataCollatorWithPadding(self.tokenizer)
    self.dataset, self.encoded_dataset = self.get_nli_dataset(self.tokenizer)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    self.precision, self.recall, self.f1, self.accuracy = evaluate.load("precision"), evaluate.load("recall"), evaluate.load("f1"), evaluate.load("accuracy")
  
  def get_nli_dataset(self, tokenizer):
    dataset = load_dataset("glue", "mnli")
    dataset["validation"], dataset["test"] = dataset.pop("validation_matched"), dataset.pop("test_matched")
    dataset.pop("validation_mismatched")
    dataset.pop("test_mismatched")
    encoded_dataset = dataset.map(lambda sample: tokenizer(sample["premise"], sample["hypothesis"], truncation=True), batched=True)
    return dataset, encoded_dataset
  
  def compute_metrics(self, eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return self.metric.compute(predictions=predictions, references=labels)
  
  def compute_metrics(self, eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision_score = self.precision.compute(references=labels, predictions=predictions)
    recall_score = self.recall.compute(references=labels, predictions=predictions)
    f1_score = self.f1.compute(references=labels, predictions=predictions)
    accuracy_score = self.accuracy.compute(references=labels, predictions=predictions)

    return {
        "precision": precision_score["precision"],
        "recall": recall_score["recall"],
        "f1": f1_score["f1"],
        "accuracy": accuracy_score["accuracy"],
    }