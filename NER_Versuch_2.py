# from datasets import load_dataset
# from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification
# import numpy as np
# from datasets import load_metric
# from transformers import TrainingArguments, Trainer
#
#
#
# # Lädt wikiann Datensatz mit dem Norwegisch-Tag ("no") und speichert ihn in der Variable dataset
# dataset = load_dataset("unimelb-nlp/wikiann", "no")
#
# # Extrahiert die NER-Tags (Labelnamen) aus dem Trainingsdatensatz und speichert sie in label_names.
# label_names = dataset["train"].features["ner_tags"].feature.names
# print(label_names)
#
# # Extrahiert die ersten zwei Beispiele aus dem Trainingsdatensatz und speichert sie in der Variable dataset['train'][:2]
# dataset['train'][:2]
# print(dataset['train'][:2])
#
# # Lädt den distilbert-base-uncased Tokenizer aus der transformers Bibliothek
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#
# # tokenisiert die tokens in jedem Beispiel des Datensatzes. is_split_into_words=True: Eingabe in Wörter gesplittet
# def tokenize_function(examples):
#     return tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True)
#
# # Wendet die tokenize_function auf den gesamten Datensatz an.
# tokenized_datasets_ = dataset.map(tokenize_function, batched=True)
#
# # Überprüft, ob die Länge der input_ids (die tokenisierten Eingaben) gleich der Länge der ner_tags (NER-Labels) ist.
# print(len(tokenized_datasets_['train'][0]['input_ids']) == len(tokenized_datasets_['train'][0]['ner_tags']))
#
#
# # Funktion zur Erhaltung der Werte für input_ids, attention_mask und angepasste Labels
# def tokenize_adjust_labels(samples):
#     # tokinisiert die Beispiele in Batches und teilt die Eingabe in Wörter
#     tokenized_samples = tokenizer.batch_encode_plus(samples["tokens"], is_split_into_words=True, truncation=True)
#
#     # Liste zur Speicherung der angepassten Labels
#     total_adjusted_labels = []
#
#     # Iteration übr alle tokinisierten Eingaben
#     for k in range(0, len(tokenized_samples["input_ids"])):
#         prev_wid = -1
#         word_ids_list = tokenized_samples.word_ids(batch_index=k) # Wort-IDs für das aktuelle Beispiel
#         existing_label_ids = samples["ner_tags"][k] # Vorhandene Labels für das aktuelle Beispiel
#         i = -1
#
#     # Liste zur Speicherung der angepassten Labels für das aktuelle Beispiel
#     adjusted_label_ids = []
#
#     # Iteration über die Wort-IDs
#     for word_idx in word_ids_list:
#     # Special tokens have a word id that is None. We set the label to -100
#     # so they are automatically ignored in the loss function.
#         if(word_idx is None):
#             adjusted_label_ids.append(-100)
#         elif(word_idx!=prev_wid):
#             i = i + 1
#             adjusted_label_ids.append(existing_label_ids[i])
#             prev_wid = word_idx
#         else:
#             label_name = label_names[existing_label_ids[i]]
#             adjusted_label_ids.append(existing_label_ids[i])
#
#         # angepassten Labels zur Gesamtliste hinzufügen
#         total_adjusted_labels.append(adjusted_label_ids)
#
#     # angepasste Labels zu den tokenisierten Beispielen hinzufügen
#     tokenized_samples["labels"] = total_adjusted_labels
#
#     return tokenized_samples
#
# # Test der Tokenisierung mit einem Beispieltext
# out = tokenizer("Fine tune NER in google colab")
# print(out)
#
# # Zeigt Wort-IDs für das Beispiel an
# print(out.word_ids(0))
#
# # Anwendung der Funktion zur Tokenisierung und Anpassung der Labels auf den gesamten Datensatz
# tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True, remove_columns=['tokens', 'ner_tags', 'langs', 'spans'])
# print(tokenized_dataset)
#
# tokenized_dataset['train'][:2]
#
# data_collator = DataCollatorForTokenClassification(tokenizer)
#
# model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_names))
#
#
# metric = load_metric("seqeval")
#
# def compute_metrics(p):
#     predictions, labels = p
#     #select predicted index with maximum logit for each token
#     predictions = np.argmax(predictions, axis=2)
#     # Remove ignored index (special tokens)
#     true_predictions = [
#         [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }
#
# batch_size = 16
# logging_steps = len(tokenized_dataset['train']) // batch_size
# epochs = 2
#
# training_args = TrainingArguments(
#     output_dir="results",
#     num_train_epochs=epochs,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     evaluation_strategy="epoch",
#     disable_tqdm=False,
#     logging_steps=logging_steps)
#
#
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# #fine tune using train method
# trainer.train()
#
# out = tokenizer("Fine tune NER in google colab!")
# print(out)
#
# out.word_ids(0)


import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer
from datasets import load_metric

# Laden des Datasets
dataset = load_dataset("unimelb-nlp/wikiann", "no")

# Label-Namen extrahieren
label_names = dataset["train"].features["ner_tags"].feature.names
print(label_names)

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Funktion zur Tokenisierung und Anpassung der Labels
def tokenize_adjust_labels(samples):
    tokenized_samples = tokenizer.batch_encode_plus(samples["tokens"], is_split_into_words=True, truncation=True,
                                                    padding="max_length")

    total_adjusted_labels = []

    for k in range(len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = samples["ner_tags"][k]
        i = -1
        adjusted_label_ids = []

        for word_idx in word_ids_list:
            # Special tokens have a word id that is None. We set the label to -100
            # so they are automatically ignored in the loss function.
            if word_idx is None:
                adjusted_label_ids.append(-100)
            elif word_idx != prev_wid:
                i += 1
                adjusted_label_ids.append(existing_label_ids[i])
                prev_wid = word_idx
            else:
                # Append the same label for subwords
                adjusted_label_ids.append(existing_label_ids[i])

        # Make sure the length matches
        while len(adjusted_label_ids) < len(tokenized_samples["input_ids"][k]):
            adjusted_label_ids.append(-100)
        total_adjusted_labels.append(adjusted_label_ids)

    # Add adjusted labels to the tokenized samples
    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples


# Tokenisierten und angepassten Datensatz erstellen
tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True,
                                remove_columns=['tokens', 'ner_tags', 'langs', 'spans'])
print(tokenized_dataset)

# Überprüfen, ob die Längen übereinstimmen
print(len(tokenized_dataset['train'][0]['input_ids']) == len(tokenized_dataset['train'][0]['labels']))

# Datenkollator für die Token-Klassifizierung
data_collator = DataCollatorForTokenClassification(tokenizer)

# Modell für die Token-Klassifizierung laden
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_names))

# Metrik laden
metric = load_metric("seqeval")


# Funktion zur Berechnung der Metriken
def compute_metrics(p):
    predictions, labels = p
    # select predicted index with maximum logit for each token
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Trainingsparameter
batch_size = 16
logging_steps = len(tokenized_dataset['train']) // batch_size
epochs = 2

training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps
)

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Feintuning durchführen
# trainer.train()

# Beispiel-Tokenisierung
out = tokenizer("Fine tune NER in google colab!")
print(out)

print(out.word_ids(0))



predictions, labels, _ = trainer.predict(tokenized_dataset["test"])

predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)

true_predictions = [
    [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

true_labels = [
    [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)

print(results)

