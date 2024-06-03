from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Laden des Tokenizers und des Modells
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# NER-Pipeline einrichten
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Beispieltext
text = "Autophagy maintains tumour growth through circulating arginine. Autophagy captures intracellular components and delivers them to lysosomes, where they are degraded and recycled to sustain metabolism and to enable survival during starvation1-5. Acute, whole-body deletion of the essential autophagy gene Atg7 in adult mice causes a systemic metabolic defect that manifests as starvation intolerance and gradual loss of white adipose tissue, liver glycogen and muscle mass1. Cancer cells also benefit from autophagy."

# NER anwenden
ner_results = nlp(text)
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']:.2f}")
