import spacy
import sys
import os

# Add the project root to Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_training.utils import normalize_text


print("\n=== Testing Trained Model ===")
# Test with the just-trained model first
test_text = """СИГНАЛ #ADA/USDT
🔑 Откройте ШОРТ по цене $0,8149 - $0,8239 с кредитным плечом X25.
🍒 Цели:
🔘 $0,8084
🔘 $0,8051
🔘 $0,7979
❗️ СТОП-ЛОСС: $0,8509"""
test_text_normalized = normalize_text(test_text)

print(f"Original test text: {test_text}")
print(f"Normalized test text: {test_text_normalized}")

nlp_loaded = spacy.load("./ner_model")
doc_loaded = nlp_loaded(test_text_normalized)

print("Entities found by loaded model:")
for ent in doc_loaded.ents:
    print(f"  '{ent.text}' ({ent.label_}) at {ent.start_char}-{ent.end_char}")

if not doc_loaded.ents:
    print("  No entities found by loaded model")
