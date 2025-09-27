import spacy
import sys
import os

# Add the project root to Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

try:
    from ai.training.utils import normalize_text

except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from ai.training.utils import normalize_text


print("\n=== Testing Trained Model ===")
# Test with the just-trained model first
test_text = """LONG #DODOUSDT from [$1.371](https://www.binance.com/ru/trade/DODOUSDT?utm_source=CScalp)

Проторговка под уровнем 1.40. На уровне стоит плотность. Выше - каскад сильных уровней. Хороший потенциал в лонг с понятной точкой и целями."""
test_text_normalized = normalize_text(test_text)

print(f"Original test text: {test_text}")
print(f"Normalized test text: {test_text_normalized}")

nlp_loaded = spacy.load("ai/models/ner_model")
doc_loaded = nlp_loaded(test_text_normalized)

print("Entities found by loaded model:")
for ent in doc_loaded.ents:
    print(f"  '{ent.text}' ({ent.label_}) at {ent.start_char}-{ent.end_char}")

if not doc_loaded.ents:
    print("  No entities found by loaded model")
