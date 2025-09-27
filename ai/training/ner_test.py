import spacy
from utils import normalize_text, relabel_numeric_entities


print("\n=== Testing Trained Model ===")
# Test with the just-trained model first
test_text = """LONG #DODOUSDT from [$1.371](https://www.binance.com/ru/trade/DODOUSDT?utm_source=CScalp)

Проторговка под уровнем 1.40. На уровне стоит плотность. Выше - каскад сильных уровней. Хороший потенциал в лонг с понятной точкой и целями."""

test_text = """LONG #ETHUSDT from [$1 080](https://www.binance.com/ru/trade/ETHUSDT?utm_source=CScalp) stop loss [$1 070](https://www.binance.com/ru/trade/ETHUSDT?utm_source=CScalp)

5m TF. Сделка на выход из консолидации. Инструмент торгуется на повышенных объемах. Ожидаю импульсивный выход в лонг.

Автор: [Market Situations](https://t.me/marketsituation)

#ПробойУровня"""
test_text_normalized = normalize_text(test_text)

print(f"Original test text: {test_text}")
print(f"Normalized test text: {test_text_normalized}")

nlp_loaded = spacy.load("ai/models/ner_model")
doc_loaded = nlp_loaded(test_text_normalized)

raw_entities = [
    {
        "text": ent.text,
        "label": ent.label_,
        "start": ent.start_char,
        "end": ent.end_char,
    }
    for ent in doc_loaded.ents
]

entities = relabel_numeric_entities(test_text_normalized, raw_entities)

if entities:
    print("Entities found by loaded model (post-processed):")
    for ent, original in zip(entities, doc_loaded.ents):
        label_display = ent["label"]
        if original.label_ != ent["label"]:
            label_display = f"{label_display} (was {original.label_})"
        print(f"  '{ent['text']}' ({label_display}) at {ent['start']}-{ent['end']}")
else:
    print("  No entities found by loaded model")
