import spacy
from utils import normalize_text


print("\n=== Testing Trained Model ===")
# Test with the just-trained model first
test_text = """LONG #DODOUSDT from [$1.371](https://www.binance.com/ru/trade/DODOUSDT?utm_source=CScalp)

Проторговка под уровнем 1.40. На уровне стоит плотность. Выше - каскад сильных уровней. Хороший потенциал в лонг с понятной точкой и целями."""

test_text = """LONG #ETHUSDT from [$1 080](https://www.binance.com/ru/trade/ETHUSDT?utm_source=CScalp) stop loss [$1 070](https://www.binance.com/ru/trade/ETHUSDT?utm_source=CScalp)

5m TF. Сделка на выход из консолидации. Инструмент торгуется на повышенных объемах. Ожидаю импульсивный выход в лонг.

Автор: [Market Situations](https://t.me/marketsituation)

#ПробойУровня"""
test_text_normalized = normalize_text(test_text, collapse_digit_spaces=True)

print(f"Original test text: {test_text}")
print(f"Normalized test text: {test_text_normalized}")

nlp_loaded = spacy.load("ai/models/ner_model")
doc_loaded = nlp_loaded(test_text_normalized)

print("Entities found by loaded model:")
for ent in doc_loaded.ents:
    print(f"  '{ent.text}' ({ent.label_}) at {ent.start_char}-{ent.end_char}")

if not doc_loaded.ents:
    print("  No entities found by loaded model")
