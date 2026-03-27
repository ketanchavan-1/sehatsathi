import sys
sys.path.insert(0, r'c:\Users\mayan\python.py')
import ml2

test_text = "Sardi khokla tap ghamoriya"
result = ml2.transliterate_to_devanagari(test_text)

print(f"Input: {test_text}")
print(f"Output: {result}")
print(f"Changed: {result != test_text}")
print(f"Has Devanagari: {any('\\u0900' <= ch <= '\\u097F' for ch in result)}")

# Test individual words
words = ["sardi", "khokla", "tap", "ghamoriya"]
for word in words:
    trans = ml2.transliterate_to_devanagari(word)
    print(f"  '{word}' -> '{trans}' (changed: {trans != word})")
