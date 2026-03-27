import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

# Load translations
with open('translations.json', 'r', encoding='utf-8') as f:
    trans = json.load(f)

# Test what we're looking for
test_text_hi = "तीव्र ज्वर"  # High fever in Hindi
test_text_mr = "उच्च ताप"   # High fever in Marathi

print("=" * 60)
print("Testing Hindi single symptm: तीव्र ज्वर")
print("=" * 60)

# Check what matches in translations 
symptoms_dict = trans.get('symptoms', {})
matched_hi = []
for eng_symptom, trans_dict in symptoms_dict.items():
    hi_name = trans_dict.get("hi", "").strip()
    if hi_name and hi_name in test_text_hi:
        matched_hi.append((eng_symptom, hi_name))
        print(f"Match found: {eng_symptom} -> {hi_name}")

if not matched_hi:
    print("No matches found!")
    print("\nSearching for 'fever' or 'high' in translations...")
    for eng_symptom, trans_dict in symptoms_dict.items():
        hi_name = trans_dict.get("hi", "").strip()
        if 'fever' in eng_symptom.lower() or 'fever' in hi_name.lower() or 'ज्वर' in hi_name:
            print(f"  English: {eng_symptom}")

print("\n" + "=" * 60)
print("Testing Marathi single symptom: उच्च ताप")
print("=" * 60)

matched_mr = []
for eng_symptom, trans_dict in symptoms_dict.items():
    mr_name = trans_dict.get("mr", "").strip()
    if mr_name and mr_name in test_text_mr:
        matched_mr.append((eng_symptom, mr_name))

if not matched_mr:
    print("No matches found!")
    print("\nSearching for fever symptoms in Marathi...")
    for eng_symptom, trans_dict in symptoms_dict.items():
        mr_name = trans_dict.get("mr", "").strip()
        if 'fever' in eng_symptom.lower() and mr_name:
            print(f"  English: {eng_symptom}")
            print(f"  Marathi: {mr_name[0:50]}")
