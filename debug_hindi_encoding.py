import json
import sys

# Read translations
with open('translations.json', 'r', encoding='utf-8') as f:
    trans = json.load(f)

# Get high fever translations
high_fever_trans = trans['symptoms']['high fever']
hindi_trans = high_fever_trans.get('hi', '')

# Test string
test_hindi = "तीव्र ज्वर"

# Compare bytes
print("Hindi translation from file:")
print(f"  Length: {len(hindi_trans)}")
print(f"  Bytes: {hindi_trans.encode('utf-8').hex()}")
print()
print("Test string:")
print(f"  Length: {len(test_hindi)}")
print(f"  Bytes: {test_hindi.encode('utf-8').hex()}")
print()

# Check if they match
print(f"Match: {hindi_trans == test_hindi}")

# Check if one is in the other
print(f"hindi_trans in test_hindi: {hindi_trans in test_hindi}")
print(f"test_hindi in hindi_trans: {test_hindi in hindi_trans}")

# Try NFD normalization (common Unicode normalization form)
import unicodedata
hindi_trans_nfd = unicodedata.normalize('NFD', hindi_trans)
test_hindi_nfd = unicodedata.normalize('NFD', test_hindi)

print()
print("After NFD normalization:")
print(f"  Match: {hindi_trans_nfd == test_hindi_nfd}")
print(f"  hindi_trans_nfd bytes: {hindi_trans_nfd.encode('utf-8').hex()}")
print(f"  test_hindi_nfd bytes:  {test_hindi_nfd.encode('utf-8').hex()}")
