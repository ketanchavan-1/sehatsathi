# Add Hindi & Marathi Language Support for Disease Prediction

The model currently accepts English symptoms and outputs English disease names. We'll add a translation layer so users can input symptoms in Hindi or Marathi and get disease predictions in their chosen language.

**Approach**: The model itself stays trained on English. We add a translation dictionary that maps Hindi/Marathi symptom names → English (for input) and English disease names → Hindi/Marathi (for output). This avoids retraining and keeps accuracy identical.

## Proposed Changes

### Translation Data

#### [NEW] [translations.json](file:///c:/Users/mayan/python.py/translations.json)

A JSON file containing two sections:
- **`symptoms`**: Maps each English symptom (132 total) to its Hindi and Marathi translations
- **`diseases`**: Maps each English disease (41 total) to its Hindi and Marathi translations

Example structure:
```json
{
  "symptoms": {
    "itching": {"hi": "खुजली", "mr": "खाज"},
    "skin_rash": {"hi": "त्वचा पर चकत्ते", "mr": "त्वचेवर पुरळ"},
    ...
  },
  "diseases": {
    "Fungal infection": {"hi": "फंगल संक्रमण", "mr": "बुरशीजन्य संसर्ग"},
    ...
  }
}
```

---

### Model Script

#### [MODIFY] [ml2.py](file:///c:/Users/mayan/python.py/ml2.py)

1. **Add `--lang` argument** (choices: `en`, `hi`, `mr`, default: `en`)
2. **Add `load_translations()` function** — loads `translations.json`
3. **Add `translate_symptoms_to_english()` function** — converts Hindi/Marathi symptom input to English symptom names for model input
4. **Add `translate_disease_from_english()` function** — converts English disease names to Hindi/Marathi for output
5. **Update [main()](file:///c:/Users/mayan/python.py/ml2.py#64-101)** — integrate translation before prediction and after getting results
6. **Update prompts** — show Hindi/Marathi prompt text when `--lang` is `hi` or `mr`

## Verification Plan

### Automated Tests

Run the script in all three languages and verify output:

```powershell
# English (existing behavior unchanged)
& .venv/Scripts/python.exe ml2.py -s "itching,skin_rash" --lang en

# Hindi input
& .venv/Scripts/python.exe ml2.py -s "खुजली,त्वचा पर चकत्ते" --lang hi

# Marathi input
& .venv/Scripts/python.exe ml2.py -s "खाज,त्वचेवर पुरळ" --lang mr
```

All three should produce valid disease predictions. Hindi/Marathi runs should output disease names in the respective language.
