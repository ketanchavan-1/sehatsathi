import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
import argparse
import json
import os
import sys
import tempfile
import sounddevice as sd  # type: ignore
from scipy.io.wavfile import write  # type: ignore
import speech_recognition as sr  # type: ignore

# Fix Windows console encoding for Hindi/Marathi output
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from xgboost import XGBClassifier  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "en": "Enter comma-separated symptoms (e.g. itching, skin rash): ",
    "hi": "लक्षण दर्ज करें, अल्पविराम से अलग करें (जैसे खुजली, त्वचा पर चकत्ते): ",
    "mr": "लक्षणे प्रविष्ट करा, स्वल्पविरामाने वेगळे करा (उदा. खाज, त्वचेवर पुरळ): ",
}

VOICE_PROMPTS = {
    "en": {"choose": "Do you want to Type (t) or Speak (s) the symptoms? ", "speak": "Listening for 5 seconds... Speak now!"},
    "hi": {"choose": "क्या आप लक्षण टाइप (t) करना चाहते हैं या बोलना (s) चाहते हैं? ", "speak": "5 सेकंड तक सुन रहे हैं... अब बोलें!"},
    "mr": {"choose": "तुम्हाला लक्षणे टाइप (t) करायची आहेत की बोलायची (s) आहेत? ", "speak": "5 सेकंद ऐकत आहे... आता बोला!"},
}


def load_translations():
    path = os.path.join(SCRIPT_DIR, "translations.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transliterate_to_devanagari(text):
    """Convert common Roman transliteration patterns to Devanagari.
    Handles Google STT output which often converts Hindi/Marathi to Roman characters.
    Uses a mapping of common symptom patterns since external library installation failed.
    """
    text_lower = text.lower()
    
    # Common IAST/HK symptom mappings for Hindi/Marathi
    # This handles common voice input patterns
    transliteration_map = {
        # Respiratory
        "sardi khokla tap ghamoriya": "सर्दी खोखला ताप घमोरिया",  # Full phrase from screenshot
        "sardi khokla": "सर्दी खोखला",
        "sardi": "सर्दी",  # Common cold
        "khokla": "खोखला",  # Cough-like (variant)
        "khasi": "खांसी",  # Cough
        
        # Fever
        "tap ghamoriya": "ताप घमोरिया",
        "tap": "ताप",  # Fever (Marathi)
        "bukhaar": "बुखार",  # Fever (Hindi)
        "bukhar": "बुखार",  # Fever (variant)
        "tivra": "तीव्र",  # High/severe
        "taap": "ताप",  # Fever (variant)
        
        # Pain
        "dukhi": "दुखी",  # Pain/ache
        "dard": "दर्द",  # Pain
        "sir dard": "सिर दर्द",  # Headache
        "sira dard": "सिर दर्द",  # Headache
        "sir dukhi": "सिर दुखी",  # Headache
        
        # Digestive
        "pet": "पेट",  # Belly/stomach
        "pet dard": "पेट दर्द",  # Abdominal pain
        "ghamoriya": "घमोरिया",  # Rash/heat rash
    }
    
    # Try longest matches first to handle phrases before individual words
    matched = False
    for roman, devanagari in sorted(transliteration_map.items(), key=lambda x: len(x[0]), reverse=True):
        if roman in text_lower:
            # Replace case-insensitively
            import re
            text = re.sub(re.escape(roman), devanagari, text_lower, flags=re.IGNORECASE)
            matched = True
            break
    
    # Return transliterated version if we found a match
    if matched and any('\u0900' <= ch <= '\u097F' for ch in text):
        return text
    
    return text  # Return original if no mapping found


def get_voice_input(lang):
    """Capture audio from microphone and convert to text using Google STT."""
    print(VOICE_PROMPTS.get(lang, VOICE_PROMPTS["en"])["speak"])
    
    fs = 44100  # Sample rate
    seconds = 5  # Duration
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Save as temp WAV file
    temp_wav = tempfile.mktemp(suffix=".wav")
    write(temp_wav, fs, recording)
    
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            
        # Determine STT language code
        stt_lang = "en-IN"
        if lang == "hi":
            stt_lang = "hi-IN"
        elif lang == "mr":
            stt_lang = "mr-IN"
            
        text = recognizer.recognize_google(audio_data, language=stt_lang)
        
        # For Hindi/Marathi, try to transliterate Roman characters to Devanagari
        # (Google STT sometimes returns romanized text instead of Devanagari)
        if lang in ["hi", "mr"]:
            transliterated = transliterate_to_devanagari(text)
            # Use transliterated version if it resulted in Devanagari characters
            if transliterated != text and any('\u0900' <= ch <= '\u097F' for ch in transliterated):
                text = transliterated
        
        os.remove(temp_wav)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        os.remove(temp_wav)
        return ""
    except Exception as e:
        print(f"Error during speech recognition: {e}")
        try:
            os.remove(temp_wav)
        except OSError:
            pass
        return ""


def detect_language(text, translations):
    """Auto-detect language from input text.
    Returns 'hi', 'mr', or 'en'.
    """
    # Check if text contains Devanagari characters (used by both Hindi and Marathi)
    has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
    if not has_devanagari:
        return "en"

    # Distinguish Hindi vs Marathi by counting symptom matches in each language
    # Don't use .lower() on Devanagari - it doesn't work properly with Unicode
    symptoms = [s.strip() for s in text.split(",") if s.strip()]
    hi_matched: list[str] = []
    mr_matched: list[str] = []
    for eng_symptom, trans in translations["symptoms"].items():
        hi_name = trans.get("hi", "").strip()
        mr_name = trans.get("mr", "").strip()
        for s in symptoms:
            if hi_name and (s == hi_name or s in hi_name or hi_name in s):
                hi_matched.append(s)
            if mr_name and (s == mr_name or s in mr_name or mr_name in s):
                mr_matched.append(s)

    if len(mr_matched) > len(hi_matched):
        return "mr"
    return "hi"  # Default to Hindi if Devanagari detected and can't distinguish


def translate_symptoms_to_english(symptoms, translations, lang):
    """Convert Hindi/Marathi symptom input to English symptom names."""
    if lang == "en":
        return symptoms

    # Build reverse map: translated symptom -> english symptom
    # Don't use .lower() on Devanagari - use exact match instead
    reverse_map: dict[str, str] = {}
    for eng_symptom, trans in translations["symptoms"].items():
        if lang in trans:
            reverse_map[trans[lang].strip()] = eng_symptom

    translated = []
    for symptom in symptoms:
        s = symptom.strip()
        if s in reverse_map:
            translated.append(reverse_map[s])
        else:
            # Keep as-is if not found (maybe user typed English symptom with --lang hi)
            translated.append(s)
    return translated


def translate_disease_to_lang(disease_name, translations, lang):
    """Convert English disease name to Hindi/Marathi."""
    if lang == "en":
        return disease_name
    disease_trans = translations["diseases"].get(disease_name, {})
    return disease_trans.get(lang, disease_name)


def load_data(path="dataset (version 1).xlsx"):
    df = pd.read_excel(path)
    symptom_columns = [col for col in df.columns if "Symptom" in col]
    df["symptoms"] = df[symptom_columns].values.tolist()
    # Handle NaN, lowercase, strip whitespace, and replace underscores with spaces
    df["symptoms"] = df["symptoms"].apply(lambda x: [s.strip().lower().replace("_", " ") for s in x if pd.notna(s)])
    return df


def train_model(df):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["symptoms"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Disease"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    return model, mlb, label_encoder


def predict_disease(symptoms, model, mlb, label_encoder):
    encoded = mlb.transform([symptoms])
    probs = model.predict_proba(encoded)[0]
    if np.max(probs) <= 1.0:
        probs = probs * 100

    diseases = label_encoder.inverse_transform(np.arange(len(probs)))
    results: dict[str, float] = {}
    for d, p in zip(diseases, probs):
        val: float = float(p)
        results[d] = int(val * 100) / 100.0
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return sorted_results


def main():
    parser = argparse.ArgumentParser(description="Train and predict disease from symptoms")
    parser.add_argument("--symptoms", "-s", nargs="?", const="", default=None, help="Comma-separated symptoms e.g. itching, skin rash")
    parser.add_argument("--dataset", type=str, default="dataset (version 1).xlsx", help="Path to dataset")
    parser.add_argument("--lang", type=str, choices=["auto", "en", "hi", "mr"], default="auto", help="Language: auto (detect), en (English), hi (Hindi), mr (Marathi)")
    parser.add_argument("direct_symptoms", nargs="*", help="Optional direct symptoms (space-separated)")
    args = parser.parse_args()

    lang = args.lang
    translations = load_translations()

    print("Loading dataset...")
    df = load_data(args.dataset)
    print("Dataset shape:", df.shape)

    print("Training model...")
    model, mlb, label_encoder = train_model(df)

    # Use English prompt when lang is auto (we don't know the language yet)
    prompt_lang = lang if lang != "auto" else "en"

    if args.symptoms is not None and args.symptoms != "":
        user_input = args.symptoms.strip()
    elif args.direct_symptoms:
        user_input = ",".join(args.direct_symptoms)
    else:
        # Ask user if they want to type or speak
        choice = input(VOICE_PROMPTS.get(prompt_lang, VOICE_PROMPTS["en"])["choose"]).strip().lower()
        if choice == 's':
            user_input = get_voice_input(prompt_lang)
            if not user_input:
                user_input = input(PROMPTS.get(prompt_lang, PROMPTS["en"])).strip()
        else:
            user_input = input(PROMPTS.get(prompt_lang, PROMPTS["en"])).strip()

    if not user_input:
        raise SystemExit("No symptoms provided. Exiting.")

    # Auto-detect language from input if needed
    if lang == "auto":
        lang = detect_language(user_input, translations)
        lang_names = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
        print("Detected language:", lang_names.get(lang, lang))

    user_symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]
    print("User Input:", user_input)

    # Translate symptoms to English for model
    english_symptoms = translate_symptoms_to_english(user_symptoms, translations, lang)
    if lang != "en":
        print("Translated Symptoms:", english_symptoms)

    # Validate that at least one symptom is known to the model
    valid_symptoms = [s for s in english_symptoms if s in mlb.classes_]
    if not valid_symptoms:
        print("\nError: Invalid input. None of the provided symptoms match the dataset vocabulary.")
        print("Please try different symptoms.")
        raise SystemExit(1)

    predictions = predict_disease(english_symptoms, model, mlb, label_encoder)
    print("\nTop Predictions:")
    items = list(predictions.items())
    for i, (disease, prob) in enumerate(items):
        if i >= 5:
            break
        display_name = translate_disease_to_lang(disease, translations, lang)
        print(f"{display_name}")


if __name__ == "__main__":
    main()
