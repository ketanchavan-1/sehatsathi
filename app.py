from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
import uvicorn  # type: ignore
import ml2  # type: ignore
import google.generativeai as genai  # type: ignore
import os
import joblib  # type: ignore
from google.api_core.exceptions import GoogleAPICallError  # type: ignore

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print(f"Gemini initialization error: {e}")
        gemini_model = None

app = FastAPI(title="Sehat Sathi - Disease Prediction API")

# Enable CORS for local HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

# Global state for ML
model = None
mlb = None
label_encoder = None
translations = None

@app.on_event("startup")
def load_ml_components():
    global model, mlb, label_encoder, translations
    print("Loading translations...")
    translations = ml2.load_translations()

    model_path = "disease_model.joblib"
    mlb_path = "mlb.joblib"
    label_encoder_path = "label_encoder.joblib"

    if all(os.path.exists(path) for path in [model_path, mlb_path, label_encoder_path]):
        print("Loading pre-trained model artifacts...")
        model = joblib.load(model_path)
        mlb = joblib.load(mlb_path)
        label_encoder = joblib.load(label_encoder_path)
    else:
        print("Loading dataset (version 1).xlsx...")
        df = ml2.load_data("dataset (version 1).xlsx")
        print("Training XGBoost model in memory...")
        model, mlb, label_encoder = ml2.train_model(df)
    print("Backend ready.")


@app.get("/")
def home():
    return FileResponse("HACK.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}

def build_fallback_advice(symptoms, predicted_diseases):
    """Provide basic non-Gemini guidance when the API is unavailable."""
    symptom_text = ", ".join(symptoms[:4]) if symptoms else "your symptoms"
    disease_text = predicted_diseases[0] if predicted_diseases else "a common illness"
    return (
        f"Your reported symptoms ({symptom_text}) may be associated with {disease_text}. "
        "Rest, stay hydrated, eat light food, and monitor whether symptoms are improving or getting worse.\n\n"
        "Seek medical help sooner if you have trouble breathing, severe chest pain, persistent vomiting, dehydration, "
        "very high fever, confusion, or symptoms that keep worsening.\n\n"
        "Please consult a qualified doctor for proper diagnosis and treatment."
    )


def get_common_case_prediction(symptoms):
    """Return safer common-condition predictions for very common symptom sets."""
    symptom_set = set(symptoms)
    red_flag_symptoms = {
        "chest pain",
        "breathlessness",
        "blood in sputum",
        "weight loss",
        "yellowing of eyes",
        "yellowish skin",
        "weakness of one body side",
        "slurred speech",
        "blackheads",  # avoid forcing common-case logic onto unrelated symptom groups
    }
    if symptom_set & red_flag_symptoms:
        return None

    common_symptom_pool = {
        "fever (>101Â°f)",
        "high fever",
        "mild fever",
        "severe headache",
        "headache",
        "cough",
        "fatigue",
        "malaise",
        "continuous sneezing",
        "runny nose",
        "congestion",
        "throat irritation",
        "joint/muscle pain",
        "muscle pain",
        "nausea/vomiting",
        "abdominal pain",
        "stomach pain",
        "diarrhoea",
        "itching",
        "skin rash",
    }
    if len(symptom_set) > 4 or not symptom_set.issubset(common_symptom_pool):
        return None

    if {"continuous sneezing", "cough"}.issubset(symptom_set) or {"runny nose", "congestion"}.issubset(symptom_set):
        return ["Common Cold / Seasonal Allergies", "Common Viral Fever"]
    if {"high fever", "severe headache"}.issubset(symptom_set) or {"fever (>101Â°f)", "severe headache"}.issubset(symptom_set):
        return ["Common Viral Fever", "Seasonal Viral Infection"]
    if {"high fever", "cough"}.issubset(symptom_set) or {"fever (>101Â°f)", "cough"}.issubset(symptom_set):
        return ["Common Viral Fever", "Common Cold / Seasonal Allergies"]
    if {"itching", "skin rash"}.issubset(symptom_set):
        return ["Common Skin Allergy", "Fungal infection"]
    if {"nausea/vomiting", "abdominal pain"}.issubset(symptom_set) or {"nausea/vomiting", "stomach pain"}.issubset(symptom_set):
        return ["Common Indigestion or Acid Reflux", "Food Poisoning"]
    if {"fatigue", "severe headache"}.issubset(symptom_set):
        return ["General Fatigue / Stress", "Common Headache"]
    if "cough" in symptom_set and "severe headache" in symptom_set:
        return ["Common Viral Fever", "Common Cold / Seasonal Allergies"]
    if "high fever" in symptom_set or "fever (>101Â°f)" in symptom_set:
        return ["Common Viral Fever"]
    if "cough" in symptom_set:
        return ["Common Cold / Seasonal Allergies"]
    if "severe headache" in symptom_set or "headache" in symptom_set:
        return ["Common Headache"]
    return None

def get_gemini_advice(symptoms, predicted_diseases):
    """Use Gemini 2.0 Flash to generate a brief explanation and health advice."""
    fallback = build_fallback_advice(symptoms, predicted_diseases)
    if not gemini_model:
        print("Gemini is disabled: missing or invalid API configuration.")
        return fallback, False, "Gemini is not configured."

    try:
        prompt = f"""You are a helpful medical AI assistant for an Indian healthcare platform called Sehat Sathi.

A patient reported these symptoms: {', '.join(symptoms)}
Our ML model predicted these possible conditions: {', '.join(predicted_diseases)}

Please provide:
1. A brief explanation (2-3 sentences) of why these symptoms might indicate the predicted condition(s).
2. Practical health advice (3-4 bullet points) including home remedies and when to see a doctor.

Keep the response concise, friendly, and easy to understand. Do NOT use markdown formatting. Use plain text only. Always end with a reminder to consult a qualified doctor."""

        response = gemini_model.generate_content(
            prompt,
            request_options={"timeout": 20},
        )
        advice_text = getattr(response, "text", "") or ""
        if advice_text.strip():
            return advice_text, True, ""
        return fallback, False, "Gemini returned an empty response."
    except GoogleAPICallError as e:
        error_message = str(e)
        print(f"Gemini API error: {error_message}")
        if "quota" in error_message.lower() or "429" in error_message:
            return fallback, False, "Gemini quota exceeded for the current API key."
        return fallback, False, error_message
    except Exception as e:
        print(f"Gemini API error: {e}")
        return fallback, False, str(e)

@app.post("/predict")
def predict_disease_api(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")
    
    text = req.text.strip()
    
    # Detect language - handle both comma-separated and natural text input
    # For natural speech input (no commas), just check if it contains Devanagari
    has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
    
    # If no Devanagari but text looks like it might be transliterated Hindi/Marathi,
    # try to convert it (this handles Google STT output for Hindi/Marathi voices)
    if not has_devanagari and any(char.isalpha() for char in text):
        # Check if it looks like it could be transliteration (has common Indic patterns)
        transliterated = ml2.transliterate_to_devanagari(text)
        if transliterated != text and any('\u0900' <= ch <= '\u097F' for ch in transliterated):
            text = transliterated
            has_devanagari = True
    
    if has_devanagari:
        # Contains Devanagari - try comma-separated format for detection
        lang = ml2.detect_language(text, translations)
    else:
        lang = "en"
    
    # Split text by commas or spaces depending on format (if user speaks naturally, there are no commas)
    # So we replace common words like "and", "or", "mera", "," with commas or spaces?
    # Simple split by space/comma:
    import re
    # Convert sentence "I have fever and headache" to list of possible symptoms
    # Or rely on ml2 logic which splits by comma. Since voice inputs natural english, "I have fever and headache"
    # To handle natural text: we should find any matching symptoms in the whole string.
    
    # Collect all possible symptom triggers from translations and english names
    user_symptoms = []
    text_lower = text.lower()
    
    # Exact reverse match logic using `ml2.py`'s vocabulary
    mlb_classes = list(mlb.classes_) if mlb else []
    
    # Simple mapping for common voice dictations that don't match exact labels
    voice_map = {
        "fever": "fever (>101°f)",
        "vomit": "nausea/vomiting",
        "nausea": "nausea/vomiting",
        "muscle pain": "joint/muscle pain",
        "joint pain": "joint/muscle pain",
        "headache": "severe headache",
        "sir dard": "severe headache",
        "sirdard": "severe headache",

        # Romanized Marathi/Hindi
        "sardi": "continuous sneezing",
        "khokla": "cough",
        "tap": "high fever",
        "bukhar": "high fever",
        "ghamoriya": "skin rash",
    }
    
    devanagari_map = {
        "सर्दी": "continuous sneezing",
        "खोखला": "cough",
        "ताप": "high fever",
        "बुखार": "high fever",
        "घमोरिया": "skin rash",
        "तेज बुखार": "high fever",
        "सिरदर्द": "severe headache",
        "सिर दर्द": "severe headache",
        "सर दर्द": "severe headache",
        "डोकेदुखी": "severe headache",
        "पेट दर्द": "abdominal pain",
        "पेटातील दुखणे": "abdominal pain",
    }

    for k, v in voice_map.items():
        if k in text_lower:
            user_symptoms.append(v)

    for k, v in devanagari_map.items():
        if k in text:
            user_symptoms.append(v)
            
    for symptom in mlb_classes:
        if symptom in text_lower:
            user_symptoms.append(symptom)
            
    # Also check hindi/marathi matched (without lowercasing Devanagari)
    if translations:
        for eng_symptom, trans in translations.get("symptoms", {}).items():
            hi_name = trans.get("hi", "").strip()
            mr_name = trans.get("mr", "").strip()
            # Check for exact match (Devanagari script is case-sensitive)
            if hi_name and hi_name in text:
                user_symptoms.append(eng_symptom)
            if mr_name and mr_name in text:
                user_symptoms.append(eng_symptom)
            
    # Deduplicate
    valid_symptoms = list(set(user_symptoms))
    
    if not valid_symptoms:
        raise HTTPException(status_code=400, detail=f"No recognizable symptoms found in: '{text}'. Please try to be more specific (e.g., 'fever', 'headache').")
        
    # Intercept single common symptoms to avoid alarming ML predictions
    if len(valid_symptoms) == 1:
        single = valid_symptoms[0]
        common_cases = {
            "severe headache": "Common Headache",
            "headache": "Common Headache",
            "high fever": "Common Viral Fever",
            "fever (>101°f)": "Common Viral Fever",
            "mild fever": "Common Viral Fever",
            "nausea/vomiting": "Common Indigestion or Acid Reflux",
            "joint/muscle pain": "Common Muscle Strain or Fatigue",
            "fatigue": "General Fatigue / Stress",
            "cough": "Common Cold / Seasonal Allergies"
        }
        if single in common_cases:
            disease_name = common_cases[single]
            # Get Gemini advice for the common case too
            advice, advice_available, advice_error = get_gemini_advice([single], [disease_name])
            return {
                "detected_language": lang,
                "matched_symptoms": valid_symptoms,
                "predictions": [{"disease": disease_name}],
                "advice": advice,
                "advice_available": advice_available,
                "advice_error": advice_error
            }

    common_case_predictions = get_common_case_prediction(valid_symptoms)
    if common_case_predictions:
        advice, advice_available, advice_error = get_gemini_advice(valid_symptoms, common_case_predictions)
        return {
            "detected_language": lang,
            "matched_symptoms": valid_symptoms,
            "predictions": [{"disease": disease_name} for disease_name in common_case_predictions[:3]],
            "advice": advice,
            "advice_available": advice_available,
            "advice_error": advice_error
        }
        
    predictions = ml2.predict_disease(valid_symptoms, model, mlb, label_encoder)
    
    # Format top 3 results
    top_results = []
    disease_names = []
    items = list(predictions.items())
    for i, (disease, prob) in enumerate(items):
        if i >= 3:
            break
        display_name = ml2.translate_disease_to_lang(disease, translations, lang)
        top_results.append({"disease": display_name})
        disease_names.append(display_name)
    
    # Get Gemini advice
    advice, advice_available, advice_error = get_gemini_advice(valid_symptoms, disease_names)
        
    return {
        "detected_language": lang,
        "matched_symptoms": valid_symptoms,
        "predictions": top_results,
        "advice": advice,
        "advice_available": advice_available,
        "advice_error": advice_error
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
