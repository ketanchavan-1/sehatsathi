from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
import uvicorn  # type: ignore
import ml2  # type: ignore
import os
import joblib  # type: ignore
import requests  # type: ignore
import sqlite3
import secrets
import hashlib
import json
from typing import Optional
import re

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
DB_PATH = os.getenv("DB_PATH", "sehatsathi.db")

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


class RegisterRequest(BaseModel):
    username: str
    name: str
    email: str
    password: str
    age: Optional[str] = ""
    gender: Optional[str] = ""
    height_cm: Optional[str] = ""
    weight_kg: Optional[str] = ""
    health_goal: Optional[str] = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class DietPlanRequest(BaseModel):
    token: str
    diet_type: str
    goal: str
    calories: str
    meals: str
    plan: list[dict]


class FoodImageRequest(BaseModel):
    image_data: str


class SurveyResponseRequest(BaseModel):
    token: str
    responses: dict

# Global state for ML
model = None
mlb = None
label_encoder = None
translations = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            age TEXT DEFAULT '',
            gender TEXT DEFAULT '',
            height_cm TEXT DEFAULT '',
            weight_kg TEXT DEFAULT '',
            health_goal TEXT DEFAULT '',
            session_token TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diet_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            diet_type TEXT NOT NULL,
            goal TEXT NOT NULL,
            calories TEXT NOT NULL,
            meals TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            responses_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    # Support legacy plaintext passwords from earlier local/dev iterations.
    if "$" not in stored_hash:
        return password == stored_hash
    try:
        salt, expected = stored_hash.split("$", 1)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return digest.hex() == expected
    except ValueError:
        return False


def get_user_by_token(token: str):
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE session_token = ?",
        (token,),
    ).fetchone()
    conn.close()
    return user

@app.on_event("startup")
def load_ml_components():
    global model, mlb, label_encoder, translations
    init_db()
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


@app.post("/auth/register")
def register_user(req: RegisterRequest):
    if not req.email.strip() or not req.password.strip() or not req.name.strip() or not req.username.strip():
        raise HTTPException(status_code=400, detail="Name, username, email, and password are required.")

    token = secrets.token_urlsafe(32)
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO users (username, name, email, password_hash, age, gender, height_cm, weight_kg, health_goal, session_token)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.username.strip(),
                req.name.strip(),
                req.email.strip().lower(),
                hash_password(req.password),
                req.age or "",
                req.gender or "",
                req.height_cm or "",
                req.weight_kg or "",
                req.health_goal or "",
                token,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Email or username already exists.")

    user = conn.execute(
        "SELECT id, username, name, email, age, gender, height_cm, weight_kg, health_goal FROM users WHERE email = ?",
        (req.email.strip().lower(),),
    ).fetchone()
    conn.close()
    return {"message": "Profile created successfully.", "token": token, "user": dict(user)}


@app.post("/auth/login")
def login_user(req: LoginRequest):
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (req.email.strip().lower(),),
    ).fetchone()

    if not user or not verify_password(req.password, user["password_hash"]):
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = secrets.token_urlsafe(32)
    if "$" not in user["password_hash"]:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(req.password), user["id"]),
        )
    conn.execute("UPDATE users SET session_token = ? WHERE id = ?", (token, user["id"]))
    conn.commit()
    updated = conn.execute(
        "SELECT id, username, name, email, age, gender, height_cm, weight_kg, health_goal FROM users WHERE id = ?",
        (user["id"],),
    ).fetchone()
    conn.close()
    return {"token": token, "user": dict(updated)}


@app.post("/auth/logout")
def logout_user(token: str):
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")

    conn = get_db_connection()
    conn.execute("UPDATE users SET session_token = '' WHERE id = ?", (user["id"],))
    conn.commit()
    conn.close()
    return {"message": "Logged out successfully."}


@app.get("/profile")
def get_profile(token: str):
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")
    return {
        "user": {
            "id": user["id"],
            "username": user["username"],
            "name": user["name"],
            "email": user["email"],
            "age": user["age"],
            "gender": user["gender"],
            "height_cm": user["height_cm"],
            "weight_kg": user["weight_kg"],
            "health_goal": user["health_goal"],
        }
    }


@app.post("/diet-plans")
def save_diet_plan(req: DietPlanRequest):
    user = get_user_by_token(req.token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")

    conn = get_db_connection()
    cursor = conn.execute(
        """
        INSERT INTO diet_plans (user_id, diet_type, goal, calories, meals, plan_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user["id"],
            req.diet_type,
            req.goal,
            req.calories,
            req.meals,
            json.dumps(req.plan),
        ),
    )
    conn.commit()
    saved_plan = conn.execute(
        """
        SELECT id, diet_type, goal, calories, meals, plan_json, created_at
        FROM diet_plans
        WHERE id = ?
        """,
        (cursor.lastrowid,),
    ).fetchone()
    conn.close()
    return {
        "message": "Diet plan saved successfully.",
        "plan": {
            "id": saved_plan["id"],
            "diet_type": saved_plan["diet_type"],
            "goal": saved_plan["goal"],
            "calories": saved_plan["calories"],
            "meals": saved_plan["meals"],
            "plan": json.loads(saved_plan["plan_json"]),
            "created_at": saved_plan["created_at"],
        },
    }


@app.get("/diet-plans")
def get_saved_diet_plans(token: str):
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")

    conn = get_db_connection()
    plans = conn.execute(
        """
        SELECT id, diet_type, goal, calories, meals, plan_json, created_at
        FROM diet_plans
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user["id"],),
    ).fetchall()
    conn.close()

    return {
        "plans": [
            {
                "id": plan["id"],
                "diet_type": plan["diet_type"],
                "goal": plan["goal"],
                "calories": plan["calories"],
                "meals": plan["meals"],
                "plan": json.loads(plan["plan_json"]),
                "created_at": plan["created_at"],
            }
            for plan in plans
        ]
    }


@app.post("/survey-responses")
def save_survey_response(req: SurveyResponseRequest):
    user = get_user_by_token(req.token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")

    conn = get_db_connection()
    cursor = conn.execute(
        """
        INSERT INTO survey_responses (user_id, responses_json)
        VALUES (?, ?)
        """,
        (
            user["id"],
            json.dumps(req.responses),
        ),
    )
    conn.commit()
    saved_response = conn.execute(
        """
        SELECT id, responses_json, created_at
        FROM survey_responses
        WHERE id = ?
        """,
        (cursor.lastrowid,),
    ).fetchone()
    conn.close()
    return {
        "message": "Survey response saved successfully.",
        "response": {
            "id": saved_response["id"],
            "responses": json.loads(saved_response["responses_json"]),
            "created_at": saved_response["created_at"],
        },
    }


@app.get("/survey-responses")
def get_survey_responses(token: str):
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")

    conn = get_db_connection()
    records = conn.execute(
        """
        SELECT id, responses_json, created_at
        FROM survey_responses
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user["id"],),
    ).fetchall()
    conn.close()

    return {
        "responses": [
            {
                "id": record["id"],
                "responses": json.loads(record["responses_json"]),
                "created_at": record["created_at"],
            }
            for record in records
        ]
    }

def build_fallback_advice(symptoms, predicted_diseases):
    """Provide basic non-LLM guidance when the API is unavailable."""
    symptom_text = ", ".join(symptoms[:4]) if symptoms else "your symptoms"
    disease_text = predicted_diseases[0] if predicted_diseases else "a common illness"
    return (
        f"Your reported symptoms ({symptom_text}) may be associated with {disease_text}. "
        "Rest, stay hydrated, eat light food, and monitor whether symptoms are improving or getting worse.\n\n"
        "Seek medical help sooner if you have trouble breathing, severe chest pain, persistent vomiting, dehydration, "
        "very high fever, confusion, or symptoms that keep worsening.\n\n"
        "Please consult a qualified doctor for proper diagnosis and treatment."
    )


def extract_json_object(text: str):
    """Extract a JSON object from a model response that may include extra text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def fallback_food_analysis():
    return {
        "summary": "We could not confidently analyze this food photo right now.",
        "foods": [],
        "total_calories": None,
        "micronutrients": [],
        "limitations": "Try a clearer top-down photo with good lighting and the full plate visible."
    }


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

    fever_variants = {"fever (>101Â°f)", "fever (>101°f)"}
    common_symptom_pool = {
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
        "nausea",
        "vomiting",
        "abdominal pain",
        "stomach pain",
        "diarrhoea",
        "itching",
        "skin rash",
    }
    common_symptom_pool.update(fever_variants)
    if len(symptom_set) > 4 or not symptom_set.issubset(common_symptom_pool):
        return None

    if {"continuous sneezing", "cough"}.issubset(symptom_set) or {"runny nose", "congestion"}.issubset(symptom_set):
        return ["Common Cold / Seasonal Allergies", "Common Viral Fever"]
    if {"high fever", "severe headache"}.issubset(symptom_set) or any({variant, "severe headache"}.issubset(symptom_set) for variant in fever_variants):
        return ["Common Viral Fever", "Seasonal Viral Infection"]
    if {"high fever", "cough"}.issubset(symptom_set) or any({variant, "cough"}.issubset(symptom_set) for variant in fever_variants):
        return ["Common Viral Fever", "Common Cold / Seasonal Allergies"]
    if {"itching", "skin rash"}.issubset(symptom_set):
        return ["Common Skin Allergy", "Fungal infection"]
    if {"vomiting", "diarrhoea"}.issubset(symptom_set) and ("abdominal pain" in symptom_set or "stomach pain" in symptom_set):
        return ["Food Poisoning", "Gastroenteritis"]
    if {"vomiting", "diarrhoea"}.issubset(symptom_set):
        return ["Gastroenteritis", "Food Poisoning"]
    if {"nausea/vomiting", "abdominal pain"}.issubset(symptom_set) or {"nausea/vomiting", "stomach pain"}.issubset(symptom_set):
        return ["Common Indigestion or Acid Reflux", "Food Poisoning"]
    if {"vomiting", "abdominal pain"}.issubset(symptom_set) or {"vomiting", "stomach pain"}.issubset(symptom_set):
        return ["Food Poisoning", "Gastroenteritis"]
    if {"fatigue", "severe headache"}.issubset(symptom_set):
        return ["General Fatigue / Stress", "Common Headache"]
    if "cough" in symptom_set and "severe headache" in symptom_set:
        return ["Common Viral Fever", "Common Cold / Seasonal Allergies"]
    if "high fever" in symptom_set or any(variant in symptom_set for variant in fever_variants):
        return ["Common Viral Fever"]
    if "cough" in symptom_set:
        return ["Common Cold / Seasonal Allergies"]
    if "severe headache" in symptom_set or "headache" in symptom_set:
        return ["Common Headache"]
    return None


def reorder_predictions_for_common_symptoms(symptoms, predictions):
    """Demote rare/severe diseases for very common low-risk symptom clusters."""
    symptom_set = set(symptoms)
    fever_variants = {"fever (>101Â°f)", "fever (>101°f)"}
    common_symptom_pool = {
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
    common_symptom_pool.update(fever_variants)
    red_flag_symptoms = {
        "chest pain",
        "breathlessness",
        "blood in sputum",
        "weight loss",
        "yellowing of eyes",
        "yellowish skin",
        "weakness of one body side",
        "slurred speech",
    }
    if len(symptom_set) > 4 or not symptom_set.issubset(common_symptom_pool) or symptom_set & red_flag_symptoms:
        return predictions

    deprioritize = {
        "AIDS",
        "hepatitis A",
        "Hepatitis B",
        "Hepatitis C",
        "Hepatitis D",
        "Hepatitis E",
        "Paralysis (brain hemorrhage)",
        "Heart attack",
        "Tuberculosis",
    }
    boosted = {
        "Common Cold",
        "Common Viral Fever",
        "Viral Fever",
        "Sinusitis",
        "Migraine",
        "Gastroenteritis",
        "Food Poisoning",
        "Fungal infection",
        "Allergy",
        "Bronchial Asthma",
    }

    adjusted = {}
    for disease, prob in predictions.items():
        score = float(prob)
        if disease in deprioritize:
            score *= 0.05
        if disease in boosted:
            score *= 1.5
        adjusted[disease] = score

    return dict(sorted(adjusted.items(), key=lambda item: item[1], reverse=True))


def build_prediction_candidate_pool(symptoms, ml_candidates=None):
    """Build a broader but still controlled candidate list for final prediction refinement."""
    symptom_set = set(symptoms)
    candidate_pool = []
    red_flag_symptoms = {
        "chest pain",
        "breathlessness",
        "blood in sputum",
        "weight loss",
        "yellowing of eyes",
        "yellowish skin",
        "weakness of one body side",
        "slurred speech",
    }
    has_red_flags = bool(symptom_set & red_flag_symptoms)

    common_case_predictions = get_common_case_prediction(symptoms) or []
    for disease in common_case_predictions:
        if disease not in candidate_pool:
            candidate_pool.append(disease)

    if {"continuous sneezing", "cough"}.issubset(symptom_set) or {"runny nose", "congestion"}.issubset(symptom_set):
        for disease in ["Common Cold / Seasonal Allergies", "Common Viral Fever", "Sinusitis", "Seasonal Viral Infection"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if {"high fever", "severe headache"}.issubset(symptom_set) or {"high fever", "cough"}.issubset(symptom_set):
        for disease in ["Common Viral Fever", "Seasonal Viral Infection", "Viral Fever", "Common Cold / Seasonal Allergies"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if {"vomiting", "diarrhoea"}.issubset(symptom_set) and ("abdominal pain" in symptom_set or "stomach pain" in symptom_set):
        for disease in ["Food Poisoning", "Gastroenteritis", "Common Indigestion or Acid Reflux"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)
    elif {"vomiting", "diarrhoea"}.issubset(symptom_set):
        for disease in ["Gastroenteritis", "Food Poisoning"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)
    elif {"vomiting", "abdominal pain"}.issubset(symptom_set) or {"vomiting", "stomach pain"}.issubset(symptom_set):
        for disease in ["Food Poisoning", "Common Indigestion or Acid Reflux", "Gastroenteritis"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if {"itching", "skin rash"}.issubset(symptom_set):
        for disease in ["Common Skin Allergy", "Fungal infection"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if "severe headache" in symptom_set or "headache" in symptom_set:
        for disease in ["Common Headache", "Migraine", "General Fatigue / Stress"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if {"yellowing of eyes", "yellowish skin"}.issubset(symptom_set):
        for disease in ["Hepatitis E", "Hepatitis A", "Hepatitis B", "Jaundice"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if {"blood in sputum", "weight loss"}.issubset(symptom_set) or {"blood in sputum", "breathlessness"}.issubset(symptom_set):
        for disease in ["Tuberculosis", "Pneumonia"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if {"chest pain", "breathlessness"}.issubset(symptom_set):
        for disease in ["Heart attack", "Pneumonia", "Bronchial Asthma"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    if {"weakness of one body side", "slurred speech"}.issubset(symptom_set):
        for disease in ["Paralysis (brain hemorrhage)"]:
            if disease not in candidate_pool:
                candidate_pool.append(disease)

    for disease in ml_candidates or []:
        if disease not in candidate_pool:
            candidate_pool.append(disease)

    if has_red_flags:
        serious_first = []
        for disease in ml_candidates or []:
            if disease not in serious_first:
                serious_first.append(disease)
        for disease in candidate_pool:
            if disease not in serious_first:
                serious_first.append(disease)
        return serious_first[:8]

    return candidate_pool[:8]

def get_groq_advice(symptoms, predicted_diseases):
    """Use Groq to generate a brief explanation and health advice."""
    fallback = build_fallback_advice(symptoms, predicted_diseases)
    if not GROQ_API_KEY:
        print("Groq is disabled: missing API configuration.")
        return fallback, False, "Groq is not configured."

    try:
        prompt = f"""You are a helpful medical AI assistant for an Indian healthcare platform called Sehat Sathi.

A patient reported these symptoms: {', '.join(symptoms)}
Our ML model predicted these possible conditions: {', '.join(predicted_diseases)}

Please provide:
1. A brief explanation (2-3 sentences) of why these symptoms might indicate the predicted condition(s).
2. Practical health advice (3-4 bullet points) including home remedies and when to see a doctor.

Keep the response concise, friendly, and easy to understand. Do NOT use markdown formatting. Use plain text only. Always end with a reminder to consult a qualified doctor."""
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You provide short, safe, plain-text health explanations."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        advice_text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if advice_text.strip():
            return advice_text, True, ""
        return fallback, False, "Groq returned an empty response."
    except requests.HTTPError as e:
        error_message = str(e)
        print(f"Groq API error: {error_message}")
        if getattr(e.response, "status_code", None) == 401:
            return fallback, False, "Groq API key is invalid."
        if getattr(e.response, "status_code", None) == 429:
            return fallback, False, "Groq rate limit exceeded for the current API key."
        return fallback, False, error_message
    except Exception as e:
        print(f"Groq API error: {e}")
        return fallback, False, str(e)


def get_groq_food_analysis(image_data: str):
    """Analyze a food image and estimate foods, calories, and micronutrients."""
    fallback = fallback_food_analysis()
    if not GROQ_API_KEY:
        print("Groq is disabled: missing API configuration.")
        return fallback, False, "Groq is not configured."

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_VISION_MODEL,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a nutrition image analyst. Estimate visible foods from a meal photo and return strict JSON. "
                            "Be careful with uncertainty."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this food image. Return JSON with keys: summary (string), foods (array of objects with "
                                    "name, portion, estimated_calories, micronutrients where micronutrients is an array of strings with estimated amounts like "
                                    "'Protein ~ 31 g' or 'Vitamin C ~ 18 mg'), total_calories (number or null), "
                                    "micronutrients (array of strings with total estimated amounts for the whole meal), limitations (string). "
                                    "If unsure, still provide rough approximate amounts and say they are estimates."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            },
                        ],
                    },
                ],
                "max_completion_tokens": 700,
            },
            timeout=35,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = extract_json_object(content)
        if not parsed:
            return fallback, False, "Groq returned an unreadable food-analysis response."

        return {
            "summary": parsed.get("summary") or fallback["summary"],
            "foods": parsed.get("foods") or [],
            "total_calories": parsed.get("total_calories"),
            "micronutrients": parsed.get("micronutrients") or [],
            "limitations": parsed.get("limitations") or fallback["limitations"],
        }, True, ""
    except requests.HTTPError as e:
        error_message = str(e)
        print(f"Groq food analysis error: {error_message}")
        if getattr(e.response, "status_code", None) == 401:
            return fallback, False, "Groq API key is invalid."
        if getattr(e.response, "status_code", None) == 429:
            return fallback, False, "Groq rate limit exceeded for the current API key."
        return fallback, False, error_message
    except Exception as e:
        print(f"Groq food analysis error: {e}")
        return fallback, False, str(e)


def get_groq_refined_predictions(symptoms, candidate_diseases):
    """Use Groq to rerank or replace weak candidates from a controlled candidate pool."""
    if not GROQ_API_KEY or not candidate_diseases:
        return candidate_diseases, False

    fallback = list(candidate_diseases)
    try:
        red_flag_symptoms = {
            "chest pain",
            "breathlessness",
            "blood in sputum",
            "weight loss",
            "yellowing of eyes",
            "yellowish skin",
            "weakness of one body side",
            "slurred speech",
        }
        has_red_flags = bool(set(symptoms) & red_flag_symptoms)
        prompt = (
            "You are reviewing disease-prediction candidates for a healthcare app.\n"
            f"Reported symptoms: {', '.join(symptoms)}\n"
            f"Candidate diseases from the ML system: {', '.join(candidate_diseases)}\n\n"
            "Return strict JSON with one key: predictions (an array of up to 3 disease names chosen only from the candidate list).\n"
            "If the top ML candidates are a poor fit, replace them with better-fitting diseases from the candidate list.\n"
            "Prefer common benign conditions for common symptoms, and avoid severe or rare diseases unless the symptom pattern strongly supports them.\n"
            f"Red-flag symptoms present: {'yes' if has_red_flags else 'no'}.\n"
            "If red-flag symptoms are present, do not force common benign diseases above serious fitting conditions."
        )
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You rerank disease candidates and return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": 250,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = extract_json_object(content) or {}
        reranked = parsed.get("predictions") or []
        cleaned = [item for item in reranked if item in candidate_diseases]
        if cleaned:
            remaining = [item for item in candidate_diseases if item not in cleaned]
            return cleaned + remaining, True
        return fallback, False
    except Exception as e:
        print(f"Groq prediction refinement error: {e}")
        return fallback, False


@app.post("/analyze-food-image")
def analyze_food_image(req: FoodImageRequest):
    if not req.image_data or not req.image_data.strip():
        raise HTTPException(status_code=400, detail="No image provided.")

    image_data = req.image_data.strip()
    if not image_data.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="Image must be a valid data URL.")

    analysis, available, error = get_groq_food_analysis(image_data)
    return {
        "analysis_available": available,
        "analysis_error": error,
        **analysis,
    }

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
    text_lower_compact = re.sub(r"\s+", "", text_lower)
    text_compact = re.sub(r"\s+", "", text)
    
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
        "ulti": "vomiting",
        "julab": "diarrhoea",
        "pot dukhi": "abdominal pain",
        "potdukhi": "abdominal pain",
        "pot dard": "abdominal pain",
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

    devanagari_map["पोटदुखी"] = "abdominal pain"
    devanagari_map["पोट दुखी"] = "abdominal pain"
    devanagari_map["उलटी"] = "vomiting"
    devanagari_map["जुलाब"] = "diarrhoea"

    for k, v in voice_map.items():
        if k in text_lower:
            user_symptoms.append(v)

    for k, v in devanagari_map.items():
        if k in text:
            user_symptoms.append(v)

    # Phrase-level guard for very common cold/viral patterns from voice input
    common_phrase_groups = [
        {
            "patterns": ["sardibukharsirdard", "sardibukharsirdard", "sardibukharsirdard"],
            "symptoms": ["continuous sneezing", "high fever", "severe headache"],
        },
        {
            "patterns": ["sardibukhar", "bukharsirdard", "sardisirdard"],
            "symptoms": ["continuous sneezing", "high fever", "severe headache"],
        },
        {
            "patterns": ["sardikhoklatap", "sardikhoklabukhar"],
            "symptoms": ["continuous sneezing", "cough", "high fever"],
        },
        {
            "patterns": ["सर्दीबुखारसिरदर्द", "सर्दीबुखारसरदर्द", "सर्दीतापसिरदर्द"],
            "symptoms": ["continuous sneezing", "high fever", "severe headache"],
        },
        {
            "patterns": ["सर्दीखोकलाताप", "सर्दीखोकलाबुखार"],
            "symptoms": ["continuous sneezing", "cough", "high fever"],
        },
    ]
    for group in common_phrase_groups:
        if any(pattern in text_lower_compact or pattern in text_compact for pattern in group["patterns"]):
            user_symptoms.extend(group["symptoms"])
            
    for symptom in mlb_classes:
        if symptom in text_lower:
            user_symptoms.append(symptom.strip())
            
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
    valid_symptoms = sorted({symptom.strip() for symptom in user_symptoms if symptom and symptom.strip()})
    
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
            # Get Groq advice for the common case too
            advice, advice_available, advice_error = get_groq_advice([single], [disease_name])
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
        common_candidate_pool = build_prediction_candidate_pool(valid_symptoms, common_case_predictions[:3])
        refined_common_predictions, _ = get_groq_refined_predictions(
            valid_symptoms,
            common_candidate_pool
        )
        final_common_predictions = refined_common_predictions[:3] or common_case_predictions[:3]
        advice, advice_available, advice_error = get_groq_advice(valid_symptoms, final_common_predictions)
        return {
            "detected_language": lang,
            "matched_symptoms": valid_symptoms,
            "predictions": [{"disease": disease_name} for disease_name in final_common_predictions],
            "advice": advice,
            "advice_available": advice_available,
            "advice_error": advice_error
        }
        
    predictions = ml2.predict_disease(valid_symptoms, model, mlb, label_encoder)
    predictions = reorder_predictions_for_common_symptoms(valid_symptoms, predictions)

    ml_candidates = [disease for disease, _ in list(predictions.items())[:6]]
    candidate_diseases = build_prediction_candidate_pool(valid_symptoms, ml_candidates)
    refined_candidate_diseases, _ = get_groq_refined_predictions(valid_symptoms, candidate_diseases)
    final_candidate_diseases = refined_candidate_diseases[:3] or candidate_diseases[:3]
    
    # Format top 3 results
    top_results = []
    disease_names = []
    for disease in final_candidate_diseases:
        display_name = ml2.translate_disease_to_lang(disease, translations, lang)
        top_results.append({"disease": display_name})
        disease_names.append(display_name)
    
    # Get Groq advice
    advice, advice_available, advice_error = get_groq_advice(valid_symptoms, disease_names)
        
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
