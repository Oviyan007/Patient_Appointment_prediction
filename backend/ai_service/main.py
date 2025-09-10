# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import os

# # Load the trained Random Forest model
# model_path = os.path.join("models", "noshow_model.joblib")
# rf = joblib.load(model_path)

# app = FastAPI()

# class AppointmentFeatures(BaseModel):
#     no_show_history: int
#     attended_history: int
#     day_of_week: int
#     hour: int
#     lead_time_days: int

# @app.post("/predict/no_show")
# def predict_risk(features: AppointmentFeatures):
#     # Prepare input for model
#     input_data = np.array([[
#         features.no_show_history,
#         features.attended_history,
#         features.day_of_week,
#         features.hour,
#         features.lead_time_days
#     ]])

#     # Get risk prediction (probability of no-show)
#     risk_score = float(rf.predict_proba(input_data)[0][1])
#      # -----------------------------
#     # Categorize risk level
#     if risk_score < 0.3:
#         risk_level = "Low"
#     elif risk_score < 0.7:
#         risk_level = "Medium"
#     else:
#         risk_level = "High"

#     # -----------------------------

#     # -----------------------------
#     # Build explanations
#     explanations = {}

#     if features.no_show_history > 2:
#         explanations["no_show_history"] = f"High past no-shows ({features.no_show_history})"
#     elif features.no_show_history > 0:
#         explanations["no_show_history"] = f"Some past no-shows ({features.no_show_history})"
#     else:
#         explanations["no_show_history"] = "No previous no-shows"

#     if features.attended_history < 3:
#         explanations["attended_history"] = f"Low attendance history ({features.attended_history})"
#     else:
#         explanations["attended_history"] = f"Good attendance history ({features.attended_history})"

#     if features.hour >= 15:
#         explanations["hour"] = f"Late afternoon slot ({features.hour}:00)"
#     else:
#         explanations["hour"] = f"Earlier slot ({features.hour}:00)"

#     if features.lead_time_days < 2:
#         explanations["lead_time_days"] = f"Short booking notice ({features.lead_time_days} days)"
#     else:
#         explanations["lead_time_days"] = f"Booked in advance ({features.lead_time_days} days)"

#     explanations["day_of_week"] = f"Day of week = {features.day_of_week} (little impact)"

#     # -----------------------------
#     return {
#         "risk": risk_score,
#         "risk_level": risk_level,
#         "factors": explanations
#     }

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware


# Load No-show model
rf = joblib.load("models/noshow_model.joblib")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Data Models
# ------------------------------
class AppointmentFeatures(BaseModel):
    no_show_history: int
    attended_history: int
    day_of_week: int
    hour: int
    lead_time_days: int

class Appointment(BaseModel):
    id: int
    patient_name: str
    doctor_id: int
    room_id: int
    duration: int  # in minutes

class Patient(BaseModel):
    id: int
    name: str
    phone: str
    risk: float = 0.0

# ------------------------------
# 1️⃣ No-show prediction
# ------------------------------
@app.post("/predict/no_show")
def predict_no_show(features: AppointmentFeatures):
    input_data = np.array([[
        features.no_show_history,
        features.attended_history,
        features.day_of_week,
        features.hour,
        features.lead_time_days
    ]])
    risk_score = float(rf.predict_proba(input_data)[0][1])

    # Risk category
    if risk_score < 0.3:
        risk_level = "Low"
    elif risk_score < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Explanations
    explanations = {}
    explanations["no_show_history"] = f"{features.no_show_history} past no-shows"
    explanations["attended_history"] = f"{features.attended_history} attended"
    explanations["hour"] = f"{features.hour}:00 appointment"
    explanations["lead_time_days"] = f"{features.lead_time_days} days lead time"
    explanations["day_of_week"] = f"Day {features.day_of_week}"

    return {"risk": risk_score, "risk_level": risk_level, "factors": explanations}

# ------------------------------
# 2️⃣ Resource Allocation (demo)
# ------------------------------
@app.post("/optimize/schedule")
def optimize_schedule(appointments: List[Appointment], doctors: List[Dict], rooms: List[Dict]):
    # Simplified demo: assign first available doctor/room
    schedule = []
    for i, appt in enumerate(appointments):
        doctor = doctors[i % len(doctors)]
        room = rooms[i % len(rooms)]
        schedule.append({
            "appointment_id": appt.id,
            "patient_name": appt.patient_name,
            "doctor": doctor["name"],
            "room": room["name"]
        })
    return {"schedule": schedule}

# ------------------------------
# 3️⃣ Wait Time Prediction (demo)
# ------------------------------
@app.post("/predict/wait_time")
def predict_wait_time(appointments: List[Appointment]):
    # Dummy logic: assume 5 min per patient ahead
    wait_times = []
    for i, appt in enumerate(appointments):
        wait_times.append({
            "appointment_id": appt.id,
            "predicted_wait_time": i * 5  # 5 minutes per patient ahead
        })
    return {"wait_times": wait_times}

# ------------------------------
# 4️⃣ Smart Reminder (demo)
# ------------------------------
@app.post("/reminder")
def generate_reminder(patients: List[Patient]):
    reminders = []
    for p in patients:
        msg = f"Hi {p.name}, your appointment is scheduled. "
        if p.risk > 0.7:
            msg += "Please confirm, as you have a high chance of missing it."
        reminders.append({"patient_id": p.id, "message": msg})
    return {"reminders": reminders}
