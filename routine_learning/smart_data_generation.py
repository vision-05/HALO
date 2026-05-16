import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def get_rainfall_probability(season):
    """Seasonal rainfall probabilities."""
    probs = {
        "Winter": 0.40,
        "Spring": 0.35,
        "Summer": 0.15,
        "Autumn": 0.45,
    }
    return probs.get(season, 0.25)


def generate_weather(season):
    """Generate weather with rain intensity."""
    rain_prob = get_rainfall_probability(season)
    is_raining = random.random() < rain_prob
    rain_intensity = 0.0

    if is_raining:
        rain_intensity = random.choice(["light", "moderate", "heavy"])
        intensity_map = {"light": 0.3, "moderate": 0.6, "heavy": 0.9}
        rain_intensity = intensity_map[rain_intensity]

    return {"is_raining": is_raining, "rain_intensity": rain_intensity}


def generate_temperature(season):
    ranges = {
        "Winter": (0, 8),
        "Spring": (8, 16),
        "Summer": (18, 30),
        "Autumn": (8, 15),
    }
    low, high = ranges[season]
    return round(random.uniform(low, high), 1)


def heating_preference(season, outside_temp):
    if season == "Summer":
        return "Off"

    if outside_temp <= 3:
        return random.choice([22, 23])
    elif outside_temp <= 8:
        return random.choice([21, 22])
    else:
        return 21


def confidence():
    return round(random.uniform(0.90, 0.99), 2)


# Adds realistic noise to time, influenced by fatigue carryover
def add_time_noise(base_hour, base_minute=0, variance_minutes=20, fatigue_minutes=0):
    """Add temporal noise with fatigue dependency.

    fatigue_minutes: carryover from previous day's late activities.
    Positive values delay wake-up times; negative values (recovery) slightly reduce delay.
    """
    noise = random.randint(-variance_minutes, variance_minutes)

    # Fatigue makes the person sleep longer on wake-up
    if base_hour < 12 and fatigue_minutes > 0:
        noise += int(fatigue_minutes * random.uniform(0.3, 0.6))

    dt = datetime(2026, 1, 1, base_hour, base_minute) + timedelta(minutes=noise)
    return dt.strftime("%H:%M")


# Occasionally skip events (10% chance)
def should_skip(probability=0.10):
    return random.random() < probability


def create_event(date, time_str, event_type, location, heating, tv_status,
                 dog_walk=False, grocery=False,
                 dance=False, movie_night=False, weather=None):
    timestamp = f"{date.strftime('%Y-%m-%d')} {time_str}"
    season = get_season(date.month)
    outside_temp = generate_temperature(season)

    if weather is None:
        weather = {"is_raining": False, "rain_intensity": 0.0}

    return {
        "timestamp": timestamp,
        "day_of_week": date.strftime("%A"),
        "is_weekend": int(date.weekday() >= 5),
        "season": season,
        "weather_temp": outside_temp,
        "is_raining": weather["is_raining"],
        "rain_intensity": weather["rain_intensity"],
        "location_status": location,
        "event_type": event_type,
        "heating_temp": heating,
        "tv_status": tv_status,
        "dog_walk": dog_walk,
        "grocery_trip": grocery,
        "dance_class": dance,
        "family_movie_night": movie_night,
        "confidence_score": confidence()
    }


def generate_dataset():
    all_events = []
    current_date = START_DATE
    fatigue_carryover = 0  # Track sleep debt across days
    previous_day_late_event = False  # Track if previous day had late activity

    while current_date <= END_DATE:
        day_name = current_date.strftime("%A")
        season = get_season(current_date.month)
        temp = generate_temperature(season)
        heating = heating_preference(season, temp)

        # Generate daily weather
        daily_weather = generate_weather(season)

        # Decay fatigue carryover (recovery effect) and apply late-event penalty
        if previous_day_late_event:
            fatigue_carryover = min(60, fatigue_carryover + random.randint(30, 60))
        else:
            fatigue_carryover = max(0, fatigue_carryover - random.randint(10, 20))

        if day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:

            # Wake up (influenced by fatigue)
            wake_time = add_time_noise(7, 0, fatigue_minutes=fatigue_carryover)
            all_events.append(create_event(
                current_date,
                wake_time,
                "WakeUp",
                "Home",
                heating,
                "Off",
                weather=daily_weather
            ))

            # Leave home for uni
            all_events.append(create_event(
                current_date,
                add_time_noise(8, 30),
                "LeaveHome",
                "Away",
                heating,
                "Off",
                weather=daily_weather
            ))

            # Return home
            all_events.append(create_event(
                current_date,
                add_time_noise(17, 0, 40),
                "ReturnHome",
                "Home",
                heating,
                "Off",
                weather=daily_weather
            ))

            # Friday family movie night
            late_movie_night = False
            if day_name == "Friday" and not should_skip(0.05):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(21, 0, 15),
                    "TVOn",
                    "Home",
                    heating,
                    "On",
                    movie_night=True,
                    weather=daily_weather
                ))

                all_events.append(create_event(
                    current_date,
                    add_time_noise(23, 0, 20),
                    "TVOff",
                    "Home",
                    heating,
                    "Off",
                    movie_night=True,
                    weather=daily_weather
                ))
                late_movie_night = True

            previous_day_late_event = late_movie_night

        elif day_name == "Saturday":

            # Saturday wake-up is later but influenced by Friday fatigue
            wake_hour = 10 if fatigue_carryover < 30 else (11 if fatigue_carryover < 60 else 12)
            all_events.append(create_event(
                current_date,
                add_time_noise(wake_hour, 0, 30, fatigue_minutes=fatigue_carryover),
                "WakeUp",
                "Home",
                heating,
                "Off",
                weather=daily_weather
            ))

            # Morning dog walk: skip if heavy rain
            rain_penalty = 0.30 if daily_weather["rain_intensity"] >= 0.6 else 0.10
            if not should_skip(rain_penalty):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(10, 20, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True,
                    weather=daily_weather
                ))

            # Evening dog walk: skip if heavy rain
            if not should_skip(rain_penalty):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(18, 0, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True,
                    weather=daily_weather
                ))

            previous_day_late_event = False

        elif day_name == "Sunday":

            # Sunday wake-up is early, but fatigue can shift it
            all_events.append(create_event(
                current_date,
                add_time_noise(7, 0, 15, fatigue_minutes=fatigue_carryover),
                "WakeUp",
                "Home",
                heating,
                "Off",
                weather=daily_weather
            ))

            # Dance class: skip if heavy rain or high fatigue
            dance_skip_prob = 0.03 + (0.20 if daily_weather["rain_intensity"] >= 0.6 else 0.0)
            if not should_skip(dance_skip_prob):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(8, 0, 10),
                    "DanceClass",
                    "Away",
                    heating,
                    "Off",
                    dance=True,
                    weather=daily_weather
                ))

            # Dog walk after dance: skip if heavy rain
            rain_penalty = 0.30 if daily_weather["rain_intensity"] >= 0.6 else 0.10
            if not should_skip(rain_penalty):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(10, 30, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True,
                    weather=daily_weather
                ))

            # Grocery shopping: delay or skip if heavy rain
            grocery_skip_prob = 0.08 + (0.25 if daily_weather["rain_intensity"] >= 0.6 else 0.0)
            if not should_skip(grocery_skip_prob):
                # Delay grocery trip if moderate rain
                hours_offset = random.randint(0, 2) if daily_weather["rain_intensity"] < 0.6 else random.randint(1, 3)
                shopping_hour = 16 + hours_offset
                all_events.append(create_event(
                    current_date,
                    add_time_noise(shopping_hour, 0, 40),
                    "GroceryShopping",
                    "Away",
                    heating,
                    "Off",
                    grocery=True,
                    weather=daily_weather
                ))

            # Evening dog walk: skip if heavy rain
            if not should_skip(rain_penalty):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(18, 0, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True,
                    weather=daily_weather
                ))

            previous_day_late_event = False

        current_date += timedelta(days=1)

    return pd.DataFrame(all_events)


if __name__ == "__main__":
    df = generate_dataset()

    df = df.sort_values("timestamp")
    df.to_csv("user_routine_synthetic_data.csv", index=False)

    print("Dataset generated successfully.")
    print(f"Total events created: {len(df)}")
    print(df.head(20))
