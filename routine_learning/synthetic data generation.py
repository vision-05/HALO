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


# Adds realistic noise to time
def add_time_noise(base_hour, base_minute=0, variance_minutes=20):
    noise = random.randint(-variance_minutes, variance_minutes)
    dt = datetime(2026, 1, 1, base_hour, base_minute) + timedelta(minutes=noise)
    return dt.strftime("%H:%M")


# Occasionally skip events (10% chance)
def should_skip(probability=0.10):
    return random.random() < probability



def create_event(date, time_str, event_type, location, heating, tv_status,
                 dog_walk=False, grocery=False,
                 dance=False, movie_night=False):

    timestamp = f"{date.strftime('%Y-%m-%d')} {time_str}"
    season = get_season(date.month)
    outside_temp = generate_temperature(season)

    return {
        "timestamp": timestamp,
        "day_of_week": date.strftime("%A"),
        "is_weekend": int(date.weekday() >= 5),
        "season": season,
        "weather_temp": outside_temp,
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

    while current_date <= END_DATE:
        day_name = current_date.strftime("%A")
        season = get_season(current_date.month)
        temp = generate_temperature(season)
        heating = heating_preference(season, temp)



        if day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:

            # Wake up
            all_events.append(create_event(
                current_date,
                add_time_noise(7, 0),
                "WakeUp",
                "Home",
                heating,
                "Off"
            ))

            # Leave home for uni
            all_events.append(create_event(
                current_date,
                add_time_noise(8, 30),
                "LeaveHome",
                "Away",
                heating,
                "Off"
            ))

            # Return home
            all_events.append(create_event(
                current_date,
                add_time_noise(17, 0, 40),
                "ReturnHome",
                "Home",
                heating,
                "Off"
            ))

            # Friday family movie night
            if day_name == "Friday" and not should_skip(0.05):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(21, 0, 15),
                    "TVOn",
                    "Home",
                    heating,
                    "On",
                    movie_night=True
                ))

                all_events.append(create_event(
                    current_date,
                    add_time_noise(23, 0, 20),
                    "TVOff",
                    "Home",
                    heating,
                    "Off",
                    movie_night=True
                ))


        elif day_name == "Saturday":

            all_events.append(create_event(
                current_date,
                add_time_noise(10, 0, 30),
                "WakeUp",
                "Home",
                heating,
                "Off"
            ))

            # Morning dog walk
            if not should_skip():
                all_events.append(create_event(
                    current_date,
                    add_time_noise(10, 20, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True
                ))

            # Evening dog walk
            if not should_skip():
                all_events.append(create_event(
                    current_date,
                    add_time_noise(18, 0, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True
                ))


        elif day_name == "Sunday":

            all_events.append(create_event(
                current_date,
                add_time_noise(7, 0, 15),
                "WakeUp",
                "Home",
                heating,
                "Off"
            ))

            # Dance class
            if not should_skip(0.03):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(8, 0, 10),
                    "DanceClass",
                    "Away",
                    heating,
                    "Off",
                    dance=True
                ))

            # Dog walk after dance
            if not should_skip():
                all_events.append(create_event(
                    current_date,
                    add_time_noise(10, 30, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True
                ))

            # Grocery shopping
            if not should_skip(0.08):
                all_events.append(create_event(
                    current_date,
                    add_time_noise(16, 0, 40),
                    "GroceryShopping",
                    "Away",
                    heating,
                    "Off",
                    grocery=True
                ))

            # Evening dog walk
            if not should_skip():
                all_events.append(create_event(
                    current_date,
                    add_time_noise(18, 0, 20),
                    "DogWalk",
                    "Away",
                    heating,
                    "Off",
                    dog_walk=True
                ))

        current_date += timedelta(days=1)

    return pd.DataFrame(all_events)



if __name__ == "__main__":
    df = generate_dataset()

    df = df.sort_values("timestamp")
    df.to_csv("user_routine_synthetic_data.csv", index=False)

    print("Dataset generated successfully.")
    print(f"Total events created: {len(df)}")
    print(df.head(20))
