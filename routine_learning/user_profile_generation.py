import pandas as pd
from collections import Counter


INPUT_FILE = ("user_routine_synthetic_data.csv")
OUTPUT_FILE = "user_routine_profile.json"

def extract_hour(timestamp):
    return pd.to_datetime(timestamp).hour


def extract_time(timestamp):
    return pd.to_datetime(timestamp).strftime("%H:%M")


def most_common(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def average_hour(values):
    if not values:
        return None
    return round(sum(values) / len(values), 2)



# Load Dataset
def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# Profile Extraction
def build_user_profile(df):
    profile = {}

    # Wake-Up Profile
    wakeups = df[df["event_type"] == "WakeUp"].copy()

    weekday_wakeups = wakeups[wakeups["day_of_week"].isin([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
    ])]

    saturday_wakeups = wakeups[wakeups["day_of_week"] == "Saturday"]
    sunday_wakeups = wakeups[wakeups["day_of_week"] == "Sunday"]

    profile["wake_up_patterns"] = {
        "weekday_average_hour": average_hour(
            weekday_wakeups["timestamp"].dt.hour.tolist()
        ),
        "saturday_average_hour": average_hour(
            saturday_wakeups["timestamp"].dt.hour.tolist()
        ),
        "sunday_average_hour": average_hour(
            sunday_wakeups["timestamp"].dt.hour.tolist()
        )
    }

    # University / Occupancy Pattern
    leave_home = df[df["event_type"] == "LeaveHome"]
    return_home = df[df["event_type"] == "ReturnHome"]

    profile["weekday_occupancy_pattern"] = {
        "usual_leave_hour": average_hour(
            leave_home["timestamp"].dt.hour.tolist()
        ),
        "usual_return_hour": average_hour(
            return_home["timestamp"].dt.hour.tolist()
        )
    }

    # Heating Preferences
    heating_df = df[
        (df["heating_temp"].notna()) &
        (df["heating_temp"] != "Off")
    ].copy()

    heating_by_season = {}

    for season in heating_df["season"].unique():
        season_data = heating_df[heating_df["season"] == season]
        heating_by_season[season] = most_common(
            season_data["heating_temp"].tolist()
        )

    profile["heating_preferences"] = heating_by_season

    # Friday Family Movie Night
    friday_movies = df[
        (df["family_movie_night"] == True) |
        (df["family_movie_night"] == "True")
    ]

    profile["family_movie_night"] = {
        "usually_happens": len(friday_movies) > 0,
        "most_common_start_hour": average_hour(
            friday_movies[
                friday_movies["event_type"] == "TVOn"
            ]["timestamp"].dt.hour.tolist())
    }

    # Dog Walking Profile
    dog_walks = df[df["dog_walk"] == True]

    saturday_walks = dog_walks[
        dog_walks["day_of_week"] == "Saturday"
    ]

    sunday_walks = dog_walks[
        dog_walks["day_of_week"] == "Sunday"
    ]

    profile["dog_walking_habits"] = {
        "saturday_walk_frequency": len(saturday_walks),
        "sunday_walk_frequency": len(sunday_walks),
        "common_evening_walk_hour": average_hour(
            dog_walks[
                dog_walks["timestamp"].dt.hour >= 17
            ]["timestamp"].dt.hour.tolist())
    }

    # Grocery Shopping Profile
    groceries = df[df["grocery_trip"] == True]

    profile["grocery_shopping"] = {
        "usually_on_sunday": len(
            groceries[groceries["day_of_week"] == "Sunday"]
        ) > 0,
        "common_shopping_hour": average_hour(
            groceries["timestamp"].dt.hour.tolist())
    }

    # Dance Class Profile
    dance = df[df["dance_class"] == True]

    profile["dance_class"] = {
        "usually_on_sunday": len(dance) > 0,
        "common_start_hour": average_hour(
            dance["timestamp"].dt.hour.tolist())
    }

    return profile


# Save Profile
def save_profile(profile):
    import json

    with open(OUTPUT_FILE, "w") as f:
        json.dump(profile, f, indent=4)



if __name__ == "__main__":
    print("Loading synthetic dataset...")
    df = load_data()

    print("Building user routine profile...")
    profile = build_user_profile(df)

    save_profile(profile)

    print("\nUser profile generated successfully.")
    print(f"Saved as: {OUTPUT_FILE}\n")

    import json
    print(json.dumps(profile, indent=4))