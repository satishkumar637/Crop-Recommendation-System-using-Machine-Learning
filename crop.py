# Crop Advisory System based on Soil & Weather Conditions

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Step 1: Agricultural dataset (sample knowledge)
farm_data = pd.DataFrame({
    "avg_temperature_c": [20, 25, 30, 35, 28, 22, 18, 32, 27, 23, 29, 34, 26, 21, 19],
    "avg_humidity_percent": [80, 70, 60, 50, 65, 85, 90, 55, 75, 82, 68, 52, 78, 88, 92],
    "annual_rainfall_mm": [200, 150, 100, 80, 120, 220, 250, 90, 140, 210, 130, 70, 160, 230, 260],
    "soil_ph": [6.5, 6.0, 7.0, 7.5, 6.8, 6.2, 5.8, 7.2, 6.4, 6.1, 6.9, 7.4, 6.6, 5.9, 5.7],
    "recommended_crop": [
        "rice", "wheat", "maize", "cotton", "maize",
        "rice", "rice", "cotton", "sugarcane", "barley",
        "millet", "cotton", "pulses", "rice", "rice"
    ]
})

# Step 2: Train crop decision model
features = farm_data[[
    "avg_temperature_c",
    "avg_humidity_percent",
    "annual_rainfall_mm",
    "soil_ph"
]]

target = farm_data["recommended_crop"]

crop_model = DecisionTreeClassifier()
crop_model.fit(features, target)

# Step 3: Input validation with real constraints
def get_valid_agri_input(message, min_val, max_val):
    """
    Ensures farmer enters realistic agricultural values.
    Prevents impossible conditions (e.g., negative rainfall).
    """
    while True:
        try:
            value = float(input(message))

            if value < min_val or value > max_val:
                print(f"Value should be between {min_val} and {max_val} based on real farm conditions.")
            else:
                return value

        except ValueError:
            print(" Invalid input! Please enter numeric value.")

# Step 4: Collect field conditions from user
print("\n Smart Crop Advisory System")
print("Provide your field conditions for best crop suggestion\n")

temperature = get_valid_agri_input("Average Temperature (°C) [0–50]: ", 0, 50)
humidity = get_valid_agri_input("Humidity (%) [0–100]: ", 0, 100)
rainfall = get_valid_agri_input("Annual Rainfall (mm) [0–500]: ", 0, 500)
soil_ph = get_valid_agri_input("Soil pH [0–14]: ", 0, 14)

# Step 5: Convert input into model format
field_conditions = pd.DataFrame({
    "avg_temperature_c": [temperature],
    "avg_humidity_percent": [humidity],
    "annual_rainfall_mm": [rainfall],
    "soil_ph": [soil_ph]
})

# Step 6: Crop recommendation
predicted_crop = crop_model.predict(field_conditions)[0]

# Step 7: Business logic message (real-world)
print("\n🌱 Recommended Crop:", predicted_crop)

# Practical advisory note
if predicted_crop == "rice":
    print("💧 Suitable for high rainfall and water-rich fields.")
elif predicted_crop == "wheat":
    print("🌤️ Best for moderate temperature and less water conditions.")
elif predicted_crop == "cotton":
    print("☀️ Requires warm climate and well-drained soil.")
elif predicted_crop == "sugarcane":
    print("🌧️ Needs high water availability and fertile soil.")
else:
    print("🌿 Suitable under balanced environmental conditions.")

print("\n Recommendation based on given soil & climate inputs.")
