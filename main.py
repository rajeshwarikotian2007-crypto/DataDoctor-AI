import csv
import statistics

with open("sample_data.csv", "r") as file:
    data = list(csv.DictReader(file))

print("🩺 DATADOCTOR AI")
print("--------------------")


# -----------------------------
# 1. MISSING VALUES
# -----------------------------

missing_total = 0

print("\n🔍 MISSING VALUES")

for column in data[0].keys():

    missing = 0

    for row in data:
        if row[column] == "":
            missing += 1

    missing_total += missing

    print(column, "→", missing, "missing values")


# -----------------------------
# 2. DUPLICATE ROWS
# -----------------------------

print("\n🔍 DUPLICATE ROWS")

seen = []
duplicates = 0

for row in data:

    row_data = tuple(row.values())

    if row_data in seen:
        duplicates += 1
    else:
        seen.append(row_data)

print("Duplicate rows →", duplicates)


# -----------------------------
# 3. OUTLIERS
# -----------------------------

print("\n🔍 OUTLIER CHECK")

outliers_total = 0

for column in data[0].keys():

    values = []

    for row in data:

        if row[column] != "":

            try:
                values.append(float(row[column]))
            except ValueError:
                pass

    if len(values) >= 3:

        mean = statistics.mean(values)

        standard_deviation = statistics.stdev(values)

        outliers = 0

        for value in values:

            if abs(value - mean) > 2 * standard_deviation:
                outliers += 1

        outliers_total += outliers

        print(column, "→", outliers, "possible outliers")


# -----------------------------
# 4. DATA HEALTH SCORE
# -----------------------------

total_rows = len(data)

score = 100

score -= missing_total * 5
score -= duplicates * 5
score -= outliers_total * 5

if score < 0:
    score = 0

print("\n🩺 DATA HEALTH SCORE")
print("--------------------")

print("Score:", score, "/100")

if score >= 80:
    print("🟢 Dataset looks healthy")

elif score >= 60:
    print("🟡 Dataset needs some improvement")

else:
    print("🔴 Dataset needs major cleaning")
# -----------------------------
# 5. DATASET STATISTICS
# -----------------------------

print("\n📊 DATASET STATISTICS")
print("--------------------")

for column in data[0].keys():

    values = []

    for row in data:

        if row[column] != "":

            try:
                values.append(float(row[column]))
            except ValueError:
                pass

    if len(values) > 0:

        average = statistics.mean(values)
        minimum = min(values)
        maximum = max(values)

        print("\n", column)
        print("Average:", round(average, 2))
        print("Minimum:", minimum)
        print("Maximum:", maximum)
# COLUMN TYPE ANALYSIS

print("\n🤖 COLUMN TYPE ANALYSIS")
print("--------------------")

text_columns = []
numeric_columns = []

for column in data[0].keys():

    is_numeric = True

    for row in data:

        value = row[column]

        if value == "":
            continue

        try:
            float(value)
        except ValueError:
            is_numeric = False
            break

    if is_numeric:
        numeric_columns.append(column)
    else:
        text_columns.append(column)

print("\n📝 Text Columns:")

for column in text_columns:
    print("•", column)

print("\n🔢 Numerical Columns:")

for column in numeric_columns:
    print("•", column)
# TARGET COLUMN DETECTION

print("\n🎯 TARGET COLUMN ANALYSIS")
print("--------------------")

if len(numeric_columns) > 0:

    target_column = numeric_columns[-1]

    print("Possible target column:", target_column)

    print("\nOther numerical columns can be used as features:")

    for column in numeric_columns:
        if column != target_column:
            print("•", column)

else:
    print("No numerical columns found.")
# -----------------------------
# 7. MACHINE LEARNING
# -----------------------------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("\n🤖 MACHINE LEARNING")
print("--------------------")

X = []
y = []

for row in data:
    if row["Age"] != "" and row["Marks"] != "":
        X.append([float(row["Age"])])
        y.append(float(row["Marks"]))

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate performance
score = r2_score(y_test, predictions)

print("Model trained successfully!")
print("Test R² Score:", round(score, 2))

# Make a prediction
prediction = model.predict([[23]])

print("Predicted marks for age 23:", round(prediction[0], 2))
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

print("\n🏆 MODEL COMPARISON")
print("--------------------")

best_model = None
best_score = float("-inf")
best_name = ""

for name, ml_model in models.items():

    ml_model.fit(X_train, y_train)

    predictions = ml_model.predict(X_test)

    model_score = r2_score(y_test, predictions)

    print(name, "→", round(model_score, 2))

    if model_score > best_score:
        best_score = model_score
        best_model = ml_model
        best_name = name

print("\n🏆 BEST MODEL:", best_name)
print("Score:", round(best_score, 2))