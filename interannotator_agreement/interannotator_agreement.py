import pandas as pd
from sklearn.metrics import cohen_kappa_score
import krippendorff

# Load the transformed CSV files
ratings1 = pd.read_csv("interannotator_agreement/transformed_a1.csv")
ratings2 = pd.read_csv("interannotator_agreement/transformed_a2.csv")

# Define the categories to compare
categories = [
    "Vanishing Subject",
    "Subject Multiplication/Reduction",
    "Temporal Subject Dysmorphia",
    "Omission Error",
    "Physical Incongruity"
]

# Initialize dictionaries to store Cohen's Kappa and Krippendorff's Alpha for each category
kappa_scores = {}
alpha_scores = {}

# Calculate Cohen's Kappa and Krippendorff's Alpha for each category
for category in categories:
    # Extract binary labels for the current category from both raters
    binary_labels1 = ratings1[category]
    binary_labels2 = ratings2[category]
    
    # Calculate Cohen's Kappa for the current category
    kappa = cohen_kappa_score(binary_labels1, binary_labels2)
    kappa_scores[category] = kappa

    # Prepare data for Krippendorff's Alpha calculation
    matrix = pd.concat([binary_labels1, binary_labels2], axis=1).to_numpy().T  # Transpose to match expected input format
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal')
    alpha_scores[category] = alpha

# Display the scores for both metrics
print("Cohen's Kappa Scores for Each Class:")
for category, kappa in kappa_scores.items():
    print(f"{category}: {kappa}")

print("\nKrippendorff's Alpha Scores for Each Class:")
for category, alpha in alpha_scores.items():
    print(f"{category}: {alpha}")

