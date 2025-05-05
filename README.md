
# Fresh Hearts

Fresh Hearts is a machine learning AI application developed using **Streamlit**. This project focuses on testing and predicting heart disease risks by analyzing clinical markers. It aims to assist medical practitioners in evaluating heart disease risk factors efficiently and effectively.

## Features

This Streamlit app evaluates the following clinical markers to predict heart disease risks:

- **Age**: Age of the patient [Numeric, in years].
- **Sex**: Gender of the patient [1: Male, 0: Female].
- **CP (Chest Pain Type)**:
  - 0: Typical Angina.
  - 1: Atypical Angina.
  - 2: Non-Anginal Pain.
  - 3: Asymptomatic.
- **Trestbps**: Resting blood pressure [Numeric, in mm Hg].
- **Chol**: Serum cholesterol level [Numeric, in mg/dl].
- **Fbs**: Fasting blood sugar [1: if fasting blood sugar > 120 mg/dl, 0: otherwise].
- **Restecg**: Resting electrocardiographic results:
  - 0: Normal.
  - 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV).
  - 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria.
- **Thalach**: Maximum heart rate achieved [Numeric value between 60 and 202].
- **Exang**: Exercise-induced angina [1: Yes, 0: No].
- **Oldpeak**: ST depression induced by exercise relative to rest [Numeric value measured in depression].
- **Slope**: Slope of the peak exercise ST segment:
  - 0: Upsloping.
  - 1: Flat.
  - 2: Downsloping.
- **Ca**: Number (0-3) of major vessels colored by fluoroscopy [0, 1, 2, 3].
- **Thal**: Thalassemia types:
  - 1: Normal.
  - 2: Fixed defect.
  - 3: Reversible defect.
- **Target**: Outcome variable for heart attack risk [1: Higher chance of heart attack, 0: Lower chance or normal].

## Installation

To set up and run the app locally, follow these steps:

1. Clone this repository:
   
   git clone https://github.com/star975/fresh_hearts.git
   
2. Navigate to the project directory:

   
   **cd fresh_hearts**
   
4. Install the required dependencies:

   
   **pip install -r requirements.txt**
   
6. Run the Streamlit app:
   
   **streamlit run app.py**
   

## Usage

1. **Create an Account**: Users are required to register for the first time by providing their details, including the names of the doctor and the patient.
2. **Input Clinical Readings**: Enter the clinical readings as prompted by the application.
3. **Predict**: Use the tool to predict the likelihood of heart disease.
4. **Download Results**: Users can download their results or share them via email.
5. **Live Dashboard**: Access the live dashboard for statistical analysis, which can also be downloaded.

## Acknowledgments

Special thanks to:
- **Mrs. Patience**: My supervisor.
- **Mr. John** and **Mr. Alex**: For their invaluable support throughout my journey in pursuing a Bachelor's Degree in Health Informatics.

---

Feel free to contribute to this project or report any issues!
On mobile money
+256774652671
