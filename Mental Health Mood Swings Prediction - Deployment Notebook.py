# Mental Health Mood Swings Prediction - Deployment Notebook

# 1. Install dependencies (run once)
!pip install xgboost ipywidgets --quiet

import pickle
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import ipywidgets as widgets

# 2. Load models, encoders, and feature names
with open('saved_models/hist_model.pkl', 'rb') as f:
    hist_model = pickle.load(f)

with open('saved_models/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('saved_models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('saved_models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# 3. Preprocessing function to encode inputs
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = -1  # Unknown category
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df

# 4. Create widgets for all variables in the model
style = {'description_width': '150px'}
layout = widgets.Layout(width='400px')

gender_widget = widgets.Dropdown(
    options=label_encoders['Gender'].classes_.tolist(),
    description='Gender:',
    style=style,
    layout=layout
)

country_widget = widgets.Dropdown(
    options=label_encoders['Country'].classes_.tolist(),
    description='Country:',
    style=style,
    layout=layout
)

occupation_widget = widgets.Dropdown(
    options=label_encoders['Occupation'].classes_.tolist(),
    description='Occupation:',
    style=style,
    layout=layout
)

self_employed_widget = widgets.Dropdown(
    options=label_encoders['self_employed'].classes_.tolist(),
    description='Self Employed:',
    style=style,
    layout=layout
)

family_history_widget = widgets.Dropdown(
    options=label_encoders['family_history'].classes_.tolist(),
    description='Family History:',
    style=style,
    layout=layout
)

treatment_widget = widgets.Dropdown(
    options=label_encoders['treatment'].classes_.tolist(),
    description='Treatment:',
    style=style,
    layout=layout
)

days_indoors_widget = widgets.Dropdown(
    options=label_encoders['Days_Indoors'].classes_.tolist(),
    description='Days Indoors:',
    style=style,
    layout=layout
)

growing_stress_widget = widgets.Dropdown(
    options=label_encoders['Growing_Stress'].classes_.tolist(),
    description='Growing Stress:',
    style=style,
    layout=layout
)

changes_habits_widget = widgets.Dropdown(
    options=label_encoders['Changes_Habits'].classes_.tolist(),
    description='Changes Habits:',
    style=style,
    layout=layout
)

mental_health_history_widget = widgets.Dropdown(
    options=label_encoders['Mental_Health_History'].classes_.tolist(),
    description='Mental Health History:',
    style=style,
    layout=layout
)

mood_swings_widget = widgets.Dropdown(
    options=label_encoders['Mood_Swings'].classes_.tolist(),
    description='Mood Swings:',
    style=style,
    layout=layout
)

coping_struggles_widget = widgets.Dropdown(
    options=label_encoders['Coping_Struggles'].classes_.tolist(),
    description='Coping Struggles:',
    style=style,
    layout=layout
)

work_interest_widget = widgets.Dropdown(
    options=label_encoders['Work_Interest'].classes_.tolist(),
    description='Work Interest:',
    style=style,
    layout=layout
)

social_weakness_widget = widgets.Dropdown(
    options=label_encoders['Social_Weakness'].classes_.tolist(),
    description='Social Weakness:',
    style=style,
    layout=layout
)

mental_health_interview_widget = widgets.Dropdown(
    options=label_encoders['mental_health_interview'].classes_.tolist(),
    description='Mental Health Interview:',
    style=style,
    layout=layout
)

care_options_widget = widgets.Dropdown(
    options=label_encoders['care_options'].classes_.tolist(),
    description='Care Options:',
    style=style,
    layout=layout
)

submit_button = widgets.Button(
    description='Predict Mood Swings',
    button_style='success',
    layout=widgets.Layout(width='400px')
)

output = widgets.Output()

# 5. Prediction function
def on_submit_clicked(b):
    with output:
        output.clear_output()
        input_data = {
            'Gender': gender_widget.value,
            'Country': country_widget.value,
            'Occupation': occupation_widget.value,
            'self_employed': self_employed_widget.value,
            'family_history': family_history_widget.value,
            'treatment': treatment_widget.value,
            'Days_Indoors': days_indoors_widget.value,
            'Growing_Stress': growing_stress_widget.value,
            'Changes_Habits': changes_habits_widget.value,
            'Mental_Health_History': mental_health_history_widget.value,
            'Mood_Swings': mood_swings_widget.value,
            'Coping_Struggles': coping_struggles_widget.value,
            'Work_Interest': work_interest_widget.value,
            'Social_Weakness': social_weakness_widget.value,
            'mental_health_interview': mental_health_interview_widget.value,
            'care_options': care_options_widget.value,
        }
        X_input = preprocess_input(input_data)
        prediction = hist_model.predict(X_input)[0]
        # Decode prediction back to original label
        mood_swings_label = label_encoders['Mood_Swings'].inverse_transform([prediction])[0]
        print(f"Predicted Mood Swings Level: {mood_swings_label}")

submit_button.on_click(on_submit_clicked)

# 6. Display interface with a nice header
display(HTML("<h2 style='color: #2E86C1;'>Mental Health Mood Swings Prediction</h2>"))
display(HTML("<p>Please fill in the following information to get a prediction of mood swings level.</p>"))

widgets_list = [
    gender_widget, country_widget, occupation_widget, self_employed_widget,
    family_history_widget, treatment_widget, days_indoors_widget, growing_stress_widget,
    changes_habits_widget, mental_health_history_widget, mood_swings_widget,
    coping_struggles_widget, work_interest_widget, social_weakness_widget,
    mental_health_interview_widget, care_options_widget,
    submit_button, output
]

for w in widgets_list:
    display(w)
