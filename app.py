import streamlit as st
import joblib
import re
import string

# ================================
# LOAD MODEL & VECTORIZER
# ================================
lr = joblib.load("logistic_model.joblib")
nb = joblib.load("nb_model.joblib")
svm = joblib.load("svm_model.joblib")
stacking = joblib.load("stacking_model.joblib")
cv = joblib.load("countvectorizer.joblib")

# ================================
# TEXT CLEANING FUNCTION
# ================================
def clean_text(text):
    # Remove numbers
    text = re.sub(r"\w*\d\w*", " ", text)

    # Remove punctuation & lowercase
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text.lower())

    return text

# ================================
# STREAMLIT UI
# ================================
def main():
    st.title("AI Sentiment Analysis App")
    st.write("Enter a sentence and the AI will predict its sentiment.")

    # Model Selection
    model_choice = st.selectbox(
        "Choose a model:",
        ("Logistic Regression", "Naive Bayes", "SVM", "Stacking")
    )

    user_input = st.text_area("Enter your text here:")

    if st.button("Predict Sentiment"):
        if user_input.strip() != "":
            
            # Clean the input
            clean_input = clean_text(user_input)

            # Vectorize
            text_vectorized = cv.transform([clean_input])

            if model_choice == "Logistic Regression":
                model = lr
            elif model_choice == "Naive Bayes":
                model = nb
            elif model_choice == "SVM":
                model = svm
            else:
                model = stacking

            # Predict
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]

            # Show result
            st.success(f"Predicted Sentiment: **{prediction.upper()}**")

            st.write("Confidence Scores:")
            for label, score in zip(model.classes_, probabilities):
                st.write(f"- {label.capitalize()}: {score:.2%}")

        else:
            st.warning("Please enter some text first!")

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    main()



