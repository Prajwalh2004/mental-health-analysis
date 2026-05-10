# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# # -------------------- CONFIG --------------------
# st.set_page_config(page_title="Mental Health App", layout="wide")

# # -------------------- LOAD FILES --------------------
# try:
#     df = pd.read_csv("cleaned_df.csv")
#     model = joblib.load("Random Forest_model.joblib")
#     model_features = joblib.load("features.joblib")
# except Exception as e:
#     st.error(f"❌ Error loading files: {e}")
#     st.stop()

# target = "treatment"

# # -------------------- HEADER --------------------
# st.title("🧠 Mental Health Analysis & Prediction")
# st.warning("⚠️ This tool is for educational purposes only. Not a medical diagnosis.")

# # -------------------- SIDEBAR --------------------
# page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🤖 Prediction"])

# # ===================== DASHBOARD ======================
# if page == "📊 Dashboard":

#     st.subheader("📊 Dataset Overview")
#     st.write(df.head())

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Stress Distribution")
#         fig, ax = plt.subplots()
#         sns.histplot(df["Growing_Stress"], kde=True, ax=ax)
#         st.pyplot(fig)

#     with col2:
#         st.subheader("Mood Swings Distribution")
#         fig, ax = plt.subplots()
#         sns.histplot(df["Mood_Swings"], kde=True, ax=ax)
#         st.pyplot(fig)

#     # Feature Importance
#     st.subheader("🔥 Feature Importance")

#     try:
#         importances = model.feature_importances_
#         feat_imp = pd.Series(importances, index=model_features).sort_values(ascending=False)

#         st.bar_chart(feat_imp.head(10))
#     except:
#         st.info("Model does not support feature importance")
#         if page == "📊 Dashboard":

#         st.subheader("📊 Dataset Overview")
#         st.write(df.head())

#         col1, col2 = st.columns(2)

#         # ---------------- STRESS ----------------
#         with col1:
#             st.subheader("Stress Distribution")
#             fig, ax = plt.subplots()
#             sns.histplot(df["Growing_Stress"], kde=True, ax=ax)
#             st.pyplot(fig)

#             # Insight
#             avg_stress = df["Growing_Stress"].mean()

#             if avg_stress > 3:
#                 st.warning("⚠️ Overall stress levels are high in the dataset")
#             else:
#                 st.success("✅ Stress levels are generally moderate/low")

#         # ---------------- MOOD ----------------
#         with col2:
#             st.subheader("Mood Swings Distribution")
#             fig, ax = plt.subplots()
#             sns.histplot(df["Mood_Swings"], kde=True, ax=ax)
#             st.pyplot(fig)

#             # Insight
#             avg_mood = df["Mood_Swings"].mean()

#             if avg_mood > 3:
#                 st.warning("⚠️ Frequent mood swings observed across users")
#             else:
#                 st.success("✅ Mood swings are relatively stable")

#         # ---------------- FAMILY HISTORY ----------------
#         if "family_history" in df.columns:
#             st.subheader("Family History Impact")

#             counts = df["family_history"].value_counts()

#             st.bar_chart(counts)

#             # Insight
#             if counts.get(1, 0) > counts.get(0, 0):
#                 st.warning("⚠️ Majority of individuals have a family history of mental illness")
#             else:
#                 st.info("ℹ️ Most individuals do not report family history")

#         # ---------------- TARGET RELATION ----------------
#         if "treatment" in df.columns:
#             st.subheader("Stress vs Treatment")

#             fig, ax = plt.subplots()
#             sns.boxplot(x="treatment", y="Growing_Stress", data=df, ax=ax)
#             st.pyplot(fig)

#             # Insight
#             treated_stress = df[df["treatment"] == 1]["Growing_Stress"].mean()
#             untreated_stress = df[df["treatment"] == 0]["Growing_Stress"].mean()

#             if treated_stress > untreated_stress:
#                 st.warning("⚠️ People receiving treatment tend to have higher stress")
#             else:
#                 st.info("ℹ️ Stress levels are similar regardless of treatment")

#         # ---------------- FEATURE IMPORTANCE ----------------
#         st.subheader("🔥 Feature Importance")

#         try:
#             importances = model.feature_importances_
#             feat_imp = pd.Series(importances, index=model_features).sort_values(ascending=False)

#             st.bar_chart(feat_imp.head(10))

#             # Insight
#             top_feature = feat_imp.idxmax()
#             st.info(f"📌 Most influential factor: **{top_feature}**")

#         except:
#             st.info("Model does not support feature importance")

# # ===================== PREDICTION =====================
# elif page == "🤖 Prediction":

#     st.subheader("🤖 Mental Health Prediction")

#     # ---------------- INPUT UI ----------------
#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", min_value=18, max_value=100, value=25)
#         gender_ui = st.selectbox("Gender", ["Male", "Female"])
#         family_history_ui = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

#     with col2:
#         stress = st.slider("Growing Stress", 0, 5, 2)
#         mood = st.slider("Mood Swings", 0, 5, 2)
#         work = st.slider("Work Interest", 0, 5, 3)

#     # ---------------- CONVERSIONS ----------------
#     gender = 1 if gender_ui == "Male" else 0
#     family_history = 1 if family_history_ui == "Yes" else 0

#     # ---------------- PREPARE INPUT ----------------
#     def prepare_input():
#         input_dict = {col: 0 for col in model_features}

#         # numeric (only if exists in model)
#         if "Age" in model_features:
#             input_dict["Age"] = age

#         if "Growing_Stress" in model_features:
#             input_dict["Growing_Stress"] = stress

#         if "Mood_Swings" in model_features:
#             input_dict["Mood_Swings"] = mood

#         if "Work_Interest" in model_features:
#             input_dict["Work_Interest"] = work

#         # gender
#         if "Gender" in model_features:
#             input_dict["Gender"] = gender
#         else:
#             col = f"Gender_{gender_ui}"
#             if col in model_features:
#                 input_dict[col] = 1

#         # family history
#         if "family_history" in model_features:
#             input_dict["family_history"] = family_history
#         else:
#             col = f"family_history_{family_history_ui}"
#             if col in model_features:
#                 input_dict[col] = 1

#         return pd.DataFrame([input_dict])

#     # ---------------- PREDICT ----------------
#     if st.button("Predict"):

#         input_df = prepare_input()

#         with st.spinner("Analyzing..."):
#             prediction = model.predict(input_df)[0]

#             try:
#                 proba = model.predict_proba(input_df)[0][1]
#             except:
#                 proba = None

#         # ---------------- RESULT ----------------
#         if prediction == 1:
#             st.error("🔴 High Risk of Mental Health Issue")
#         else:
#             st.success("🟢 Low Risk")

#         if proba is not None:
#             st.metric("Risk Probability", f"{proba:.2f}")

#         # ---------------- INSIGHTS ----------------
#         st.subheader("💡 Insights")

#         if proba is not None:
#             if proba > 0.7:
#                 st.error("Strong indicators of mental health risk detected.")
#             elif proba > 0.4:
#                 st.warning("Moderate risk — consider lifestyle improvements.")
#             else:
#                 st.success("Low risk — maintain healthy habits.")

#         if stress > 3:
#             st.warning("⚠️ High stress detected")

#         if mood > 3:
#             st.warning("⚠️ Frequent mood swings observed")

#         if work < 2:
#             st.warning("⚠️ Low work interest may indicate burnout")

#         if family_history == 1:
#             st.warning("⚠️ Family history increases risk factor")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Mental Health App", layout="wide")

# -------------------- LOAD FILES --------------------
try:
    df = pd.read_csv("cleaned_df.csv")
    model = joblib.load("Random Forest_model.joblib")
    model_features = joblib.load("features.joblib")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# -------------------- HEADER --------------------
st.title("🧠 Mental Health Analysis & Prediction")
st.warning("⚠️ This tool is for educational purposes only. Not a medical diagnosis.")

# -------------------- SIDEBAR --------------------
page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🤖 Prediction"])

# ===================== DASHBOARD ======================
# =====================================================
if page == "📊 Dashboard":

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    col1, col2 = st.columns(2)

    # -------- Stress --------
    with col1:
        st.subheader("Stress Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Growing_Stress"], kde=True, ax=ax)
        st.pyplot(fig)

        avg_stress = df["Growing_Stress"].mean()
        if avg_stress > 3:
            st.warning("⚠️ Overall stress levels are high in the dataset")
        else:
            st.success("✅ Stress levels are moderate/low")

    # -------- Mood --------
    with col2:
        st.subheader("Mood Swings Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Mood_Swings"], kde=True, ax=ax)
        st.pyplot(fig)

        avg_mood = df["Mood_Swings"].mean()
        if avg_mood > 3:
            st.warning("⚠️ Frequent mood swings observed")
        else:
            st.success("✅ Mood swings are relatively stable")

    # -------- Family History --------
    if "family_history" in df.columns:
        st.subheader("Family History Impact")

        counts = df["family_history"].value_counts()
        st.bar_chart(counts)

        if counts.get(1, 0) > counts.get(0, 0):
            st.warning("⚠️ Many individuals report family history of mental illness")
        else:
            st.info("ℹ️ Most individuals do not report family history")

    # -------- Stress vs Treatment --------
    if "treatment" in df.columns:
        st.subheader("Stress vs Treatment")

        fig, ax = plt.subplots()
        sns.boxplot(x="treatment", y="Growing_Stress", data=df, ax=ax)
        st.pyplot(fig)

        treated = df[df["treatment"] == 1]["Growing_Stress"].mean()
        untreated = df[df["treatment"] == 0]["Growing_Stress"].mean()

        if treated > untreated:
            st.warning("⚠️ Higher stress observed in treated individuals")
        else:
            st.info("ℹ️ Similar stress levels across groups")

    # -------- Feature Importance --------
    st.subheader("🔥 Feature Importance")

    try:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=model_features).sort_values(ascending=False)

        st.bar_chart(feat_imp.head(10))

        st.info(f"📌 Most influential factor: {feat_imp.idxmax()}")
    except:
        st.info("Model does not support feature importance")

# ===================== PREDICTION =====================
# =====================================================
elif page == "🤖 Prediction":

    st.subheader("🤖 Mental Health Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 25)
        gender_ui = st.selectbox("Gender", ["Male", "Female"])
        family_history_ui = st.selectbox("Family History", ["No", "Yes"])

    with col2:
        stress = st.slider("Growing Stress", 0, 5, 2)
        mood = st.slider("Mood Swings", 0, 5, 2)
        work = st.slider("Work Interest", 0, 5, 3)

    # -------- Convert --------
    gender = 1 if gender_ui == "Male" else 0
    family_history = 1 if family_history_ui == "Yes" else 0

    # -------- Prepare Input --------
    def prepare_input():
        input_dict = {col: 0 for col in model_features}

        if "Age" in model_features:
            input_dict["Age"] = age

        if "Growing_Stress" in model_features:
            input_dict["Growing_Stress"] = stress

        if "Mood_Swings" in model_features:
            input_dict["Mood_Swings"] = mood

        if "Work_Interest" in model_features:
            input_dict["Work_Interest"] = work

        if "Gender" in model_features:
            input_dict["Gender"] = gender
        else:
            col = f"Gender_{gender_ui}"
            if col in model_features:
                input_dict[col] = 1

        if "family_history" in model_features:
            input_dict["family_history"] = family_history
        else:
            col = f"family_history_{family_history_ui}"
            if col in model_features:
                input_dict[col] = 1

        return pd.DataFrame([input_dict])

    # -------- Predict --------
    if st.button("Predict"):

        input_df = prepare_input()

        with st.spinner("Analyzing..."):
            prediction = model.predict(input_df)[0]

            try:
                proba = model.predict_proba(input_df)[0][1]
            except:
                proba = None

        # -------- Output --------
        if prediction == 1:
            st.error("🔴 High Risk")
        else:
            st.success("🟢 Low Risk")

        if proba is not None:
            st.metric("Risk Probability", f"{proba:.2f}")

        # # -------- Insights --------
        # st.subheader("💡 Insights")

        # if proba:
        #     if proba > 0.7:
        #         st.error("High likelihood of mental health risk")
        #     elif proba > 0.4:
        #         st.warning("Moderate risk detected")
        #     else:
        #         st.success("Low risk")

        # if stress > 3:
        #     st.warning("⚠️ High stress detected")

        # if mood > 3:
        #     st.warning("⚠️ Frequent mood swings")

        # if work < 2:
        #     st.warning("⚠️ Possible burnout (low work interest)")

        # if family_history == 1:
        #     st.warning("⚠️ Family history increases risk factor")

        # -------- Insights --------
        st.subheader("💡 Insights")

        if proba:
            if proba > 0.7:
                st.error("High likelihood of mental health risk")
            elif proba > 0.4:
                st.warning("Moderate risk detected")
            else:
                st.success("Low risk")

        if stress > 3:
            st.warning("⚠️ High stress detected")

        if mood > 3:
            st.warning("⚠️ Frequent mood swings")

        if work < 2:
            st.warning("⚠️ Possible burnout (low work interest)")

        if family_history == 1:
            st.warning("⚠️ Family history increases risk factor")


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Built as a Data Science Project 🚀")
