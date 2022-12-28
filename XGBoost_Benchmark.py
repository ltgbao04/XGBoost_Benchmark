import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

MODELS = {
    "LogisticRegression": LogisticRegression(),
    "XGBoost": XGBClassifier(
        n_estimators=20, max_depth=2, learning_rate=0.01, objective="binary:logistic"
    ),
    "SVM": SVC(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
}


st.markdown("# CS116")

st.markdown("### Upload your `.csv` file here")

uploaded_file = st.file_uploader("Choose a file", type=[".csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("### Your uploaded data")
    st.write(df)
    st.markdown("### Pick input feature(s)")
    st.write("Last column will be used as output.")
    selected_features = st.multiselect("Select", list(df.columns.values)[:-1])
    if len(selected_features) > 0:
        bad_columns = []
        for selected_feature in selected_features:
            if (
                not pd.to_numeric(df[selected_feature], errors="coerce").notnull().all()
                or selected_feature == list(df.columns.values)[-1]
            ):
                bad_columns.append(selected_feature)
        if len(bad_columns) > 0:
            st.write(
                "These selected column(s) may contain not numeric value or can't be used as input features, please check again:",
                " ".join([bad_column for bad_column in bad_columns]),
            )
        else:
            model_selection = st.radio(
                "Select model",
                [model_name for model_name, _ in MODELS.items()],
            )
            if model_selection:
                all_classes = df[df.columns[-1]].unique()
                if model_selection in MODELS:
                    model = MODELS[model_selection]
                    X = df[selected_features].values
                    y = df.iloc[:, -1].values
                    test_ratio = st.number_input(
                        "Select test size (train size will be `1 - test_size`)",
                        min_value=0.0,
                        max_value=1.0,
                    )
                    if test_ratio:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_ratio, random_state=42
                        )
                        if model_selection == "XGBoost":
                            from sklearn.preprocessing import LabelEncoder

                            le = LabelEncoder()
                            y_train = le.fit_transform(y_train)
                            y_test = le.fit_transform(y_test)
                        model.fit(X_train, y_train)
                        y_test_pred = model.predict(X_test)
                        test_scores = classification_report(
                            y_test,
                            y_test_pred,
                            target_names=all_classes,
                            output_dict=True,
                        )
                        st.write("Test scores")
                        st.dataframe(test_scores)
                        st.write("Testset Confusion matrix")
                        plot_confusion_matrix(
                            model,
                            X_test,
                            y_test,
                            display_labels=all_classes,
                        )
                        st.set_option("deprecation.showPyplotGlobalUse", False)
                        st.pyplot()
                    else:
                        st.write("Unknown option")
