import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go

st.sidebar.title("Machine Learning Model Comparison & Tuning")
mode = st.sidebar.radio("STEPS", ["Compare Models", "Tune Model"])

if mode == "Compare Models":
    st.subheader("Compare Models")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Dataset for Classification", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        # Rename columns to handle spaces
        data.columns = data.columns.str.replace(" ", "_")

    if uploaded_file:
        st.write("Dataset Preview")
        st.dataframe(data.head())

        # Specify the target column
        target_col = "LUNG_CANCER"  # Explicitly defined
        feature_cols = [col for col in data.columns if col != "LUNG_CANCER"]  # All other columns as features

        # Prepare features and target
        X = data[feature_cols]
        y = data[target_col]

        # Check if target variable contains missing values
        if y.isnull().sum() > 0:
            y = y.fillna(y.mean())  # Fill missing values

        # Handle categorical target for classification (not required here since it's already numeric binary)
        if y.dtype == 'object' or len(np.unique(y)) < 20:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            task = "Classification"

        # Handle missing values in features as well
        X.fillna(X.mean(), inplace=True)

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define classification models
        models = {
            "Decision Tree (CART)": DecisionTreeClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "AdaBoost (Gradient Boosting)": AdaBoostClassifier(),
            "K-Nearest Neighbors (K-NN)": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Multi-Layer Perceptron (MLP)": MLPClassifier(),
            "Perceptron": Perceptron(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machines (SVM)": SVC(),
        }

        metric = accuracy_score
        metric_label = "Accuracy"

        # Evaluate models
        st.subheader(f"{task} Model Performance")
        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = metric(y_test, y_pred)
            results[name] = score

        # Convert results to DataFrame
        results_df = pd.DataFrame(results.items(), columns=["Model", metric_label])

        # Highlight the best and worst rows in the table
        best_model = results_df.loc[results_df[metric_label].idxmax()]
        worst_model = results_df.loc[results_df[metric_label].idxmin()]

        def highlight_best_and_worst(row):
            if row["Model"] == best_model["Model"]:
                return ["background-color: #48bb78; color: white"] * len(row)
            elif row["Model"] == worst_model["Model"]:
                return ["background-color: #f56565; color: white"] * len(row)
            else:
                return [""] * len(row)

        st.write(results_df.style.apply(highlight_best_and_worst, axis=1))

        # Generate the bar chart using Plotly
        def plot_highlighted_bar_chart(df, metric_label, best_model, worst_model):
            colors = [
                "#48bb78" if model == best_model["Model"]  # Green for the best model
                else "#f56565" if model == worst_model["Model"]  # Red for the worst model
                else "#63b3ed"  # Blue for other models
                for model in df["Model"]
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=df["Model"],
                    y=df[metric_label],
                    marker=dict(color=colors),
                    text=[f"{v:.4f}" for v in df[metric_label]],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title=f"{task} Performance Visualization",
                xaxis_title="Machine Learning Algorithm",
                yaxis_title=metric_label,
                xaxis_tickangle=-45,
                template="plotly_white",
            )
            
            return fig

        # Call the function to generate the chart
        st.plotly_chart(plot_highlighted_bar_chart(results_df, metric_label, best_model, worst_model))

        best_model_name = best_model["Model"]
        best_model_instance = models[best_model_name]

        # Train the best model on the full training data
        best_model_instance.fit(X_train, y_train)

        # Save the model to a .joblib file
        joblib_filename = f"{best_model_name}.joblib"
        joblib.dump(best_model_instance, joblib_filename)

        # Provide download link
        #st.download_button(
            #label="Download the Best Model",
            #data=open(joblib_filename, "rb").read(),
            #file_name=joblib_filename,
           # mime="application/octet-stream",
        #)

        st.session_state.best_model_name = best_model["Model"]
        st.session_state.data = data


elif mode == "Tune Model":

    if "best_model_name" not in st.session_state or "data" not in st.session_state:
        st.error("Please compare models first to select a model for tuning.")
    else:
        
        best_model_name = st.session_state.best_model_name
        data = st.session_state.data

        st.subheader(f"Tuning **{best_model_name}**")
        # Assuming the last column is the target
        target_col = data.columns[-1]
        feature_cols = data.columns[:-1]

        # Prepare features and target
        X = data[feature_cols]
        y = data[target_col]

        # Handle missing values
        if y.isnull().sum() > 0:
            y = y.fillna(y.mean())
        if y.dtype == 'object' or len(np.unique(y)) < 20:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        X.fillna(X.mean(), inplace=True)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if best_model_name == "Decision Tree":
            st.write("**Tuning Decision Tree Parameters**")
            
            # Parameter grid
            max_depth_values = [3, 5, 10, 15, 20]
            min_samples_split_values = [2, 5, 10]
            min_samples_leaf_values = [1, 2, 4, 6]

            # Generate combinations
            param_combinations = list(itertools.product(max_depth_values, min_samples_split_values, min_samples_leaf_values))
            results = []

            # Loop through combinations
            for max_depth, min_samples_split, min_samples_leaf in param_combinations:
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([max_depth, min_samples_split, min_samples_leaf, accuracy])

            # Convert results to a DataFrame
            results_df = pd.DataFrame(results, columns=["Max Depth", "Min Samples Split", "Min Samples Leaf", "Accuracy"])

            # Display results
            st.write("**Tuning Results Table**")
            st.dataframe(results_df)

            # Highlight the best parameters
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"**Best Parameters:** {best_params.to_dict()}")

            # Save the best model
            best_model = DecisionTreeClassifier(
                max_depth=best_params["Max Depth"],
                min_samples_split=best_params["Min Samples Split"],
                min_samples_leaf=best_params["Min Samples Leaf"],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_decision_tree.joblib")

            # Plot results
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=results_df["Max Depth"],
                    y=results_df["Min Samples Split"],
                    z=results_df["Min Samples Leaf"],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=results_df["Accuracy"],
                        colorscale="Viridis",
                        colorbar=dict(title="Accuracy")
                    ),
                    text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
                )
            ])
            fig.update_layout(
                title="Decision Tree Parameter Tuning",
                scene=dict(
                    xaxis_title="Max Depth",
                    yaxis_title="Min Samples Split",
                    zaxis_title="Min Samples Leaf"
                ),
                template="plotly_white"
            )
            st.plotly_chart(fig)

        elif best_model_name == "Gaussian Naive Bayes":
            
            var_smoothing_values = [1e-15, 1e-13, 1e-11,1e-9,1e-7]
            model_names = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]

            results = []
            results1 = []

            # Loop through combinations
            for model_name, var_smoothing in zip(model_names, var_smoothing_values):
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([var_smoothing, accuracy])
                results1.append([model_name, f"{var_smoothing:.1e}", accuracy])

            results_df1 = pd.DataFrame(results1, columns=["Model", "Var Smoothing", "Accuracy"])
            results_df = pd.DataFrame(results, columns=["Var Smoothing", "Accuracy"])
            st.write("**Tuning Results Table**")
            st.dataframe(results_df1)

            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            best_model_name = results_df1.loc[results_df["Accuracy"].idxmax(), "Model"]
            best_model = GaussianNB(var_smoothing=best_params["Var Smoothing"])
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_gaussian_nb.joblib")

            fig = go.Figure(data=[
                go.Bar(
                    x=results_df1["Model"],  # x-axis: Var Smoothing
                    y=results_df["Accuracy"],  # y-axis: Accuracy
                    marker=dict(
                        color=results_df["Accuracy"],
                        colorscale="Viridis",
                        colorbar=dict(title="Accuracy")
                    ),
                    text=results_df["Accuracy"].apply(lambda acc: f"{acc:.3f}"),  # Show accuracy as text
                    textposition="outside"  # Position text outside bars
                )
            ])

            fig.update_layout(
                title="Tuning Results Visualization",
                xaxis_title="Model",
                yaxis_title="Accuracy",
                template="plotly_white",
                xaxis=dict(type="category")  # Ensure x-axis is treated as categorical for Var Smoothing
            )

            st.plotly_chart(fig)

            joblib.dump(best_model, "best_tuned_gaussian.joblib")
            # Repeat similar structure for other models: AdaBoost, K-NN, Logistic Regression

            st.download_button(
                label=f"Download Tuned {best_model_name}",
                data=open(f"best_tuned_gaussian.joblib", "rb").read(),
                file_name=f"best_tuned_gaussian.joblib",
                mime="application/octet-stream",
            )

        elif best_model_name == "AdaBoost (Gradient Boosting)":
            st.write("Tuning AdaBoost Parameters")

            # Define the parameter grid
            n_estimators_values = [50, 100, 200]
            learning_rate_values = [0.01, 0.1, 1.0]

            # Store results
            results = []

            # Iterate through all parameter combinations
            for n_estimators, learning_rate in itertools.product(n_estimators_values, learning_rate_values):
                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([n_estimators, learning_rate, accuracy])

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results, columns=["N Estimators", "Learning Rate", "Accuracy"])

            # Define a function for styling the best and worst accuracy rows
            def highlight_best_worst(row):
                best = results_df["Accuracy"].max()  # Best accuracy
                worst = results_df["Accuracy"].min()  # Worst accuracy
                if row["Accuracy"] == best:
                    return ['background-color: #48bb78'] * len(row)  # Green for best result
                else:
                    return [''] * len(row)  # No color for others

            # Apply the styling to the DataFrame
            styled_df = results_df.style.apply(highlight_best_worst, axis=1)
            # Display the results table
            st.write("Tuning Results Table:")
            st.dataframe(styled_df)

            # Plot the results
            fig = go.Figure(data=[go.Scatter(
                x=results_df["N Estimators"],
                y=results_df["Learning Rate"],
                mode='markers',
                marker=dict(
                    size=10,
                    color=results_df["Accuracy"],
                    colorscale="Viridis",
                    colorbar=dict(title="Accuracy")
                ),
                text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
            )])
            fig.update_layout(
                title="Tuning AdaBoost Parameters",
                xaxis_title="N Estimators",
                yaxis_title="Learning Rate",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Optionally, you can save the best model
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"Best Parameters: {best_params.to_dict()}")
            best_model = AdaBoostClassifier(
                n_estimators=int(best_params["N Estimators"]), 
                learning_rate=best_params["Learning Rate"]
            )
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_adaboost.joblib")

            st.download_button(
                label="Download Tuned Model",
                data=open("best_tuned_adaboost.joblib", "rb").read(),
                file_name="best_tuned_adaboost.joblib",
                mime="application/octet-stream",
            )

        elif best_model_name == "K-Nearest Neighbors (K-NN)":
            st.write("Tuning K-Nearest Neighbors Parameters")

            # Define the parameter grid
            n_neighbors_values = [3, 5, 10]
            weights_values = ["uniform", "distance"]
            p_values = [1, 2]  # 1 for Manhattan, 2 for Euclidean

            # Store results
            results = []

            # Iterate through all parameter combinations
            for n_neighbors, weights, p in itertools.product(n_neighbors_values, weights_values, p_values):
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([n_neighbors, weights, p, accuracy])

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results, columns=["Neighbors", "Weights", "P", "Accuracy"])

            # Display the results table
            st.write("Tuning Results Table:")
            st.dataframe(results_df)

            # Plot the results
            fig = go.Figure(data=[go.Scatter(
                x=results_df["Neighbors"],
                y=results_df["Accuracy"],
                mode='markers',
                marker=dict(
                    size=10,
                    color=results_df["Accuracy"],
                    colorscale="Viridis",
                    colorbar=dict(title="Accuracy")
                ),
                text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
            )])
            fig.update_layout(
                title="Tuning K-Nearest Neighbors Parameters",
                xaxis_title="Neighbors",
                yaxis_title="Accuracy",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Optionally, you can save the best model
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"Best Parameters: {best_params.to_dict()}")
            best_model = KNeighborsClassifier(
                n_neighbors=best_params["Neighbors"], 
                weights=best_params["Weights"], 
                p=best_params["P"]
            )
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_knn.joblib")

            st.download_button(
                label="Download Tuned Model",
                data=open("best_tuned_knn.joblib", "rb").read(),
                file_name="best_tuned_knn.joblib",
                mime="application/octet-stream",
            )
            
        elif best_model_name == "Logistic Regression":
            st.write("Tuning Logistic Regression Parameters")

            # Define the parameter grid
            penalty_values = ['l1', 'l2']
            C_values = [0.01, 0.1, 1.0, 10.0]

            # Store results
            results = []

            # Iterate through all parameter combinations
            for penalty, C in itertools.product(penalty_values, C_values):
                model = LogisticRegression(penalty=penalty, C=C, solver='liblinear', random_state=42)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([penalty, C, accuracy])

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results, columns=["Penalty", "C", "Accuracy"])

            # Display the results table
            st.write("Tuning Results Table:")
            st.dataframe(results_df)

            # Plot the results
            fig = go.Figure(data=[go.Scatter(
                x=results_df["C"],
                y=results_df["Accuracy"],
                mode='markers',
                marker=dict(
                    size=10,
                    color=results_df["Accuracy"],
                    colorscale="Viridis",
                    colorbar=dict(title="Accuracy")
                ),
                text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
            )])
            fig.update_layout(
                title="Tuning Logistic Regression Parameters",
                xaxis_title="C",
                yaxis_title="Accuracy",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Optionally, you can save the best model
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"Best Parameters: {best_params.to_dict()}")
            best_model = LogisticRegression(
                penalty=best_params["Penalty"],
                C=best_params["C"],
                solver='liblinear',
                random_state=42
            )
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_logistic_regression.joblib")

            st.download_button(
                label="Download Tuned Model",
                data=open("best_tuned_logistic_regression.joblib", "rb").read(),
                file_name="best_tuned_logistic_regression.joblib",
                mime="application/octet-stream",
            )

        elif best_model_name == "Multi-Layer Perceptron (MLP)":
            st.write("Tuning Multi-Layer Perceptron Parameters")

            # Define the parameter grid
            hidden_layer_sizes_values = [(50,), (100,), (50, 50), (100, 50)]
            alpha_values = [0.0001, 0.001, 0.01]
            learning_rate_init_values = [0.001, 0.01, 0.1]

            # Store results
            results = []

            # Iterate through all parameter combinations
            for hidden_layer_sizes, alpha, learning_rate_init in itertools.product(hidden_layer_sizes_values, alpha_values, learning_rate_init_values):
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=1000,
                    random_state=42
                )
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([hidden_layer_sizes, alpha, learning_rate_init, accuracy])

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results, columns=["Hidden Layers", "Alpha", "Learning Rate Init", "Accuracy"])

            # Display the results table
            st.write("Tuning Results Table:")
            st.dataframe(results_df)

            # Plot the results
            fig = go.Figure(data=[go.Scatter3d(
                x=results_df["Alpha"],
                y=results_df["Learning Rate Init"],
                z=results_df["Accuracy"],
                mode='markers',
                marker=dict(
                    size=10,
                    color=results_df["Accuracy"],
                    colorscale="Viridis",
                    colorbar=dict(title="Accuracy")
                ),
                text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
            )])
            fig.update_layout(
                title="Tuning Multi-Layer Perceptron Parameters",
                scene=dict(
                    xaxis_title="Alpha",
                    yaxis_title="Learning Rate Init",
                    zaxis_title="Accuracy"
                ),
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Optionally, you can save the best model
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"Best Parameters: {best_params.to_dict()}")
            best_model = MLPClassifier(
                hidden_layer_sizes=best_params["Hidden Layers"],
                alpha=best_params["Alpha"],
                learning_rate_init=best_params["Learning Rate Init"],
                max_iter=1000,
                random_state=42
            )
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_mlp.joblib")

            st.download_button(
                label="Download Tuned Model",
                data=open("best_tuned_mlp.joblib", "rb").read(),
                file_name="best_tuned_mlp.joblib",
                mime="application/octet-stream",
            )

        elif best_model_name == "Perceptron":
            st.write("Tuning Perceptron Parameters")

            # Define the parameter grid
            penalty_values = ['l2', 'l1', 'elasticnet']
            alpha_values = [0.0001, 0.001, 0.01]
            max_iter_values = [1000, 2000]

            # Store results
            results = []

            # Iterate through all parameter combinations
            for penalty, alpha, max_iter in itertools.product(penalty_values, alpha_values, max_iter_values):
                model = Perceptron(
                    penalty=penalty, 
                    alpha=alpha, 
                    max_iter=max_iter,
                    random_state=42
                )
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([penalty, alpha, max_iter, accuracy])

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results, columns=["Penalty", "Alpha", "Max Iter", "Accuracy"])

            # Display the results table
            st.write("Tuning Results Table:")
            st.dataframe(results_df)

            # Plot the results
            fig = go.Figure(data=[go.Scatter(
                x=results_df["Alpha"],
                y=results_df["Max Iter"],
                mode='markers',
                marker=dict(
                    size=10,
                    color=results_df["Accuracy"],
                    colorscale="Viridis",
                    colorbar=dict(title="Accuracy")
                ),
                text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
            )])
            fig.update_layout(
                title="Tuning Perceptron Parameters",
                xaxis_title="Alpha",
                yaxis_title="Max Iter",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Optionally, you can save the best model
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"Best Parameters: {best_params.to_dict()}")
            best_model = Perceptron(
                penalty=best_params["Penalty"],
                alpha=best_params["Alpha"],
                max_iter=best_params["Max Iter"],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_perceptron.joblib")

            st.download_button(
                label="Download Tuned Model",
                data=open("best_tuned_perceptron.joblib", "rb").read(),
                file_name="best_tuned_perceptron.joblib",
                mime="application/octet-stream",
            )

        elif best_model_name == "Random Forest":
            st.write("Tuning Random Forest Parameters")

            # Define the parameter grid
            n_estimators_values = [50, 100, 200]
            max_depth_values = [10, 20, None]

            # Store results
            results = []

            # Iterate through all parameter combinations
            for n_estimators, max_depth in itertools.product(n_estimators_values, max_depth_values):
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([n_estimators, max_depth, accuracy])

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results, columns=["N Estimators", "Max Depth", "Accuracy"])

            # Display the results table
            st.write("Tuning Results Table:")
            st.dataframe(results_df)

            # Plot the results
            fig = go.Figure(data=[go.Scatter(
                x=results_df["N Estimators"],
                y=results_df["Accuracy"],
                mode='markers',
                marker=dict(
                    size=10,
                    color=results_df["Accuracy"],
                    colorscale="Viridis",
                    colorbar=dict(title="Accuracy")
                ),
                text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
            )])
            fig.update_layout(
                title="Tuning Random Forest Parameters",
                xaxis_title="N Estimators",
                yaxis_title="Accuracy",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Optionally, you can save the best model
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"Best Parameters: {best_params.to_dict()}")
            best_model = RandomForestClassifier(
                n_estimators=best_params["N Estimators"],
                max_depth=best_params["Max Depth"],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_random_forest.joblib")

            st.download_button(
                label="Download Tuned Model",
                data=open("best_tuned_random_forest.joblib", "rb").read(),
                file_name="best_tuned_random_forest.joblib",
                mime="application/octet-stream",
            )

        elif best_model_name == "Support Vector Machines (SVM)":
            st.write("Tuning Support Vector Machines Parameters")

            # Define the parameter grid
            C_values = [0.1, 1, 10]
            kernel_values = ['linear', 'rbf']

            # Store results
            results = []

            # Iterate through all parameter combinations
            for C, kernel in itertools.product(C_values, kernel_values):
                model = SVC(C=C, kernel=kernel, random_state=42)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results.append([C, kernel, accuracy])

            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results, columns=["C", "Kernel", "Accuracy"])

            # Display the results table
            st.write("Tuning Results Table:")
            st.dataframe(results_df)

            # Plot the results
            fig = go.Figure(data=[go.Scatter(
                x=results_df["C"],
                y=results_df["Accuracy"],
                mode='markers',
                marker=dict(
                    size=10,
                    color=results_df["Accuracy"],
                    colorscale="Viridis",
                    colorbar=dict(title="Accuracy")
                ),
                text=results_df.apply(lambda row: f"Accuracy: {row['Accuracy']:.3f}", axis=1)
            )])
            fig.update_layout(
                title="Tuning Support Vector Machines Parameters",
                xaxis_title="C",
                yaxis_title="Accuracy",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Optionally, you can save the best model
            best_params = results_df.loc[results_df["Accuracy"].idxmax()]
            st.write(f"Best Parameters: {best_params.to_dict()}")
            best_model = SVC(C=best_params["C"], kernel=best_params["Kernel"], random_state=42)
            best_model.fit(X_train, y_train)
            joblib.dump(best_model, "best_tuned_svm.joblib")

            st.download_button(
                label="Download Tuned Model",
                data=open("best_tuned_svm.joblib", "rb").read(),
                file_name="best_tuned_svm.joblib",
                mime="application/octet-stream",
            )


        

            
