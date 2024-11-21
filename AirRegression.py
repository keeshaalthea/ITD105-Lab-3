import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import AdaBoostRegressor
import plotly.graph_objects as go

st.sidebar.title("Machine Learning Model Comparison & Tuning")
mode = st.sidebar.radio("Choose Mode", ["Compare Models", "Tune Models"])

if mode == "Compare Models":
    st.subheader("Compare Models")
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Dataset for Regression", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

    if uploaded_file:
        st.write("Dataset Preview")
        st.dataframe(data.head())

        # Assuming the last column is the target
        target_col = data.columns[-1]  # Last column as target
        feature_cols = data.columns[:-1]  # All other columns as features

        # Prepare features and target
        X = data[feature_cols]
        y = data[target_col]

        # Handle missing values in the target variable
        if y.isnull().sum() > 0:
            y = y.fillna(y.mean())  # Fill missing values

        # Handle missing values in features as well
        X.fillna(X.mean(), inplace=True)

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Decision Tree (CART)": DecisionTreeRegressor(),
            "Elastic Net": ElasticNet(),
            "AdaBoost (Gradient Boosting)": GradientBoostingRegressor(),
            "K-Nearest Neighbors (K-NN)": KNeighborsRegressor(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge(),
            "Linear Regression": LinearRegression(),
            "Multi-Layer Perceptron (MLP)": MLPRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Machines (SVM)": SVR(),
        }
        metric = mean_absolute_error
        metric_label = "MAE"

        # Evaluate models
        st.write(f"Regression Model Performance")
        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # For regression, round predictions to avoid mismatch errors
            y_pred = np.round(y_pred, 2)
            
            score = metric(y_test, y_pred)
            results[name] = score

        # Convert results to DataFrame
        results_df = pd.DataFrame(results.items(), columns=["Model", metric_label])

        # Highlight the best and worst rows in the table
        best_model = results_df.loc[results_df[metric_label].idxmin()]  # Best model is the one with the lowest MAE
        worst_model = results_df.loc[results_df[metric_label].idxmax()]  # Worst model is the one with the highest MAE

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
                title="Regression Model Performance",
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
        joblib_filename = f"{best_model_name}_best_model.joblib"
        joblib.dump(best_model_instance, joblib_filename)

        # Provide download link
        #st.download_button(
            #label="Download the Best Model",
            #data=open(joblib_filename, "rb").read(),
            #file_name=joblib_filename,
            #mime="application/octet-stream",
        #)
        
        st.session_state.best_model_name = best_model["Model"]
        st.session_state.data = data


elif mode == "Tune Models":
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
        X.fillna(X.mean(), inplace=True)
        
        if best_model_name == "Decision Tree":
            st.write("**Tuning Decision Tree Parameters**")
            
            # Parameter grid
            max_depth_values = [3, 5, 10, 15, 20]
            min_samples_split_values = [2, 5, 10]
            min_samples_leaf_values = [1, 2, 4, 6]
            kfold = KFold(n_splits=20, shuffle=True, random_state=42)
            
            param_combinations = list(itertools.product(max_depth_values, min_samples_split_values, min_samples_leaf_values))
            results = []
            for max_depth, min_samples_split, min_samples_leaf in param_combinations:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'Max Depth': max_depth,
                    'Min Samples Split': min_samples_split,
                    'Min Samples Leaf': min_samples_leaf,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            # Find the best parameters based on Mean MAE
            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'Max Depth': int(best_params['Max Depth']),
                'Min Samples Split': int(best_params['Min Samples Split']),
                'Min Samples Leaf': int(best_params['Min Samples Leaf']),
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train the best model on the entire dataset
            best_model = DecisionTreeRegressor(
                max_depth=int(best_params['Max Depth']),
                min_samples_split=int(best_params['Min Samples Split']),
                min_samples_leaf=int(best_params['Min Samples Leaf']),
                random_state=42
            )
            best_model.fit(X, y)

            # Save the tuned model
            tuned_model_filename = f"Tuned_{best_model_name}_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download link for the tuned model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )
            
        if best_model_name == "Elastic Net":
            st.write("**Tuning Elastic Net Parameters**")
            
            # Elastic Net tuning parameters
            alphas = [0.01, 0.1, 1, 10]
            l1_ratios = [0.1, 0.5, 0.9, 1]
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            param_combinations = list(itertools.product(alphas, l1_ratios))
            results = []
            for alpha, l1_ratio in param_combinations:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'Alpha': alpha,
                    'L1 Ratio': l1_ratio,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })
                
            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'Alpha': best_params['Alpha'],
                'L1 Ratio': best_params['L1 Ratio'],
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train best model
            best_model = ElasticNet(alpha=best_params['Alpha'], l1_ratio=best_params['L1 Ratio'])
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_ElasticNet_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )

        elif best_model_name == "AdaBoost (Gradient Boosting)":
            st.write("**Tuning AdaBoost (Gradient Boosting) Parameters**")
            
            # AdaBoost (Gradient Boosting) tuning parameters
            n_estimators_values = [50, 100, 150, 200]
            learning_rate_values = [0.01, 0.1, 1]
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            param_combinations = list(itertools.product(n_estimators_values, learning_rate_values))
            results = []
            for n_estimators, learning_rate in param_combinations:
                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'N Estimators': n_estimators,
                    'Learning Rate': learning_rate,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'N Estimators': best_params['N Estimators'],
                'Learning Rate': best_params['Learning Rate'],
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train best model
            best_model = GradientBoostingRegressor(
                n_estimators=int(best_params['N Estimators']), 
                learning_rate=best_params['Learning Rate']
            )
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_AdaBoost_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )

        elif best_model_name == "K-Nearest Neighbors (K-NN)":
            st.write("**Tuning K-Nearest Neighbors (K-NN) Parameters**")
            
            # K-NN tuning parameters
            n_neighbors_values = [3, 5, 7, 9]
            weights_values = ['uniform', 'distance']
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            param_combinations = list(itertools.product(n_neighbors_values, weights_values))
            results = []
            for n_neighbors, weights in param_combinations:
                model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'N Neighbors': n_neighbors,
                    'Weights': weights,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'N Neighbors': best_params['N Neighbors'],
                'Weights': best_params['Weights'],
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train best model
            best_model = KNeighborsRegressor(
                n_neighbors=best_params['N Neighbors'],
                weights=best_params['Weights']
            )
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_KNN_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )

        elif best_model_name == "Lasso Regression":
            st.write("**Tuning Lasso Regression Parameters**")
            
            # Lasso tuning parameters
            alphas = [0.01, 0.1, 1, 10]
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            results = []
            for alpha in alphas:
                model = Lasso(alpha=alpha)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'Alpha': alpha,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'Alpha': best_params['Alpha'],
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train best model
            best_model = Lasso(alpha=best_params['Alpha'])
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_Lasso_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )

        elif best_model_name == "Ridge Regression":
            st.write("**Tuning Ridge Regression Parameters**")
            
            # Ridge tuning parameters
            alphas = [0.01, 0.1, 1, 10]
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            results = []
            for alpha in alphas:
                model = Ridge(alpha=alpha)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'Alpha': alpha,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'Alpha': best_params['Alpha'],
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train best model
            best_model = Ridge(alpha=best_params['Alpha'])
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_Ridge_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )

        elif best_model_name == "Linear Regression":
            st.write("**Tuning Linear Regression Parameters**")
            
            # Linear Regression doesn't require hyperparameter tuning in its basic form, so it's just fit
            model = LinearRegression()
            model.fit(X, y)
            
            # Save the model
            tuned_model_filename = f"Tuned_LinearRegression_model.joblib"
            joblib.dump(model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )

        elif best_model_name == "Multi-Layer Perceptron (MLP)":
            st.write("**Tuning Multi-Layer Perceptron (MLP) Parameters**")
            
            # MLP tuning parameters
            hidden_layer_sizes = [(50,), (100,), (200,), (300,)]
            activation_values = ['relu', 'tanh']
            alpha_values = [0.0001, 0.001, 0.01]
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            param_combinations = list(itertools.product(hidden_layer_sizes, activation_values, alpha_values))
            results = []
            for hidden_layer_size, activation, alpha in param_combinations:
                model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation=activation, alpha=alpha, max_iter=1000)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'Hidden Layer Size': hidden_layer_size,
                    'Activation': activation,
                    'Alpha': alpha,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'Hidden Layer Size': best_params['Hidden Layer Size'],
                'Activation': best_params['Activation'],
                'Alpha': best_params['Alpha'],
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train best model
            best_model = MLPRegressor(
                hidden_layer_sizes=best_params['Hidden Layer Size'],
                activation=best_params['Activation'],
                alpha=best_params['Alpha'],
                max_iter=1000
            )
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_MLP_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )

        elif best_model_name == "Random Forest":

            # Random Forest tuning parameters
            n_estimators_values = [50, 100, 150, 200]
            max_depth_values = [5, 10, 15, 20]
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            param_combinations = list(itertools.product(n_estimators_values, max_depth_values))
            results = []
            for n_estimators, max_depth in param_combinations:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'Model': f'Model {len(results) + 1}',
                    'N Estimators': n_estimators,
                    'Max Depth': max_depth,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            results_df = pd.DataFrame(results)
            st.write("Tuning Results Table")
            st.dataframe(results_df)

            # Find best parameters
            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]

            # Bar plot using Plotly
            fig = go.Figure()

            # Add bars for Mean MAE
            fig.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Mean MAE'],
                text=results_df['Mean MAE'].round(4),
                textposition='auto',
                marker_color=['green' if x == best_params['Mean MAE'] else 'blue' for x in results_df['Mean MAE']],
                name='Mean MAE'
            ))

            # Highlight the best model with red color
            fig.update_layout(
                title="Random Forest Model Tuning Results",
                xaxis_title="Models",
                yaxis_title="Mean MAE",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            )

            # Display plot
            st.plotly_chart(fig)

            # Train the best model
            best_model = RandomForestRegressor(
                n_estimators=best_params['N Estimators'],
                max_depth=best_params['Max Depth'],
                random_state=42
            )
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_RandomForest_model_{results_df['Model'][results_df['Mean MAE'].idxmin()]}.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for the model
            st.download_button(
                label=f"Download Tuned Model {results_df['Model'][results_df['Mean MAE'].idxmin()]}", 
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )


        elif best_model_name == "Support Vector Machines (SVM)":
            st.write("**Tuning Support Vector Machines (SVM) Parameters**")
            
            # SVM tuning parameters
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            c_values = [0.1, 1, 10]

            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            param_combinations = list(itertools.product(kernels, c_values))
            results = []
            for kernel, c in param_combinations:
                model = SVR(kernel=kernel, C=c) 
                scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
                mean_score = -scores.mean()
                std_dev = scores.std()
                results.append({
                    'Kernel': kernel,
                    'C': c,
                    'Mean MAE': mean_score,
                    'Std Dev': std_dev
                })

            results_df = pd.DataFrame(results)

            # Highlight the best and worst rows
            def highlight_best_and_worst(row):
                best = results_df["Mean MAE"].min()  # Best accuracy
                if row["Mean MAE"] == best:
                        return ['background-color: #48bb78'] * len(row)  # Green for best result
                else:
                        return [''] * len(row)  # No color for others
            
            styled_results_df = results_df.style.apply(highlight_best_and_worst, axis=1)

            st.write("Tuning Results Table")
            st.dataframe(styled_results_df)

            best_params = results_df.loc[results_df['Mean MAE'].idxmin()]
            st.write("Best Parameters:")
            st.json({
                'Kernel': best_params['Kernel'],
                'C': best_params['C'],
                'Mean MAE': round(best_params['Mean MAE'], 4)
            })

            # Train best model
            best_model = SVR(kernel=best_params['Kernel'], C=best_params['C'])
            best_model.fit(X, y)

            # Save the model
            tuned_model_filename = f"Tuned_SVM_model.joblib"
            joblib.dump(best_model, tuned_model_filename)

            # Provide download button for model
            st.download_button(
                label="Download the Tuned Model",
                data=open(tuned_model_filename, "rb").read(),
                file_name=tuned_model_filename,
                mime="application/octet-stream",
            )



