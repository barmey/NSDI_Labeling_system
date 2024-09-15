# Data Processing
import time

import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt

import utils
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def convert_dict_to_df(dict_classiciation_devices):
    features_dict = utils.enriched_field_to_field_dict.keys()
    features = [key for key in features_dict]
    labels = utils.read_csv_single_column_to_list(utils.types_path)
    labels = [label.lower() for label in labels]

    num_labels = len(labels)
    num_features = len(features)

    columns = [f"{feature}_{label}" for feature in features for label in labels]
    columns.append('device_name')
    data = []
    for device in dict_classiciation_devices:
       data.append(dict_classiciation_devices[device] + [device])

    df = pd.DataFrame(data, columns=columns)
    return df

def train_decision_tree_XGBClassifier(df_devices):
    data = df_devices
    # Separate features and labels
    X = data.drop(columns=['function', 'device_name'])  # Dropping both 'function' and 'device_name'
    y_str = data['function']

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the labels
    y_encoded = label_encoder.fit_transform(y_str)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost classifier
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def train_decision_tree(df_devices):
    data = df_devices
    # Separate features and labels
    X = data.drop(columns='function')
    X = X.drop(columns='device_name')
    y_str = data['function']

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the labels
    Y = label_encoder.fit_transform(y_str)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    param_dist = {'n_estimators': randint(50, 500),
                  'max_depth': randint(1, 20)}

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf,
                                     param_distributions=param_dist,
                                     n_iter=30,
                                     cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)

    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_)

    # Generate predictions with the best model
    y_pred = best_rf.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    feature_names = list(X.columns)

    if False:
        # Create the confusion matrix
        # cm = confusion_matrix(y_test, y_pred)

        # ConfusionMatrixDisplay(confusion_matrix=cm).plot()

        importances = best_rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=feature_names)
        # Sort the feature importances in descending order
        forest_importances_sorted = forest_importances.sort_values(ascending=False)

        # Select the top 20 feature importances
        top_20_importances = forest_importances_sorted.head(20)

        # Get the standard deviations corresponding to the top 20 importances
        #std_top_20 = std[forest_importances_sorted.index][:20]  # Slice the std array using top 20 importance indices
        std_top_20 = [std[feature] for feature in top_20_importances.index]

        fig, ax = plt.subplots()
        top_20_importances.plot.bar(yerr=std_top_20, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

    result = permutation_importance(
        best_rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    #elapsed_time = time.time() - start_time
    #print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    #print("Precision:", precision)
    #print("Recall:", recall)

    if False:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    for i in range(3):
        tree = best_rf.estimators_[i]
        dot_data = export_graphviz(tree,
                                   feature_names=X_train.columns,
                                   filled=True,
                                   max_depth=3,
                                   impurity=False,
                                   proportion=True)
        graph = graphviz.Source(dot_data)
        #display(graph)
        graph.view()
    sleep(100)

