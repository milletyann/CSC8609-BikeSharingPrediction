# import systèmes
import argparse
import os
import warnings

# import mathématiques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# sklearn et lightgbm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
)

# métriques
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    make_scorer,
)

# modèles
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
)
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

### ------------------ VARIABLES ------------------- ###
models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Bagging": BaggingRegressor(),
    "Boosting": GradientBoostingRegressor(),
}
n_samples_to_plot = 2000

### ------------------ FONCTIONS ------------------- ###
warnings.filterwarnings("ignore")


# Data preprocessing
def prepare_data(display_plots):
    day_df = pd.read_csv("datasets/day.csv")
    hour_df = pd.read_csv("datasets/hour.csv")

    day_df["dteday"] = pd.to_datetime(day_df["dteday"]).astype(np.int64)
    hour_df["dteday"] = pd.to_datetime(hour_df["dteday"]).astype(np.int64)

    target_col = "cnt"
    cat_cols = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    num_cols = ["temp", "atemp", "hum", "windspeed"]

    if display_plots:
        numerical_cols = num_cols + ["hr", "casual", "registered"] + [target_col]
        categorical_cols = cat_cols + ["dteday"] + [target_col]

        plot_distrib_temp(day_df)
        plot_boxplots(hour_df)
        plot_imbalanced(hour_df)
        plot_pairplot(
            hour_df, numerical_cols, categorical_cols, target_col, n_samples_to_plot
        )
        plot_scatterplot(day_df, hour_df, target_col)

    # feature engineering
    # day.csv
    poly_day = PolynomialFeatures(degree=2, include_bias=False)
    poly_day_features = poly_day.fit_transform(day_df[["temp", "hum", "windspeed"]])
    poly_day_df = pd.DataFrame(
        poly_day_features, columns=poly_day.get_feature_names_out()
    ).drop(columns=["temp", "hum", "windspeed"])
    day_df = pd.concat([day_df, poly_day_df], axis=1)
    # hour.csv
    poly_hour = PolynomialFeatures(degree=2, include_bias=False)
    poly_hour_features = poly_hour.fit_transform(hour_df[["temp", "hum", "windspeed"]])
    poly_hour_df = pd.DataFrame(
        poly_hour_features, columns=poly_hour.get_feature_names_out()
    ).drop(columns=["temp", "hum", "windspeed"])
    hour_df = pd.concat([hour_df, poly_hour_df], axis=1)

    # rolling data
    day_df["rolling7d_avg"] = day_df["cnt"].rolling(window=7, min_periods=1).mean()
    hour_df["rolling24h_avg"] = hour_df["cnt"].rolling(window=24, min_periods=1).mean()

    # final datasets
    # day.csv
    X_day = day_df.drop(
        columns=[target_col, "dteday", "casual", "registered", "instant", "atemp"]
    )
    y_day = day_df[target_col]

    # hour.csv
    X_hour = hour_df.drop(
        columns=[target_col, "dteday", "casual", "registered", "instant", "atemp"]
    )
    y_hour = hour_df[target_col]

    # train_test_split
    X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(
        X_day, y_day, random_state=42, test_size=0.2
    )
    X_hour_train, X_hour_test, y_hour_train, y_hour_test = train_test_split(
        X_hour, y_hour, random_state=42, test_size=0.2
    )

    return (X_day, y_day), (X_hour, y_hour)


def ablationDataset(X_day, y_day, X_hour, y_hour):
    X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(
        X_day, y_day, random_state=42, test_size=0.2
    )
    X_hour_train, X_hour_test, y_hour_train, y_hour_test = train_test_split(
        X_hour, y_hour, random_state=42, test_size=0.2
    )

    # fabrication des données calendaires
    X_day_train_cal = X_day_train.drop(
        columns=[
            "weathersit",
            "temp",
            "hum",
            "windspeed",
            "temp^2",
            "temp hum",
            "temp windspeed",
            "hum^2",
            "hum windspeed",
            "windspeed^2",
        ]
    )
    X_day_test_cal = X_day_test.drop(
        columns=[
            "weathersit",
            "temp",
            "hum",
            "windspeed",
            "temp^2",
            "temp hum",
            "temp windspeed",
            "hum^2",
            "hum windspeed",
            "windspeed^2",
        ]
    )
    X_hour_train_cal = X_hour_train.drop(
        columns=[
            "weathersit",
            "temp",
            "hum",
            "windspeed",
            "temp^2",
            "temp hum",
            "temp windspeed",
            "hum^2",
            "hum windspeed",
            "windspeed^2",
        ]
    )
    X_hour_test_cal = X_hour_test.drop(
        columns=[
            "weathersit",
            "temp",
            "hum",
            "windspeed",
            "temp^2",
            "temp hum",
            "temp windspeed",
            "hum^2",
            "hum windspeed",
            "windspeed^2",
        ]
    )

    # fabrication des données météo
    X_day_train_meteo = X_day_train.drop(
        columns=["season", "yr", "mnth", "holiday", "weekday", "workingday"]
    )
    X_day_test_meteo = X_day_test.drop(
        columns=["season", "yr", "mnth", "holiday", "weekday", "workingday"]
    )
    X_hour_train_meteo = X_hour_train.drop(
        columns=["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday"]
    )
    X_hour_test_meteo = X_hour_test.drop(
        columns=["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday"]
    )

    return (
        X_day_train_cal,
        X_day_test_cal,
        X_hour_train_cal,
        X_hour_test_cal,
        X_day_train_meteo,
        X_day_test_meteo,
        X_hour_train_meteo,
        X_hour_test_meteo,
        y_day_train,
        y_day_test,
        y_hour_train,
        y_hour_test,
    )


# Helpers plotting functions
def plot_distrib_temp(day_df):
    print("")
    print("Histogrammes des distributions de température réelle et ressentie (day.csv)")
    _ = day_df[["temp", "atemp"]].hist(figsize=(15, 4))
    plt.title("Histogrammes de températures (10 bins)")
    plt.show()

    print("")
    print(
        "Histogrammes de la distribution de température réelle avec une meilleure précision (day.csv)"
    )
    day_df[["temp"]].plot.hist(bins=30)
    plt.title("Histogramme de température (30 bins)")
    plt.show()

    print("")
    print("Density plots des températures réelles et ressenties")
    day_df[["atemp", "temp"]].plot.kde()
    plt.title("Densité des variables température")
    plt.show()


def plot_boxplots(hour_df):
    print("")
    print("Boxplots de données météo pour détecter des outliers (hour.csv)")
    hour_df.boxplot(column=["windspeed", "temp", "atemp", "hum"])
    plt.title("Boxplots de données météo")
    plt.show()


def plot_imbalanced(hour_df):
    hour_df["holiday"].value_counts().plot(kind="bar")
    plt.xlabel = "Catégories"
    plt.ylabel = "Fréquence"
    plt.title("Équilibre de la classe holiday")
    plt.show()

    hour_df["workingday"].value_counts().plot(kind="bar")
    plt.xlabel = "Catégories"
    plt.ylabel = "Fréquence"
    plt.title("Équilibre de la classe workingday")
    plt.show()

    hour_df["weathersit"].value_counts().plot(kind="bar")
    plt.xlabel = "Catégories"
    plt.ylabel = "Fréquence"
    plt.title("Équilibre de la classe weathersit")
    plt.show()


def plot_pairplot(hour_df, num_cols, cat_cols, target_col, n_samples_to_plot):
    print("")
    print("Pairplots d'attributs numériques")
    _ = sns.pairplot(
        data=hour_df[:n_samples_to_plot],
        vars=num_cols,
        hue=target_col,
        plot_kws={"alpha": 0.3},
        height=3,
        diag_kind="hist",
        diag_kws={"bins": 30},
    )
    plt.show()

    print("")
    print("Pairplots d'attributs catégoriques")
    _ = sns.pairplot(
        data=hour_df[:n_samples_to_plot],
        vars=cat_cols,
        hue=target_col,
        plot_kws={"alpha": 0.3},
        height=3,
        diag_kind="hist",
        diag_kws={"bins": 30},
    )
    plt.show()


def plot_scatterplot(day_df, hour_df, target_col):
    print("")
    print("Scatterplot des locations en fonctions de la date")
    _ = sns.scatterplot(
        x="dteday",
        y="cnt",
        data=day_df,
        hue=target_col,
        alpha=0.5,
    )
    plt.show()

    print("")
    print("Scatterplot des locations en fonctions des vacances")
    _ = sns.scatterplot(
        x="holiday",
        y="cnt",
        data=hour_df,
        hue=target_col,
        alpha=0.5,
    )
    plt.show()


def displayLRWeights(X_train, X_test, y_train, y_test):
    linear_weights = []
    k_outer = 5
    k_inner = 3
    outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=42)
    linear_results = []
    for train_idx, val_idx in outer_cv.split(X_train):
        X_train_fold, X_val_fold = (
            X_train.iloc[train_idx],
            X_train.iloc[val_idx],
        )
        y_train_fold, y_val_fold = (
            y_train.iloc[train_idx],
            y_train.iloc[val_idx],
        )

        model = LinearRegression()
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_val_fold)
        rmse = root_mean_squared_error(y_val_fold, predictions)
        r2 = r2_score(y_val_fold, predictions)

        linear_results.append({"RMSE": rmse, "R2": r2})
        linear_weights.append(model.coef_)

    # Conversion en dataframe
    linear_weights_df = pd.DataFrame(linear_weights, columns=X_train.columns)

    # Plot des poids
    linear_weights_mean = linear_weights_df.mean().sort_values()
    plt.figure(figsize=(12, 8))
    linear_weights_mean.plot(kind="bar")
    plt.title("Coefficients moyens de la régression linéaire sur la CV")
    plt.show()


def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=42,
    )
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training error")
    plt.plot(train_sizes, val_scores_mean, "o-", color="g", label="Validation error")
    plt.title(f"Learning Curve: {title}")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


# Models testing/evaluating functions
def evaluate_model(model, X_train, y_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    RMSE = cross_val_score(
        model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf
    )
    R2 = cross_val_score(model, X_train, y_train, scoring="r2", cv=kf)
    return -1 * np.mean(RMSE), np.mean(R2)


def test_models(models, data_day, data_hour):
    X_day_train, X_day_test, y_day_train, y_day_test = data_day
    X_hour_train, X_hour_test, y_hour_train, y_hour_test = data_hour

    results_day = {}
    results_hour = {}

    for name, model in models.items():
        print("    Modèle en cours de traitement: {}".format(name))
        # day.csv
        rmse_day, r2_day = evaluate_model(model, X_day_train, y_day_train)
        results_day[name] = {"RMSE": rmse_day, "R²": r2_day}

        # hour.csv
        rmse_hour, r2_hour = evaluate_model(model, X_hour_train, y_hour_train)
        results_hour[name] = {"RMSE": rmse_hour, "R²": r2_hour}

    print("")
    print("    Day Dataset Results:")
    for model, metrics in results_day.items():
        print(
            "        {}: RMSE={}, R²={}".format(
                model, round(metrics["RMSE"], 3), round(metrics["R²"], 3)
            )
        )

    print("\n    Hour Dataset Results:")
    for model, metrics in results_hour.items():
        print(
            "        {}: RMSE={}, R²={}".format(
                model, round(metrics["RMSE"], 3), round(metrics["R²"], 3)
            )
        )


def train_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Métriques
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("        RMSE: {}".format(rmse))
    print("        R^2: {}".format(r2))
    print("        MAPE: {}".format(mape))


# Models functions
def nested_cross_validation(
    model, param_grid, X, y, search_method, outer_folds=5, inner_folds=3
):
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    results = []

    # pour chaque fold extérieur
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # distinguer les types de search pour avoir une fonction unique
        if search_method == "grid":
            search = GridSearchCV(
                model,
                param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
        elif search_method == "random":
            search = RandomizedSearchCV(
                model,
                param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_iter=20,
                n_jobs=-1,
                random_state=42,
            )

        search.fit(X_train, y_train)

        # recup du meilleur estimateur pour le fold intér en cours
        # TODO: voir si c'est pas mieux de faire au most_frequent plutôt qu'au best_estimator
        best_model = search.best_estimator_

        # métriques calculées sur test set de ce fold (données jamais vues)
        test_rmse = root_mean_squared_error(y_test, best_model.predict(X_test))
        test_r2 = r2_score(y_test, best_model.predict(X_test))

        results.append(
            {
                "Best_Params": search.best_params_,
                "Test_RMSE": test_rmse,
                "Test_R2": test_r2,
            }
        )

    results_df = pd.DataFrame(results)
    # Sortir automatiquement les meilleurs paramètres
    best_params = results_df["Best_Params"].tolist()
    params = {
        key: Counter(d[key] for d in best_params).most_common(1)[0][0]
        for key in best_params[0]
    }
    return results_df, params


def get_ncv_params_grids():
    ridge_params = {"alpha": [0.001, 0.01, 0.1, 1.0]}
    boosting_params_day = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.3],
        "max_depth": [2, 3, 5],
    }
    random_forest_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_split": [3, 5, 7],
    }
    boosting_params_hour = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [5, 7, 10, 20],
    }
    svr_params_day = {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.15],
        "gamma": [0.1, 1],
        "kernel": ["rbf"],
    }
    svr_params_hour = {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.03, 0.1],
        "gamma": [0.1, 1],
        "kernel": ["rbf"],
    }

    return (
        ridge_params,
        boosting_params_day,
        random_forest_params,
        boosting_params_hour,
        svr_params_day,
        svr_params_hour,
    )


def load_models_params():
    boosting_params_day = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
    }
    boosting_params_hour = {
        "max_depth": 20,
        "n_estimators": 300,
        "learning_rate": 0.1,
        "lambda_l2": 0,
        "min_child_samples": 10,
    }
    svr_params_day = {
        "C": 100,
        "epsilon": 0.15,
        "gamma": 0.1,
        "kernel": "rbf",
    }
    svr_params_hour = {
        "C": 100,
        "epsilon": 0.03,
        "gamma": 0.1,
        "kernel": "rbf",
    }
    return (
        boosting_params_day,
        boosting_params_hour,
        svr_params_day,
        svr_params_hour,
    )


def main(display_plots, run_nested_crossval, dev_debug):
    # Préparation des données
    print("[INFO] Préparation des données...")
    data_day, data_hour = prepare_data(display_plots)

    # Séparer les données en split
    print("[INFO] Création des variables de données...")
    X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(
        data_day[0], data_day[1], random_state=42, test_size=0.2
    )
    X_hour_train, X_hour_test, y_hour_train, y_hour_test = train_test_split(
        data_hour[0], data_hour[1], random_state=42, test_size=0.2
    )

    # Lancer les expériences
    print("[INFO]: Tests de différents modèles...")
    test_models(
        models,
        (X_day_train, X_day_test, y_day_train, y_day_test),
        (X_hour_train, X_hour_test, y_hour_train, y_hour_test),
    )

    # Plot des poids de la régression linéaire
    if display_plots:
        displayLRWeights(X_day_train, X_day_test, y_day_train, y_day_test)

    # NCV
    if run_nested_crossval:
        print("[INFO]: Nested Cross Validation running...")
        (
            ridge_params,
            boosting_params_day,
            random_forest_params,
            boosting_params_hour,
            svr_params_day,
            svr_params_hour,
        ) = get_ncv_params_grids()

        # day.csv
        linear_results, ridge_params = nested_cross_validation(
            Ridge(), ridge_params, data_day[0], data_day[1], search_method="grid"
        )
        boosting_results_day, boosting_params_day = nested_cross_validation(
            GradientBoostingRegressor(),
            boosting_params_day,
            data_day[0],
            data_day[1],
            search_method="random",
        )
        svr_results_day, svr_params_day = nested_cross_validation(
            SVR(), svr_params_day, data_day[0], data_day[1], search_method="random"
        )

        # hour.csv
        random_forest_results, random_forest_params = nested_cross_validation(
            RandomForestRegressor(),
            random_forest_params,
            X_hour,
            y_hour,
            search_method="grid",
        )
        # Commentée car bug
        # boosting_results_hour, boosting_params_hour = nested_cross_validation(LGBMRegressor(),boosting_params_hour,data_hour[0],data_hour[1],search_method="random")
        boosting_params_hour = {
            "max_depth": 20,
            "n_estimators": 300,
            "learning_rate": 0.1,
            "lambda_l2": 0,
            "min_child_samples": 10,
        }
        svr_results_hour, svr_params_hour = nested_cross_validation(
            SVR(), svr_params_hour, data_hour[0], data_hour[1], search_method="random"
        )

        print("    Linear Regression Params (Day Dataset):")
        print(ridge_params)
        print("\n    Boosting Params (Day Dataset):")
        print(boosting_params_day)
        print("\n    SVR Params (Day Dataset):")
        print(svr_params_day)
        print("\n    Random Forest Params (Hour Dataset):")
        print(random_forest_params)
        print("\n    Boosting Params (Hour Dataset):")
        print(boosting_params_hour)
        print("\n    SVR Params (Hour Dataset):")
        print(svr_params_hour)
    else:
        boosting_params_day, boosting_params_hour, svr_params_day, svr_params_hour = (
            load_models_params()
        )

    # Learning Curves
    # Commentée car 'str is not callable' (fonctionne dans le notebook)
    # if display_plots:
    if False:
        print("[INFO]: Plotting Learning Curves...")
        # gradient boosting day.csv
        boosting_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.1, random_state=42
        )
        plot_learning_curve(
            boosting_model, X_day_train, y_day_train, "Gradient Boosting (Day Dataset)"
        )
        # lightgbm hour.csv
        lgbm_model = LGBMRegressor(
            n_estimators=300,
            max_depth=20,
            min_child_samples=5,
            lambda_l2=0.1,
            learning_rate=0.1,
            random_state=42,
        )
        plot_learning_curve(
            lgbm_model, X_hour_train, y_hour_train, "LGBM (Hour Dataset)"
        )

    # Initialisation des modèles finaux
    day_model = GradientBoostingRegressor(**boosting_params_day, random_state=42)
    hour_model = LGBMRegressor(**boosting_params_hour, random_state=42)
    svr_day = SVR(**svr_params_day)
    svr_hour = SVR(**svr_params_hour)

    # Étude d'ablation
    print("[INFO]: Étude d'ablation...")
    (
        X_day_train_cal,
        X_day_test_cal,
        X_hour_train_cal,
        X_hour_test_cal,
        X_day_train_meteo,
        X_day_test_meteo,
        X_hour_train_meteo,
        X_hour_test_meteo,
        y_day_train,
        y_day_test,
        y_hour_train,
        y_hour_test,
    ) = ablationDataset(data_day[0], data_day[1], data_hour[0], data_hour[1])
    print("")
    print("    Données météo supprimées:")
    train_evaluate(
        day_model,
        X_day_train_cal,
        X_day_test_cal,
        y_day_train,
        y_day_test,
    )
    print("")
    print("    Données calendaires supprimées:")
    train_evaluate(
        day_model,
        X_day_train_meteo,
        X_day_test_meteo,
        y_day_train,
        y_day_test,
    )
    print("")
    print("#######################################################")
    print("#######################################################")
    print("#####                                            ######")
    print("#####  Les performances des modèles de Boosting  ######")
    print("#####      avec features tronquées ne sont       ######")
    print("#####   plus les mêmes que dans le notebook de   ######")
    print("#####       développement (cause inconnue)       ######")
    print("#####                                            ######")
    print("#######################################################")
    print("#######################################################")
    print("")

    # Entraînement et évaluation des modèles finaux
    print("")
    print("[INFO]: Entraînement des modèles et évaluation...")
    print("")
    print("    Boosting (day.csv):")
    train_evaluate(day_model, X_day_train, X_day_test, y_day_train, y_day_test)
    print("    Boosting (hour.csv):")
    train_evaluate(hour_model, X_hour_train, X_hour_test, y_hour_train, y_hour_test)
    print("")
    print("#######################################################")
    print("#######################################################")
    print("#####                                            ######")
    print("#####  Les performances des modèles de Boosting  ######")
    print("#####     diffèrent légèrement de celles du      ######")
    print("#####         notebook de développement          ######")
    print("#####                                            ######")
    print("#######################################################")
    print("#######################################################")
    print("")

    print("[INFO]: Modèle supplémentaire...")
    print("")
    print("    SVR (day.csv):")
    train_evaluate(svr_day, X_day_train, X_day_test, y_day_train, y_day_test)
    print("    SVR (hour.csv):")
    train_evaluate(svr_hour, X_hour_train, X_hour_test, y_hour_train, y_hour_test)
    print("")
    print("#######################################################")
    print("#######################################################")
    print("#####                                            ######")
    print("#####    Les performances des SVR ne sont pas    ######")
    print("#####  aussi bonnes que dans le notebook de dev  ######")
    print("#####                                            ######")
    print("#######################################################")
    print("#######################################################")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer les expériences du projet.")
    # gérer l'argument de l'affichage des plots
    parser.add_argument(
        "--display-plots",
        action="store_true",
        default=False,
        help="Ajouter cet argument pour afficher les plots pendant les expériences",
    )
    # gérer l'argument des cross-validations croisées
    parser.add_argument(
        "--run-nested-crossval",
        action="store_true",
        default=False,
        help="Ajouter cet argument pour ajouter les cross-validation croisées pendant les expériences",
    )
    # argument dev-debug (pour activer un test à la fois)
    parser.add_argument(
        "--dev-debug",
        action="store_true",
        default=False,
        help="Ajouter cet argument pour ajouter les cross-validation croisées pendant les expériences",
    )
    args = parser.parse_args()

    main(args.display_plots, args.run_nested_crossval, args.dev_debug)
