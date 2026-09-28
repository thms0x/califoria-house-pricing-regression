import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)


RANDOM_STATE = 42
TARGET_SCALE = 100_000


def print_section(title: str) -> None:
    print(f"\n{'-' * 15} {title} {'-' * 15}")


def load_data():
    X, y = fetch_california_housing(
        as_frame=True,
        data_home="data/",
        return_X_y=True,
    )

    return X, y


def explore_data(X: pd.DataFrame, y: pd.Series) -> None:
    print_section("Pre-Training Data Analysis")

    print("\nFeatures:")
    print(X.head())

    print("\nShape:")
    print(X.shape)

    print("\nInfo:")
    X.info()

    print("\nFeature statistics:")
    print(X.describe())

    print("\nMissing values:")
    print(X.isna().sum())

    print("\nTarget preview:")
    print(y.head())

    print("\nTarget statistics:")
    print(y.describe())


def build_models() -> dict:
    return {
        "Dummy": DummyRegressor(
            strategy="median"
        ),

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def evaluate_models(models, X_train, y_train):
    print_section("Model Evaluation")

    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        mae = -scores.mean()

        results.append({
            "model": name,
            "cv_mae": mae,
            "cv_mae_dollars": mae * TARGET_SCALE,
        })

        print(
            f"{name:20} "
            f"${mae * TARGET_SCALE:,.0f}"
        )

    return pd.DataFrame(results)


def tune_gradient_boosting(X_train, y_train):
    print_section("Hyperparameter Tuning")

    model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE
    )

    param_distributions = {
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
            0.2,
        ],

        "max_iter": [
            100,
            200,
            300,
            500,
            800,
        ],

        "max_leaf_nodes": [
            15,
            31,
            63,
            127,
        ],

        "max_depth": [
            None,
            3,
            5,
            10,
            20,
        ],

        "min_samples_leaf": [
            10,
            20,
            30,
            50,
        ],

        "l2_regularization": [
            0.0,
            0.1,
            1.0,
            10.0,
        ],

        "max_features": [
            0.6,
            0.8,
            1.0,
        ],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        "\nBest CV MAE:",
        f"${-search.best_score_ * TARGET_SCALE:,.0f}",
    )

    return search.best_estimator_


def evaluate_final_model(
    model,
    X_test,
    y_test,
):
    print_section("Post-Training Prediction Analysis")

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    results = pd.DataFrame({
        "Actual Price": y_test * TARGET_SCALE,
        "Predicted Price": predictions * TARGET_SCALE,
    })

    print(results.head())

    print(
        f"\nTest MAE: "
        f"${mae * TARGET_SCALE:,.0f}"
    )

    return predictions


def plot_predictions(
    y_test,
    predictions,
):
    minimum = min(
        y_test.min(),
        predictions.min(),
    )

    maximum = max(
        y_test.max(),
        predictions.max(),
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.4,
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
    )

    plt.xlabel(
        "Actual Price"
    )

    plt.ylabel(
        "Predicted Price"
    )

    plt.title(
        "Actual vs Predicted House Prices"
    )

    plt.tight_layout()
    plt.show()


def main():
    X, y = load_data()

    explore_data(
        X,
        y,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    evaluation_results = evaluate_models(
        models,
        X_train,
        y_train,
    )

    print("\nModel comparison:")
    print(
        evaluation_results
        .sort_values("cv_mae")
        .reset_index(drop=True)
    )

    best_model = tune_gradient_boosting(
        X_train,
        y_train,
    )

    predictions = evaluate_final_model(
        best_model,
        X_test,
        y_test,
    )

    plot_predictions(
        y_test,
        predictions,
    )


if __name__ == "__main__":
    main()