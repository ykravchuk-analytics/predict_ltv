import pandas as pd
import numpy as np
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter
import matplotlib.pyplot as plt
from lifetimes.utils import summary_data_from_transaction_data, _customer_lifetime_value

DEFAULT_FREQ = "W"
HORIZONS_WEEKS = 52

def read_file(file_path):
    print(f"Reading file: {file_path}")

    return pd.read_csv(file_path)

def write_result(file_path, df):
    print(f"Writing result: {file_path}")

    df.to_csv(file_path, index=False)

def parse_product_id(product_id):
    # Format: weekly.2.49
    subscription, price_dollar, price_cents = product_id.split(".")

    return subscription, float(price_dollar) + int(price_cents) / 100

def prepare_data(df):
    """
    Following next structure of csv:
        user_id,event_timestamp,first_purchase_time,cohort_week,lifetime_weeks,product_id
    """
    # 1. Cleanup all rows where user_id is not defined/null
    result = df[df['user_id'].notnull()].copy()

    # 2. Parse product id into subscription, revenue
    parsed = result["product_id"].apply(parse_product_id)
    result["subscription_type"] = parsed.apply(lambda x: x[0])
    result["revenue"] = parsed.apply(lambda x: x[1]).astype(float)

    return result


def calculate_simple_ltv(df):
    """
    LTV grouped by user_id.

    Get TOTAL revenue by each user
    """
    result = df.copy().groupby(["user_id"], as_index=False)["revenue"].sum()

    total_rev = float(df["revenue"].sum())
    n_users = df["user_id"].nunique()

    return result, total_rev / n_users

def fit_lifetimes_bgnbd_gammagamma(df):
    observation_end = df["event_timestamp"].max()

    summary = summary_data_from_transaction_data(
        transactions=df,
        customer_id_col="user_id",
        datetime_col="event_timestamp",
        monetary_value_col="revenue",
        observation_period_end=observation_end,
        freq=DEFAULT_FREQ,
    )

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    # Gamma-Gamma only if there is enough users with freq > 0 and monetary > 0
    mask = (summary["frequency"] > 0) & (summary["monetary_value"] > 0)
    if mask.sum() >= 50:
        ggf = GammaGammaFitter(penalizer_coef=0.001)
        ggf.fit(summary.loc[mask, "frequency"], summary.loc[mask, "monetary_value"])
    else:
        ggf = None  # Not enough data for GG

    return bgf, ggf, summary


def ltv_predictive_curve_mean_lifetimes(
    bgf,
    ggf,
    summary,
    discount_rate_annual = 0.0,
):
    rows = []

    # discount per period. freq is week
    discount_rate = (1 + discount_rate_annual) ** (1 / HORIZONS_WEEKS) - 1

    for w in range(0, HORIZONS_WEEKS + 1):
        time = int(w)

        if ggf is not None:
            clv = _customer_lifetime_value(
                transaction_prediction_model=bgf,
                frequency=summary["frequency"],
                recency=summary["recency"],
                T=summary["T"],
                monetary_value=summary["monetary_value"],
                time=time,
                discount_rate=discount_rate,
                freq=DEFAULT_FREQ,
            )
            pred_mean = float(clv.mean())
        else:
            # fallback: expected purchases * avg monetary_value
            exp_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
                time, summary["frequency"], summary["recency"], summary["T"]
            )
            avg_money = summary["monetary_value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pred_mean = float((exp_purchases * avg_money).mean())

        rows.append({"lifetime_weeks": w, "ltv_pred_mean": pred_mean})

    return pd.DataFrame(rows)


def ltv_cohort_weekly_curve_actual(df):
    """
    LTV by cohorts:
      cohort_week, lifetime_weeks, ltv_actual_cum (cumulative revenue per user)
    """
    out = df.copy()

    cohort_sizes = out.groupby("cohort_week")["user_id"].nunique().rename("cohort_users")

    weekly_rev = (
        out.groupby(["cohort_week", "lifetime_weeks"])["revenue"]
        .sum()
        .reset_index()
        .merge(cohort_sizes.reset_index(), on="cohort_week", how="left")
    )
    weekly_rev["rev_per_user"] = weekly_rev["revenue"] / weekly_rev["cohort_users"]
    weekly_rev = weekly_rev.sort_values(["cohort_week", "lifetime_weeks"])

    weekly_rev["ltv_actual_cum"] = weekly_rev.groupby("cohort_week")["rev_per_user"].cumsum()
    weekly_rev = weekly_rev[weekly_rev["lifetime_weeks"] <= HORIZONS_WEEKS]

    return weekly_rev[["cohort_week", "lifetime_weeks", "ltv_actual_cum"]]


def ltv_cohort_mean_curve(cohort_curve_actual):
    mean_curve = cohort_curve_actual.groupby("lifetime_weeks", as_index=False)["ltv_actual_cum"].mean()
    return mean_curve.rename(columns={"ltv_actual_cum": "ltv_actual_mean"})


def plot_actual_vs_predicted_mean_curve(
    actual_mean,
    predicted_mean,
):
    merged = pd.merge(actual_mean, predicted_mean, on="lifetime_weeks", how="outer").sort_values("lifetime_weeks")
    merged = merged[merged["lifetime_weeks"] <= HORIZONS_WEEKS]

    plt.figure(figsize=(10, 6))
    plt.plot(merged["lifetime_weeks"], merged["ltv_actual_mean"], label="Actual LTV (mean cohorts)")
    plt.plot(merged["lifetime_weeks"], merged["ltv_pred_mean"], label="Predicted LTV (lifetimes)")
    plt.grid(True)
    plt.title("LTV curve: Actual vs Predicted")
    plt.xlabel("Lifetime (weeks)")
    plt.ylabel("Cumulative LTV per user")
    plt.legend()
    plt.show()


def main():
    # 1. Read dataframe
    dataframe = read_file('data.csv') # TODO: cleanup

    # 1.1 Cleanup dataframe
    prepared_data = prepare_data(dataframe)

    # 2. Calculate simple ltv for all user lifetime
    simple_ltv, avarage_simple = calculate_simple_ltv(prepared_data)
    print(f"AVG for simple ltv is: {avarage_simple}")
    write_result('results/simple_ltv.csv', simple_ltv)

    # 3. Calculate cohorts
    cohort_curve_actual = ltv_cohort_weekly_curve_actual(prepared_data)
    actual_mean_curve = ltv_cohort_mean_curve(cohort_curve_actual)

    # 4. Calculate predictive mean curve
    bgf, ggf, summary = fit_lifetimes_bgnbd_gammagamma(prepared_data)
    pred_mean_curve = ltv_predictive_curve_mean_lifetimes(
        bgf=bgf,
        ggf=ggf,
        summary=summary,
        discount_rate_annual=0.0,
    )
    write_result('results/summary_grammagamma.csv', summary)
    write_result('results/ltv_predictive_curve_mean.csv', pred_mean_curve)

    # 5. Visualize
    plot_actual_vs_predicted_mean_curve(actual_mean_curve, pred_mean_curve)

if __name__ == '__main__':
    main()
