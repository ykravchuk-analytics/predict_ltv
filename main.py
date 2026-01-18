import pandas as pd
import numpy as np


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



    return pd.DataFrame(rows)



def main():
    # 1. Read dataframe
    dataframe = read_file('data.csv') # TODO: cleanup

    # 1.1 Cleanup dataframe
    prepared_data = prepare_data(dataframe)

    # 2. Calculate simple ltv for all user lifetime
    simple_ltv, avarage_simple = calculate_simple_ltv(prepared_data)
    print(f"AVG for simple ltv is: {avarage_simple}")
    write_result('results/simple_ltv.csv', simple_ltv)

if __name__ == '__main__':
    main()
