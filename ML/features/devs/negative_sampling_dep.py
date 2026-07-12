
# features/negative_sampling.py
import pandas as pd
import numpy as np

def generate_negative_samples(
    transactions, 
    sku_metadata, 
    customer_metadata,
    n_skus,
    max_negatives=20,
    category_weight=0.5,
    trending_weight=0.3,
    fallback_levels=["Town", "City", "State", "Global"],
    use_trending=True,
    use_category_relevance=True
):
    """
    Generates negative samples (non-purchases) using:
    - Adaptive sampling ratio
    - Hierarchical localized popularity (Town > City > State > Global)
    - Hard negative mining (trending/category-relevant SKUs)
    """
    # Precompute popularity stats once
    popularity_stats = _precompute_popularity_stats(transactions, sku_metadata, fallback_levels)
    
    # Precompute trending SKUs (last 4 weeks)
    trending_skus = (
        transactions[transactions["Week"] > pd.to_datetime(transactions["Week"].max()) - pd.Timedelta(weeks=4)]
        ["SKUID"].value_counts(normalize=True)
        .reindex(np.arange(n_skus), fill_value=0)
    ) if use_trending else None

    negative_samples = []

    # Group by (Week, CustomerID)
    grouped = transactions.groupby(["Week", "CustomerID"])
    
    for (week, customer_id), group in grouped:
        purchased_skus = group["SKUID"].unique()
        non_purchased = np.setdiff1d(np.arange(n_skus), purchased_skus)
        
        if len(non_purchased) == 0:
            continue  # No non-purchased SKUs to sample

        # Get customer location info
        customer_info = customer_metadata[customer_metadata["CustomerID"] == customer_id].iloc[0]
        location_weights = _get_location_weights(customer_info, popularity_stats, fallback_levels)

        # Combine weights: location + category + trending
        combined_weights = location_weights.copy()
        
        if use_category_relevance:
            category_mask = _get_category_mask(group, sku_metadata, non_purchased)
            combined_weights += category_weight * category_mask

        if use_trending:
            combined_weights += trending_weight * trending_skus.values[non_purchased]

        # Normalize final weights
        combined_weights = _normalize_weights(combined_weights)

        # Adaptive sampling ratio
        n_negatives = min(len(purchased_skus) * 2, max_negatives)

        # Sample negatives
        try:
            sampled_skus = np.random.choice(
                non_purchased,
                size=min(n_negatives, len(non_purchased)),
                p=combined_weights,
                replace=False
            )
        except ValueError:
            # Fallback to uniform sampling if weights invalid
            sampled_skus = np.random.choice(
                non_purchased,
                size=min(n_negatives, len(non_purchased)),
                replace=False
            )

        # Append to list
        for sku in sampled_skus:
            negative_samples.append([week, customer_id, sku])

    # Create DataFrame and merge with metadata
    neg_df = pd.DataFrame(negative_samples, columns=["Week", "CustomerID", "SKUID"])
    neg_df["Label"] = 0
    
    # Merge with metadata
    neg_df = neg_df.merge(sku_metadata, on="SKUID", how="left")
    neg_df = neg_df.merge(customer_metadata, on="CustomerID", how="left")
    
    return neg_df

# --------------------------
# Helper Functions
# --------------------------

def _precompute_popularity_stats(transactions, sku_metadata, fallback_levels):
    """Precompute popularity stats at multiple levels"""
    stats = {}
    for level in fallback_levels:
        if level == "Global":
            stats[level] = transactions["SKUID"].value_counts(normalize=True)
        else:
            stats[level] = (
                transactions.groupby(level)["SKUID"]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )
    return stats

def _get_location_weights(customer_info, popularity_stats, fallback_levels):
    """Get popularity weights based on customer's location"""
    for level in fallback_levels:
        if level == "Global":
            return popularity_stats[level].reindex(np.arange(1000), fill_value=0).values
        elif customer_info[level] in popularity_stats[level].index:
            return popularity_stats[level].loc[customer_info[level]].values
    return popularity_stats["Global"].reindex(np.arange(1000), fill_value=0).values

def _get_category_mask(group, sku_metadata, non_purchased):
    """Create mask for category relevance"""
    customer_categories = group.merge(sku_metadata, on="SKUID")["Category"].unique()
    category_mask = sku_metadata.set_index("SKUID").loc[non_purchased, "Category"].isin(customer_categories)
    return category_mask.astype(int).values

def _normalize_weights(weights):
    """Normalize weights, handle zero-sum cases"""
    total = weights.sum()
    if total == 0:
        return np.ones_like(weights) / len(weights)
    return weights / total



# ------------------------------ OLDER VERSIONS ------------------------------

def generate_negative_samples_v3(
    transactions, 
    sku_metadata, 
    customer_metadata,
    n_skus,
    max_negatives=20,           # Cap on negatives per customer-week
    category_weight=0.5,         # Weight for category relevance
    trending_weight=0.3,         # Weight for trending SKUs
    local_fallback="state"       # Fallback level if town/city missing
):
    """
    Generates negative samples (non-purchases) using:
    - Adaptive sampling ratio
    - Localized popularity (Town > City > State > Global)
    - Hard negative mining (trending/category-relevant SKUs)
    """
    # Precompute popularity stats at different levels
    def compute_local_popularity(group_col):
        return (
            transactions.groupby(group_col)["SKUID"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

    town_popularity = compute_local_popularity("Town")
    city_popularity = compute_local_popularity("City")
    state_popularity = compute_local_popularity("State")
    global_popularity = (
        transactions["SKUID"].value_counts(normalize=True)
    )

    all_skus = np.arange(n_skus)
    negative_samples = []

    # Group by (Week, CustomerID)
    grouped = transactions.groupby(["Week", "CustomerID"])
    
    for (week, customer_id), group in grouped:
        purchased_skus = group["SKUID"].unique()
        non_purchased = np.setdiff1d(all_skus, purchased_skus)
        
        if len(non_purchased) == 0:
            continue  # No non-purchased SKUs to sample

        # Get customer location
        customer_info = customer_metadata[
            customer_metadata["CustomerID"] == customer_id
        ].iloc[0]
        town = customer_info["Town"]
        city = customer_info["City"]
        state = customer_info["State"]

        # Determine popularity weights (hierarchical fallback)
        if town in town_popularity.index:
            popularity = town_popularity.loc[town]
        elif city in city_popularity.index:
            popularity = city_popularity.loc[city]
        elif state in state_popularity.index:
            popularity = state_popularity.loc[state]
        else:
            popularity = global_popularity

        # Normalize weights
        popularity = popularity.reindex(all_skus, fill_value=0)
        popularity = popularity / popularity.sum() if popularity.sum() > 0 else 1 / n_skus

        # Category relevance (weight SKUs in categories customer buys)
        customer_categories = group.merge(sku_metadata, on="SKUID")["Category"].unique()
        category_mask = sku_metadata.set_index("SKUID").loc[non_purchased, "Category"].isin(customer_categories)
        category_weights = np.where(category_mask, category_weight, 0)

        # Trending SKUs (last 4 weeks)
        recent_trend = (
            transactions[
                (transactions["Week"] > pd.to_datetime(week) - pd.Timedelta(weeks=4))
            ]["SKUID"]
            .value_counts(normalize=True)
            .reindex(all_skus, fill_value=0)
        )
        trending_weights = recent_trend.values * trending_weight

        # Combine weights
        combined_weights = (
            (1 - category_weight - trending_weight) * popularity.values +
            category_weights + 
            trending_weights
        )
        combined_weights = combined_weights / combined_weights.sum()

        # Adaptive sampling ratio
        n_positives = len(purchased_skus)
        n_negatives = min(n_positives * 2, max_negatives)  # 2x positives, capped

        # Sample negatives
        try:
            sampled_skus = np.random.choice(
                non_purchased,
                size=min(n_negatives, len(non_purchased)),
                p=combined_weights[non_purchased],
                replace=False
            )
        except ValueError:
            # Fallback to uniform sampling if weights invalid
            sampled_skus = np.random.choice(
                non_purchased,
                size=min(n_negatives, len(non_purchased)),
                replace=False
            )

        # Append to list
        for sku in sampled_skus:
            negative_samples.append([week, customer_id, sku])

    # Create DataFrame and merge with metadata
    neg_df = pd.DataFrame(negative_samples, columns=["Week", "CustomerID", "SKUID"])
    neg_df["Label"] = 0
    
    # Merge with metadata
    neg_df = neg_df.merge(sku_metadata, on="SKUID", how="left")
    neg_df = neg_df.merge(customer_metadata, on="CustomerID", how="left")
    
    return neg_df


def generate_negative_samples_v2(transactions, sku_metadata, customer_metadata, n_skus, ratio=10):
    """Create negative samples and merge with metadata"""
    all_skus = np.arange(n_skus)
    negative_samples = []
    
    grouped = transactions.groupby(["Week", "CustomerID"])
    for (week, customer_id), group in grouped:
        purchased_skus = group["SKUID"].unique()
        non_purchased = np.setdiff1d(all_skus, purchased_skus)

        if len(non_purchased) == 0:
            continue  # No non-purchased SKUs to sample

        # Compute global popularity weights
        sku_counts = transactions["SKUID"].value_counts(normalize=True)
        weights = sku_counts.reindex(all_skus, fill_value=0).values

        # Get weights for non-purchased SKUs
        p_weights = weights[non_purchased]
        total_weight = p_weights.sum()

        # Fall back to uniform sampling if no weight is available
        if total_weight == 0:
            p = None  # Uniform probability
        else:
            p = p_weights / total_weight  # Normalize

        # Sample negatives
        sampled_skus = np.random.choice(
            non_purchased,
            size=min(ratio * len(purchased_skus), len(non_purchased)),
            p=p,
            replace=False
        )

        for sku in sampled_skus:
            negative_samples.append([week, customer_id, sku])

    # Create DataFrame and merge with metadata
    neg_df = pd.DataFrame(negative_samples, columns=["Week", "CustomerID", "SKUID"])
    neg_df["Label"] = 0
    
    # Merge with metadata
    neg_df = neg_df.merge(sku_metadata, on="SKUID", how="left")
    neg_df = neg_df.merge(customer_metadata, on="CustomerID", how="left")
    
    return neg_df



def generate_negative_samples_v1(transactions, n_skus, ratio=10):
    """Create negative samples (non-purchases) for training"""
    all_skus = np.arange(n_skus)
    negative_samples = []
    
    grouped = transactions.groupby(["Week", "CustomerID"])
    for (week, customer_id), group in grouped:
        purchased_skus = group["SKUID"].unique()
        non_purchased = np.setdiff1d(all_skus, purchased_skus)

        if len(non_purchased) == 0:
            continue  # No non-purchased SKUs to sample

        # Compute global popularity weights
        sku_counts = transactions["SKUID"].value_counts(normalize=True)
        weights = sku_counts.reindex(all_skus, fill_value=0).values

        # Get weights for non-purchased SKUs
        p_weights = weights[non_purchased]
        total_weight = p_weights.sum()

        # Fall back to uniform sampling if no weight is available
        if total_weight == 0:
            p = None  # Uniform probability
        else:
            p = p_weights / total_weight  # Normalize

        # Sample negatives
        try:
            sampled_skus = np.random.choice(
                non_purchased,
                size=min(ratio * len(purchased_skus), len(non_purchased)),
                p=p,
                replace=False
            )
        except ValueError:
            # Fallback: if sampling fails (e.g., all probabilities zero or NaN),
            # fall back to uniform sampling again
            sampled_skus = np.random.choice(
                non_purchased,
                size=min(ratio * len(purchased_skus), len(non_purchased)),
                replace=False
            )

        for sku in sampled_skus:
            negative_samples.append([week, customer_id, sku])

    neg_df = pd.DataFrame(negative_samples, columns=["Week", "CustomerID", "SKUID"])
    neg_df["Label"] = 0
    return neg_df