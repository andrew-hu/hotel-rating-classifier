import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sborn
from enum import Enum, unique
from collections import defaultdict
import time
import math

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.neighbors import RadiusNeighborsRegressor

CACHE_FILE_NAME = "hotel_reviews_cache.h5"
CSV_FILE = "Hotel_Reviews.csv"


def _load_dataset(file):
    """
    Load CSV file if cache does not exist,               (~5.8 seconds)
    If cache already exists then load dataset from cache (~1.9 seconds)
    @param file : CSV file to open and load
    """

    # If cache already exists, load from cache
    if os.path.exists(CACHE_FILE_NAME):
        dframe = pd.read_hdf(CACHE_FILE_NAME, "dataset")
    else:
        dframe = pd.DataFrame(pd.read_csv(file))
        dataset_cache = pd.HDFStore(CACHE_FILE_NAME)
        dataset_cache["dataset"] = dframe
        dataset_cache.close()

    return dframe


"""
def _generate_histograms(dframe, headers):
    """
    Generates histograms for each numerical attribute
    @param dframe : Dataframe to analyze
    """
    headers = dframe.columns

    fig = plt.figure(figsize=(30, 20))
    plt.xlabel("Trip_distance" , fontsize=15)
    plt.ylabel("Frequency"     , fontsize=15)

    for h in headers:
        if is_numeric_dtype(dframe[h]):
            column_min = dframe.at[dframe[h].idxmin(), h]
            column_max = dframe.at[dframe[h].idxmax(), h]
            plt.xlim([column_min, column_max])

            dframe.hist(column=h)
            plt.savefig("histograms/" + h + ".png")
            plt.close()


def _generate_heatmap(dframe):
    """
    Creates and opens a heat map diagram
    @param dframe : Dataframe to analyze
    """
    corr = (dframe.astype(float).corr())

    sborn.heatmap(
        corr, 
        cbar        = True, 
        square      = True, 
        annot       = True, 
        fmt         = '.2f', 
        annot_kws   = {'size': 5},
        xticklabels = corr.columns.values,
        yticklabels = corr.columns.values,
    )
    plt.savefig("heatmap.png")
    plt.close()


def _drop_categorical_attributes(dframe, headers):
    """
    Drops the attributes where the type is not numeric
    @param dframe : Dataframe to modify
    """
    headers = dframe.columns

    for h in headers:
        if not is_numeric_dtype(dframe[h]):
            dframe.drop(h, axis=1, inplace=True)


def _drop_specific_attributes(dframe, headers):
    """
    Drops a set of attributes that are probably not useful
    @param dframe : Dataframe to modify
    """
    DROP_LIST = [
        "Negative_Review",
        "Positive_Review",
        "Reviewer_Score",
        "lat",              # Should not be imputed
        "lng",              # Should not be imputed
        "Tags",             # Can encode this
    ]
    for header in DROP_LIST:
        dframe.drop(header, axis=1, inplace=True)


def _convert_strings_to_numerical(dframe, headers):
    """
    """
    for h in headers:
        if not is_numeric_dtype(dframe[h]):
            dframe[h] = LabelEncoder().fit_transform(dframe[h])


def _impute_missing_values(dframe, headers):
    """
    """
    for h in headers:
        if dframe[h].isnull().sum() > 0:
            print(h, dframe[h].isnull().sum())
            dframe[h].fillna(dframe[h].mean(), inplace=True)


def _run_dtree_classifier(num_features, training_data, testing_data, training_results, testing_results):
    """
    """
    classifier = RandomForestClassifier(
        bootstrap                = True, 
        class_weight             = None, 
        criterion                = 'gini',
        max_depth                = 15, 
        max_features             = num_features,
        max_leaf_nodes           = None,
        min_impurity_decrease    = 0.0, 
        min_impurity_split       = None,
        min_samples_leaf         = 50, 
        min_samples_split        = 40,
        min_weight_fraction_leaf = 0.0, 
        n_estimators             = 1000, 
        n_jobs                   = 2,
        oob_score                = True, 
        random_state             = 27, 
        verbose                  = 0,
        warm_start               = False
    )

    # Train the classifier
    classifier.fit(training_data, training_results)

    # Run cross validation
    scores = cross_val_score(classifier, testing_data, testing_results, cv=5)
    print(depth, scores.mean(), scores.std() * 2)


"""


def _run_random_forest_regressor(training_data, testing_data, training_results, testing_results):
    """
    """

    for m in [training_data, testing_data, training_results, testing_results]:
        assert np.any(np.isnan(m.as_matrix().astype(np.float))) == False, "{}".format(m)
        assert np.all(np.isfinite(m.as_matrix().astype(np.float))) == True, "{}".format(m)

    print("-" * 100)
    print(training_data.shape)
    print(training_data.columns)
    print("-" * 100)

    regressor = RandomForestRegressor(
        bootstrap                = True, 
        oob_score                = True, 
        max_depth                = 12, 
        max_features             = len(training_data.columns),
        max_leaf_nodes           = None,
        min_impurity_decrease    = 0.0, 
        min_impurity_split       = None,
        min_samples_leaf         = 10, 
        min_samples_split        = 20,
        min_weight_fraction_leaf = 0.0,
        n_estimators             = 100,
        n_jobs                   = 2,
        random_state             = 27, 
        verbose                  = 0,
        warm_start               = False
    )

    regressor.fit(training_data, training_results.values.ravel())

    # Run cross validation
    scores = cross_val_score(regressor, testing_data, testing_results.values.ravel(), scoring="neg_mean_squared_error", cv=10)
    print(scores.mean(), scores.std() * 2)


def _one_hot_encode(dframe, attribute):
    """
    """
    return pd.get_dummies(
        data    = dframe, 
        columns = [attribute]
    )


def _custom_one_hot_encode_tags(dframe, attribute):
    """
    To one hot encode all the tags
    """

    # Only works with Tags
    assert(attribute == "Tags")

    TAG_FREQUENCY_THRESHOLD = 5000

    def sanitize_tags(unsanitized):
        unsanitized = unsanitized.replace("[", "")            # Opening bracket
        unsanitized = unsanitized.replace("]", "")            # Closing bracket
        unsanitized = unsanitized.replace(" \', \' ", ",")    # Comma separators
        unsanitized = unsanitized.replace("\' ", "")          # Opening quote
        unsanitized = unsanitized.replace(" \'", "")          # Closing quote
        unsanitized = unsanitized.split(",")
        return unsanitized

    all_tags = { }

    # First pass : Goes through and populates all_tags
    for i, value in dframe.iterrows():
        tag_list = sanitize_tags(dframe.loc[i, attribute])
        for tag in tag_list:
            if tag not in all_tags:
                all_tags[tag] = 1
            else:
                all_tags[tag] += 1

    """
    plt.figure(figsize=(20, 10))
    plt.xticks([])
    plt.bar(list(all_tags.keys()), all_tags.values())
    plt.savefig("tag_histogram.png")
    """

    all_tags_reduced = set()

    # Reduce the original tags pool down to only the tags that have the minimum frequency
    for tag, count in all_tags.items():
        if count >= TAG_FREQUENCY_THRESHOLD:
            all_tags_reduced.add(tag)

    # For each tag add a column of zeros
    for tag in all_tags_reduced:
        dframe[tag] = np.zeros(len(dframe))

    five_percent = math.ceil(len(dframe) * 0.05)
    # Second pass : For each tag in the tag list, populate the unique tag with a 1
    for i, value in dframe.iterrows():
        tag_list = sanitize_tags(dframe.loc[i, attribute])
        if i % five_percent == 0:
            print(math.floor(i / len(dframe)) * 100, "%")
            sys.stdout.flush()
        for tag in tag_list:
            dframe.at[i, tag] = 1

    # Remove the Tags column since it has been split into other columns
    dframe.drop(attribute, axis=1, inplace=True)

    """
    index = 0
    tags_map_reduced = { }

    # Reduce the original tags pool down to only the tags that have a minimum frequency
    for tag, values in tags_map.items():
        if values["count"] >= TAG_FREQUENCY_THRESHOLD:
            tags_map_reduced[tag] = index
            index += 1

    # Second pass goes through and replaces the features with their vectored encoding
    for i, value in dframe.iterrows():
        attribute_list = sanitize_tags(dframe.loc[i, attribute])
        encoding = ["0" for i in range(len(tags_map_reduced))]
        # For each attribute that was chosen in the reduced attribute list
        for tag in attribute_list:
            if tag in tags_map_reduced:
                # Set the "bit" (more of a char) in the encoded string
                bit_index = tags_map_reduced[tag]
                encoding[bit_index] = "1"
        # Replace attribute with encoded attribute
        dframe.at[i, attribute] = "".join(encoding)
    """


def _remove_suffix_from_attribute(dframe, attribute):
    """
    """
    for i, value in dframe.iterrows():
        dframe.at[i, attribute] = dframe.at[i, attribute].split(" ")[0]


def _combine_two_attributes(dframe, attributes):
    """
    """
    if len(attributes) != 2:
        raise ValueError("Size of attributes was not 2 : {}".format(len(attributes)))
    dframe["coordinates"] = dframe[attributes[0]].astype(str) + "," + dframe[attributes[1]].astype(str)


def _drop_attribute(dframe, attribute):
    dframe.drop(attribute, axis=1, inplace=True)


CONTINUOUS_ATTRIBUTES = [
    "Additional_Number_of_Scoring",
    "Average_Score",
    "Review_Total_Negative_Word_Counts",
    "Total_Number_of_Reviews",
    "Review_Total_Positive_Word_Counts",
    "Total_Number_of_Reviews_Reviewer_Has_Given",
    "Reviewer_Score",
]

DROPPED_ATTRIBUTES = [
    "Positive_Review",
    "Negative_Review",
    "Tags",
    "lat",           # No point when using Hotel_Name
    "lng",           # No point when using Hotel_Name
    "Hotel_Address", # No point when using Hotel_Name
]

FACTORIZED_ATTRIBUTES = [
    "Hotel_Name",
    "Reviewer_Nationality",
    "Review_Date",
]

ONE_HOT_ATTRIBUTES = [
]


def _plot_missing_numbers(dframe):
    import missingno as msno
    msno.bar(
        dframe,
        sort     = True,
        figsize  = (30,8),
        color    = "#34495e",
        fontsize = 15,
        labels   = True
    )
    plt.show()


def main():
    """
    Main function
    """

    start_time = time.time()

    dframe = _load_dataset(CSV_FILE)

    # Only use first bit
    dframe = dframe.sample(
        n = 100000,
        random_state = 27,
    )

    for attribute in dframe.columns:

        if attribute == "days_since_review":
            _remove_suffix_from_attribute(dframe, attribute)
        
        elif attribute in FACTORIZED_ATTRIBUTES:
            dframe[[attribute]] = dframe[[attribute]].apply(lambda col : pd.factorize(col)[0])

        elif attribute in DROPPED_ATTRIBUTES:
            _drop_attribute(dframe, attribute)

        elif attribute in ONE_HOT_ATTRIBUTES:
            pass

        elif attribute in CONTINUOUS_ATTRIBUTES:
            pass

        else:
            raise ValueError("Attribute not bucketed : {}".format(attribute))

    # Get results column
    results = dframe["Reviewer_Score"].copy()

    _drop_attribute(dframe, "Reviewer_Score")

    # Convert results to string from float
    results = results.apply(lambda x: float(x))
    results = pd.DataFrame(results)

    # Split the dataset
    training_data, testing_data, training_results, testing_results = train_test_split(
        dframe, 
        results, 
        test_size=0.33, 
        random_state=27
    )

    _run_random_forest_regressor(
        training_data, 
        testing_data, 
        training_results, 
        testing_results
    )

    # _generate_histograms(dframe, headers)

    # _generate_heatmap(dframe)

    end_time = time.time()
    print("Program took {} to execute...".format(end_time - start_time))


if __name__ == '__main__':
    main()
