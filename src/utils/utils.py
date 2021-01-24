import pandas as pd


def get_class_weights(train_dir):
    """
    Calculates class weights based on inverse class frequency

    Args:
        train_dir (str): Directory with subdirectories containing training images
    
    Returns:
        class_weight (dict): Class labels are encoded as dictionary keys, and
            with class weights as the corresponding values
    """
    class_freq_df = get_class_frequencies(train_dir)
    class_freq_df["weight"] = 1 / class_freq_df["freq"]
    class_freq_df["weight_scaled"] = (
        class_freq_df["weight"] / class_freq_df["weight"].sum()
    )
    for index, row in class_freq_df.iterrows():
        class_weight[index] = row["weight_scaled"]
    return class_weight


def get_class_frequencies(train_dir):
    """
    Counts class frequencies in training directory

    Args:
        train_dir (str): Directory with subdirectories containing training images
    
    Returns:
        class_freq_df (pd.DataFrame): Dataframe with columns "taxon" and "freq"
            denoting plankton groups and the corresponding number of images within
            the training directory, respectively
            
    """
    taxon = []
    num_images = []
    for subdir in Path(train_dir).iterdir():
        taxon.append(subdir.name)
        num_images.append(len(list(subdir.glob("*.jpg"))))

    class_freq_df = pd.DataFrame({"taxon": taxon, "freq": num_images})
    return class_freq_df
∏