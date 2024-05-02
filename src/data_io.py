import sys
sys.path.append('../daaad_fixed/src/')
sys.path.append('../')

import numpy as np
import pandas as pd
from daaad_fixed.src.dataset.data_set import DataSet, DataReal
from typing import Tuple, List

def get_log_modal_mass_y(reaction_force_y: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """Computes the log modal mass of the y direction.

    Args:
        reaction_force_y: The reaction force in the y direction.
        frequency: The frequency of the mode.

    Returns:
        The log modal mass of the y direction.
    """
    log_modal_mass = 2 * np.log10(np.abs(reaction_force_y)) - 4 * np.log10(
        2 * np.pi * frequency
    )
    return log_modal_mass

def get_spiral_data(num_modes: int = 3, data_fraction: float=1, only_freqs: bool=False) -> Tuple[DataSet, List]:
    """Reads in the spiral data and returns a DataSet object, that can be used for training.

    Args:
        num_modes: How many modes are loaded as performance attributes. Defaults to 3.
        data_fraction: How much of the data should be loaded. Defaults to 1.
        only_freqs: Whether only frequencies or also modal masses should be considered. Defaults to False.

    Raises:
        ValueError: If data_fraction is not between 0 and 1.

    Returns:
        DataSet object containing the spiral data.
    """
    data_path = "../data/spiral-resonators/"
    input_data = pd.read_csv(data_path + "Inputs.csv", index_col="design")
    # Apply log to columns such that they are more uniform
    apply_log = ["cs_half_width", "cs_half_height", "spiral_turns", "cs_scale"]
    input_data[apply_log] = np.log10(input_data[apply_log])
    output_data = pd.read_csv(data_path + "Outputs.csv", index_col="design")
    # We get rid of the designs with only very high frequency modes.
    # These were the ones, where the spiral barely had an effect.
    valid_designs = (
        output_data[[f"M{imode + 1}_F" for imode in range(12)]].min(axis=1) < 8e3
    )
    input_data = input_data[valid_designs]
    output_data = output_data[valid_designs]
    # Load fraction of data
    if data_fraction != 1:
        if data_fraction <= 0 or data_fraction > 1:
            raise ValueError("Value must be between 0 and 1.")
        num_data_points = int(len(input_data) * data_fraction)
        random_index = np.sort(
            np.random.choice(
                input_data.index.to_numpy(), num_data_points, replace=False
            )
        )
        input_data = input_data.loc[random_index]
        output_data = output_data.loc[random_index]

    num_modes_total = 12
    freq_columns = [f"M{i+1}_F" for i in range(num_modes_total)]
    reaction_force_columns = [f"M{i+1}_RFY" for i in range(num_modes_total)]
    freqs = output_data[freq_columns]
    reaction_force_y = output_data[reaction_force_columns]
    # Compute log modal mass
    log_modal_mass_y = get_log_modal_mass_y(
        reaction_force_y.to_numpy(), freqs.to_numpy()
    )
    # First we find the modes with the highest modal mass
    modes_of_interest = np.argsort(log_modal_mass_y, axis=1)[:, -num_modes:]
    freqs_of_interest = np.take_along_axis(freqs.to_numpy(), modes_of_interest, axis=1)
    # Then we sort the modes by frequency
    sort_idx = np.argsort(freqs_of_interest, axis=1)
    modes_of_interest = np.take_along_axis(modes_of_interest, sort_idx, axis=1)
    # Construct dictionary of design variables
    design_variables = {}
    for column in input_data.columns:
        design_variables[column] = DataReal.from_data(
            input_data[column].to_numpy(), column
        )

    # Construct dictionary of performance attributes
    performance_attributes = {}
    for imode in range(num_modes):
        mode_freqs = output_data[freq_columns].to_numpy()[
            np.arange(output_data.shape[0]), modes_of_interest[:, imode]
        ]
        performance_attributes[f"log_M{imode+1}_F"] = DataReal.from_data(
            np.log10(mode_freqs), f"log_M{imode+1}_F"
        )
        if not only_freqs:
            log_modal_mass_y_tmp = log_modal_mass_y[
                np.arange(output_data.shape[0]), modes_of_interest[:, imode]
            ]
            performance_attributes[f"log_M{imode+1}_MMY"] = DataReal.from_data(
                log_modal_mass_y_tmp, f"log_M{imode+1}_MMY"
            )
    # Create dataset
    dataset = DataSet.from_features_list(
        list(design_variables.values()), list(performance_attributes.values())
    )
    return dataset, apply_log