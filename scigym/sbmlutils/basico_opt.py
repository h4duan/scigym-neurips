import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from basico import *  # noqa
from basico.callbacks import create_default_handler
from basico.petab import load_petab  # noqa
from petab.v1 import Problem  # noqa

from scigym.api import ModifySpeciesAction
from scigym.sbml import SBML


def sample_uniform(low, high, size=1):
    """Sample from a uniform distribution."""
    return np.random.uniform(low, high, size)


def make_petab_problem_files(
    pred_sbml: SBML,
    inco_sbml: SBML,
    list_of_actions_and_obs: List[Tuple[List[ModifySpeciesAction], pd.DataFrame]],
    fit_all_parameters=False,
):
    # STEP 1: Figure out which parameters are in pred_sbml but not in inco_sbml
    if fit_all_parameters:
        params_to_fit = pred_sbml.get_parameter_ids()
    else:
        params_to_fit = set(pred_sbml.get_parameter_ids()) - set(inco_sbml.get_parameter_ids())

    params_values = pred_sbml.get_initial_parameter_values()
    print(f"Parameters to fit: {params_to_fit}")
    print(f"All Parameters: {pred_sbml.get_parameter_ids()}")

    if len(params_to_fit) == 0:
        raise ValueError("No parameters to fit. Please check the SBML files.")

    # STEP 2: Create tables for the PETAB problem
    parameter_rows = []
    condition_rows = []
    observable_rows = []
    measurement_rows = []
    noise_formula_weights = {}
    observable_species_ids = pred_sbml.get_species_ids()
    default_condition_row = {sid: None for sid in pred_sbml.get_species_ids()}

    # Create parameter rows for each parameter
    for param_id in pred_sbml.get_parameter_ids():
        estimate = param_id in params_to_fit
        param_value = params_values.get(param_id, 1.0)
        parameter_rows.append(
            {
                "parameterId": param_id,
                "parameterScale": "log10",
                "lowerBound": 0,
                "upperBound": 10000,
                # "lowerBound": 0.0,
                # "upperBound": 100000.0,
                "nominalValue": float(param_value),
                "estimate": int(estimate),
            }
        )

    for i, (actions, obs) in enumerate(list_of_actions_and_obs):
        # Create condition id
        condition_id = f"condition_{i+1}"

        # For each action, create a row with the same condition id, setting its initial concentration
        for action in actions:
            condition_rows.append(
                {
                    "conditionId": condition_id,
                    **default_condition_row,
                    action.species_id: float(action.value),
                }
            )

        # Calculate the noise formula weights for each species column in obs
        for sid in obs.columns:
            if sid == "time":
                continue
            l2_norm = math.sqrt(sum(map(lambda x: pow(x, 2), obs[sid])))
            inv_l2_norm = 1 / l2_norm if l2_norm != 0 else 1
            noise_formula_weights[f"ex_{i+1}_sp_{sid}"] = inv_l2_norm

        # Reshape the dataframe to long format (excluding 'time' column)
        melted_obs = obs.melt(id_vars=["time"], var_name="observableId", value_name="measurement")

        # Skip any 'time' entries that might have been caught
        melted_obs = melted_obs[melted_obs["observableId"] != "time"]

        # Rename the observableId to match the expected format
        rename_obs_id_fn = lambda x: f"ex_{i+1}_sp_{x}"
        melted_obs["observableId"] = melted_obs["observableId"].apply(rename_obs_id_fn)

        # Add conditionId to melted_obs
        melted_obs["simulationConditionId"] = condition_id

        # Add all rows to measurement_rows at once
        measurement_rows.extend(melted_obs.to_dict("records"))

    # Create observable rows for each observed species id
    for species_id in observable_species_ids:
        for i in range(len(list_of_actions_and_obs)):
            observable_id = f"ex_{i+1}_sp_{species_id}"
            noise_formula = noise_formula_weights[observable_id]
            observable_rows.append(
                {
                    "observableId": observable_id,
                    "observableFormula": species_id,
                    "noiseFormula": noise_formula,
                }
            )

    # Make dataframes and save as temporary .tsv files
    df_parameter = pd.DataFrame(parameter_rows)
    df_condition = pd.DataFrame(condition_rows)
    df_observable = pd.DataFrame(observable_rows)
    df_measurement = pd.DataFrame(measurement_rows)

    # Create a temporary directory to store the files
    temp_dir = tempfile.mkdtemp()

    # Save sbml file to the temporary directory
    pred_sbml.save(os.path.join(temp_dir, "model.xml"))

    # Save each dataframe to a .tsv file in the temporary directory
    df_parameter.to_csv(os.path.join(temp_dir, "petab_parameters.tsv"), sep="\t", index=False)
    df_condition.to_csv(os.path.join(temp_dir, "petab_conditions.tsv"), sep="\t", index=False)
    df_observable.to_csv(os.path.join(temp_dir, "petab_observables.tsv"), sep="\t", index=False)
    df_measurement.to_csv(os.path.join(temp_dir, "petab_measurements.tsv"), sep="\t", index=False)

    # Create a PEtab YAML file
    petab_yaml = {
        "format_version": 1,
        "parameter_file": "petab_parameters.tsv",
        "problems": [
            {
                "condition_files": ["petab_conditions.tsv"],
                "observable_files": ["petab_observables.tsv"],
                "measurement_files": ["petab_measurements.tsv"],
                "sbml_files": ["model.xml"],
            }
        ],
    }
    with open(os.path.join(temp_dir, "petab_problem.yaml"), "w") as f:
        yaml.dump(petab_yaml, f)

    # Create a PEtab problem object
    # petab_problem = Problem.from_yaml(os.path.join(temp_dir, 'petab_problem.yaml'))

    # Validate the PEtab problem
    # validation_results = petab_problem.validate()
    # if len(validation_results) > 0:
    #     raise ValueError(f"PEtab problem validation failed: {validation_results}")

    # Return the PEtab problem and the temporary directory
    return temp_dir


def get_path_to_copasi_file_from_petab_dir(temp_dir: str):
    import subprocess

    result = subprocess.run(
        ["copasi_petab_import", str(Path(temp_dir) / "petab_problem.yaml"), temp_dir]
    )
    assert result.returncode == 0, f"Failed to run copasi_petab_import: {result.stderr}"
    cps_file = Path(temp_dir) / "petab_problem.yaml.cps"
    # cps_file = load_petab(
    #     str(Path(temp_dir) / "petab_problem.yaml"),
    #     temp_dir,
    # )
    assert Path(cps_file).exists(), f"Failed to create CPS file: {cps_file}"
    return cps_file


def read_into_copasi_and_optimize(
    cps_file: Path, method="Levenberg - Marquardt", **kwargs
) -> pd.DataFrame:
    try:
        model = load_model(cps_file)
        fit_info = get_fit_parameters(model)
        assert fit_info is not None, "Failed to get fit parameters from Copasi model"
        print(f"Fitting the following parameters")
        print(fit_info.head())
        # print(get_task_settings(task_parameterestimation.TASK_PARAMETER_ESTIMATION))
        # print(get_experiment_mapping(0))
        # breakpoint()
        create_default_handler()
        results_df = run_parameter_estimation(
            method=method,
            update_model=True,
            calculate_statistics=True,
            **kwargs,
        )
        assert results_df is not None, "Failed to run parameter estimation"
        print(f"Results")
        print(results_df.head())
        statistics = get_fit_statistic(include_fitted=True)
        print(f"Fit Statistics")
        print(statistics["variables"])
        return results_df
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise e


def fit_parameter_using_basico(
    pred_sbml: SBML,
    inco_sbml: SBML,
    list_of_actions_and_obs: List[Tuple[List[ModifySpeciesAction], pd.DataFrame]],
    fit_all_parameters=False,
    method=task_parameterestimation.PE.LEVENBERG_MARQUARDT,
    settings={
        "method": {
            "name": "Levenberg - Marquardt",
            "Iteration Limit": 5000,
            "Tolerance": 1e-18,
        }
    },
    **kwargs,
) -> pd.DataFrame:
    # Create a PEtab problem
    temp_dir = make_petab_problem_files(
        pred_sbml,
        inco_sbml,
        list_of_actions_and_obs,
        fit_all_parameters,
    )
    # Get the copasi file from the temporary PEtab directory
    cps_file = get_path_to_copasi_file_from_petab_dir(temp_dir)
    # Execute the optimization in copasi
    results_df = read_into_copasi_and_optimize(cps_file, method=method, settings=settings, **kwargs)
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    return results_df


def create_fitting_function(inco_sbml: str, list_of_data: List[pd.DataFrame]):
    def fit_parameters(
        pred_sbml: str,
    ) -> str:
        pred_sbml_mod = SBML(pred_sbml)
        inco_sbml_mod = SBML(inco_sbml)
        list_of_actions_and_obs = []
        for data in list_of_data:
            species_names = [col for col in data.columns if col != "Time"]
            initial_values = {species: data.loc[0, species] for species in species_names}
            perturbations = [
                ModifySpeciesAction(sid, value=noise) for sid, noise in initial_values.items()
            ]
            df_obs = data.rename(columns={"Time": "time"})
            list_of_actions_and_obs.append((perturbations, df_obs))
        results_df = fit_parameter_using_basico(
            pred_sbml_mod, inco_sbml_mod, list_of_actions_and_obs
        )
        # Update predicted model with fitted parameters
        for id, row in results_df.iterrows():
            param_id = str(id).removeprefix("Values[").removesuffix("]")
            param_value = row["sol"]
            param: libsbml.Parameter = pred_sbml_mod.model.getParameter(param_id)
            print(f"Initial value for {param.getId()}: {param.getValue()}")
            print(f"Updated value for {param.getId()}: {param_value}")
            param.setValue(param_value)
        return pred_sbml_mod.to_string()

    return fit_parameters
