import json
import sys

from pathlib import Path

from src.data_loading.load_from_table import get_x_arr_for_scikit_from_table
from src.pipelines.train_eval_pipelines import (
    train_eval_gridsearch_loocv_with_outer_n_loop,
)

# Determine the number of CPU cores to use
n_monte_carlo_jobs = 20

# UCM MEG study: 233 subjects in the AD spectrum selected from a large cohort.
# Groups: HC, SCD, MCI (non-converter, nC), MCI (converter, C), AD dementia.
# Source reconstruction applied to MEG recordings; generative effective connectivity
# estimated via multivariate Ornstein-Uhlenbeck modelling (Berjaga-Buisan et al. 2025).
# Features: FDT violation and Entropy Production on the GEC matrix.
data_files = {
    "meg_power_conn": "UCM_power_conn_features_ML_ready.csv",
    "meg_all": "UCM_power_conn_gec_env_features_ML_ready.csv",
    "meg_thermo": "UCM_gec_env_features_ML_ready.csv",
    "gene_meg_thermo": "UCM_gene_gec_env_features_ML_ready.csv",
    "gene_meg_all": "UCM_gene_all_neurophys_features_ML_ready.csv",
    "gmv": "UCM_gmv_features_ML_ready.csv",
    "gmv_meg_thermo": "UCM_gmv_gec_env_features_ML_ready.csv",
}

classifier = "LogReg"
classifications = [
    "HC_vs_AD",
    "HC_vs_MCI_nC",
    "HC_vs_MCI_C",
    # "SCD_vs_AD",
    "SCD_vs_MCI_nC",
    "SCD_vs_MCI_C",
    "MCI_nC_vs_MCI_C",
    "MCI_C_vs_AD",
]

parameter_filenames = {
    feature_set: f"parameters_{classifier}_{'UCMGMV' if feature_set == 'gmv' else 'MEG'}.json"
    for feature_set in data_files
}

group_labels = {
    "HC": "HC", "SCD": "SCD", "MCI_nC": "MCI (nC)", "MCI_C": "MCI (C)", "AD": "AD"
}

if __name__ == "__main__":
    # Main execution
    path_repo = Path(Path(__file__).parent / ".." / "..").resolve()
    excel_folder = path_repo / "Data" / "fdt-eeg"
    param_folder = path_repo / "Parameters"

    # Optional CLI arg: index into the classifications list (for SLURM job arrays)
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        run_classifications = [classifications[idx]]
    else:
        run_classifications = classifications

    for classification in run_classifications:
        for data_type in data_files.keys():
            print(f"Training {classifier} with parameters_{classifier}.json")

            results_folder = (
                    path_repo / "Results" / "3d_gs_classification_ucm_meg_fdt_gec" /
                    classifier / classification / data_type
            )
            results_folder.mkdir(exist_ok=True, parents=True)

            data_file = excel_folder / data_files[data_type]

            # Load the parameters file
            param_file = path_repo / "Parameters" / parameter_filenames[data_type]
            with open(param_file, "r") as file:
                parameters = json.load(file)
            parameters["N_JOBS"] = 1  # For MRMR and other processes, best keep at 1

            # Parse classification string (e.g. "HC_vs_MCI_nC") into group labels
            group_a_key, group_b_key = classification.split("_vs_")
            parameters["GROUPS"] = {
                group_labels[group_a_key]: 0, group_labels[group_b_key]: 1
            }

            print(f"Training N fold with {classifier} Grid Search and LOOCV {data_file} Data")

            # We load the data from the file
            x, y, feature_columns = get_x_arr_for_scikit_from_table(
                data_file,
                parameters["GROUPS"],
                parameters["ID_KEY"],
                parameters["GROUP_KEY"]
            )

            # And we run the train_eval_knn gridsearch which performs gridsearch,
            # trains on best parameters, and evaluates the model and reports the
            # performances.
            train_eval_gridsearch_loocv_with_outer_n_loop(
                x, y, feature_columns, parameters, results_folder,
                n=20,
                n_monte_carlo_jobs=n_monte_carlo_jobs,
            )
