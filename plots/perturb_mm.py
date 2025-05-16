import libsedml
from utils import *  # noqa

from scigym.api import *  # noqa
from scigym.sbml import SBML
from scigym.simulator import Simulator


def setup_sedml(sbml):
    """
    Set up the SED-ML document for the SBML model.
    """
    sedml_document = libsedml.SedDocument()
    sedml_document.setId("id69")
    tc = sedml_document.createUniformTimeCourse()
    tc.setInitialTime(0)
    tc.setOutputStartTime(0)
    tc.setOutputEndTime(90)
    tc.setNumberOfSteps(51)
    tc.setId("id69")
    alg = tc.createAlgorithm()
    alg.setKisaoID("KISAO:0000019")
    sbml.sedml_document = sedml_document
    sbml.sed_simulation = tc.clone()


def plot_data(data: ExperimentResult, save_path: str):
    """
    Plot the simulation data.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    assert data.result
    df = pd.DataFrame(
        {
            "time": data.time,
            **data.result,
        }
    )
    df = df.set_index("time")
    df = df.reset_index()
    df = df.melt(id_vars=["time"], var_name="species", value_name="M")
    fig, ax = plt.subplots(figsize=(3, 2.75))
    sns.lineplot(data=df, x="time", y="M", hue="species", ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def main():
    """
    Main function to run the simulation.
    """
    sbml = SBML("mm.xml")

    setup_sedml(sbml)

    if 1:
        simulation = Simulator(sbml)

        data = simulation.run(observed_species=sbml.get_species_ids())

        plot_data(data, "mm.baseline.svg")

    ic = sbml.get_initial_concentrations()

    perturbations = [
        NullifySpeciesAction(species_id="E"),
        ModifySpeciesAction(species_id="S", value=2e-6),
    ]

    int_sbml = sbml._apply_experiment_actions(sbml, perturbations)

    setup_sedml(int_sbml)

    simulation = Simulator(int_sbml)

    data = simulation.run(observed_species=int_sbml.get_species_ids())

    plot_data(data, "mm.perturb.svg")


if __name__ == "__main__":
    main()
