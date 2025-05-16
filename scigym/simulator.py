import logging
import warnings
from typing import Callable, Dict, List, Optional

import libsedml
import numpy as np
import roadrunner as rr
from deprecation import deprecated
from tellurium.sedml.tesedml import (
    SEDMLCodeFactory,
    getKisaoStringFromVal,
    getKisaoValFromString,
)

from scigym.api import ExperimentResult
from scigym.sbml import SBML

logger = logging.getLogger(__name__)


def onEventTrigger(model: rr.ExecutableModel, eventIndex: int, eventId: str):
    print("event {} was triggered at time {}".format(eventId, model.getTime()))


def onEventAssignment(model: rr.ExecutableModel, eventIndex: int, eventId: str):
    print(dict(zip(model.getFloatingSpeciesIds(), model.getFloatingSpeciesConcentrations())))


def run_simulation(
    _rr: rr.RoadRunner,
    observed_species: List[str],
    observed_parameters: List[str],
    noise: float = 0.0,
    rm_concentration_brackets: bool = False,
    sed_simulation: Optional[libsedml.SedUniformTimeCourse] = None,
) -> ExperimentResult:
    # Add brackets around species for concentration matching
    observed_species_with_brackets = [f"[{name}]" for name in observed_species]

    # Set the simulation selections to the observed variables
    observed_variables = observed_species_with_brackets + observed_parameters
    _rr.timeCourseSelections = ["time"] + observed_variables

    if sed_simulation is not None:
        initialTime = sed_simulation.getInitialTime()
        outputStartTime = sed_simulation.getOutputStartTime()
        outputEndTime = sed_simulation.getOutputEndTime()
        numberOfSteps = sed_simulation.getNumberOfSteps()
    else:
        print("Using default simulation parameters")
        initialTime = 0
        outputStartTime = 0
        outputEndTime = 10
        numberOfSteps = 100

    try:
        # throw some points away
        if abs(outputStartTime - initialTime) > 1e-6:
            _rr.simulate(start=initialTime, end=outputStartTime, points=2)
        # real simulation
        result = _rr.simulate(start=outputStartTime, end=outputEndTime, steps=numberOfSteps)
    except Exception as e:
        # logger.error(f"Simulation failed with error: {str(e)}")
        raise

    colnames = result.colnames
    data = np.array(result.data)
    # data[data < 1e-25] = 0

    # Add time to the results
    experiment_time: List[float] = data.copy()[:, 0].tolist()  # type: ignore

    # Add the observed species data to the results
    time_series = {}
    for sid, sidwb in zip(observed_species, observed_species_with_brackets):
        if sidwb in colnames:
            i = colnames.index(sidwb)
            if i > -1:
                key = sid if rm_concentration_brackets else sidwb
                time_series[key] = data[:, i].tolist()

    # Add the observed parameters data to the results
    for pid in observed_parameters:
        try:
            i = colnames.index(pid)
            if i > -1:
                time_series[pid] = data[:, i].tolist()
        except ValueError:
            # logger.warning(f"Column {pid} not found in simulation results")
            continue

    return ExperimentResult(time=experiment_time, result=time_series, raw_result=result)


class Simulator:
    """Class for simulating experiments on SBML models using tellurium."""

    _rr: rr.RoadRunner
    event_ids: List[str]
    event_tracker: Dict[str, bool]
    simulation: libsedml.SedUniformTimeCourse

    def __init__(self, sbml: SBML, **kwargs) -> None:
        if not isinstance(sbml.sed_simulation, libsedml.SedUniformTimeCourse):
            raise ValueError(
                "The simulation configuration must be set on your SBML object to run the simulator."
            )
        self.sbml = sbml
        self.event_tracker = {}
        self.simulation = sbml.sed_simulation.clone()
        self._rr = SBML._get_rr_instance(sbml.to_string())
        self.event_ids = self._rr.model.getEventIds()

        Simulator._setup_integrator(
            self.simulation,
            self._rr,
            onTrigger=self.onEventTrigger,
            **kwargs,
        )
        # self._rr.setIntegrator('gillespie')

    def onEventTrigger(self, model: rr.ExecutableModel, eventIndex: int, eventId: str):
        self.event_tracker[eventId] = True

    def prepare_simulation(self) -> None:
        self._rr.resetSeed()
        self._rr.resetToOrigin()
        self.event_tracker = dict(zip(self.event_ids, [False] * len(self.event_ids)))

    def getSimulationParameters(self) -> Dict[str, float]:
        return dict(
            initialTime=self.simulation.getInitialTime(),
            outputStartTime=self.simulation.getOutputStartTime(),
            outputEndTime=self.simulation.getOutputEndTime(),
            numberOfSteps=self.simulation.getNumberOfPoints(),
        )

    def updateSimulationParameters(
        self,
        initialTime: Optional[float] = None,
        outputStartTime: Optional[float] = None,
        outputEndTime: Optional[float] = None,
        numberOfSteps: Optional[int] = None,
    ) -> None:
        """Update the simulation parameters"""
        if initialTime is not None:
            self.simulation.setInitialTime(initialTime)
        if outputStartTime is not None:
            self.simulation.setOutputStartTime(outputStartTime)
        if outputEndTime is not None:
            self.simulation.setOutputEndTime(outputEndTime)
        if numberOfSteps is not None:
            self.simulation.setNumberOfPoints(numberOfSteps)

    def run(
        self,
        observed_species: List[str] | None = None,
        observed_parameters: List[str] = [],
        noise: float = 0.0,
        rm_concentration_brackets: bool = False,
    ) -> ExperimentResult:
        """Run the simulation and return the results"""
        self.prepare_simulation()
        try:
            if not observed_species:
                observed_species = self.sbml.get_species_ids()
            else:
                observed_species = [
                    species
                    for species in observed_species
                    if species in self.sbml.get_species_ids()
                ]

            result = run_simulation(
                self._rr,
                observed_species=observed_species,
                observed_parameters=observed_parameters,
                noise=noise,
                rm_concentration_brackets=rm_concentration_brackets,
                sed_simulation=self.simulation,
            )
            return result
        except Exception as e:
            raise

    @staticmethod
    def _setup_integrator(
        simulation: libsedml.SedSimulation,
        _rr: rr.RoadRunner,
        onTrigger: Optional[Callable] = None,
        onAssignment: Optional[Callable] = None,
        force_cvode: bool = False,
    ):
        """Sets up the integrator for the simulation"""
        algorithm = simulation.getAlgorithm()
        if algorithm is None:
            warnings.warn("Algorithm missing on simulation, defaulting to 'cvode: KISAO_0000019'")
            algorithm = simulation.createAlgorithm()
            algorithm.setKisaoID(getKisaoStringFromVal(19))

        kisao = getKisaoValFromString(algorithm.getKisaoID())
        kisaoname = ""
        if algorithm.isSetName():
            kisaoname = " (" + algorithm.getName() + ")"

        # Check if algorithm is supported
        if force_cvode or not SEDMLCodeFactory.isSupportedAlgorithmForSimulationType(
            kisao=kisao, simType=libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE
        ):
            warnings.warn(
                "Algorithm {}{} unsupported for {} simulation '{}'.  Using CVODE.".format(
                    kisao, kisaoname, simulation.getElementName(), simulation.getId()
                )
            )
            algorithm = simulation.createAlgorithm()
            algorithm.setKisaoID(getKisaoStringFromVal(19))
            kisao = getKisaoValFromString(algorithm.getKisaoID())
            kisaoname = " (" + algorithm.getName() + ")"

        # set integrator/solver
        integratorName = SEDMLCodeFactory.getIntegratorNameForKisaoID(kisao)
        if not integratorName:
            raise RuntimeError("No integrator exists for {} in roadrunner".format(kisao))

        _rr.setIntegrator(integratorName)

        if integratorName == "gillespie":
            _rr.integrator.setValue("variable_step_size", False)

        # Setup event listener using class-level callbacks
        eventIds = _rr.model.getEventIds()
        for eid in eventIds:
            try:
                e = _rr.model.getEvent(eid)
                if onTrigger is not None:
                    e.setOnTrigger(onTrigger)
                if onAssignment is not None:
                    e.setOnAssignment(onAssignment)
            except Exception as ex:
                logger.warning(f"Failed to set event listener for {eid}: {str(ex)}")

        # Set solver-specific options
        if kisao == 288:  # BDF
            _rr.integrator.setValue("stiff", True)
        elif kisao == 280:  # Adams-Moulton
            _rr.integrator.setValue("stiff", False)

        # integrator/solver settings (AlgorithmParameters)
        for par in algorithm.getListOfAlgorithmParameters():
            try:
                pkey = SEDMLCodeFactory.algorithmParameterToParameterKey(par)
                # only set supported algorithm paramters
                if pkey:
                    if pkey.dtype is str:
                        value = "'{}'".format(pkey.value)
                    else:
                        value = pkey.value

                    if value == str("inf") or pkey.value == float("inf"):
                        value = "float('inf')"

                    if pkey.key == "conserved_moiety_analysis":
                        if pkey.dtype == str and pkey.value.isdigit():
                            _rr.conservedMoietyAnalysis = bool(int(pkey.value))
                        else:
                            _rr.conservedMoietyAnalysis = pkey.value
                    else:
                        _rr.integrator.setValue(pkey.key, value)
            except Exception as ex:
                logger.warning(f"Failed to set algorithm parameter: {str(ex)}")

        if getattr(_rr, "conservedMoietyAnalysis", False) == True:
            _rr.conservedMoietyAnalysis = False

    @deprecated
    @staticmethod
    def return_experiment_data(
        sbml: SBML,
        observed_species: List[str] = [],
        observed_parameters: List[str] = [],
        rm_concentration_brackets: bool = False,
        noise=0.0,
    ) -> ExperimentResult:
        """
        Run a simulation based on the given SBML model and return data.

        Args:
            sbml: SBML model object with a tellurium model
            observed_species: Species to observe
            observed_parameters: Parameters to observe
            rm_concentration_brackets: Whether to remove brackets from species names
            noise: Amount of noise to add to the results

        Returns:
            ExperimentResult containing time series data
        """
        if not isinstance(sbml.sed_simulation, libsedml.SedUniformTimeCourse):
            raise ValueError(
                "The simulation configuration must be set on your SBML object to run the simulator."
            )
        try:
            _rr = rr.RoadRunner(sbml.to_string())
            Simulator._setup_integrator(sbml.sed_simulation, _rr)
        except Exception as e:
            logger.error(f"Error setting up simulation: {str(e)}")
            raise

        return run_simulation(
            _rr,
            observed_species=observed_species,
            observed_parameters=observed_parameters,
            noise=noise,
            rm_concentration_brackets=rm_concentration_brackets,
            sed_simulation=sbml.sed_simulation,
        )
