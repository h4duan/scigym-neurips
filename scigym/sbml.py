from __future__ import annotations

import os
import random
import traceback
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple

import libsbml
import libsedml
import roadrunner
import tellurium as te
from deprecation import deprecated
from libsbml import Model, SBase, SBMLDocument

from scigym.api import (
    ExperimentConstraint,
    ModifyReactionAction,
    ModifySpeciesAction,
    NullifyReactionAction,
    NullifySpeciesAction,
)
from scigym.constants import DEFAULT_METADATA_REMOVAL_CONFIG
from scigym.exceptions import ApplyExperimentActionError, ParseExperimentActionError
from scigym.sbmlutils.scrambleIds import SIdScrambler
from scigym.utils import *  # noqa


class SBML:
    """
    Class for handling SBML (Systems Biology Markup Language) models.
    Provides methods for manipulating SBML files by removing or masking components.
    """

    model: Model
    document: SBMLDocument
    sed_simulation: libsedml.SedUniformTimeCourse
    sedml_document: libsedml.SedDocument | None = None

    def __init__(
        self,
        sbml_string_or_file: str,
        sedml_string_or_file: str | None = None,
    ):
        """
        Initialize with an SBML file path or string.

        Args:
            sbml_string_or_file: Path to the SBML file
            sedml_string_or_file: Path to SEDML file providing rr simulation parameters
        """
        self.load_sbml_from_string_or_file(sbml_string_or_file)
        self.load_sedml_from_string_or_file(sedml_string_or_file)
        self.experiment_action_to_fn = {
            ModifySpeciesAction: self.change_initial_concentration,
            NullifySpeciesAction: self.nullify_species,
            ModifyReactionAction: self.change_reaction_rate,
            NullifyReactionAction: self.nullify_reaction,
        }

    def load_sbml_from_string_or_file(self, sbml_string_or_file: Any) -> int:
        reader = libsbml.SBMLReader()

        doc: SBMLDocument
        model: Model
        if os.path.isfile(sbml_string_or_file):
            if not os.path.exists(sbml_string_or_file):
                raise ValueError(f"File {sbml_string_or_file} does not exist.")
            doc = reader.readSBMLFromFile(str(sbml_string_or_file))  # type: ignore
        else:
            doc = reader.readSBMLFromString(sbml_string_or_file)

        for i in range(doc.getNumErrors()):
            error: libsbml.SBMLError = doc.getError(i)
            if error.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                raise ValueError(error.getMessage())

        # Convert local parameters to global ones
        converter = libsbml.SBMLLocalParameterConverter()

        if converter.setDocument(doc) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError("Failed to set document for local parameter converter")

        if converter.convert() != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError("Failed to convert local parameters to global ones")

        # # Remove unused unit definitions and convert to standard units
        # converter = libsbml.SBMLUnitsConverter()
        # if converter.setDocument(doc) != libsbml.LIBSBML_OPERATION_SUCCESS:
        #     raise RuntimeError("Failed to set document for units converter")
        # if converter.convert() != libsbml.LIBSBML_OPERATION_SUCCESS:
        #     raise RuntimeError("Failed to remove unused units and set standard units")

        # Inline function definitions and initial assignments
        if not doc.expandInitialAssignments():
            raise RuntimeError("Failed to expand initial assignments")

        if not doc.expandFunctionDefinitions():
            raise RuntimeError("Failed to expand function definitions")

        # Get the SBML model from the document
        model = doc.getModel()

        if model is None:
            raise ValueError(f"Model object does not exist for {doc}")

        # Assign initial values to unset global parameters
        for i in range(model.getNumParameters()):
            parameter: libsbml.Parameter = model.getParameter(i)
            if not parameter.isSetValue():
                parameter.setValue(0.0)

        self.document = doc
        self.model = model

        return True

    def load_sedml_from_string_or_file(self, sedml_string_or_file: Any | None) -> int:
        if not sedml_string_or_file:
            return False

        self.sedml_document = load_sedml_from_string_or_file(sedml_string_or_file)
        if self.sedml_document.getNumModels() != 1:
            print([model.getId() for model in self.sedml_document.getListOfModels()])
            print(sedml_string_or_file)

        sed_simulation = None
        outputEndTime = -float("inf")
        simulations: libsedml.SedListOfSimulations = self.sedml_document.getListOfSimulations()

        for i in range(simulations.size()):
            sim: libsedml.SedSimulation = simulations.get(i)
            simType: int = sim.getTypeCode()
            if simType is not libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE:
                continue
            assert isinstance(sim, libsedml.SedUniformTimeCourse)
            if sed_simulation is None or sim.getOutputEndTime() > outputEndTime:
                sed_simulation = sim
                outputEndTime = sim.getOutputEndTime()

        if sed_simulation is None:
            warnings.warn("No SedUniformTimeCourse simulation found in the SEDML document")
            tc = self.sedml_document.createUniformTimeCourse()
            tc.setInitialTime(0)
            tc.setOutputStartTime(0)
            tc.setOutputEndTime(10)
            tc.setNumberOfSteps(51)
            tc.setId("69")
            alg = tc.createAlgorithm()
            alg.setKisaoID("KISAO:0000019")
            self.sed_simulation = tc.clone()
        else:
            self.sed_simulation = sed_simulation.clone()

        return True

    def get_parameter_ids(self) -> List[str]:
        parameters: libsbml.ListOfParameters = self.model.getListOfParameters()
        return [parameters.get(j).getId() for j in range(parameters.size())]

    def get_functions_ids(self) -> List[str]:
        functions: libsbml.ListOfFunctionDefinitions = self.model.getListOfFunctionDefinitions()
        return [functions.get(j).getId() for j in range(functions.size())]

    def get_reaction_ids(self) -> List[str]:
        reactions: libsbml.ListOfReactions = self.model.getListOfReactions()
        return [reactions.get(j).getId() for j in range(reactions.size())]

    def get_species_ids(self, floating_only=False, boundary_only=False) -> List[str]:
        floating_sids = []
        boundary_sids = []
        all_species: libsbml.ListOfSpecies = self.model.getListOfSpecies()
        for i in range(all_species.size()):
            species: libsbml.Species = all_species.get(i)
            if species.getBoundaryCondition():
                boundary_sids.append(species.getId())
            else:
                floating_sids.append(species.getId())
        if floating_only:
            return floating_sids
        elif boundary_only:
            return boundary_sids
        return floating_sids + boundary_sids

    def get_experiment_constraints(
        self,
    ) -> Tuple[List[ExperimentConstraint], List[ExperimentConstraint]]:
        """Returns all experimental constraints in the SBML model."""
        reaction_constraints = []
        for i in range(self.model.getNumReactions()):
            reaction: libsbml.Reaction = self.model.getReaction(i)
            reaction_constraints.append(get_experimental_constraint(reaction))
        species_constraints = []
        for i in range(self.model.getNumSpecies()):
            species: libsbml.Species = self.model.getSpecies(i)
            species_constraints.append(get_experimental_constraint(species))
        return reaction_constraints, species_constraints

    def get_initial_parameter_values(
        self, _rr: roadrunner.RoadRunner | None = None
    ) -> Dict[str, float]:
        _rr = SBML._get_rr_instance(self.to_string()) if _rr is None else _rr
        names = _rr.getGlobalParameterIds()
        values = _rr.getGlobalParameterValues()
        return {
            name: float(value) for name, value in zip(names, values) if name not in ["time", "t"]
        }

    def change_initial_parameter_value(
        self,
        pid: str,
        value: float,
        _rr: roadrunner.RoadRunner,
    ) -> int:
        assert pid in self.get_parameter_ids(), f"Parameter {pid} not found in the model"
        try:
            return _rr.setGlobalParameterByName(pid, value)
        except Exception as e:
            raise ApplyExperimentActionError(
                f"Failed to set initial parameter value for {pid} to {value}: {e}"
            )

    def get_initial_concentrations(
        self, _rr: roadrunner.RoadRunner | None = None
    ) -> Dict[str, float]:
        _rr = SBML._get_rr_instance(self.to_string()) if _rr is None else _rr
        floating_arr = _rr.getFloatingSpeciesConcentrationsNamedArray()
        boundary_arr = _rr.getBoundarySpeciesConcentrationsNamedArray()
        floating_conditions = dict(zip(floating_arr.colnames, floating_arr.tolist()[0]))
        boundary_conditions = dict(zip(boundary_arr.colnames, boundary_arr.tolist()[0]))
        initial_concentrations: Dict[str, float] = floating_conditions | boundary_conditions
        for sid, initc in initial_concentrations.items():
            if initc < 0:
                warnings.warn(f"Found negative initial concentration for {sid}: {initc}")
        return initial_concentrations

    def get_kinetic_law(self, reaction_id) -> libsbml.KineticLaw | None:
        reaction: libsbml.Reaction | None = self.model.getReaction(reaction_id)
        assert reaction is not None, f"Reaction {reaction_id} not found in the model"
        return reaction.getKineticLaw()

    def shuffle_all(self):
        """Shuffle all components of the SBML model."""
        shuffle_parameters(self.model)
        shuffle_reactions(self.model)
        shuffle_species(self.model)
        shuffle_compartments(self.model)
        # shuffle_function_definitions(self.model)
        # shuffle_unit_definitions(self.model)
        # shuffle_initial_assignments(self.model)
        # shuffle_rules(self.model)
        # shuffle_constraints(self.model)

    def remove_parameter(self, param_id: str | List[str]) -> int:
        param_ids = [param_id] if isinstance(param_id, str) else param_id
        valid_ids = self.get_parameter_ids()
        return_status = []
        for pid in param_ids:
            if pid not in valid_ids:
                raise ValueError(f"Parameter {pid} not found in the model")
            param: libsbml.Parameter = self.model.getParameter(pid)
            objects_to_remove = find_parameter_initializations(self.model, pid)
            for object in objects_to_remove:
                assert object.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS
            return_status.append(param.unsetValue())
        return return_status.count(libsbml.LIBSBML_OPERATION_SUCCESS)

    def remove_kinetic_law(self, reaction_id: str | List[str]) -> int:
        react_ids = [reaction_id] if isinstance(reaction_id, str) else reaction_id
        valid_ids = self.get_reaction_ids()
        removal_counter = 0
        for rid in react_ids:
            if rid not in valid_ids:
                raise ValueError(f"Reaction {rid} not found in the model")
            react: libsbml.Reaction = self.model.getReaction(rid)
            if not react.isSetKineticLaw():
                raise ValueError(f"Reaction {rid} does not have a kinetic law")
            removal_counter += self._remove_kinetic_law(rid)
        return removal_counter

    def _remove_kinetic_law(self, reaction_id: str) -> int:
        reaction: libsbml.Reaction = self.model.getReaction(reaction_id)
        if not reaction.isSetKineticLaw():
            return 1
        kinetic_law: libsbml.KineticLaw = reaction.getKineticLaw()
        assert kinetic_law.getNumLocalParameters() == 0
        assert kinetic_law.getNumParameters() == 0
        if not kinetic_law.isSetMath():
            return 1
        ast_node: libsbml.ASTNode = kinetic_law.getMath()
        objects_to_check: Dict[str, libsbml.SBase] = self._recursively_parse_ast_node(ast_node)
        assert kinetic_law.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS
        for key, object in objects_to_check.items():
            assert key == object.getId()
            if self._count_usages(key) <= 1:
                assert object.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS
        return 1

    def _recursively_parse_ast_node(self, ast_node: libsbml.ASTNode) -> Dict[str, SBase]:
        objects = defaultdict(SBase)
        name = ast_node.getName()
        units = ast_node.getUnits()
        if name is not None:
            for o in [
                self.model.getParameter(name),
                self.model.getFunctionDefinition(name),
                self.model.getUnitDefinition(units),
                # self.model.getCompartment(name),
                # self.model.getSpecies(name),
            ]:
                if isinstance(o, libsbml.SBase):
                    objects[o.getId()] = o

        for j in range(ast_node.getNumChildren()):
            objects |= self._recursively_parse_ast_node(ast_node.getChild(j))

        return objects

    def _count_usages(self, sid: str) -> int:
        if sid is None or sid == "":
            return 0
        sbml_string = self.to_string()
        return (
            sbml_string.count(f" {sid} ")
            + sbml_string.count(f"'{sid}'")
            + sbml_string.count(f'"{sid}"')
        )

    def remove_reaction(self, reaction_id: str | List[str]) -> int:
        react_ids = [reaction_id] if isinstance(reaction_id, str) else reaction_id
        valid_ids = self.get_reaction_ids()
        return_success = 0
        for rid in react_ids:
            if rid not in valid_ids:
                raise ValueError(f"Reaction {rid} not found in the model")
            self._remove_kinetic_law(rid)
            return_success += self._remove_reaction(rid)
            # assert self._count_usages(rid) == 0
        return return_success

    def _remove_reaction(self, reaction_id: str) -> int:
        reaction: libsbml.Reaction = self.model.getReaction(reaction_id)
        reaction_refs = find_reaction_references(self.model, reaction_id)
        for object in reaction_refs:
            assert object.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS
        return reaction.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS

    def remove_species(self, species_id: str | List[str]):
        species_ids = [species_id] if isinstance(species_id, str) else species_id
        valid_ids = self.get_species_ids()
        return_success = 0
        for sid in species_ids:
            if sid not in valid_ids:
                raise ValueError(f"Species {sid} not found in the model")
            return_success += self._remove_species(sid)
            # assert self._count_usages(sid) == 0
        return return_success

    def _remove_species(self, species_id: str) -> int:
        species: libsbml.Species = self.model.getSpecies(species_id)
        species_refs = find_species_references(self.model, species_id)
        for object in species_refs:
            assert object.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS
        return species.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS

    def change_reaction_rate(
        self,
        action: ModifyReactionAction,
        verify_constraints=False,
        **kwargs,
    ) -> int:
        reaction_id = action.reaction_id
        multiply_factor = action.multiply_factor
        assert (
            reaction_id in self.get_reaction_ids()
        ), f"Reaction {reaction_id} not found in the model"
        reaction: libsbml.Reaction = self.model.getReaction(reaction_id)
        if verify_constraints:
            constraint = get_experimental_constraint(reaction)
            if not constraint.can_modify:
                raise ApplyExperimentActionError(
                    f"Reaction rate for {reaction_id} cannot be modified"
                )
        if not reaction.isSetKineticLaw():
            warnings.warn(f"Reaction {reaction_id} does not have a set kinetic law")
            raise ApplyExperimentActionError(
                f"Reaction rate for {reaction_id} could not be modified"
            )
        kinetic_law: libsbml.KineticLaw = reaction.getKineticLaw()
        if not kinetic_law.isSetMath():
            warnings.warn(f"Kinetic law for reaction {reaction_id} does not have a math expression")
            return 0
        ast_node: libsbml.ASTNode = kinetic_law.getMath()
        multiplier_node = libsbml.ASTNode(libsbml.AST_REAL)
        times_node = libsbml.ASTNode(libsbml.AST_TIMES)
        ast_node_formula: str = libsbml.formulaToL3String(ast_node)
        ast_node_clone: libsbml.ASTNode = libsbml.parseL3Formula(ast_node_formula)
        try:
            assert multiplier_node.setValue(multiply_factor) == libsbml.LIBSBML_OPERATION_SUCCESS
            assert multiplier_node.isWellFormedASTNode()
            assert ast_node_clone.isWellFormedASTNode()
            assert times_node.insertChild(0, multiplier_node) == libsbml.LIBSBML_OPERATION_SUCCESS
            assert times_node.insertChild(1, ast_node_clone) == libsbml.LIBSBML_OPERATION_SUCCESS
            assert times_node.isWellFormedASTNode()
            assert kinetic_law.setMath(times_node) == libsbml.LIBSBML_OPERATION_SUCCESS
            return 1
        except AssertionError as e:
            raise ApplyExperimentActionError(
                f"Failed to set kinetic law for reaction {reaction_id} to {multiply_factor} * {ast_node_formula}"
            )

    def nullify_reaction(
        self,
        action: NullifyReactionAction,
        verify_constraints=False,
        **kwargs,
    ) -> int:
        reaction_id = action.reaction_id
        assert (
            reaction_id in self.get_reaction_ids()
        ), f"Reaction {reaction_id} not found in the model"
        reaction: libsbml.Reaction = self.model.getReaction(reaction_id)
        if verify_constraints:
            constraint = get_experimental_constraint(reaction)
            if not constraint.can_nullify:
                raise ApplyExperimentActionError(
                    f"Reaction rate for {reaction_id} cannot be nullified"
                )
        if not reaction.isSetKineticLaw():
            warnings.warn(f"Reaction {reaction_id} does not have a set kinetic law")
            raise ApplyExperimentActionError(
                f"Reaction rate for {reaction_id} could not be nullified"
            )
        kinetic_law: libsbml.KineticLaw = reaction.getKineticLaw()
        if not kinetic_law.isSetMath():
            warnings.warn(f"Kinetic law for reaction {reaction_id} does not have a math expression")
            raise ApplyExperimentActionError(
                f"Reaction rate for {reaction_id} could not be nullified"
            )
        try:
            zero_node = libsbml.ASTNode(libsbml.AST_REAL)
            assert zero_node.setValue(0.0) == libsbml.LIBSBML_OPERATION_SUCCESS
            assert kinetic_law.setMath(zero_node) == libsbml.LIBSBML_OPERATION_SUCCESS
            return 1
        except AssertionError as e:
            raise ApplyExperimentActionError(
                f"Failed to set kinetic law for reaction {reaction_id} to zero"
            )

    def change_initial_concentration(
        self,
        action: ModifySpeciesAction,
        _rr: roadrunner.RoadRunner,
        verify_constraints=False,
        **kwargs,
    ) -> int:
        """Perturbs the initial concentration of a species in the SBML model."""
        species_id = action.species_id
        value = action.value
        assert species_id in self.get_species_ids(), f"Species {species_id} not found in the model"
        species: libsbml.Species = self.model.getSpecies(species_id)
        if verify_constraints:
            constraint = get_experimental_constraint(species)
            if not constraint.can_modify:
                raise ApplyExperimentActionError(f"Species {species_id} cannot be modified")
        try:
            if species.getConstant():
                raise ApplyExperimentActionError(f"Cannot modify a constant species {species_id}")
            elif species.getBoundaryCondition():
                raise ApplyExperimentActionError(f"Cannot modify a boundary species {species_id}")
            return _rr.setInitConcentration(species_id, value, forceRegenerate=False)
        except ApplyExperimentActionError as e:
            raise
        except Exception as e:
            print(e)
            raise ApplyExperimentActionError(
                f"Failed to set initial concentration for {species_id} to {value}"
            )

    def nullify_species(
        self,
        action: NullifySpeciesAction,
        verify_constraints=False,
        **kwargs,
    ) -> int:
        species_id = action.species_id
        assert species_id in self.get_species_ids(), f"Species {species_id} not found in the model"
        species: libsbml.Species = self.model.getSpecies(species_id)
        if verify_constraints:
            constraint = get_experimental_constraint(species)
            if not constraint.can_nullify:
                raise ApplyExperimentActionError(f"Species {species_id} cannot be modified")
        try:
            ref_objects = find_species_knockout_references(self.model, species_id)
            for obj in ref_objects:
                assert obj.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS
            dangling_objects = find_dangling_objects(self.model)
            for obj in dangling_objects:
                assert obj.removeFromParentAndDelete() == libsbml.LIBSBML_OPERATION_SUCCESS
            assert species.setBoundaryCondition(False) == libsbml.LIBSBML_OPERATION_SUCCESS
            assert species.setHasOnlySubstanceUnits(True) == libsbml.LIBSBML_OPERATION_SUCCESS
            assert species.setConstant(True) == libsbml.LIBSBML_OPERATION_SUCCESS
            return species.setInitialAmount(0.0) == libsbml.LIBSBML_OPERATION_SUCCESS
        except Exception as e:
            raise ApplyExperimentActionError(f"Failed to nullify species {species_id}")

    @classmethod
    def apply_experiment_actions(
        cls,
        sbml: SBML,
        experiment_actions: List[str],
        valid_species_ids: List[str],
        valid_reaction_ids: List[str],
        **kwargs,
    ) -> SBML:
        actions: List[ExperimentAction] = []
        for action in experiment_actions:
            try:
                action = parse_experiment_action(
                    action,
                    valid_species_ids,
                    valid_reaction_ids,
                )
                actions.append(action)
            except ParseExperimentActionError as e:
                raise
        return SBML._apply_experiment_actions(sbml, actions, **kwargs)

    @classmethod
    def _apply_experiment_actions(
        cls, sbml: SBML, actions: Sequence[ExperimentAction], **kwargs
    ) -> SBML:
        new_sbml = deepcopy(sbml)

        actions_by_type = defaultdict(list)
        for action in actions:
            actions_by_type[type(action)].append(action)

        modify_actions: List[ModifySpeciesAction] = actions_by_type.get(ModifySpeciesAction, [])
        nullify_actions: List[NullifySpeciesAction] = actions_by_type.get(NullifySpeciesAction, [])

        for a in nullify_actions:
            try:
                new_sbml.nullify_species(a, **kwargs)
            except ApplyExperimentActionError as e:
                raise

        _rr = SBML._get_rr_instance(new_sbml.to_string())

        for a in modify_actions:
            try:
                new_sbml.change_initial_concentration(a, _rr=_rr, **kwargs)
            except ApplyExperimentActionError as e:
                raise

        if len(modify_actions) > 0:
            try:
                new_sbml_string = str(_rr.getCurrentSBML())
                new_sbml = SBML(new_sbml_string, new_sbml.to_sedml_string())
            except Exception as e:
                raise ApplyExperimentActionError(f"Failed to apply experiment actions: {e}")

        return new_sbml

    @classmethod
    def add_noise_to_initial_concentrations(
        cls,
        sbml,
    ) -> SBML:
        new_sbml = cls(sbml.to_string(), sbml.to_sedml_string())
        _rr = SBML._get_rr_instance(new_sbml.to_string())
        init_concentrations = new_sbml.get_initial_concentrations(_rr)
        experiment_actions: List[ModifySpeciesAction] = []
        for species, concentration in init_concentrations.items():
            noise = random.uniform(noise_lower_bound, noise_upper_bound)
            experiment_actions.append(ModifySpeciesAction(species, noise * concentration))
        return SBML._apply_experiment_actions(new_sbml, experiment_actions)

    @classmethod
    def eval_add_noise_to_initial_concentrations(cls, true_sbml, inco_sbml, pred_sbml, noise):
        # Create copies of the input models to avoid modifying the originals
        new_true_sbml = cls(true_sbml.to_string(), true_sbml.to_sedml_string())
        new_inco_sbml = cls(inco_sbml.to_string(), inco_sbml.to_sedml_string())
        new_pred_sbml = cls(pred_sbml.to_string(), pred_sbml.to_sedml_string())

        # Get the initial concentrations for each model
        _rr_inco = SBML._get_rr_instance(new_inco_sbml.to_string())
        inco_concentrations = new_inco_sbml.get_initial_concentrations(_rr_inco)

        # Generate noise for each species in inco_sbml and store in a dictionary
        noise_dict = {}
        for species, concentration in inco_concentrations.items():
            Species: libsbml.Species = new_inco_sbml.model.getSpecies(species)
            if Species.getConstant() or Species.getBoundaryCondition():
                continue
            noise_dict[species] = perturb_concentration_proportional(concentration, noise)

        # Apply noise to all three models
        models = [new_true_sbml, new_inco_sbml, new_pred_sbml]
        noised_models = []

        for model in models:
            _rr = SBML._get_rr_instance(model.to_string())
            init_concentrations = model.get_initial_concentrations(_rr)
            experiment_actions: List[ModifySpeciesAction] = []

            for species, concentration in init_concentrations.items():
                if species in noise_dict:
                    experiment_actions.append(ModifySpeciesAction(species, noise_dict[species]))

            noised_model = SBML._apply_experiment_actions(model, experiment_actions)
            noised_models.append(noised_model)

        return noised_models

    def to_string(self) -> str:
        """
        Convert an SBML model to a string representation.

        Args:
            sbml_model: The SBML model

        Returns:
            String representation of the SBML model
        """
        return libsbml.writeSBMLToString(self.document)

    def save(self, path: str) -> int:
        """
        Save an SBML model to an xml file.

        Args:
            path: The path to save the model to
        """
        return libsbml.writeSBMLToFile(self.document, path)

    def to_sedml_string(self) -> str | None:
        if not self.sedml_document:
            return None
        return libsedml.writeSedMLToString(self.sedml_document)

    def save_sedml(self, path: str) -> int:
        return libsedml.writeSedMLToFile(self.sedml_document, path)

    @default_document_parameter
    def _remove_metadata(
        self,
        object: SBase,
        touched_elements: Set[int] = set(),
        config: Dict[Any, Dict[str, bool]] = DEFAULT_METADATA_REMOVAL_CONFIG,
    ) -> Set[int]:
        """
        Removes notes, annotations, and names that may leak
        information about the biomodel that can be recalled
        from pre-training context rather than reasoning.
        """
        elements: libsbml.SBaseList = object.getListOfAllElements()

        for j in range(elements.getSize() + 1):
            if j == elements.getSize():
                element: SBase = object
            else:
                element: SBase = elements.get(j)

            settings = config["default"] | config.get(element.getTypeCode(), {})

            if settings["del_name"] and element.isSetName():
                element.unsetName()

            if settings["del_notes"] and element.isSetNotes():
                element.unsetNotes()

            if settings["del_annotations"] and element.isSetAnnotation():
                element.unsetAnnotation()

            # if element.isSetAnnotation():
            #     process_annotation(element.getAnnotation())

            if settings["del_history"] and element.isSetModelHistory():
                element.unsetModelHistory()

            if settings["del_sbo_terms"] and element.isSetSBOTerm():
                element.unsetSBOTerm()

            if settings["del_cv_terms"] and element.getNumCVTerms() > 0:
                element.unsetCVTerms()

            if settings["del_created_date"] and element.isSetCreatedDate():
                element.unsetCreatedDate()

            if settings["del_modified_date"] and element.isSetModifiedDate():
                element.unsetModifiedDates()

            if settings["del_user_data"] and element.isSetUserData():
                element.unsetUserData()

            touched_elements.add(element.getMetaId())

            if settings["del_metaid"] and element.isSetMetaId():
                element.unsetMetaId()

        return touched_elements

    def _scramble_ids(self, type_codes_to_ignore: List[int] = []) -> Dict[int, Dict[str, str]]:
        """
        Scrambles the ids of the SBML model, maintaining uniqueness
        while removing any suspicious references that might leak info
        """
        self._assign_unique_metaids()
        allElements = self.document.getListOfAllElements()
        oldIds = getAllIds(allElements=allElements)
        idScrambler = SIdScrambler(oldIds, type_codes_to_ignore)
        self.model.renameIDs(elements=allElements, idTransformer=idScrambler)
        self._remove_metaids()
        # assert self._validate_references(idScrambler.real_to_fake_ids)
        return idScrambler.real_to_fake_ids

    @default_document_parameter
    def _assign_unique_metaids(self, object: SBase, metaids: Set[str] = set()):
        elements: libsbml.SBaseList = object.getListOfAllElements()
        for j in range(elements.getSize()):
            element: SBase = elements.get(j)
            new_metaid = generate_new_id(prefix="metaid_", ban_list=metaids)
            element.setMetaId(new_metaid)
            metaids.add(new_metaid)

    @default_document_parameter
    def _remove_metaids(self, object: SBase, metaids: Set[str] = set()):
        elements: libsbml.SBaseList = object.getListOfAllElements()
        for j in range(elements.getSize()):
            element: SBase = elements.get(j)
            if element.isSetMetaId():
                element.unsetMetaId()
                metaids.add(element.getMetaId())

    def _canonicalize_names(
        self, type_codes_to_include: List[int] = []
    ) -> Dict[int, Dict[str, str]]:
        """
        For the SBML type codes passed in, canonicalizes the names by
        removing any special characters and removing any weird memorizable signatures for an LLM
        """
        real_to_fake_names: Dict[int, Dict[str, str]] = defaultdict(dict)
        allElements: libsbml.SBaseList = self.document.getListOfAllElements()
        for j in range(allElements.getSize()):
            element: libsbml.SBase = allElements.get(j)
            type_code = element.getTypeCode()
            if type_code not in type_codes_to_include:
                continue
            if not element.isSetName():
                oldName = element.getId()
            else:
                oldName = element.getName()
            if oldName == "" or oldName is None:
                continue
            # newName = canonicalize_name(oldName)
            newName = oldName
            if newName != oldName:
                assert element.setName(newName) == libsbml.LIBSBML_OPERATION_SUCCESS
                real_to_fake_names[type_code][oldName] = newName
        return real_to_fake_names

    def _validate_references(self, real_to_fake_ids: Dict[int, Dict[str, str]] = {}) -> bool:
        """Validates that all references have been updated correctly"""
        sbml_string = self.to_string()
        # Find all formulas containing old IDs
        for type_code in real_to_fake_ids.keys():
            for old_id in real_to_fake_ids[type_code].keys():
                # Search raw XML for any remaining instances of old_id
                if old_id in ["time"]:
                    continue
                for query in [f'"{old_id}"', f"'{old_id}'", f" {old_id} "]:
                    if query in sbml_string:
                        where = sbml_string.find(query)
                        print(
                            f"WARNING: Found old ID '{query}' still in document at position {where}"
                        )
                        return False
        return True

    @classmethod
    def _get_rr_instance(cls, sbml_string_or_file: str, *args, **kwargs) -> roadrunner.RoadRunner:
        return te.loadSBMLModel(sbml_string_or_file)
        # return roadrunner.RoadRunner(sbml_string_or_file, *args, **kwargs)

    @classmethod
    def code_to_sbml(
        cls, sbml: "SBML", executable_file_as_string: str, safe_globals: Dict[str, Any] = {}
    ) -> SBML:
        """
        Runs the executable_file_as_string LLM code on the sbml.to_string() input,
        then captures the LLM output (which is a modified sbml string), and returns
        new SBML object. If any step in this pipeline fails, then return an error
        message for the LLM to handle

        Args:
            sbml (SBML): partial SBML input object
            executable_file_as_string (str): string containing executable code

        Returns:
            SBML: modified SBML on success
        """

        try:
            # Get the old sbml string and feed as input to the llm script
            code = (
                "import numpy as np\nimport pandas as pd\nimport math\nimport scipy\nimport sklearn\nimport jax\nimport tellurium as te\nimport roadrunner as rr\nimport libsbml\n"
                + executable_file_as_string
            )

            # Add the input_sbml_string to local_vars so the executable code can access it
            safe_globals["modified_sbml_string"] = None

            # Execute the code with stdout capture (for debugging only)
            exec(code, safe_globals)

            modified_sbml_string: str = safe_globals["modified_sbml_string"]  # type: ignore
            return SBML(modified_sbml_string)

        except Exception as e:
            # Capture the error message
            error_message = traceback.format_exc()
            raise ValueError(error_message)

    def __deepcopy__(self, memo):
        """
        Override the deepcopy behavior.

        Args:
            memo: Dictionary that keeps track of objects already copied to
                  avoid infinite recursion with circular references
        """
        # Check if this object is already in the memo dictionary
        if id(self) in memo:
            return memo[id(self)]

        # Create a new instance of this class without calling __init__
        result = SBML(
            sbml_string_or_file=self.to_string(),
            sedml_string_or_file=self.to_sedml_string(),
        )
        return result

    @deprecated
    def _remove_ast_node(
        self,
        object: libsbml.ASTNode,
        remove_ref_parameters=True,
        remove_ref_function_definitions=True,
        remove_ref_unit_definitions=False,
        remove_ref_compartments=False,
        remove_ref_species=False,
    ):
        if object.isName():
            name = object.getName()
            # Try removing the element by name from each possible container
            # Only one of these will succeed if the name exists in that container
            if remove_ref_parameters:
                self.model.removeParameter(name)
            if remove_ref_function_definitions:
                self.model.removeFunctionDefinition(name)
            if remove_ref_compartments:
                self.model.removeCompartment(name)
            if remove_ref_species:
                self.model.removeSpecies(name)

        if object.hasUnits():
            if remove_ref_unit_definitions:
                self.model.removeUnitDefinition(object.getUnits())

        for j in range(object.getNumChildren()):
            self._remove_ast_node(
                object.getChild(j),
                remove_ref_parameters=True,
                remove_ref_function_definitions=True,
                remove_ref_unit_definitions=False,
                remove_ref_compartments=False,
                remove_ref_species=False,
            )

    @deprecated
    def get_kinetic_laws(self, whitelist=None) -> List[libsbml.KineticLaw]:
        reaction_ids = self.get_reaction_ids() if whitelist is None else whitelist
        laws = [self.get_kinetic_law(rid) for rid in reaction_ids]
        return [law for law in laws if law is not None]

    @deprecated
    @classmethod
    def get_initial_conditions(
        cls,
        sbml: "SBML",
        as_amounts=True,
        whitelist=None,
    ) -> Dict[str, float]:
        """
        Obtain the initial conditions for each species from the SBML model.

        Args:
            as_amounts: If True, all values are converted to amounts. If False, all are kept as concentrations.
            whitelist: List of species ids for which to get the initial conditions
        """
        _rr = SBML._get_rr_instance(sbml.to_string())

        if as_amounts:
            floating_arr = _rr.getFloatingSpeciesAmountsNamedArray()
            boundary_arr = _rr.getBoundarySpeciesAmountsNamedArray()
        else:
            floating_arr = _rr.getFloatingSpeciesConcentrationsNamedArray()
            boundary_arr = _rr.getBoundarySpeciesConcentrationsNamedArray()

        floating_conditions = dict(zip(floating_arr.colnames, floating_arr.tolist()[0]))
        boundary_conditions = dict(zip(boundary_arr.colnames, boundary_arr.tolist()[0]))
        initial_conditions = floating_conditions | boundary_conditions

        if whitelist is None or len(whitelist) == 0:
            return initial_conditions

        result = {}
        for sid in whitelist:
            if sid not in initial_conditions:
                raise ValueError(f"The requested species {sid} was not found in the system")
            result[sid] = initial_conditions[sid]
        return result

    @deprecated
    @classmethod
    def set_initial_conditions(
        cls, sbml: "SBML", initial_conditions: Dict[str, float], as_amounts=True
    ) -> "SBML":
        """
        Sets the initial conditions for species with ids defined in the keys of initial_conditions

        Args:
            sbml (SBML): The SBML object to modify
            initial_conditions (Dict[str, float]): Species id to initial amount or concentration
            as_amounts (bool, optional): Whether the values in initial conditions are amounts or concentrations

        Returns:
            SBML: A new SBML class object with the modified initial conditions
        """
        _rr = SBML._get_rr_instance(sbml.to_string())

        floating_species_ids = _rr.getFloatingSpeciesIds()
        boundary_species_ids = _rr.getBoundarySpeciesIds()
        all_ids = floating_species_ids + boundary_species_ids
        initial_assignment_ids = _rr.getInitialAssignmentIds()
        assignment_rule_ids = _rr.getAssignmentRuleIds()
        rate_rule_ids = _rr.getRateRuleIds()

        assert len(floating_species_ids) + len(boundary_species_ids) == sbml.model.getNumSpecies()

        for sid, value in initial_conditions.items():
            if sid not in all_ids:
                raise ValueError(f"Could not match {sid} to any species id in the model")
            if sid in initial_assignment_ids:
                raise ValueError(
                    f"Cannot set initial condition of species {sid} with an initial assignment function"
                )
            if sid in assignment_rule_ids:
                raise ValueError(
                    f"Cannot set initial condition of species {sid} with an assignment rule"
                )
            if sid in boundary_species_ids:
                warnings.warn(
                    f"Setting initial condition on boundary species {sid} is allowed, but may lead to unexpected behaviour"
                )
            if sid in rate_rule_ids:
                warnings.warn(
                    f"Setting initial condition on species {sid} with a rate rule is allowed, but may lead to unexpected behavior"
                )
            if as_amounts:
                _rr.setInitAmount(sid, value, forceRegenerate=False)
            else:
                _rr.setInitConcentration(sid, value, forceRegenerate=False)

        # NOTE: This removes initial assignment rules, but that's okay since this method is for subsequent simulation
        return SBML(_rr.getCurrentSBML())

    @deprecated
    @classmethod
    def apply_interventions(
        cls, sbml: "SBML", conditions: Dict[str, float], as_amounts=True
    ) -> "SBML":
        """
        Sets the initial conditions for species with ids defined in the keys of conditions after removing all its dependencies

        Args:
            sbml (SBML): The SBML object to modify
            conditions (Dict[str, float]): Species id to intervention amount or concentration
            as_amounts (bool, optional): Whether the values in initial conditions are amounts or concentrations
        """
        _rr = SBML._get_rr_instance(sbml.to_string())

        _document: SBMLDocument = sbml.document.clone()
        _model: Model = sbml.model.clone()

        floating_species_ids = _rr.getFloatingSpeciesIds()
        boundary_species_ids = _rr.getBoundarySpeciesIds()
        initial_assignment_ids = _rr.getInitialAssignmentIds()
        assignment_rule_ids = _rr.getAssignmentRuleIds()
        rate_rule_ids = _rr.getRateRuleIds()

        all_species_ids = floating_species_ids + boundary_species_ids

        for sid in conditions.keys():
            if sid not in all_species_ids:
                raise ValueError(f"Requested sid {sid} does not exist in the model")

            # Get the species related to this sid
            species: libsbml.Species = _model.getSpecies(sid)

            # Remove initial assignment for this sid if exists
            if sid in initial_assignment_ids:
                assert _model.removeInitialAssignment(sid)

            # Remove assignment rule for this sid if exists
            if sid in assignment_rule_ids:
                assignment_rule: libsbml.AssignmentRule = _model.getAssignmentRuleByVariable(sid)
                assert assignment_rule.removeFromParentAndDelete()

            # Remove rate rules for this sid if exists
            if sid in rate_rule_ids:
                rate_rule: libsbml.RateRule = _model.getRateRuleByVariable(sid)
                assert rate_rule.removeFromParentAndDelete()

            # Remove reactions that have this sid as product
            rids_to_remove = []
            reactions: libsbml.ListOfReactions = _model.getListOfReactions()
            for i in range(reactions.size()):
                reaction: libsbml.Reaction = reactions.get(i)
                products: libsbml.ListOfSpeciesReferences = reaction.getListOfProducts()
                for j in range(products.size()):
                    product: libsbml.SpeciesReference = products.get(j)
                    if product.getSpecies() == sid:
                        rids_to_remove.append(reaction.getId())

            for rid in rids_to_remove:
                _model.removeReaction(rid)

        # Recreate a new sbml object from this intervened model
        _document.setModel(_model)
        new_sbml = SBML(_document.toSBML())

        # Set the initial conditions on these intervened species and return the final sbml
        return SBML.set_initial_conditions(
            new_sbml, initial_conditions=conditions, as_amounts=as_amounts
        )

    @deprecated
    @default_model_parameter
    def unset_parameter_values(self, object: SBase, whitelist: List[str] = []):
        """
        Removes all global and local parameter values from the SBML file

        Args:
            whitelist: List of parameter ids to spare
        """
        # TODO: Maybe need to also remove units associated to parameters
        if isinstance(object, Model):
            self.unset_parameter_values(object.getListOfParameters(), whitelist=whitelist)
        elif isinstance(object, libsbml.ListOfParameters):
            for j in range(object.size()):
                self.unset_parameter_values(object.get(j), whitelist=whitelist)
        elif isinstance(object, libsbml.Parameter):
            if object.getId() not in whitelist:
                object.unsetValue()

    @deprecated
    @default_model_parameter
    def remove_kinetic_laws(self, object: SBase, whitelist=[], agressive=True, **kwargs) -> int:
        """
        Removes kinetic laws from the SBML model, which includes function definitions,
        unit definitions, and parameters referenced by the kinetic law's math expression

        Args:
            whitelist: List of reaction ids whose kinetic laws we wish to spare
            agressive: If True, removes all functions and global parameters used in kinetic law, even if used elsewhere
        """
        if isinstance(object, Model):
            return self.remove_kinetic_laws(
                object.getListOfReactions(), whitelist, agressive, **kwargs
            )
        elif isinstance(object, (libsbml.ListOfReactions, libsbml.ListOfParameters)):
            return sum(
                self.remove_kinetic_laws(object.get(j), whitelist, agressive, **kwargs)
                for j in range(object.size())
            )
        elif isinstance(object, libsbml.Reaction):
            if object.getId() not in whitelist:
                self.remove_kinetic_laws(object.getKineticLaw(), whitelist, agressive, **kwargs)
                return object.unsetKineticLaw()
        elif isinstance(object, libsbml.KineticLaw):
            assert object.getNumParameters() == 0
            assert object.getNumLocalParameters() == 0
            self.remove_kinetic_laws(object.getMath(), whitelist, agressive, **kwargs)
        elif isinstance(object, libsbml.ASTNode):
            if agressive:
                self._remove_ast_node(object, **kwargs)
        elif isinstance(object, libsbml.Parameter):
            object.removeFromParentAndDelete()
        return 0

    @deprecated
    def remove_reactions(self, whitelist=[], **kwargs):
        """
        Removes an entire Reaction from the SBML model, which includes referenced
        function definitions, unit definitions, and parameters

        Args:
            whitelist: List of Reaction ids to spare
        """
        reaction_ids = self.get_reaction_ids()
        reactions: libsbml.ListOfReactions = self.model.getListOfReactions()
        for rid in reaction_ids:
            if rid not in whitelist:
                reaction: libsbml.Reaction = reactions.get(rid)
                self.remove_kinetic_laws(reaction, agressive=True, **kwargs)
                reaction.removeFromParentAndDelete()
