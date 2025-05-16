import math
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import libsbml
import networkx as nx
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from scigym.api import EvaluationResult
from scigym.constants import SBML_GRAPH_PARAMS
from scigym.sbml import SBML
from scigym.sbmlutils.sbmldiagram import *  # noqa
from scigym.simulator import Simulator
from scigym.utils import compute_dict_smape, compute_mean_error


def extract_reaction_edges(reaction: libsbml.Reaction) -> List[Tuple[str, str]]:
    """Extracts the reactant-product edges from the reaction."""
    reactant_ids: List[str] = [
        reaction.getReactant(j).getSpecies() for j in range(reaction.getNumReactants())
    ]
    product_ids: List[str] = [
        reaction.getProduct(k).getSpecies() for k in range(reaction.getNumProducts())
    ]
    edges: List[Tuple[str, str]] = []
    for reactant_id in reactant_ids:
        for product_id in product_ids:
            edges.append((reactant_id, product_id))
    return edges


def get_reaction_hash(reaction: libsbml.Reaction) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
    reactant_ids: List[str] = [
        reaction.getReactant(j).getSpecies() for j in range(reaction.getNumReactants())
    ]
    product_ids: List[str] = [
        reaction.getProduct(k).getSpecies() for k in range(reaction.getNumProducts())
    ]
    modifier_ids: List[str] = [
        reaction.getModifier(l).getSpecies() for l in range(reaction.getNumModifiers())
    ]
    rp_hash = (hash(frozenset(reactant_ids)), hash(frozenset(product_ids)))
    rpm_hash = (
        hash(frozenset(reactant_ids)),
        hash(frozenset(product_ids)),
        hash(frozenset(modifier_ids)),
    )
    return rpm_hash, rp_hash


def extract_reaction_hashes(
    model: libsbml.Model,
) -> Tuple[Set[Tuple[int, int, int]], Set[Tuple[int, int]]]:
    reactions: libsbml.ListOfReactions = model.getListOfReactions()

    reaction_hashes_no_modifier: Set[Tuple[int, int]] = set()
    reaction_hashes_full: Set[Tuple[int, int, int]] = set()

    for i in range(reactions.size()):
        rpm_hash, rp_hash = get_reaction_hash(reactions.get(i))
        reaction_hashes_no_modifier.add(rp_hash)
        reaction_hashes_full.add(rpm_hash)

    return reaction_hashes_full, reaction_hashes_no_modifier


def get_correctly_predicted_reaction_ids(
    true_model: libsbml.Model,
    inco_model: libsbml.Model,
    pred_model: libsbml.Model,
) -> Tuple[Set[str], Set[str]]:
    """Get the correctly predicted reaction IDs from predicted model that aren't in the incomplete model"""
    true_rpm_hashes, true_rp_hashes = extract_reaction_hashes(true_model)
    inco_rpm_hashes, inco_rp_hashes = extract_reaction_hashes(inco_model)
    rp_rids, rpm_rids = set(), set()
    for reaction in pred_model.getListOfReactions():
        rpm_hash, rp_hash = get_reaction_hash(reaction)
        if rpm_hash in true_rpm_hashes and rpm_hash not in inco_rpm_hashes:
            rpm_rids.add(reaction.getId())
        if rp_hash in true_rp_hashes and rp_hash not in inco_rp_hashes:
            rp_rids.add(reaction.getId())
    return rpm_rids, rp_rids


def plot_sbml_diagram(
    true_sbml_model: SBML,
    pred_sbml_model: SBML,
    inco_sbml_model: SBML,
    save_file_path: str,
) -> None:
    inco_species_ids = set(inco_sbml_model.get_species_ids())
    inco_reaction_ids = set(inco_sbml_model.get_reaction_ids())

    correct_pred_rpm_rids, correct_pred_rp_rids = get_correctly_predicted_reaction_ids(
        true_sbml_model.model, inco_sbml_model.model, pred_sbml_model.model
    )
    diagram = MySBMLDiagram(
        pred_sbml_model.model,
        inco_reaction_ids=inco_reaction_ids,
        inco_species_ids=inco_species_ids,
        correct_pred_rpm_rids=correct_pred_rpm_rids,
        correct_pred_rp_rids=correct_pred_rp_rids,
        **SBML_GRAPH_PARAMS,
    )
    diagram.draw_and_save(str(save_file_path))


def reaction_retrieval_score(
    true_eset: Set[Tuple[str, str]],
    pred_eset: Set[Tuple[str, str]],
) -> Dict[str, float]:
    """Calculates the precision, recall, and F1 score for the retrieval of reactions"""
    tp = len(true_eset.intersection(pred_eset))
    fp = len(pred_eset.difference(true_eset))
    fn = len(true_eset.difference(pred_eset))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(precision=precision, recall=recall, f1=f1)


def reaction_jaccard_distance(
    true_eset: Set[Tuple[str, str]],
    pred_eset: Set[Tuple[str, str]],
) -> float:
    """Calculates the Jaccard distance between two sets of reactions"""
    intersection = len(true_eset.intersection(pred_eset))
    union = len(true_eset.union(pred_eset))
    jaccard_distance = 1 - (intersection / union) if union > 0 else 1
    return jaccard_distance


def calculate_hausdorff_score(
    true_model: libsbml.Model,
    pred_model: libsbml.Model,
) -> Tuple[float, float]:
    """Calculate the bi-directional Hausdorff distance between the reaction set of two models.
    (->) For each true reaction, find the closest predicted reaction.
    (<-) For each predicted reaction, find the closest true reaction.
    """
    # Extract the reactions from both models
    reaction1_edge_sets = {}
    reaction2_edge_sets = {}

    reactions1: libsbml.ListOfReactions = true_model.getListOfReactions()
    reactions2: libsbml.ListOfReactions = pred_model.getListOfReactions()

    for i in range(reactions1.size()):
        reaction: libsbml.Reaction = reactions1.get(i)
        reaction1_edge_sets[reaction.getId()] = set(extract_reaction_edges(reaction))

    for j in range(reactions2.size()):
        reaction: libsbml.Reaction = reactions2.get(j)
        reaction2_edge_sets[reaction.getId()] = set(extract_reaction_edges(reaction))

    # If there are no reactions in the true model
    if len(reaction1_edge_sets) == 0 and len(reaction2_edge_sets) == 0:
        return 0, 0
    if len(reaction1_edge_sets) == 0:
        return 1, 1
    if len(reaction2_edge_sets) == 0:
        return 1, 1

    # Calculate all pairwise distances between reactions
    pairwise_reaction_distances = np.zeros(
        (
            len(reaction1_edge_sets),
            len(reaction2_edge_sets),
        )
    )
    for i, true_eset in enumerate(reaction1_edge_sets.values()):
        for j, pred_eset in enumerate(reaction2_edge_sets.values()):
            # f1 = reaction_retrieval_score(true_eset, pred_eset)["f1"]
            jaccard_dist = reaction_jaccard_distance(true_eset, pred_eset)
            pairwise_reaction_distances[i, j] = jaccard_dist

    # Extract the (->) forward minimum distances for each row (true reaction)
    forward_min_distances = np.nanmin(pairwise_reaction_distances, axis=1)
    forward_average_hausdorff = np.nanmean(forward_min_distances)
    assert len(forward_min_distances) == len(reaction1_edge_sets)

    # Extract the (<-) reverse minimum distances for each column (predicted reaction)
    reverse_minimum_distances = np.nanmin(pairwise_reaction_distances, axis=0)
    reverse_average_hausdorff = np.nanmean(reverse_minimum_distances)
    assert len(reverse_minimum_distances) == len(reaction2_edge_sets)

    return float(forward_average_hausdorff), float(reverse_average_hausdorff)


def extract_rid_to_kinetic_law(model: libsbml.Model) -> Dict[str, libsbml.KineticLaw | None]:
    rid_to_klaw: Dict[str, libsbml.KineticLaw | None] = {}
    reactions: libsbml.ListOfReactions = model.getListOfReactions()
    for j in range(reactions.size()):
        reaction: libsbml.Reaction = reactions.get(j)
        kinetic_law: libsbml.KineticLaw | None = None
        if reaction.isSetKineticLaw():
            kinetic_law = reaction.getKineticLaw()
        rid_to_klaw[reaction.getId()] = kinetic_law
    return rid_to_klaw


def get_vector_norm(series: List[float], type: str = "l2") -> float:
    if type == "l0":
        norm = max(series)
    elif type == "l1":
        norm = sum(map(abs, series))
    elif type == "l2":
        norm = math.sqrt(sum(map(lambda x: pow(x, 2), series)))
    else:
        raise ValueError(f"Norm type not recognized: {type}")
    return norm


def apply_vector_norm(series: List[float], norm: float) -> List[float]:
    if norm > 0:
        return list(map(lambda x: x / norm, series))
    return series


def compute_ged(eval_g: nx.MultiDiGraph, target_g: nx.MultiDiGraph, timeout=30) -> int:
    ged = nx.similarity.graph_edit_distance(
        eval_g,
        target_g,
        node_match=node_match,
        edge_match=edge_match,
        timeout=timeout,
    )
    return ged


def normalized_metric(pred_metric: float, inco_metric: float) -> float:
    """
    Normalize the metric by dividing by the incoherent metric.
    """
    if inco_metric == pred_metric:
        return 1.0
    elif inco_metric == 0:
        return np.inf
    return pred_metric / inco_metric


def tokenize_math_formula(formula):
    formula = re.sub(r"([+\-*/^()=,])", r" \1 ", formula)
    tokens = []
    for token in formula.split():
        tokens.append(token)
    return tokens


def calculate_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    smoothing = SmoothingFunction().method1
    return sentence_bleu(  # type: ignore
        [reference], candidate, weights=weights, smoothing_function=smoothing
    )


def get_average_bleu_from_klaw(k1: libsbml.KineticLaw, k2: libsbml.KineticLaw) -> float:
    if k1 is None or k2 is None:
        return 0
    t1 = tokenize_math_formula(k1.getFormula())
    t2 = tokenize_math_formula(k2.getFormula())
    b1 = calculate_bleu(t1, t2)
    b2 = calculate_bleu(t2, t1)
    return (b1 + b2) / 2


def extract_species_edges_with_modifiers(model: libsbml.Model) -> Set[Tuple[str, str]]:
    """
    Extract undirected edges between species from an SBML model.
    Each edge represents a relationship where two species appear in the same reaction,
    including as modifiers.

    Args:
        model: The SBML model to extract edges from

    Returns:
        A set of tuples, each representing an undirected edge between two species
    """
    edges: Set[Tuple[str, str]] = set()
    reactions: libsbml.ListOfReactions = model.getListOfReactions()

    for i in range(reactions.size()):
        reaction: libsbml.Reaction = reactions.get(i)

        # Get reactants, products, and modifiers
        reactant_ids: List[str] = [
            reaction.getReactant(j).getSpecies() for j in range(reaction.getNumReactants())
        ]
        product_ids: List[str] = [
            reaction.getProduct(k).getSpecies() for k in range(reaction.getNumProducts())
        ]
        modifier_ids: List[str] = [
            reaction.getModifier(l).getSpecies() for l in range(reaction.getNumModifiers())
        ]

        # Combine all species involved in this reaction
        all_species = reactant_ids + product_ids + modifier_ids

        # Create undirected edges between all pairs of species
        for idx1 in range(len(all_species)):
            for idx2 in range(idx1 + 1, len(all_species)):
                # Skip if they're the same species
                if all_species[idx1] != all_species[idx2]:
                    # Create undirected edge (sort to ensure consistent ordering)
                    edge = tuple(sorted([all_species[idx1], all_species[idx2]]))
                    edges.add(edge)

    return edges


def evaluate_species_interaction_f1(
    true_model: libsbml.Model, pred_model: libsbml.Model
) -> Dict[str, float]:
    """
    Evaluate the F1 score for species interactions (edges) between the true and predicted models.
    Considers undirected edges only and includes modifiers in edge calculations.

    Args:
        true_model: The ground truth SBML model
        pred_model: The predicted SBML model

    Returns:
        Dictionary containing precision, recall, and F1 scores
    """
    # Extract undirected edges from both models including modifiers
    true_edges = extract_species_edges_with_modifiers(true_model)
    pred_edges = extract_species_edges_with_modifiers(pred_model)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(true_edges.intersection(pred_edges))
    false_positives = len(pred_edges.difference(true_edges))
    false_negatives = len(true_edges.difference(pred_edges))

    # Calculate precision, recall, and F1
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if len(true_edges) == 0:
        precision = None
        recall = None
        f1 = None

    # Return metrics
    return {
        "species_edges_undirected_precision": precision,
        "species_edges_undirected_recall": recall,
        "species_edges_undirected_f1": f1,
    }


def extract_typed_species_edges(model: libsbml.Model) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Extract undirected edges between species from an SBML model, categorized by relationship type.

    Args:
        model: The SBML model to extract edges from

    Returns:
        Dictionary with three edge sets: 'reactant_product', 'reactant_modifier', and 'modifier_product'
    """
    edge_sets = {"reactant_product": set(), "reactant_modifier": set(), "modifier_product": set()}

    reactions: libsbml.ListOfReactions = model.getListOfReactions()

    for i in range(reactions.size()):
        reaction: libsbml.Reaction = reactions.get(i)

        # Get reactants, products, and modifiers
        reactant_ids: List[str] = [
            reaction.getReactant(j).getSpecies() for j in range(reaction.getNumReactants())
        ]
        product_ids: List[str] = [
            reaction.getProduct(k).getSpecies() for k in range(reaction.getNumProducts())
        ]
        modifier_ids: List[str] = [
            reaction.getModifier(l).getSpecies() for l in range(reaction.getNumModifiers())
        ]

        # Create reactant-product edges
        for r_id in reactant_ids:
            for p_id in product_ids:
                edge = tuple([r_id, p_id])
                edge_sets["reactant_product"].add(edge)

        # Create reactant-modifier edges
        for r_id in reactant_ids:
            for m_id in modifier_ids:
                edge = tuple([r_id, m_id])
                edge_sets["reactant_modifier"].add(edge)

        # Create modifier-product edges
        for m_id in modifier_ids:
            for p_id in product_ids:
                edge = tuple([m_id, p_id])
                edge_sets["modifier_product"].add(edge)

    return edge_sets


def evaluate_typed_species_interaction_f1(
    true_model: libsbml.Model,
    pred_model: libsbml.Model,
) -> Dict[str, float]:
    """
    Evaluate the F1 scores for different types of species interactions.

    Args:
        true_model: The ground truth SBML model
        pred_model: The predicted SBML model

    Returns:
        Dictionary containing precision, recall, and F1 scores for each edge type
    """
    # Extract typed edges from both models
    true_edge_sets = extract_typed_species_edges(true_model)
    pred_edge_sets = extract_typed_species_edges(pred_model)

    metrics = {}

    # Calculate metrics for each edge type
    for edge_type in ["reactant_product", "reactant_modifier", "modifier_product"]:
        true_edges = true_edge_sets[edge_type]
        pred_edges = pred_edge_sets[edge_type]

        # Calculate true positives, false positives, and false negatives
        true_positives = len(true_edges.intersection(pred_edges))
        false_positives = len(pred_edges.difference(true_edges))
        false_negatives = len(true_edges.difference(pred_edges))

        # Calculate precision, recall, and F1
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Special case: if there are no true edges
        if len(true_edges) == 0:
            precision = None
            recall = None
            f1 = None
        # Add to metrics dictionary
        metrics[f"{edge_type}_precision"] = precision
        metrics[f"{edge_type}_recall"] = recall
        metrics[f"{edge_type}_f1"] = f1

    return metrics


class Evaluator:
    """
    Class for evaluating LLM responses against ground truth.
    """

    def __init__(
        self,
        true_sbml: SBML,
        incomplete_sbml: SBML,
        incomplete_runnable_sbml: SBML,
        mse_norm_type: str | None = "l2",
        ged_timeout: int = 30,
        mse_round: int = 50,
    ):
        self.true_sbml = true_sbml
        self.incomplete_sbml = incomplete_sbml
        self.mse_round = mse_round

        incomplete_runnable_sbml.load_sedml_from_string_or_file(true_sbml.to_sedml_string())
        self.incomplete_runnable_sbml = incomplete_runnable_sbml

        # EXACT REACTION RECOVERY
        self.true_rpm_hashes, self.true_rp_hashes = extract_reaction_hashes(true_sbml.model)
        self.inco_rpm_hashes, self.inco_rp_hashes = extract_reaction_hashes(incomplete_sbml.model)
        self.missing_rp_hashes = self.true_rp_hashes.difference(self.inco_rp_hashes)
        self.missing_rpm_hashes = self.true_rpm_hashes.difference(self.inco_rpm_hashes)

        # SOFT RETRIEVAL
        self.inco_forward_hausdorff, self.inco_reverse_hausdorff = calculate_hausdorff_score(
            self.true_sbml.model,
            self.incomplete_sbml.model,
        )
        self.inco_mean_hausdorff = (self.inco_forward_hausdorff + self.inco_reverse_hausdorff) / 2

        # MSE
        self.mse_norm_type = mse_norm_type
        self.setup_mse_evaluation()

        # GRAPH EDIT DISTANCE
        self.ged_timeout = ged_timeout
        self.inco_rids = set(incomplete_sbml.get_reaction_ids())
        self.inco_sids = set(incomplete_sbml.get_species_ids())
        self.true_graph = self.get_sbml_nx_graph(true_sbml)
        self.inco_graph = self.get_sbml_nx_graph(incomplete_sbml)
        self.inco_ged = compute_ged(self.inco_graph, self.true_graph, timeout=ged_timeout)

        # KINETIC LAW SIMILARITY
        self.true_rid_to_klaw = extract_rid_to_kinetic_law(true_sbml.model)
        self.inco_rid_to_klaw = extract_rid_to_kinetic_law(incomplete_sbml.model)
        self.missing_rid_to_klaw = {
            r: k
            for r, k in self.true_rid_to_klaw.items()
            if r in self.inco_rid_to_klaw and self.inco_rid_to_klaw[r] == None
        }

    def setup_mse_evaluation(self) -> None:
        true_traj = self.run_simulations(self.true_sbml)
        inco_traj = self.run_simulations(self.incomplete_runnable_sbml)
        """
        norms = {}
        if self.mse_norm_type is not None:
            for sid in inco_traj:
                norm = get_vector_norm(true_traj[sid], self.mse_norm_type)
                inco_traj[sid] = apply_vector_norm(inco_traj[sid], norm)
                true_traj[sid] = apply_vector_norm(true_traj[sid], norm)
                norms[sid] = norm
        """
        self.inco_traj = inco_traj
        self.true_traj = true_traj

    def evaluate_mse(self, pred_sbml: SBML) -> Dict[str, float]:
        pred_sbml.load_sedml_from_string_or_file(self.true_sbml.to_sedml_string())
        pred_traj = self.run_simulations(pred_sbml)

        pred_species = [f"[{sid}]" for sid in pred_sbml.get_species_ids()]
        inco_species = [f"[{sid}]" for sid in self.incomplete_sbml.get_species_ids()]

        if not all([x in pred_species for x in inco_species]):
            raise ValueError(
                f"Predicted species {pred_species} do not contain all incomplete species {inco_species}"
            )

        pred_mse = compute_mean_error([self.true_traj], [pred_traj], species_ids=inco_species)
        return dict(observe_mse=pred_mse)

    def evaluate_smape(self, pred_sbml: SBML) -> Dict[str, float]:
        pred_sbml.load_sedml_from_string_or_file(self.true_sbml.to_sedml_string())
        pred_traj = self.run_simulations(pred_sbml)

        pred_species = [f"[{sid}]" for sid in pred_sbml.get_species_ids()]
        pred_smape = compute_dict_smape(self.true_traj, pred_traj)
        return dict(observe_smape=pred_smape)

    def evaluate_smape_noise(self, pred_sbml: SBML, noise: float) -> Dict[str, Dict]:
        results = []
        pred_sbml.load_sedml_from_string_or_file(self.true_sbml.to_sedml_string())
        results = {}
        results[f"perturb_mse_{str(noise)}"] = {}
        for num_iter in range(self.mse_round):
            (
                noise_true_sbml,
                noise_inco_sbml,
                noise_pred_sbml,
            ) = SBML.eval_add_noise_to_initial_concentrations(
                self.true_sbml, self.incomplete_sbml, pred_sbml, noise
            )
            try:
                pred_traj = self.run_simulations(noise_pred_sbml)
                true_traj = self.run_simulations(noise_true_sbml)
            except:
                continue

            pred_species = [f"[{sid}]" for sid in pred_sbml.get_species_ids()]
            inco_species = [f"[{sid}]" for sid in self.incomplete_sbml.get_species_ids()]

            if not all([x in pred_species for x in inco_species]):
                raise ValueError(
                    f"Predicted species {pred_species} do not contain all incomplete species {inco_species}"
                )

            pred_species = [f"[{sid}]" for sid in pred_sbml.get_species_ids()]
            pred_smape = compute_dict_smape(self.true_traj, pred_traj)
            results[f"perturb_mse_{str(noise)}"][f"perturbation_{num_iter}"] = pred_smape
        pred_mse_values = []

        # Iterate through all keys in the results["mse"] dictionary
        for key in results[f"perturb_mse_{str(noise)}"]:
            # Check if the key is a perturbation key
            if key.startswith("perturbation_"):
                # Extract values and append to our lists
                pred_mse = results[f"perturb_mse_{str(noise)}"][key]
                pred_mse_values.append(pred_mse)

        # Compute statistics
        stats = {
            "mean_pred_mse": sum(pred_mse_values) / len(pred_mse_values) if pred_mse_values else 0,
            "max_pred_mse": max(pred_mse_values) if pred_mse_values else 0,
            "min_pred_mse": min(pred_mse_values) if pred_mse_values else 0,
        }

        # Store the stats in the results dictionary
        results[f"perturb_mse_{str(noise)}"]["stats"] = stats
        return results

    def evaluate_solution_complexity(self, pred_sbml: SBML) -> Dict[str, float]:
        return dict(
            pred_length=len(pred_sbml.to_string()),
            true_length=len(self.true_sbml.to_string()),
            inco_length=len(self.incomplete_sbml.to_string()),
        )

    def evaluate_kinetic_law_similarity(self, pred_sbml: SBML) -> Dict[str, float]:
        metrics: Dict[str, List[float]] = defaultdict(list)
        pred_rid_to_klaw = extract_rid_to_kinetic_law(pred_sbml.model)
        for rid, true_klaw in self.missing_rid_to_klaw.items():
            assert true_klaw is not None
            pred_klaw = pred_rid_to_klaw.get(rid, None)
            if pred_klaw is None:
                metrics["bleu"].append(0.0)
            else:
                metrics["bleu"].append(get_average_bleu_from_klaw(true_klaw, pred_klaw))
        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def evaluate_hausdorff_reaction_recovery(self, pred_sbml: SBML) -> Dict[str, float]:
        pred_forward_hausdorff, pred_reverse_hausdorff = calculate_hausdorff_score(
            self.true_sbml.model,
            pred_sbml.model,
        )
        pred_mean_hausdorff = (pred_forward_hausdorff + pred_reverse_hausdorff) / 2
        normalized_mean_hausdorff = normalized_metric(
            pred_mean_hausdorff,
            self.inco_mean_hausdorff,
        )
        return dict(
            normalized_hausdorff=normalized_mean_hausdorff,
            pred_hausdorff=pred_mean_hausdorff,
            inco_hausdorff=self.inco_mean_hausdorff,
        )

    def evaluate_exact_reaction_recovery(self, pred_sbml: SBML) -> Dict[str, float | List[int]]:
        pred_rpm_hashes, pred_rp_hashes = extract_reaction_hashes(pred_sbml.model)

        added_rp_hashes = pred_rp_hashes.difference(self.inco_rp_hashes)
        added_rpm_hashes = pred_rpm_hashes.difference(self.inco_rpm_hashes)

        rp_precision, rp_recall, rp_f1 = 0, 0, 0
        rpm_precision, rpm_recall, rpm_f1 = 0, 0, 0

        # Lists to store reactant and product counts for found reactions
        found_reaction_reactant_counts = []
        found_reaction_product_counts = []

        # Lists to store reactant and product counts for all reactions
        true_reactant_counts = []
        true_product_counts = []
        pred_reactant_counts = []
        pred_product_counts = []

        # Get counts for all reactions in true model
        for i in range(self.true_sbml.model.getNumReactions()):
            reaction = self.true_sbml.model.getReaction(i)
            true_reactant_counts.append(reaction.getNumReactants())
            true_product_counts.append(reaction.getNumProducts())

        # Get counts for all reactions in predicted model
        for i in range(pred_sbml.model.getNumReactions()):
            reaction = pred_sbml.model.getReaction(i)
            pred_reactant_counts.append(reaction.getNumReactants())
            pred_product_counts.append(reaction.getNumProducts())

        if len(self.missing_rp_hashes) > 0 and len(added_rp_hashes):
            found_rp_hashes = added_rp_hashes.intersection(self.missing_rp_hashes)
            rp_recall = len(found_rp_hashes) / len(self.missing_rp_hashes)
            rp_precision = len(found_rp_hashes) / len(added_rp_hashes)
            rp_f1 = (
                2 * rp_precision * rp_recall / (rp_precision + rp_recall)
                if (rp_precision + rp_recall) > 0
                else 0.0
            )

            # Now we need to find the reactions that correspond to the found hashes
            # and count their reactants and products
            for i in range(pred_sbml.model.getNumReactions()):
                reaction = pred_sbml.model.getReaction(i)
                rpm_hash, rp_hash = get_reaction_hash(reaction)
                if rp_hash in found_rp_hashes:
                    found_reaction_reactant_counts.append(reaction.getNumReactants())
                    found_reaction_product_counts.append(reaction.getNumProducts())

        if len(self.missing_rpm_hashes) > 0 and len(added_rpm_hashes):
            found_rpm_hashes = added_rpm_hashes.intersection(self.missing_rpm_hashes)
            rpm_recall = len(found_rpm_hashes) / len(self.missing_rpm_hashes)
            rpm_precision = len(found_rpm_hashes) / len(added_rpm_hashes)
            rpm_f1 = (
                2 * rpm_precision * rpm_recall / (rpm_precision + rpm_recall)
                if (rpm_precision + rpm_recall) > 0
                else 0.0
            )

        return dict(
            rp_precision=rp_precision,
            rp_recall=rp_recall,
            rp_f1=rp_f1,
            rpm_precision=rpm_precision,
            rpm_recall=rpm_recall,
            rpm_f1=rpm_f1,
        )

    def get_sbml_nx_graph(self, sbml: SBML) -> nx.MultiDiGraph:
        d = MySBMLDiagram(
            sbml.model,
            inco_species_ids=self.inco_sids,
            inco_reaction_ids=self.inco_rids,
            **SBML_GRAPH_PARAMS,
        )
        return nx.nx_agraph.from_agraph(d.g, create_using=nx.MultiDiGraph)

    def evaluate_graph_edit_distance(self, pred_sbml: SBML) -> Dict[str, float]:
        pred_graph = self.get_sbml_nx_graph(pred_sbml)
        pred_ged = compute_ged(pred_graph, self.true_graph, self.ged_timeout)
        normalized_ged = normalized_metric(pred_ged, self.inco_ged)
        return dict(pred_ged=pred_ged, inco_ged=self.inco_ged, normalized_ged=normalized_ged)

    def evaluate_species_interaction(self, pred_sbml: SBML) -> Dict[str, float]:
        """
        Evaluate species interactions between predicted and true models,
        reporting metrics for different types of relationships.

        Args:
            pred_sbml: The predicted SBML model

        Returns:
            Dictionary of evaluation metrics for different edge types
        """
        # Get metrics for different types of edges
        metrics = evaluate_typed_species_interaction_f1(self.true_sbml.model, pred_sbml.model)

        return metrics

    def run_simulations(self, sbml: SBML) -> Dict[str, List[float]]:
        """
        Run simulations with the given SBML model under various perturbed initial conditions.

        Args:
            sbml: SBML model
            simulator: Simulator object

        Returns:
            Array of simulation results with shape (time points, species, initial conditions)
        """
        # sbml = SBML.add_noise_to_initial_concentrations(sbml)
        simulation = Simulator(sbml)
        data = simulation.run(observed_species=sbml.get_species_ids())
        assert data.result is not None
        return data.result

    def __call__(
        self, pred_sbml: SBML, difficulty_level: str, perturb: bool = False, noise: float = None
    ) -> EvaluationResult:
        """
        Evaluate the LLM's SBML model against the ground truth.

        Args:
            pred_sbml: The SBML model produced by the LLM
            difficulty_level: Level chosen between "fully_observable" and "partially_observable"

        Returns:
            EvaluationResult containing evaluation scores
        """
        if perturb:
            metrics = self.evaluate_smape_noise(pred_sbml, noise)
            return EvaluationResult(detailed_scores=metrics)
        metrics = self.evaluate_smape(pred_sbml)
        metrics |= self.evaluate_species_interaction(pred_sbml)
        # metrics |= self.evaluate_mse_noise(pred_sbml)
        metrics |= self.evaluate_solution_complexity(pred_sbml)

        if difficulty_level == "fully_observable":
            # metrics |= self.evaluate_hausdorff_reaction_recovery(pred_sbml)
            metrics |= self.evaluate_exact_reaction_recovery(pred_sbml)

        # metrics |= self.evaluate_graph_edit_distance(pred_sbml)

        return EvaluationResult(detailed_scores=metrics)
