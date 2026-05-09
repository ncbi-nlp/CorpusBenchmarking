from __future__ import annotations

import logging
import collections
from typing import Set, Dict, Iterable

from corpus_benchmark.context import MetricTarget, get_identifier_links
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.registry import register_terminology_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)

PRECISION = 8
NOT_FOUND_LIMIT = 10
_GLOBAL_BRANCH_COUNT_CACHE: Dict[int, tuple[int, Dict[str, float]]] = {}
_GLOBAL_DEPTH_COUNT_CACHE: Dict[int, tuple[int, Dict[int, float]]] = {}

# TODO make these metrics resource-aware: only do ID lookups in the associated terminology
# TODO report IDs not found as <resource>:<accession>

def _get_not_found_text(not_found: Set[str]):
    not_found_list = list(not_found)[:NOT_FOUND_LIMIT]
    not_found_texts = [f"\"{not_found_item}\"" for not_found_item in not_found_list]
    not_found_text = ", ".join(not_found_texts)
    if len(not_found_list) < len(not_found):
        more_count = len(not_found) - len(not_found_list) 
        not_found_text += f" ...(+{more_count} more)"
    return not_found_text

def _identifier_links_for_terminology(
    target: MetricTarget,
    terminology: TerminologyResource,
    annotation_filter_name: str | None,
):
    return [
        link
        for link in get_identifier_links(target, annotation_filter_name)
        if link.identifier is not None and terminology.accepts_resource(link.resource)
    ]


def _count_by_branch(terminology: TerminologyResource, ids: Iterable[str]) -> Dict[str, float]:
    counts = collections.defaultdict(float)
    not_found = set()
    for ui in ids:
        concepts = terminology.resolve_to_tree_concepts(ui)
        if not concepts:
            not_found.add(ui)
            continue
        keys = [
            key
            for concept in concepts
            for key in terminology.top_ancestor_ids(concept.ui)
        ]
        if not keys:
            continue
        weight = 1.0 / len(keys)
        for key in keys:
            counts[key] += weight
    if len(not_found) > 0:
        logger.warning("No concept found for: {}".format(_get_not_found_text(not_found)))
    return dict(sorted(counts.items()))


def _count_by_depth(terminology: TerminologyResource, ids: Iterable[str]) -> Dict[int, float]:
    counts = collections.defaultdict(float)
    not_found = set()
    for ui in ids:
        concepts = terminology.resolve_to_tree_concepts(ui)
        if not concepts:
            not_found.add(ui)
            continue
        depths = [terminology.depth_for_concept(concept) for concept in concepts]
        weight = 1.0 / len(depths)
        for depth in depths:
            counts[depth] += weight
    if len(not_found) > 0:
        logger.warning("No concept found for: {}".format(_get_not_found_text(not_found)))
    return dict(sorted(counts.items()))


def _get_global_counts_by_branch(terminology: TerminologyResource) -> Dict[str, float]:
    cache_key = id(terminology)
    concept_count = len(terminology.concepts)
    cached = _GLOBAL_BRANCH_COUNT_CACHE.get(cache_key)
    if cached is not None and cached[0] == concept_count:
        return cached[1]
    target_ids = [c.ui for c in terminology.concepts.values()]
    counts = _count_by_branch(terminology, target_ids)
    _GLOBAL_BRANCH_COUNT_CACHE[cache_key] = (concept_count, counts)
    return counts


def _get_global_counts_by_depth(terminology: TerminologyResource) -> Dict[int, float]:
    cache_key = id(terminology)
    concept_count = len(terminology.concepts)
    cached = _GLOBAL_DEPTH_COUNT_CACHE.get(cache_key)
    if cached is not None and cached[0] == concept_count:
        return cached[1]
    target_ids = [c.ui for c in terminology.concepts.values()]
    counts = _count_by_depth(terminology, target_ids)
    _GLOBAL_DEPTH_COUNT_CACHE[cache_key] = (concept_count, counts)
    return counts


def _branch_label(terminology: TerminologyResource, branch_code: str) -> str:
    concept = terminology.get_concept(branch_code)
    if concept is not None:
        return concept.name
    for concept in terminology.concepts.values():
        if branch_code in getattr(concept, "tree_numbers", []):
            return concept.name
    return terminology.treetop_names.get(branch_code, branch_code)


@register_terminology_metric("high_level_concept_counts", supports_annotation_scope=True)
def high_level_concept_counts(target: MetricTarget, result_name: str, terminology: TerminologyResource, annotation_filter_name: str | None = None, **params) -> SubsetMetricResult:
    identifier_links = _identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]
    missing_ids = [ui for ui in ids if terminology.get_concept(ui) is None]
    missing_ids = sorted(list(set(missing_ids)))

    corpus_counts = _count_by_branch(terminology, ids)
    global_counts = _get_global_counts_by_branch(terminology)

    all_branches = sorted(corpus_counts.keys())
    rows = []
    for branch_code in all_branches:
        count = corpus_counts.get(branch_code, 0.0)
        terminology_total = global_counts.get(branch_code, 0.0)
        proportion = count / terminology_total if terminology_total > 0 else 0.0

        rows.append(
            {
                "branch_code": branch_code,
                "label": _branch_label(terminology, branch_code),
                "treetop": branch_code.split(".")[0],
                "treetop_name": terminology.treetop_names.get(branch_code.split(".")[0]) or _branch_label(terminology, branch_code),
                "count": round(count, PRECISION),
                "terminology_total_count": round(terminology_total, PRECISION),
                "mesh_total_count": round(terminology_total, PRECISION),
                "proportion": round(proportion, PRECISION),
            }
        )

    return SubsetMetricResult(
        result_name=result_name,
        metric_name="high_level_concept_counts",
        subset_name=target.name,
        value=rows,
        details={
            "n_input_ids": len(ids),
            "n_missing_ids": len(missing_ids),
            "missing_ids": missing_ids,
            "terminology": terminology.name,
            "resource_aliases": terminology.aliases,
        },
    )


@register_terminology_metric("concept_depth_counts", supports_annotation_scope=True)
def concept_depth_counts(target: MetricTarget, result_name: str, terminology: TerminologyResource, annotation_filter_name: str | None = None, **params) -> SubsetMetricResult:
    identifier_links = _identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]

    corpus_counts = _count_by_depth(terminology, ids)
    global_counts = _get_global_counts_by_depth(terminology)

    all_depths = sorted(set(corpus_counts.keys()) | set(global_counts.keys()))
    rows = []
    for d in all_depths:
        c_count = corpus_counts.get(d, 0.0)
        m_count = global_counts.get(d, 0.0)
        rows.append(
            {
                "depth": d,
                "count": round(c_count, PRECISION),
                "terminology_total_count": round(m_count, PRECISION),
                "mesh_total_count": round(m_count, PRECISION),
                "proportion": round(c_count / m_count, PRECISION) if m_count > 0 else 0.0,
            }
        )

    missing_ids = sorted({ui for ui in ids if terminology.get_concept(ui) is None})
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="concept_depth_counts",
        subset_name=target.name,
        value=rows,
        details={
            "n_input_ids": len(ids),
            "n_missing_ids": len(missing_ids),
            "missing_ids": missing_ids,
            "terminology": terminology.name,
            "resource_aliases": terminology.aliases,
        },
    )
