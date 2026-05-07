from __future__ import annotations

from collections.abc import Sequence

from corpus_benchmark.models.terminologies import TerminologyConcept
from corpus_benchmark.models.terminologies import TerminologyResource


def _add_weighted_counts(
    target: dict[str, float],
    source: dict[str, float],
    weight: float,
) -> None:
    for name, count in source.items():
        target[name] = target.get(name, 0.0) + count * weight


def _topic_parent_ids(concept: TerminologyConcept) -> list[str]:
    if concept.parent_ids:
        return concept.parent_ids
    return concept.mapped_ui_ids


class JournalMeSHTopicRootCounter:
    def __init__(
        self,
        terminology: TerminologyResource,
        mesh_term_overrides: dict[str, str],
        journal_name_topics: dict[str, list[str]],
    ) -> None:
        self.terminology = terminology
        self.journal_name_topics = journal_name_topics
        self.mesh_concept_overrides = self._build_mesh_concept_overrides(mesh_term_overrides)
        self.root_id_cache: dict[str, dict[str, float]] = {}
        self.root_name_cache: dict[str, dict[str, float]] = {}

    def root_counts(
        self,
        journal_name: str,
        mesh_topics: Sequence[str] | None,
    ) -> dict[str, float]:
        if mesh_topics:
            return self._journal_mesh_topic_counts(mesh_topics)
        return self._journal_name_topic_root_counts(journal_name)

    def mesh_topic_root_counts(self, mesh_topic_name: str) -> dict[str, float]:
        return self._mesh_topic_root_counts_by_name(mesh_topic_name)

    def _build_mesh_concept_overrides(self, mesh_term_overrides: dict[str, str]) -> dict[str, str]:
        mesh_concept_overrides: dict[str, str] = {}
        for mesh_term, topic in mesh_term_overrides.items():
            for concept_id in self.terminology.get_concept_ids_by_name(mesh_term):
                mesh_concept_overrides[concept_id] = topic
        return mesh_concept_overrides

    def _mesh_topic_root_counts_by_id(
        self,
        ui: str,
        active: set[str],
    ) -> dict[str, float]:
        if ui in self.root_id_cache:
            return self.root_id_cache[ui]
        if ui in self.mesh_concept_overrides:
            self.root_id_cache[ui] = {self.mesh_concept_overrides[ui]: 1.0}
            return self.root_id_cache[ui]
        if ui in active:
            return {}

        concept = self.terminology.get_concept(ui)
        if concept is None:
            self.root_id_cache[ui] = {}
            return self.root_id_cache[ui]

        active.add(ui)
        parent_ids = _topic_parent_ids(concept)
        if parent_ids:
            parent_weight = 1.0 / len(parent_ids)
            counts: dict[str, float] = {}
            for parent_id in parent_ids:
                parent_counts = self._mesh_topic_root_counts_by_id(parent_id, active)
                _add_weighted_counts(counts, parent_counts, parent_weight)
        else:
            counts = {concept.name: 1.0}
        active.remove(ui)

        self.root_id_cache[ui] = counts
        return counts

    def _mesh_topic_root_counts_by_name(self, mesh_topic_name: str) -> dict[str, float]:
        if mesh_topic_name in self.root_name_cache:
            return self.root_name_cache[mesh_topic_name]

        concept_ids = self.terminology.get_concept_ids_by_name(mesh_topic_name)
        counts: dict[str, float] = {}
        if concept_ids:
            concept_weight = 1.0 / len(concept_ids)
            for concept_id in concept_ids:
                concept_counts = self._mesh_topic_root_counts_by_id(concept_id, set())
                _add_weighted_counts(counts, concept_counts, concept_weight)

        self.root_name_cache[mesh_topic_name] = counts
        return counts

    def _journal_mesh_topic_counts(self, mesh_topics: Sequence[str]) -> dict[str, float]:
        topic_weight = 1.0 / len(mesh_topics)
        journal_counts: dict[str, float] = {}
        for mesh_topic in mesh_topics:
            topic_counts = self._mesh_topic_root_counts_by_name(mesh_topic)
            _add_weighted_counts(journal_counts, topic_counts, topic_weight)
        return {
            name: count
            for name, count in sorted(
                journal_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        }

    def _journal_name_topic_root_counts(self, journal_name: str) -> dict[str, float]:
        topics = self.journal_name_topics.get(journal_name, [])
        if not topics:
            return {}

        topic_weight = 1.0 / len(topics)
        return {
            topic: topic_weight
            for topic in sorted(topics)
        }
