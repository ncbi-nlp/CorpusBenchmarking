from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
_NAME_INDEX_CACHE: Dict[int, tuple[int, Dict[str, List[str]]]] = {}
_ID_INDEX_CACHE: Dict[int, tuple[int, Dict[str, str]]] = {}
_DEPTH_CACHE: Dict[int, tuple[int, Dict[str, int]]] = {}
_TOP_ANCESTOR_CACHE: Dict[int, tuple[int, Dict[str, List[str]]]] = {}


@dataclass(slots=True)
class TerminologyConcept:
    ui: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    tree_numbers: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    scope_note: Optional[str] = None
    mapped_ui_ids: List[str] = field(default_factory=list)
    alt_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TerminologyResource:
    name: str
    concepts: Dict[str, TerminologyConcept] = field(default_factory=dict)
    tree_to_ids: Dict[str, List[str]] = field(default_factory=dict)
    treetop_names: Dict[str, str] = field(default_factory=dict)
    resource_aliases: List[str] = field(default_factory=list)
    id_prefix: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.resource_aliases:
            self.resource_aliases = [self.name]

    @property
    def aliases(self) -> List[str]:
        return list(getattr(self, "resource_aliases", None) or [self.name])

    @property
    def prefix(self) -> Optional[str]:
        return getattr(self, "id_prefix", None)

    @staticmethod
    def normalize_identifier(ui: str | None) -> str | None:
        if ui is None:
            return None
        value = str(ui).strip()
        if not value:
            return None
        return value.replace("_", ":")

    @staticmethod
    def normalize_resource(resource: str | None) -> str | None:
        if resource is None:
            return None
        value = str(resource).strip()
        if not value:
            return None
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def accepts_resource(self, resource: str | None) -> bool:
        normalized = self.normalize_resource(resource)
        if normalized is None:
            return False
        return normalized in {self.normalize_resource(alias) for alias in self.aliases}

    def _candidate_ids(self, ui: str | None) -> List[str]:
        normalized = self.normalize_identifier(ui)
        if normalized is None:
            return []
        candidates = [normalized]
        if self.prefix:
            prefix = self.prefix.rstrip(":")
            if not normalized.casefold().startswith(prefix.casefold() + ":"):
                candidates.append(f"{prefix}:{normalized}")
        return candidates

    def get_concept(self, ui: str) -> Optional[TerminologyConcept]:
        id_index = self._id_index()
        for candidate in self._candidate_ids(ui):
            primary_id = id_index.get(candidate)
            if primary_id is not None:
                return self.concepts.get(primary_id)
        return None

    def _id_index(self) -> Dict[str, str]:
        cache_key = id(self)
        concept_count = len(self.concepts)
        cached = _ID_INDEX_CACHE.get(cache_key)
        if cached is not None and cached[0] == concept_count:
            return cached[1]

        index: Dict[str, str] = {}
        for concept in self.concepts.values():
            ids = [concept.ui, *getattr(concept, "alt_ids", [])]
            for candidate in ids:
                for normalized_candidate in self._candidate_ids(candidate):
                    index[normalized_candidate] = concept.ui
        _ID_INDEX_CACHE[cache_key] = (concept_count, index)
        return index

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.split()).casefold()

    def _name_index(self) -> Dict[str, List[str]]:
        cache_key = id(self)
        concept_count = len(self.concepts)
        cached = _NAME_INDEX_CACHE.get(cache_key)
        if cached is not None and cached[0] == concept_count:
            return cached[1]

        index: Dict[str, List[str]] = {}
        for concept in self.concepts.values():
            index.setdefault(self._normalize_name(concept.name), []).append(concept.ui)
        _NAME_INDEX_CACHE[cache_key] = (concept_count, index)
        return index

    def get_concept_ids_by_name(self, name: str) -> List[str]:
        """Return concept identifiers whose preferred name exactly matches name."""
        return list(self._name_index().get(self._normalize_name(name), []))

    def resolve_to_tree_concepts(self, ui: str) -> List[TerminologyConcept]:
        """
        Resolve an input ID to one or more concepts that carry tree numbers.
        (Similar to resolve_descriptor_records in MeSH)
        """
        concept = self.get_concept(ui)
        if concept is None:
            return []
        tree_numbers = getattr(concept, "tree_numbers", [])
        parent_ids = getattr(concept, "parent_ids", [])
        mapped_ui_ids = getattr(concept, "mapped_ui_ids", [])
        if tree_numbers or parent_ids or not mapped_ui_ids:
            return [concept]

        resolved = []
        for mapped_id in mapped_ui_ids:
            mapped = self.get_concept(mapped_id)
            if mapped and getattr(mapped, "tree_numbers", []):
                resolved.append(mapped)
        return resolved

    def depth_for_concept(self, concept: TerminologyConcept) -> int:
        cache_key = id(self)
        concept_count = len(self.concepts)
        cached = _DEPTH_CACHE.get(cache_key)
        if cached is None or cached[0] != concept_count:
            cached = (concept_count, {})
            _DEPTH_CACHE[cache_key] = cached
        depth_cache = cached[1]
        if concept.ui not in depth_cache:
            depth_cache[concept.ui] = self._depth_for_concept(concept, set())
        return depth_cache[concept.ui]

    def _depth_for_concept(self, concept: TerminologyConcept, seen: set[str]) -> int:
        if concept.ui in seen:
            return 1
        seen.add(concept.ui)
        tree_numbers = getattr(concept, "tree_numbers", [])
        parent_ids = getattr(concept, "parent_ids", [])
        if tree_numbers:
            return min(len(tree.split(".")) for tree in tree_numbers)
        if not parent_ids:
            return 1
        parent_depths = [
            self._depth_for_concept(parent, seen.copy())
            for parent_id in parent_ids
            for parent in [self.get_concept(parent_id)]
            if parent is not None and parent.ui != concept.ui
        ]
        return (min(parent_depths) + 1) if parent_depths else 1

    def top_ancestor_ids(self, ui: str) -> List[str]:
        cache_key = id(self)
        concept_count = len(self.concepts)
        cached = _TOP_ANCESTOR_CACHE.get(cache_key)
        if cached is None or cached[0] != concept_count:
            cached = (concept_count, {})
            _TOP_ANCESTOR_CACHE[cache_key] = cached
        ancestor_cache = cached[1]
        normalized = self.normalize_identifier(ui) or ui
        if normalized not in ancestor_cache:
            ancestor_cache[normalized] = self._top_ancestor_ids(normalized, set())
        return ancestor_cache[normalized]

    def _top_ancestor_ids(self, ui: str, seen: set[str]) -> List[str]:
        if ui in seen:
            return []
        seen.add(ui)
        concept = self.get_concept(ui)
        if concept is None:
            return []
        tree_numbers = getattr(concept, "tree_numbers", [])
        parent_ids = getattr(concept, "parent_ids", [])
        if tree_numbers:
            roots = []
            for tree in tree_numbers:
                root = tree.split(".")[0]
                for candidate in self.tree_to_ids.get(root, []):
                    if candidate not in roots:
                        roots.append(candidate)
            return roots or [concept.ui]
        if not parent_ids:
            return [concept.ui]
        roots = []
        for parent_id in parent_ids:
            for root in self._top_ancestor_ids(parent_id, seen.copy()):
                if root not in roots:
                    roots.append(root)
        return roots or [concept.ui]
