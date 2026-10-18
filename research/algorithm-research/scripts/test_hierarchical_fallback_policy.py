#!/usr/bin/env python3
"""Synthetic regression tests for causal hierarchical fallback policies."""

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from hierarchical_fallback_policy import (  # noqa: E402
    FallbackPolicy,
    Hierarchy,
    ParentEdge,
    action_utility,
    add_outcome,
    attach_open_mass,
    build_hierarchy,
    deduplicate_equivalent_service_homes,
    experimental_rank_score,
    greedy_menu,
    menu_value,
    path_support,
    posterior_from_scores,
    rank_proxy_posterior,
    useful_fallback_actions,
)


def target(identifier, kind="page", *, parent=None, bundle=None, locator=None):
    value = {"id": identifier, "kind": kind}
    if parent is not None:
        value["parentID"] = parent
    if bundle is not None:
        value["bundleID"] = bundle
    if locator is not None:
        value["locator"] = locator
    return value


def branch_hierarchy(*, executable=None):
    metadata = {
        "home": target("home"),
        "section": target("section", parent="home"),
        "a": target("a", parent="section"),
        "b": target("b", parent="section"),
        "c": target("c", parent="section"),
        "other": target("other"),
    }
    return build_hierarchy(metadata.values(), executable=executable, include_owner_apps=False)


class HierarchyConstructionTests(unittest.TestCase):
    def test_infers_recorded_url_ancestor_and_derives_unvisited_ancestors(self):
        parent = target("parent", locator="https://example.test/work")
        unrelated = target("child", locator="https://example.test/work/item")
        early = build_hierarchy([parent, unrelated], include_owner_apps=False)
        self.assertEqual(early.paths("child")["parent"], ("url_parent",))

        confirmed = target("child", parent="parent", locator=unrelated["locator"])
        late = build_hierarchy([parent, confirmed], include_owner_apps=False)
        self.assertEqual(late.paths("child")["parent"], ("explicit_parent",))

        missing_parent = build_hierarchy([confirmed], include_owner_apps=False)
        paths = missing_parent.paths("child")
        derived_locators = {
            missing_parent.metadata[action].get("locator")
            for action in missing_parent.derived_parent_ids
        }
        self.assertIn("https://example.test/work/", derived_locators)
        self.assertIn("https://example.test/", derived_locators)
        self.assertTrue(missing_parent.derived_parent_ids.issubset(missing_parent.executable))
        self.assertTrue(any(path == ("url_parent",) for action, path in paths.items()
                            if action in missing_parent.derived_parent_ids))

    def test_generic_site_root_and_nested_url_parents_are_usable_fallbacks(self):
        home = target("bili-home", locator="https://www.bilibili.com/")
        section = target("bili-videos", locator="https://www.bilibili.com/video/")
        clip = target(
            "bili-clip",
            locator="https://www.bilibili.com/video/BV1234567890?part=2#time",
        )
        hierarchy = build_hierarchy([home, section, clip], include_owner_apps=False)
        policy = FallbackPolicy("product", "product")

        self.assertEqual(hierarchy.paths("bili-clip")["bili-videos"], ("url_parent",))
        self.assertEqual(hierarchy.paths("bili-clip")["bili-home"], ("service_home",))
        self.assertAlmostEqual(
            action_utility(hierarchy, "bili-videos", "bili-clip", policy), .8,
        )
        self.assertAlmostEqual(
            action_utility(hierarchy, "bili-home", "bili-clip", policy), .6,
        )
        self.assertIn(
            "bili-home",
            useful_fallback_actions(
                hierarchy, "bili-clip", policy, current_action="bili-clip",
            ),
        )
        self.assertNotIn(
            "bili-home",
            useful_fallback_actions(
                hierarchy, "bili-clip", policy, current_action="bili-home",
            ),
        )

    def test_url_parent_requires_same_origin_and_component_boundary(self):
        home = target("home", locator="https://example.test/" )
        sibling = target("sibling", locator="https://example.test/workshop")
        child = target("child", locator="https://example.test/work/item")
        other_host = target("other-host", locator="https://other.example.test/work")
        hierarchy = build_hierarchy([home, sibling, child, other_host], include_owner_apps=False)
        ancestors = hierarchy.ancestors("child")
        self.assertIn("home", ancestors)
        self.assertIn("https://example.test/work/", {
            hierarchy.metadata[action].get("locator")
            for action in ancestors
        })
        self.assertNotIn("sibling", hierarchy.ancestors("child"))
        self.assertNotIn("other-host", hierarchy.ancestors("child"))

    def test_folder_and_file_targets_use_recorded_and_derived_ancestors(self):
        work = target("work", "directory", locator="file:///tmp/Workspace%20Files/")
        project = target(
            "project", "directory", locator="file://localhost/tmp/Workspace%20Files/Project",
        )
        notes = target(
            "notes", "directory", locator="file:///tmp/Workspace%20Files/Project/Notes/",
        )
        document = target(
            "document", "file",
            locator="file:///tmp/Workspace%20Files/Project/Notes/outline.md",
        )
        hierarchy = build_hierarchy([work, project, notes, document], include_owner_apps=False)

        self.assertEqual(hierarchy.paths("notes")["project"], ("path_parent",))
        self.assertEqual(hierarchy.paths("document")["notes"], ("path_parent",))
        self.assertEqual(
            hierarchy.paths("document")["work"],
            ("path_parent", "path_parent", "path_parent"),
        )
        self.assertAlmostEqual(
            action_utility(
                hierarchy, "work", "document", FallbackPolicy("product", "product"),
            ),
            .8 ** 3,
        )

        unknown = target(
            "unknown-directory", "directory",
            locator="file:///tmp/Unseen%20Parent/Project/Notes/",
        )
        derived = build_hierarchy([unknown], include_owner_apps=False)
        derived_paths = derived.paths("unknown-directory")
        direct_parent = next(
            action for action in derived.derived_parent_ids
            if derived.metadata[action].get("locator") == "file:///tmp/Unseen%20Parent/Project"
        )
        self.assertEqual(derived_paths[direct_parent], ("path_parent",))
        self.assertAlmostEqual(
            action_utility(
                derived, direct_parent, "unknown-directory",
                FallbackPolicy("product", "product"),
            ),
            .8,
        )

    def test_parent_must_be_executable_even_when_its_locator_is_known(self):
        home = target("home", locator="https://example.test/")
        child = target("child", locator="https://example.test/page")
        hierarchy = build_hierarchy(
            [home, child], executable={"child"}, include_owner_apps=False,
        )
        self.assertEqual(
            useful_fallback_actions(
                hierarchy, "child", FallbackPolicy("product", "product"),
            ),
            {},
        )

    def test_unvisited_directory_parents_are_derived_without_scanning(self):
        child = target("child-dir", "directory", locator="file:///tmp/never-observed/sub")
        hierarchy = build_hierarchy([child], include_owner_apps=False)
        derived_locators = {
            hierarchy.metadata[action].get("locator")
            for action in hierarchy.derived_parent_ids
        }
        self.assertIn("file:///tmp/never-observed", derived_locators)
        self.assertIn("file:///tmp", derived_locators)
        self.assertEqual(hierarchy.derived_parent_ids, hierarchy.executable - {"child-dir"})

        known_parent = target("known-parent", "directory", locator="file:///tmp/known")
        separate_child = target("separate-child", "directory", locator="file:///tmp/knownish/sub")
        unrelated = build_hierarchy([known_parent, separate_child], include_owner_apps=False)
        self.assertNotIn("known-parent", unrelated.ancestors("separate-child"))

    def test_derived_site_home_is_usable_without_a_homepage_visit(self):
        child = target(
            "child", locator="https://www.example.test/a/page?mode=full#part",
        )
        hierarchy = build_hierarchy([child], include_owner_apps=False)
        home = next(
            action for action in hierarchy.derived_parent_ids
            if hierarchy.metadata[action].get("locator") == "https://www.example.test/"
        )
        self.assertEqual(hierarchy.paths("child")[home], ("service_home",))
        self.assertAlmostEqual(
            action_utility(
                hierarchy, home, "child", FallbackPolicy("product", "product"),
            ),
            .6,
        )
        self.assertEqual(
            hierarchy.metadata["child"]["locator"],
            "https://www.example.test/a/page?mode=full#part",
        )

    def test_unvisited_intermediate_url_scores_for_a_new_child_on_its_path(self):
        known = target("known", locator="https://example.test/projects/old")
        hierarchy = build_hierarchy([known], include_owner_apps=False)
        middle = next(
            action for action in hierarchy.derived_parent_ids
            if hierarchy.metadata[action].get("locator") == "https://example.test/projects/"
        )
        policy = FallbackPolicy("product0.8", "product", content_retention=0.8)
        same_path = target("future", locator="https://example.test/projects/new")
        other_path = target("other", locator="https://example.test/other/new")

        self.assertAlmostEqual(
            experimental_rank_score([middle], same_path, hierarchy, policy)["top1"],
            0.8,
        )
        self.assertEqual(
            experimental_rank_score([middle], other_path, hierarchy, policy)["top1"],
            0.0,
        )

    def test_chatgpt_home_and_owner_app_are_alternative_confirmed_parents(self):
        app = target("app:browser", "app", bundle="browser")
        home = target("chat-home", bundle="browser", locator="https://chatgpt.com/")
        conversation = target(
            "chat-child",
            bundle="browser",
            locator="https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc",
        )
        hierarchy = build_hierarchy([app, home, conversation])
        self.assertEqual(hierarchy.paths("chat-child")["chat-home"], ("service_home",))
        self.assertEqual(hierarchy.paths("chat-child")["app:browser"], ("owner_app",))

    def test_multiple_service_home_actions_are_stable_and_never_parent_each_other(self):
        first = target("home-a", locator="https://chatgpt.com/")
        second = target("home-b", locator="https://chatgpt.com/")
        third = target("home-c", locator="https://chatgpt.com/")
        fourth = target("home-d", locator="https://chatgpt.com/")
        child = target(
            "chat-child", locator="https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc",
        )
        forward = build_hierarchy(
            [first, second, third, fourth, child], include_owner_apps=False,
        )
        reverse = build_hierarchy(
            [child, fourth, third, second, first], include_owner_apps=False,
        )
        self.assertEqual(forward.parents, reverse.parents)
        self.assertEqual(forward.ancestors("home-a"), set())
        self.assertEqual(forward.ancestors("home-b"), set())
        child_ancestors = forward.ancestors("chat-child")
        self.assertTrue({"home-a", "home-b", "home-c", "home-d"}.issubset(
            child_ancestors,
        ))
        self.assertEqual(
            {
                forward.metadata[action].get("locator")
                for action in child_ancestors & forward.derived_parent_ids
            },
            {"https://chatgpt.com/c/"},
        )

    def test_owner_app_requires_a_real_matching_app_action(self):
        fake = target("app:browser", "page", bundle="browser")
        child = target("child", bundle="browser")
        hierarchy = build_hierarchy([fake, child])
        self.assertNotIn("app:browser", hierarchy.ancestors("child"))

    def test_add_outcome_uses_known_parent_without_making_truth_executable(self):
        base = build_hierarchy([target("parent")], include_owner_apps=False)
        expanded = add_outcome(base, target("future-truth", parent="parent"))
        self.assertEqual(expanded.paths("future-truth")["parent"], ("explicit_parent",))
        self.assertNotIn("future-truth", expanded.executable)

    def test_add_outcome_derives_parents_without_making_future_truth_executable(self):
        home = target("known-home", locator="https://example.test/")
        known_folder = target("known-folder", "directory", locator="file:///tmp/known")
        base = build_hierarchy([home, known_folder], include_owner_apps=False)
        expanded = add_outcome(
            base,
            target("future-page", locator="https://example.test/a/b?keep=exact"),
        )
        expanded = add_outcome(
            expanded,
            target("future-folder", "directory", locator="file:///tmp/known/sub"),
        )
        self.assertEqual(expanded.paths("future-page")["known-home"], ("service_home",))
        self.assertEqual(expanded.paths("future-folder")["known-folder"], ("path_parent",))
        self.assertNotIn("future-page", expanded.executable)
        self.assertNotIn("future-folder", expanded.executable)
        new_derived = expanded.derived_parent_ids - base.derived_parent_ids
        self.assertTrue(new_derived.isdisjoint(expanded.executable))

    def test_cycles_are_rejected(self):
        metadata = {"a": target("a"), "b": target("b")}
        with self.assertRaisesRegex(ValueError, "cycle"):
            Hierarchy(metadata, {
                "a": [ParentEdge("b", "explicit_parent")],
                "b": [ParentEdge("a", "explicit_parent")],
            })


class PathSupportTests(unittest.TestCase):
    def test_arrivals_add_support_to_unvisited_ancestors_without_counting_them_as_arrivals(self):
        first = target("first", locator="https://example.test/docs/a")
        second = target("second", locator="https://example.test/docs/b")
        hierarchy = build_hierarchy([first, second], include_owner_apps=False)
        derived_by_locator = {
            hierarchy.metadata[action].get("locator"): action
            for action in hierarchy.derived_parent_ids
        }
        section = derived_by_locator["https://example.test/docs/"]
        home = derived_by_locator["https://example.test/"]

        support = path_support([
            ("arrival-a", "first", 1.0),
            ("arrival-b", "second", 1.0),
        ], hierarchy)

        self.assertEqual(support["exact"], {"first": 1.0, "second": 1.0})
        self.assertEqual(support["path"][section], 2.0)
        self.assertEqual(support["path"][home], 2.0)
        self.assertNotIn(section, support["exact"])
        self.assertNotIn(home, support["exact"])

    def test_path_support_counts_each_structural_ancestor_once_per_outcome(self):
        hierarchy = branch_hierarchy()
        support = path_support([
            ("outcome-1", "a", 1.0),
            ("outcome-1", "a", 1.0),
            ("outcome-2", "b", 1.0),
            ("outcome-3", "section", 1.0),
        ], hierarchy)

        self.assertEqual(support["exact"], {"a": 1.0, "b": 1.0, "section": 1.0})
        self.assertEqual(support["path"], {"section": 2.0, "home": 3.0})
        self.assertEqual(support["combined"], {
            "a": 1.0, "b": 1.0, "section": 3.0, "home": 3.0,
        })
        self.assertEqual(support["unique_outcomes"]["n"], 3)

    def test_asof_catalog_and_strict_maturity_are_not_backfilled(self):
        parent = target("parent")
        child = target("child", parent="parent")
        early_hierarchy = build_hierarchy([child], include_owner_apps=False)
        late_hierarchy = build_hierarchy([parent, child], include_owner_apps=False)
        arrivals = [(9, "old", "child"), (10, "equal", "child"), (11, "future", "child")]

        def support_before(cutoff, hierarchy):
            mature = [(outcome, name, 1.0) for ready, outcome, name in arrivals if ready < cutoff]
            return path_support(mature, hierarchy)

        at_ten = support_before(10, early_hierarchy)
        self.assertEqual(at_ten["exact"]["child"], 1.0)
        self.assertNotIn("parent", at_ten["combined"])
        at_eleven = support_before(11, late_hierarchy)
        self.assertEqual(at_eleven["exact"]["child"], 2.0)
        self.assertEqual(at_eleven["path"]["parent"], 2.0)

    def test_duplicate_arrival_identity_cannot_change_target_or_weight(self):
        hierarchy = branch_hierarchy()
        with self.assertRaisesRegex(ValueError, "arrival ID"):
            path_support([("same", "a", 1.0), ("same", "b", 1.0)], hierarchy)

    def test_content_and_owner_app_support_are_kept_separate(self):
        app = target("app:browser", "app", bundle="browser")
        parent = target("parent", bundle="browser")
        child = target("child", parent="parent", bundle="browser")
        hierarchy = build_hierarchy([app, parent, child])
        support = path_support([("one", "child", 1.0)], hierarchy)
        self.assertEqual(support["content_path"], {"parent": 1.0})
        self.assertEqual(support["owner_app_support"], {"app:browser": 1.0})

    def test_one_url_arrival_adds_support_to_recorded_url_ancestors(self):
        home = target("home", locator="https://example.test/")
        section = target("section", locator="https://example.test/notes/")
        page = target("page", locator="https://example.test/notes/item")
        hierarchy = build_hierarchy([home, section, page], include_owner_apps=False)
        support = path_support([("arrival-1", "page", 1.0)], hierarchy)

        self.assertEqual(support["exact"], {"page": 1.0})
        self.assertEqual(support["path"], {"section": 1.0, "home": 1.0})
        self.assertEqual(support["combined"]["page"], 1.0)
        self.assertEqual(support["combined"]["section"], 1.0)
        self.assertEqual(support["combined"]["home"], 1.0)

    def test_equivalent_home_ids_contribute_one_semantic_parent_support(self):
        first = target("home-a", locator="https://chatgpt.com/")
        second = target("home-b", locator="https://chatgpt.com/")
        third = target("home-c", locator="https://chatgpt.com/")
        fourth = target("home-d", locator="https://chatgpt.com/")
        child = target(
            "chat-child",
            locator="https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc",
        )
        hierarchy = build_hierarchy(
            [first, second, third, fourth, child], include_owner_apps=False,
        )
        support = path_support([("arrival", "chat-child", 1.0)], hierarchy)
        self.assertEqual(support["exact"], {"chat-child": 1.0})
        derived_path = next(iter(hierarchy.derived_parent_ids))
        self.assertEqual(support["path"], {"home-a": 1.0, derived_path: 1.0})


class ScoreSemanticsTests(unittest.TestCase):
    def test_declared_probabilities_and_logits_convert_but_rankings_are_rejected(self):
        self.assertEqual(
            posterior_from_scores({"a": 0.6, "b": 0.4}, "probabilities"),
            {"a": 0.6, "b": 0.4},
        )
        logits = posterior_from_scores({"a": 0.0, "b": math.log(2.0)}, "logits")
        self.assertAlmostEqual(logits["a"], 1.0 / 3.0)
        self.assertAlmostEqual(logits["b"], 2.0 / 3.0)
        with self.assertRaisesRegex(ValueError, "not probabilities"):
            posterior_from_scores({"a": 0.6, "b": 0.4}, "ranking")

    def test_rank_proxy_is_scale_invariant_and_explicit(self):
        small = rank_proxy_posterior({"a": 5.0, "b": 3.0, "c": 1.0})
        large = rank_proxy_posterior({"a": 50.0, "b": 30.0, "c": 10.0})
        self.assertEqual(small, large)
        self.assertAlmostEqual(sum(small.values()), 1.0)
        self.assertGreater(small["a"], small["b"])
        self.assertGreater(small["b"], small["c"])
        with self.assertRaisesRegex(ValueError, "nonempty and finite"):
            rank_proxy_posterior({})
        with self.assertRaisesRegex(ValueError, "nonempty and finite"):
            rank_proxy_posterior({"a": math.nan})

    def test_open_mass_stays_unknown_under_eligible_parents(self):
        hierarchy = branch_hierarchy()
        posterior, expanded = attach_open_mass(
            {"a": 0.75, "other": 0.25}, hierarchy, 0.4,
            {"section": 3.0, "home": 1.0, "missing": 100.0},
        )
        self.assertAlmostEqual(posterior["a"], 0.45)
        self.assertAlmostEqual(posterior["other"], 0.15)
        unknown_by_parent = {
            edges[0].parent: key
            for key, edges in expanded.parents.items()
            if key.startswith("hierarchy:unknown-child:")
        }
        self.assertAlmostEqual(posterior[unknown_by_parent["section"]], 0.3)
        self.assertAlmostEqual(posterior[unknown_by_parent["home"]], 0.1)
        self.assertTrue(set(unknown_by_parent.values()).isdisjoint(expanded.executable))
        self.assertAlmostEqual(sum(posterior.values()), 1.0)
        self.assertTrue(all(value >= 0 for value in posterior.values()))

    def test_open_mass_uses_app_retention_for_an_app_parent(self):
        app = target("app:browser", "app", bundle="browser")
        hierarchy = build_hierarchy([app])
        posterior, expanded = attach_open_mass(
            {"app:browser": 1.0}, hierarchy, 0.3, {"app:browser": 1.0},
        )
        unknown = next(key for key in posterior if key.startswith("hierarchy:unknown-child:"))
        policy = FallbackPolicy("product-80", "product", content_retention=0.8, app_retention=0.6)
        self.assertAlmostEqual(
            menu_value(["app:browser"], {unknown: 1.0}, expanded, policy), 0.6,
        )

    def test_open_mass_rounding_never_creates_negative_probability(self):
        hierarchy = branch_hierarchy()
        posterior, _ = attach_open_mass(
            {"a": 0.1, "b": 0.2, "c": 0.7}, hierarchy, 0.37,
            {"home": 0.1, "section": 0.2, "other": 0.7},
        )
        self.assertTrue(all(value >= 0 for value in posterior.values()))
        self.assertAlmostEqual(math.fsum(posterior.values()), 1.0)


class HierarchicalChoiceTests(unittest.TestCase):
    def setUp(self):
        self.hierarchy = branch_hierarchy()
        self.policy = FallbackPolicy("product-80", "product", content_retention=0.8)

    def test_unvisited_home_can_win_by_aggregating_known_child_probability(self):
        first = target("first", locator="https://example.test/a/item")
        second = target("second", locator="https://example.test/b/item")
        other = target("other")
        hierarchy = build_hierarchy([first, second, other], include_owner_apps=False)
        home = next(
            action for action in hierarchy.derived_parent_ids
            if hierarchy.metadata[action].get("locator") == "https://example.test/"
        )

        menu, _ = greedy_menu(
            {"first": 0.4, "second": 0.4, "other": 0.2},
            hierarchy,
            FallbackPolicy("product", "product"),
            size=1,
        )

        self.assertEqual(menu, [home])

    def test_product_and_saturating_policies_decrease_with_distance(self):
        product = self.policy
        saturating = FallbackPolicy(
            "saturating-80", "saturating", content_retention=0.8, floor=0.4,
        )
        self.assertEqual(product.path_utility([]), 1.0)
        self.assertAlmostEqual(product.path_utility(["explicit_parent"]), 0.8)
        self.assertAlmostEqual(product.path_utility(["explicit_parent"] * 2), 0.64)
        self.assertAlmostEqual(saturating.path_utility(["explicit_parent"]), 0.8)
        self.assertGreater(
            saturating.path_utility(["explicit_parent"] * 2),
            product.path_utility(["explicit_parent"] * 2),
        )

    def test_service_home_stays_at_the_agreed_point_six_for_every_curve(self):
        product = FallbackPolicy(
            "product-80", "product", content_retention=0.8,
        )
        saturating = FallbackPolicy(
            "saturating-80", "saturating", content_retention=0.8, floor=0.4,
        )
        self.assertAlmostEqual(product.path_utility(["service_home"]), 0.6)
        self.assertAlmostEqual(saturating.path_utility(["service_home"]), 0.6)
        self.assertAlmostEqual(
            product.path_utility(["service_home", "explicit_parent"]),
            0.48,
        )

    def test_close_child_support_prefers_child(self):
        menu, values = greedy_menu(
            {"a": 0.95, "b": 0.05}, self.hierarchy, self.policy, size=2,
        )
        self.assertEqual(menu[0], "a")
        self.assertAlmostEqual(values[0], 0.95)

        tied, _ = greedy_menu({"a": 1.0}, self.hierarchy, self.policy, size=1)
        self.assertEqual(tied, ["a"])

    def test_greedy_menu_stops_when_remaining_actions_add_no_value(self):
        menu, values = greedy_menu(
            {"a": 1.0}, self.hierarchy, self.policy, size=4,
        )
        self.assertEqual(menu, ["a"])
        self.assertEqual(values, [1.0])

    def test_unrelated_ties_keep_the_supplied_model_order(self):
        hierarchy = build_hierarchy(
            [target("a"), target("b"), target("c")],
            include_owner_apps=False,
        )
        menu, _ = greedy_menu(
            {"c": 1 / 3, "b": 1 / 3, "a": 1 / 3},
            hierarchy,
            self.policy,
            size=3,
            candidate_order=["a", "b", "c"],
        )
        self.assertEqual(menu, ["a", "b", "c"])

    def test_equivalent_service_homes_satisfy_each_other_without_duplicate_slots(self):
        first = target("home-a", locator="https://chatgpt.com/")
        second = target("home-b", locator="https://chatgpt.com/")
        child = target(
            "chat-child",
            locator="https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc",
        )
        hierarchy = build_hierarchy(
            [first, second, child], include_owner_apps=False,
        )

        self.assertEqual(
            deduplicate_equivalent_service_homes(
                ["chat-child", "home-b", "home-a"], hierarchy,
            ),
            ["chat-child", "home-b"],
        )

        self.assertEqual(
            menu_value(["home-a"], {"home-b": 1.0}, hierarchy, self.policy),
            1.0,
        )
        self.assertEqual(
            useful_fallback_actions(hierarchy, "home-b", self.policy),
            {"home-a": ("equivalent_service_home",)},
        )
        menu, _ = greedy_menu(
            {"home-a": 0.4, "home-b": 0.3, "chat-child": 0.3},
            hierarchy,
            self.policy,
            size=4,
        )
        self.assertEqual(len(set(menu) & {"home-a", "home-b"}), 1)

        self.assertEqual(
            menu_value(
                ["home-b"], {"chat-child": 1.0}, hierarchy, self.policy,
                current_action="home-a",
            ),
            0.0,
        )
        current_menu, _ = greedy_menu(
            {"home-a": 0.4, "home-b": 0.3, "chat-child": 0.3},
            hierarchy,
            self.policy,
            size=4,
            current_action="home-a",
        )
        self.assertTrue(set(current_menu).isdisjoint({"home-a", "home-b"}))

    def test_fragmented_children_and_frequent_branch_prefer_parent(self):
        menu, values = greedy_menu(
            {"a": 0.4, "b": 0.35, "c": 0.25},
            self.hierarchy, self.policy, size=1,
        )
        self.assertEqual(menu, ["section"])
        self.assertAlmostEqual(values[0], 0.8)

        popular_parent, _ = greedy_menu(
            {"a": 0.34, "b": 0.30, "c": 0.21, "other": 0.15},
            self.hierarchy, self.policy, size=1,
        )
        self.assertEqual(popular_parent, ["section"])

    def test_parent_child_concentration_boundary_is_the_retention_value(self):
        below, _ = greedy_menu(
            {"a": 0.79, "b": 0.21}, self.hierarchy, self.policy, size=1,
        )
        above, _ = greedy_menu(
            {"a": 0.81, "b": 0.19}, self.hierarchy, self.policy, size=1,
        )
        self.assertEqual(below, ["section"])
        self.assertEqual(above, ["a"])

    def test_direct_parent_arrivals_compete_with_child_support(self):
        child_wins, _ = greedy_menu(
            {"section": 0.1, "a": 0.9}, self.hierarchy, self.policy, size=1,
        )
        parent_wins, _ = greedy_menu(
            {"section": 0.2, "a": 0.8}, self.hierarchy, self.policy, size=1,
        )
        self.assertEqual(child_wins, ["a"])
        self.assertEqual(parent_wins, ["section"])

    def test_current_app_and_current_location_are_zero_but_other_parent_can_help(self):
        browser = target("app:browser", "app", bundle="browser")
        parent = target("parent", bundle="browser")
        child = target("child", parent="parent", bundle="browser")
        current = target("current", bundle="browser")
        hierarchy = build_hierarchy([browser, parent, child, current])
        posterior = {"child": 1.0}

        self.assertEqual(
            menu_value(["app:browser"], posterior, hierarchy, self.policy,
                       current_action="current", current_app="app:browser"),
            0.0,
        )
        self.assertEqual(
            menu_value(["current"], posterior, hierarchy, self.policy,
                       current_action="current", current_app="app:browser"),
            0.0,
        )
        self.assertAlmostEqual(
            menu_value(["parent"], posterior, hierarchy, self.policy,
                       current_action="current", current_app="app:browser"),
            0.8,
        )

    def test_top4_can_keep_parent_and_child_when_both_add_help(self):
        expanded, unknowns = self.hierarchy.with_unknown_children({"section": 0.34})
        unknown = next(iter(unknowns))
        posterior = {"a": 0.56, unknown: 0.34, "other": 0.10}
        menu, values = greedy_menu(posterior, expanded, self.policy, size=4)

        self.assertEqual(menu[:2], ["section", "a"])
        self.assertAlmostEqual(values[0], 0.72)
        self.assertAlmostEqual(values[1], 0.832)

    def test_parent_and_child_use_best_help_per_outcome_not_sum(self):
        expanded, unknowns = self.hierarchy.with_unknown_children({"section": 0.34})
        unknown = next(iter(unknowns))
        posterior = {"a": 0.56, unknown: 0.34, "other": 0.10}

        self.assertAlmostEqual(menu_value(["section"], posterior, expanded, self.policy), 0.72)
        self.assertAlmostEqual(menu_value(["a"], posterior, expanded, self.policy), 0.56)
        self.assertAlmostEqual(menu_value(["section", "a"], posterior, expanded, self.policy), 0.832)
        self.assertAlmostEqual(menu_value(["section", "section", "a"], posterior, expanded, self.policy), 0.832)
        self.assertEqual(menu_value(["section", "a"], {"a": 1.0}, expanded, self.policy), 1.0)

    def test_experimental_rank_score_uses_prefix_best_and_current_zero(self):
        truth = target("a", parent="section")
        score = experimental_rank_score(["section", "a"], truth, self.hierarchy, self.policy)
        self.assertAlmostEqual(score["top1"], 0.8)
        self.assertEqual(score["top2"], 1.0)
        self.assertEqual(score["top4"], 1.0)
        self.assertAlmostEqual(score["rank_score"], 0.95)

        current = experimental_rank_score(
            ["section", "a"], truth, self.hierarchy, self.policy,
            current_action="section",
        )
        self.assertEqual(current["top1"], 0.0)
        self.assertEqual(current["top2"], 1.0)
        self.assertAlmostEqual(current["rank_score"], 0.75)

    def test_nonexecutable_future_truth_cannot_score_as_a_prediction(self):
        hierarchy = build_hierarchy([target("parent")], include_owner_apps=False)
        truth = target("future", parent="parent")
        exact = experimental_rank_score(["future"], truth, hierarchy, self.policy)
        parent = experimental_rank_score(["parent"], truth, hierarchy, self.policy)
        self.assertEqual(exact["top1"], 0.0)
        self.assertAlmostEqual(parent["top1"], 0.8)

    def test_report_cutoff_is_fixed_to_four(self):
        with self.assertRaisesRegex(ValueError, "cutoff=4"):
            experimental_rank_score(["a"], target("a"), self.hierarchy, self.policy, cutoff=2)


if __name__ == "__main__":
    unittest.main()
