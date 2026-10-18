from copy import deepcopy
import json
import unittest

import semantic_annotations as semantics


def page(url, **extra):
    return {'kind': 'page', 'locator': url, 'bundleID': 'browser', **extra}


def local(path, kind='file', **extra):
    return {'kind': kind, 'locator': 'file://' + path, **extra}


def compare(left, right):
    return semantics.compare_annotations(semantics.annotate_target(left), semantics.annotate_target(right))


class SemanticAnnotationTests(unittest.TestCase):
    def test_new_resources_in_same_explicit_course_have_strong_auditable_scope(self):
        a = page('https://school.example/courses/CS101/lectures/week1.pdf')
        b = page('https://school.example/courses/CS101/assignments/hw2.html')
        result = compare(a, b)
        self.assertEqual(result['level'], 'strong')
        self.assertFalse(result['identity_equal'])
        course = next(t for t in result['matched_tags'] if t['kind'] == 'course_scope')
        self.assertTrue(course['counts_as_similarity_evidence'])
        self.assertIn('explicit_named_scope_marker', course['basis']['left'][0]['rule'])
        self.assertEqual(result['independent_evidence_units'], 1)

    def test_same_host_alone_is_only_weak_and_generic_tags_are_not_extra_votes(self):
        result = compare(page('https://school.example/a.pdf'), page('https://school.example/b.pdf'))
        self.assertEqual(result['level'], 'weak')
        self.assertTrue(result['weak'])
        self.assertFalse(result['strong'])
        active = [t for t in result['matched_tags'] if t['counts_as_similarity_evidence']]
        self.assertEqual([t['kind'] for t in active], ['site'])
        self.assertEqual(result['score'], max(t['score'] for t in active))

    def test_generic_kind_extension_application_and_role_cannot_create_weak(self):
        a = {'kind': 'file', 'fileType': 'pdf', 'name': 'PDF', 'bundleID': 'reader'}
        b = {'kind': 'file', 'fileType': 'pdf', 'name': 'PDF', 'bundleID': 'reader'}
        result = compare(a, b)
        self.assertEqual(result['level'], 'none')
        self.assertEqual(result['score'], 0)
        self.assertTrue(result['matched_tags'])
        self.assertFalse(any(t['counts_as_similarity_evidence'] for t in result['matched_tags']))

    def test_course_scope_does_not_cross_hosts(self):
        result = compare(page('https://alpha.example/courses/CS101/notes.pdf'),
                         page('https://beta.example/courses/CS101/notes.pdf'))
        self.assertEqual(result['level'], 'none')

    def test_account_query_and_user_paths_isolate_explicit_courses(self):
        pairs = [
            ('https://school.example/courses/CS101/files/a.pdf?user=alice',
             'https://school.example/courses/CS101/files/b.pdf?user=bob'),
            ('https://school.example/users/alice/download?course=CS101',
             'https://school.example/users/bob/download?course=CS101'),
            ('https://school.example/users/alice/courses/CS101/a.pdf',
             'https://school.example/users/bob/courses/CS101/b.pdf')]
        for a, b in pairs:
            result = compare(page(a), page(b))
            self.assertFalse(result['strong'])
            self.assertFalse(any(t['kind'] == 'course_scope' for t in result['matched_tags']))

    def test_same_project_local_new_files_and_directory_have_strong_scope(self):
        a = local('/Users/alice/Projects/smartapp/src/main.swift')
        b = local('/Users/alice/Projects/smartapp/tests/test.swift')
        self.assertEqual(compare(a, b)['level'], 'strong')
        folder = local('/Users/alice/Projects/smartapp', kind='directory')
        self.assertEqual(compare(folder, a)['level'], 'strong')

    def test_local_user_homes_and_remote_file_hosts_do_not_cross(self):
        a = local('/Users/alice/Projects/smartapp/notes.pdf')
        b = local('/Users/bob/Projects/smartapp/notes.pdf')
        self.assertEqual(compare(a, b)['level'], 'none')
        a = {'kind': 'file', 'locator': 'file://server-a/Projects/smartapp/notes.pdf'}
        b = {'kind': 'file', 'locator': 'file://server-b/Projects/smartapp/notes.pdf'}
        self.assertEqual(compare(a, b)['level'], 'none')

    def test_downloads_and_root_do_not_imply_projects_or_strong_scopes(self):
        for meta in (local('/Users/alice/Downloads', kind='directory'),
                     local('/', kind='directory'), local('/Users/alice', kind='directory')):
            ann = semantics.annotate_target(meta)
            self.assertFalse(any(t['kind'] == 'project_scope' for t in ann['tags']))
            self.assertFalse(any(t['strength'] * t['confidence'] >= .7 for t in ann['tags']))
        result = compare(local('/Users/alice/Downloads/report.pdf'),
                         local('/Users/alice/Downloads/schedule.pdf'))
        self.assertEqual(result['level'], 'none')

    def test_bare_numeric_paths_never_fabricate_courses(self):
        ann = semantics.annotate_target(page('https://school.example/content/123456/7890.pdf'))
        self.assertFalse(any(t['kind'] == 'course_scope' for t in ann['tags']))
        explicit = semantics.annotate_target(page('https://school.example/courses/123456/7890.pdf'))
        tag = next(t for t in explicit['tags'] if t['kind'] == 'course_scope')
        self.assertIn('identifier kept opaque', tag['basis'][0]['rule'])

    def test_repository_scope_retains_owner_and_repository(self):
        a = page('https://github.com/alice/smartapp/blob/main/README.md')
        b = page('https://github.com/alice/smartapp/issues/2')
        c = page('https://github.com/bob/smartapp/issues/2')
        self.assertEqual(compare(a, b)['level'], 'strong')
        self.assertFalse(compare(a, c)['strong'])
        self.assertFalse(compare(a, page('https://github.com/alice/other/issues/2'))['strong'])

    def test_query_fragment_and_exact_locator_are_never_normalized_or_replaced(self):
        url = 'https://school.example/courses/CS101/a.pdf?download=1&token=EXAMPLE_ONLY#page=2'
        original = page(url, id='stable-observed-id', origin='https://school.example')
        saved = deepcopy(original)
        ann = semantics.annotate_target(original)
        self.assertEqual(ann['identity']['locator'], url)
        self.assertEqual(original, saved)
        self.assertNotIn('EXAMPLE_ONLY', json.dumps(ann['tags']))
        changed = page(url.replace('#page=2', '#page=3'), id='stable-observed-id')
        self.assertFalse(compare(original, changed)['identity_equal'])
        self.assertEqual(compare(original, original)['level'], 'exact')

    def test_fragment_routes_are_explicit_but_tenant_scoped(self):
        a = page('https://school.example/app#/courses/CS101/files/one.pdf?tenant=a')
        b = page('https://school.example/app#/courses/CS101/files/two.pdf?tenant=a')
        c = page('https://school.example/app#/courses/CS101/files/two.pdf?tenant=b')
        self.assertTrue(compare(a, b)['strong'])
        self.assertFalse(compare(a, c)['strong'])

    def test_same_named_local_and_online_resource_do_not_merge(self):
        result = compare(local('/Users/alice/Downloads/CS101-notes.pdf', name='CS101-notes.pdf'),
                         page('https://school.example/files/CS101-notes.pdf', name='CS101-notes.pdf'))
        self.assertFalse(result['identity_equal'])
        self.assertFalse(result['strong'])
        self.assertEqual(result['level'], 'none')

    def test_parent_hash_is_weak_without_specific_parent_locator(self):
        result = compare({'kind': 'file', 'id': 'A', 'parentID': 'opaque-shared-parent'},
                         {'kind': 'file', 'id': 'B', 'parentID': 'opaque-shared-parent'})
        self.assertEqual(result['level'], 'weak')
        self.assertFalse(result['strong'])

    def test_repeated_correlated_tags_do_not_accumulate_independent_support(self):
        a = semantics.annotate_target(page('https://school.example/courses/CS101/files/a.pdf'))
        b = semantics.annotate_target(page('https://school.example/courses/CS101/files/b.pdf'))
        expected = semantics.compare_annotations(a, b)
        b['tags'] *= 10
        result = semantics.compare_annotations(a, b)
        self.assertEqual(result, expected)
        self.assertEqual(result['independent_evidence_units'], 1)

    def test_no_global_fit_or_future_metadata_affects_current_annotation(self):
        meta = page('https://school.example/courses/CS101/files/a.pdf')
        baseline = semantics.annotate_target(meta)
        semantics.annotate_target(page('https://future.example/projects/unseen/x.pdf'))
        contaminated = {**meta, 'body': 'future text', 'future_catalog': {'course': 'OTHER'},
                        'observed_outcome': {'label': 'future'}, 'background': ['other']}
        self.assertEqual(semantics.annotate_target(contaminated), baseline)
        current = semantics.annotate_current_origin(meta)
        self.assertEqual(current['tags'], baseline['tags'])
        self.assertEqual(current['annotation_role'], 'current_input_origin')
        self.assertEqual(json.dumps(baseline, sort_keys=True), json.dumps(semantics.annotate_target(meta), sort_keys=True))

    def test_malformed_or_ambiguous_paths_do_not_guess_strong_semantics(self):
        for locator in ('https://[broken', 'https://example.com:bad/courses/CS101/a',
                        'https://example.com/courses/CS101%2fOTHER/a.pdf'):
            ann = semantics.annotate_target(page(locator))
            self.assertTrue(ann['issues'])
            self.assertFalse(any(t['strength'] * t['confidence'] >= .7 for t in ann['tags']))

    def test_transient_window_ids_do_not_become_exact_semantic_identity(self):
        window = {'kind': 'window', 'id': 'window@123', 'bundleID': 'browser'}
        self.assertFalse(compare(window, window)['identity_equal'])

    def test_annotation_weights_and_version_are_explicit(self):
        ann = semantics.annotate_target(page('https://school.example/courses/CS101/notes.pdf'))
        for tag in ann['tags']:
            self.assertTrue(0 <= tag['strength'] <= 1 and 0 <= tag['confidence'] <= 1)
            self.assertTrue(tag['basis'])
        with self.assertRaises(ValueError):
            semantics.compare_annotations(ann, {**ann, 'version': 'future-version'})


if __name__ == '__main__':
    unittest.main()
