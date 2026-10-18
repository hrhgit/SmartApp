"""Deterministic target semantics derived only from already-observed metadata.

No file, directory, page body, network, model, or global fitted state is read.
Annotations are private derived data: some scope keys contain URL/path parents.
They do not replace target identity, establish intent, or create extra examples.
"""

import json
import re
from urllib.parse import parse_qsl, unquote, urlsplit


VERSION = 'target-metadata-semantics-v1'
KINDS = {'app', 'window', 'page', 'directory', 'file'}
GENERIC = {
    'users', 'user', 'home', 'documents', 'document', 'desktop', 'downloads',
    'download', 'tmp', 'temp', 'private', 'var', 'volumes', 'mnt', 'library',
    'applications', 'school', 'myfiles', 'files', 'file', 'resources', 'resource',
    'content', 'contents', 'assets', 'static', 'public', 'media', 'pdf', 'html',
    'index', 'index.html', 'index.htm', 'view', 'viewcontent', 'default',
    'courses', 'course', 'projects', 'project', 'repos', 'repo', 'repositories',
    'repository', 'src', 'source', 'tests', 'test', 'docs', 'notebooks', 'data',
    'api', 'v1', 'v2', 'v3', 'www', 'http', 'https', 'localhost',
    'application', 'app', 'window', 'windows', 'browser', 'page',
}
SCOPE_MARKERS = {
    'course': 'course_scope', 'courses': 'course_scope',
    'project': 'project_scope', 'projects': 'project_scope',
    'repo': 'repository_scope', 'repos': 'repository_scope',
    'repository': 'repository_scope', 'repositories': 'repository_scope',
}
QUERY_SCOPE_KEYS = {
    'course': 'course_scope', 'courseid': 'course_scope', 'course_id': 'course_scope',
    'project': 'project_scope', 'projectid': 'project_scope', 'project_id': 'project_scope',
    'repo': 'repository_scope', 'repository': 'repository_scope',
    'repoid': 'repository_scope', 'repo_id': 'repository_scope',
}
ACCOUNT_KEYS = {
    'user', 'userid', 'user_id', 'account', 'accountid', 'account_id',
    'tenant', 'tenantid', 'tenant_id', 'workspace', 'workspaceid', 'workspace_id',
    'org', 'orgid', 'org_id', 'organization', 'organizationid', 'organization_id',
}
ROLE_MARKERS = {
    'assignment': 'assignment', 'assignments': 'assignment', 'homework': 'assignment',
    'lecture': 'lecture', 'lectures': 'lecture', 'slides': 'lecture',
    'syllabus': 'syllabus', 'quiz': 'assessment', 'quizzes': 'assessment',
    'exam': 'assessment', 'exams': 'assessment', 'grades': 'grades',
    'discussion': 'discussion', 'discussions': 'discussion', 'announcements': 'announcement',
    'issues': 'repository_issue', 'pull': 'repository_change', 'pulls': 'repository_change',
    'commit': 'repository_change', 'commits': 'repository_change',
    'src': 'source_code', 'source': 'source_code', 'tests': 'tests',
    'notebooks': 'notebook', 'readme': 'documentation', 'docs': 'documentation',
    'download': 'download', 'downloads': 'download',
}
REPOSITORY_HOSTS = {'github.com', 'gitlab.com', 'bitbucket.org'}
REPOSITORY_RESERVED = {'login', 'signup', 'explore', 'settings', 'search', 'topics',
                       'organizations', 'users', 'groups', 'dashboard', 'help'}
COURSE_CODE = re.compile(r'^[A-Za-z]{2,8}[-_ ]?\d{2,4}[A-Za-z]?$')
OPAQUE = re.compile(r'^(?:[0-9]+|[0-9a-fA-F]{12,}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})$')


def _serial(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _text(value):
    return value if isinstance(value, str) and value else None


class _Tags:
    def __init__(self):
        self.tags = {}

    def add(self, kind, value, strength, confidence, source, rule):
        key = kind + ':' + _serial(value)
        basis = {'source': source, 'rule': rule}
        new = {'key': key, 'kind': kind, 'strength': strength,
               'confidence': confidence, 'basis': [basis]}
        old = self.tags.get(key)
        if old:
            bases = {_serial(x): x for x in old['basis'] + [basis]}
            if old['strength'] * old['confidence'] > strength * confidence:
                new = {**old}
            new['basis'] = [bases[x] for x in sorted(bases)]
        self.tags[key] = new

    def result(self):
        return [self.tags[key] for key in sorted(self.tags)]


def _parts(path):
    raw = [part for part in path.split('/') if part]
    parts = [unquote(part) for part in raw]
    # Preserve the exact locator outside annotations. Do not guess how a server
    # handles encoded separators or dot navigation when inferring parent scopes.
    if any(part in {'.', '..'} or '/' in part or '\\' in part for part in parts):
        return None
    return parts


def _specific(part):
    return bool(part and part.casefold() not in GENERIC and not OPAQUE.fullmatch(part))


def _extension(value):
    if not value:
        return None
    value = value.casefold().lstrip('.')
    mime = {'application/pdf': 'pdf', 'com.adobe.pdf': 'pdf', 'text/plain': 'txt',
            'text/html': 'html', 'application/json': 'json'}
    value = mime.get(value, value)
    return value if re.fullmatch(r'[a-z][a-z0-9]{0,9}', value) else None


def _tokens(value):
    value = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
    tokens = re.findall(r'[^\W_]+', value.casefold(), flags=re.UNICODE)
    return sorted({t for t in tokens if 2 <= len(t) <= 40 and t not in GENERIC and t not in ROLE_MARKERS
                   and not OPAQUE.fullmatch(t)})[:16]


def _path_scopes(tags, namespace, parts, source, local=False, resource_kind=None):
    for i, part in enumerate(parts[:-1]):
        scope = SCOPE_MARKERS.get(part.casefold())
        if local and parts[0].casefold() in {'users', 'home', 'volumes', 'mnt'} and i < 2:
            scope = None
        identifier = parts[i + 1]
        # An explicit /courses/<id> marker can identify an opaque course scope,
        # but never decodes that number into a course name. Bare numeric paths
        # have no such meaning. Generic containers do not name projects/courses.
        if scope and identifier.casefold() not in GENERIC:
            tags.add(scope, [namespace, parts[:i + 2]], .94, .92,
                     source, 'explicit_named_scope_marker; identifier kept opaque')
        if local and part.casefold() in {'school', 'classes', 'coursework'} and COURSE_CODE.fullmatch(identifier):
            tags.add('course_scope', [namespace, parts[:i + 2]], .90, .88,
                     source, 'education_container_followed_by_explicit_course_code')
    if local:
        # A user home or mounted volume root is not a specific project/folder.
        minimum_depth = 3 if parts and parts[0].casefold() in {'users', 'home', 'volumes', 'mnt'} else 2
        last_depth = len(parts) if resource_kind == 'directory' else len(parts) - 1
        for depth in range(1, last_depth + 1):
            prefix = parts[:depth]
            specific = depth >= minimum_depth and _specific(prefix[-1])
            tags.add('local_directory_scope', [namespace, prefix], .88 if specific else .12,
                     .96, source, 'specific_observed_path_parent' if specific else 'generic_or_root_path_parent')
    else:
        for depth in range(1, len(parts)):
            prefix = parts[:depth]
            specific = depth >= 2 and _specific(prefix[-1])
            tags.add('url_parent', [namespace, prefix], .65 if specific else .12,
                     .90, source, 'named_url_parent' if specific else 'generic_url_parent')
    for part in parts:
        stem = part.rsplit('.', 1)[0].casefold()
        role = ROLE_MARKERS.get(stem)
        if role:
            tags.add('resource_role', role, .20, .80, source, 'explicit_resource_role_segment')


def _web(tags, parsed, issues, source='locator'):
    host = (parsed.hostname or '').casefold()
    if not host:
        issues.add('missing_web_host')
        return None
    port = parsed.port  # Invalid ports deliberately fail the caller's safe parse.
    origin = parsed.scheme.casefold() + '://' + host + (':' + str(port) if port is not None else '')
    tags.add('site', origin, .18, .99, source, 'same_observed_scheme_host_port_only')
    try:
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=512)
    except ValueError:
        issues.add('query_scope_unparsed')
        return _parts(parsed.path)
    accounts = sorted((key.casefold(), value) for key, value in query_pairs if key.casefold() in ACCOUNT_KEYS)
    namespace = ['web', origin, accounts]
    parts = _parts(parsed.path)
    if parts is None:
        issues.add('ambiguous_path_encoding')
    else:
        _path_scopes(tags, namespace, parts, source + '.path')
        if (host in REPOSITORY_HOSTS and len(parts) >= 2
                and parts[0].casefold() not in REPOSITORY_RESERVED
                and _specific(parts[0]) and _specific(parts[1])):
            # GitLab nested groups are not guessed from its first two segments.
            # A /-/ separator explicitly delimits the repository path there.
            repo_parts = parts[:parts.index('-')] if host == 'gitlab.com' and '-' in parts else parts[:2]
            if host != 'gitlab.com' or len(parts) == 2 or '-' in parts:
                tags.add('repository_scope', [namespace, repo_parts], .96, .95,
                         source + '.path', 'recognized_repository_host_and_explicit_owner_repository_route')
    scopes = {}
    for key, value in query_pairs:
        scope = QUERY_SCOPE_KEYS.get(key.casefold())
        if scope and value:
            scopes.setdefault(scope, set()).add(value)
    for scope, values in scopes.items():
        if len(values) == 1:
            identifier = next(iter(values))
            tags.add(scope, [namespace, 'explicit_query_scope', parsed.path, identifier], .91, .88,
                     source + '.query', 'explicit_scope_parameter; identifier kept opaque')
        else:
            issues.add('ambiguous_' + scope + '_query')
    # Plain anchors are retained in exact identity, not treated as topic labels.
    if parsed.fragment.startswith('/'):
        fragment = urlsplit(parsed.fragment)
        fragment_parts = _parts(fragment.path)
        try:
            fragment_accounts = sorted((key.casefold(), value) for key, value in
                parse_qsl(fragment.query, keep_blank_values=True, max_num_fields=512)
                if key.casefold() in ACCOUNT_KEYS)
        except ValueError:
            issues.add('fragment_scope_query_unparsed')
            fragment_parts = None
        if fragment_parts is not None:
            _path_scopes(tags, ['fragment_route', namespace, parsed.path, fragment_accounts], fragment_parts,
                         source + '.fragment_route')
    return parts


def _name_namespace(parsed, parts, bundle, kind):
    """Weak lexical matches still respect visible site/account/user boundaries."""
    if parsed is None:
        return ['application_metadata', bundle, kind]
    if parsed.scheme.casefold() in {'http', 'https'}:
        try:
            accounts = sorted((k.casefold(), v) for k, v in
                parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=512)
                if k.casefold() in ACCOUNT_KEYS)
            owner = []
            for i, part in enumerate((parts or [])[:-1]):
                if part.casefold() in {'users', 'user', 'u', 'accounts', 'account', 'tenants',
                                       'organizations', 'orgs', 'workspaces'}:
                    owner = parts[:i + 2]
                    break
            if parsed.hostname in REPOSITORY_HOSTS and parts:
                owner = parts[:1]
            return ['web', parsed.scheme.casefold(), (parsed.hostname or '').casefold(),
                    parsed.port, accounts, owner]
        except ValueError:
            return ['unparsed_web_name', parsed.netloc]
    if parsed.scheme.casefold() == 'file' or not parsed.scheme and parsed.path.startswith('/'):
        depth = 2 if parts and parts[0].casefold() in {'users', 'home', 'volumes', 'mnt'} else 1
        return ['local', (parsed.hostname or '').casefold(), (parts or [])[:depth]]
    return ['application_metadata', bundle, kind]


def annotate_target(meta):
    """Annotate one observed target; pass only metadata known at the input time.

    Strength/confidence are fixed heuristic weights, not calibrated probabilities.
    Full locator/query/fragment strings stay unchanged in ``identity``. Metadata
    such as page body, global catalogs, later observations, or context inventories
    is ignored. This function performs no discovery or eligibility admission.
    """
    if not isinstance(meta, dict):
        raise TypeError('Target metadata must be a dictionary')
    tags, issues = _Tags(), set()
    kind = meta.get('kind') if meta.get('kind') in KINDS else 'unknown'
    bundle = _text(meta.get('bundleID')) or _text(meta.get('_bundle'))
    locator, origin = _text(meta.get('locator')), _text(meta.get('origin'))
    identity = {key: meta[key] for key in ('id', 'kind', 'locator', 'parentID', 'origin') if key in meta}
    if bundle:
        identity['bundleID'] = bundle
        tags.add('application', bundle, .18, .95, 'bundleID', 'observed_owner_application_only')
    tags.add('resource_kind', kind, .08, .99, 'kind', 'generic_object_type_only')
    parts, parsed = None, None
    if locator:
        try:
            parsed = urlsplit(locator)
            if parsed.scheme.casefold() in {'http', 'https'}:
                parts = _web(tags, parsed, issues)
            elif parsed.scheme.casefold() == 'file' or not parsed.scheme and locator.startswith('/'):
                path = parsed.path if parsed.scheme else locator
                parts = _parts(path)
                if parts is None:
                    issues.add('ambiguous_path_encoding')
                elif kind in {'file', 'directory'}:
                    _path_scopes(tags, ['local', (parsed.hostname or '').casefold()], parts,
                                 'locator.path', local=True, resource_kind=kind)
            else:
                issues.add('unsupported_locator_scheme')
        except (ValueError, UnicodeError):
            issues.add('unparseable_locator')
    elif origin:
        try:
            parsed = urlsplit(origin)
            if parsed.scheme.casefold() in {'http', 'https'} and parsed.hostname:
                host = parsed.scheme.casefold() + '://' + parsed.hostname.casefold()
                if parsed.port is not None:
                    host += ':' + str(parsed.port)
                tags.add('site', host, .18, .95, 'origin', 'origin_metadata_only_without_locator')
        except (ValueError, UnicodeError):
            issues.add('unparseable_origin')
    parent = _text(meta.get('parentID'))
    if parent:
        # A hash does not reveal whether the parent is Downloads or a project.
        tags.add('observed_parent_id', parent, .22, .95, 'parentID', 'opaque_parent_identity_only')
    extension = _extension(_text(meta.get('fileType')))
    if extension:
        tags.add('extension', extension, .12, .95, 'fileType', 'declared_resource_extension_only')
    if parts:
        filename = parts[-1]
        if '.' in filename:
            extension = _extension(filename.rsplit('.', 1)[1])
            if extension:
                tags.add('extension', extension, .12, .95, 'locator.path', 'filename_extension_only')
        for token in _tokens(filename):
            tags.add('name_token', [_name_namespace(parsed, parts, bundle, kind), token], .18, .45,
                     'locator.basename', 'unfitted_lexical_token_only; isolated by visible owner namespace')
    name = _text(meta.get('name'))
    # Page names in this collector often equal the full URL. Never tokenize its
    # query credentials or arbitrary identifiers as weak semantic words.
    if name and '://' not in name and not name.startswith('/'):
        for token in _tokens(name):
            tags.add('name_token', [_name_namespace(parsed, parts, bundle, kind), token], .18, .45,
                     'name', 'unfitted_lexical_token_only; isolated by visible owner namespace')
    return {'version': VERSION, 'identity': identity, 'tags': tags.result(),
            'issues': sorted(issues), 'evidence_unit': 'one_observed_target',
            'weights_are_calibrated_probabilities': False}


def annotate_current_origin(current_meta):
    """Prediction-input adapter: call with the current target's as-of metadata.

    It deliberately accepts no catalog, session history, actual next target, or
    later repair. Missing metadata simply yields fewer/weaker annotations.
    """
    result = annotate_target(current_meta)
    return {**result, 'annotation_role': 'current_input_origin'}


def _identity_equal(left, right):
    if left.get('kind') not in KINDS or left.get('kind') != right.get('kind') or left.get('kind') == 'window':
        return False
    if left.get('locator') or right.get('locator'):
        return bool(left.get('locator') and left.get('locator') == right.get('locator'))
    if left.get('kind') == 'app' and left.get('bundleID'):
        return left.get('bundleID') == right.get('bundleID')
    return bool(left.get('id') and left.get('id') == right.get('id'))


def compare_annotations(left, right):
    """Single-pair similarity: strongest shared evidence, never sum tag votes.

    ``strong`` includes an exact stable identity or a strong explicit shared
    scope; ``weak`` means some similarity exists without such strong evidence.
    Multiple matched labels remain one comparison of two observed records.
    """
    if left.get('version') != VERSION or right.get('version') != VERSION:
        raise ValueError('Semantic annotation versions must match this implementation')
    a = {tag['key']: tag for tag in left['tags']}
    b = {tag['key']: tag for tag in right['tags']}
    matched = []
    for key in sorted(a.keys() & b.keys()):
        strength = min(a[key]['strength'], b[key]['strength'])
        confidence = min(a[key]['confidence'], b[key]['confidence'])
        matched.append({'key': key, 'kind': a[key]['kind'], 'strength': strength,
                        'confidence': confidence, 'score': strength * confidence,
                        'basis': {'left': a[key]['basis'], 'right': b[key]['basis']}})
    # Generic type, format, application and role never create even weak support.
    # Generic filesystem roots likewise cannot connect arbitrary users/files.
    evidence_kinds = {'site', 'url_parent', 'local_directory_scope', 'observed_parent_id',
                      'name_token', 'course_scope', 'project_scope', 'repository_scope'}
    eligible = [tag for tag in matched if tag['kind'] in evidence_kinds
                and not (tag['kind'] == 'local_directory_scope' and tag['strength'] < .35)]
    for tag in matched:
        tag['counts_as_similarity_evidence'] = tag in eligible
    semantic_score = max((tag['score'] for tag in eligible), default=0.)
    exact = _identity_equal(left['identity'], right['identity'])
    score = 1. if exact else semantic_score
    strong = bool(exact or semantic_score >= .7)
    level = 'exact' if exact else 'strong' if strong else 'weak' if score else 'none'
    return {'version': VERSION, 'score': score, 'semantic_score': semantic_score,
            'level': level, 'strong': strong, 'weak': bool(score and not strong),
            'identity_equal': exact, 'matched_tags': matched,
            'independent_evidence_units': 1,
            'aggregation': 'maximum_shared_tag; never count correlated tags as separate observations'}
