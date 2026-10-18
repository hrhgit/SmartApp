"""Causal replay of paired inactivity; original event times are never overwritten."""

from uuid import UUID

VERSION = 'known-inactivity-v1'


class InactivityClock:
    def __init__(self):
        self.session = None
        self.context = None
        self.pause = None
        self.excluded_scope = None
        self.excluded = 0.

    def consume(self, event):
        if event.get('sessionID') != self.session or event.get('kind') == 'session.start':
            self.__init__()
            self.session = event.get('sessionID')
        now = event['monotonicMs']
        data = event.get('data', {})
        disposition = 'normal'
        if event.get('kind') == 'observation.scope':
            phase, scope_id = data.get('phase'), data.get('scopeID')
            try:
                UUID(scope_id)
                valid = data.get('scopeVersion') == 'excluded-foreground-v1'
            except (ValueError, TypeError, AttributeError):
                valid = False
            if valid and phase == 'begin' and self.excluded_scope is None:
                self.excluded_scope = (scope_id, now)
                self.context = None
                disposition = 'masked'
            elif valid and phase == 'end' and self.excluded_scope and self.excluded_scope[0] == scope_id:
                self.excluded += max(0., now - self.excluded_scope[1])
                self.excluded_scope = None
                disposition = 'masked'
            else:
                self.excluded_scope = None
                disposition = 'boundary'
        elif event.get('kind') == 'capture.gap':
            self.excluded_scope = None
            phase = data.get('pausePhase')
            reasons = data.get('pauseReasons', [])
            try:
                UUID(data.get('pauseID', ''))
                marker = data.get('pauseVersion') == VERSION
            except (ValueError, TypeError, AttributeError):
                marker = False
            if (marker and phase == 'begin' and self.pause is None and self.context
                    and reasons and set(reasons) <= {'system_sleep', 'screen_locked'}):
                self.pause = {'id': data['pauseID'], 'start': now, 'end': None,
                              'wall': event['timestampMs'], 'context': self.context}
                disposition = 'suspended'
            elif (marker and phase == 'end' and self.pause
                  and self.pause['id'] == data['pauseID'] and self.pause['end'] is None
                  and now >= self.pause['start'] and event['timestampMs'] >= self.pause['wall']):
                self.pause['end'] = now
                disposition = 'suspended'
            else:
                disposition = 'boundary'
        elif event.get('kind') == 'session.end':
            disposition = 'boundary'
        elif self.excluded_scope:
            if event.get('kind') == 'context.change':
                self.excluded_scope = None
                disposition = 'boundary'
            else:
                disposition = 'suspended'
        elif self.pause:
            if event.get('kind') == 'context.change':
                old = self.pause['context']
                valid = (self.pause['end'] is not None
                         and data.get('continuityPauseID') == self.pause['id']
                         and data.get('resolvesContextEventID') == old['id']
                         and data.get('reason') == 'resolution_update'
                         and data.get('targetID') == old['data'].get('targetID')
                         and data.get('entranceID') == old['data'].get('entranceID'))
                disposition = 'resumed' if valid else 'boundary'
            else:
                disposition = 'suspended'
        if disposition in {'resumed', 'boundary'}:
            if self.pause:
                end = self.pause['end'] if self.pause['end'] is not None else now
                self.excluded += max(0., min(now, end) - self.pause['start'])
            self.pause = None
            self.context = None
        active = now - self.excluded
        if self.excluded_scope:
            active -= max(0., now - self.excluded_scope[1])
        if self.pause:
            end = self.pause['end'] if self.pause['end'] is not None else now
            active -= max(0., min(now, end) - self.pause['start'])
        if event.get('kind') == 'context.change':
            self.context = event
        return {**event, 'activeMonotonicMs': active, 'continuityDisposition': disposition}


def annotate_events(events):
    clock = InactivityClock()
    return [clock.consume(event) for event in events]


def active_time(event):
    return event.get('activeMonotonicMs', event['monotonicMs'])


def is_boundary(event):
    return (event.get('continuityDisposition', 'boundary') == 'boundary'
            and event.get('kind') in {'capture.gap', 'session.end', 'observation.scope'}
            or event.get('kind') == 'observation.scope')
