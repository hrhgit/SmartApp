"""Predict learned destinations without an externally supplied candidate set."""
import math


class UnrestrictedPrediction:
    def __init__(self, model):
        self.model = model
        self.learned_labels = set()

    def learn(self, record):
        self.model.learn(record)
        self.learned_labels.add(record['label'])

    def predict(self, observation):
        # labels is only the existing scorer API's vocabulary argument. It is
        # derived exclusively from matured learning, never inventories or truth.
        query = {k: observation[k] for k in ('origin', 'app', 'timestamp', 'x')}
        query['labels'] = sorted(self.learned_labels)
        scores = self.model.scores(query)
        if any(not math.isfinite(float(v)) for v in scores.values()):
            raise ValueError('Non-finite prediction')
        if not set(scores) <= self.learned_labels:
            raise ValueError('Model produced a target it has not learned')
        ranking = sorted(scores, key=lambda k: (-scores[k], k))
        return {'predictions': ranking[:4], 'raw_ranking': ranking}


def remove_evidence_gate(config):
    """Do not replace the removed inventory restriction with a same-origin gate."""
    if config['family'] == 'same_origin_evidence_gate':
        return remove_evidence_gate(config['parameters']['base'])
    return config
