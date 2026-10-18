import contextlib
import io
import unittest

import evaluate_context_knn as core
import tune_dwell_knn as tuning
from test_evaluate_context_knn import observation


class OptimizedSearchTests(unittest.TestCase):
    def test_cached_search_matches_reference_with_ties_and_recency_fallback(self):
        items=[observation(i,truth=['B','C','F'][i%3],
            bg={'browser':{str(i%2):1}} if i%3 else {},
            quality={'browser':.5} if i%3 else {}) for i in [1,2,3,10,11,12]]
        # Include different histories and timestamps so both weight and decay change ranks.
        for i,q in enumerate(items):
            q.history={'recent:'+str(i%3):1};q.timestamp=i*3600000
        pairs=[[(j,core.similarity(q,items[j])) for j in core.eligible_neighbors(items,i)] for i,q in enumerate(items)]
        with contextlib.redirect_stdout(io.StringIO()):
            grid=tuning.grid_search(core,items,pairs,[3,4,5],weights=[0.,.35,1.],ks=[1,3,10],half_lives=[.25,6.])
        for row in grid:
            actual,_,_=core.evaluate(items,pairs,[3,4,5],row['config'])
            self.assertEqual(row['validation'],actual)


if __name__=='__main__':unittest.main()
