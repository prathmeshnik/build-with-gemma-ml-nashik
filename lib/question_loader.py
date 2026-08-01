import json
import random


class QuestionBank:
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.title = data["title"]
        self.description = data.get("description", "")
        self.min_score_to_explain = data.get("min_score_to_explain", 4)
        self._all = data["questions"]
        self._asked_ids = set()

    @property
    def total(self):
        return len(self._all)

    @property
    def remaining(self):
        return [q for q in self._all if q["id"] not in self._asked_ids]

    @property
    def done(self):
        return len(self._asked_ids)

    def next(self):
        pool = self.remaining
        if not pool:
            return None
        q = random.choice(pool)
        self._asked_ids.add(q["id"])
        return q

    def get(self, qid):
        for q in self._all:
            if q["id"] == qid:
                return q
        return None

    def reset(self):
        self._asked_ids.clear()
