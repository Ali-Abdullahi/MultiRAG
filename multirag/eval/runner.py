"""Run a gold set against the retriever and score what comes back.

Retrieval-only: no VLM is called and nothing here costs money, which is the
point. Ranking quality is the upstream variable - a generator cannot answer from
a page that was never retrieved - so it is worth measuring on its own, often,
before anything expensive is wired in.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from multirag import config
from multirag.eval import metrics
from multirag.eval.goldset import GoldSet, PageRef
from multirag.index import manifest as index_manifest
from multirag.retrieve import Hit


@dataclass(frozen=True)
class QuestionResult:
    """One question's retrieval, scored.

    Keeps the hits themselves, not just the numbers. A score of 0 is a question,
    not an answer: the next thing anyone wants to know is what *was* returned
    instead, and reconstructing that means re-running the query.
    """

    qid: str
    question: str
    relevant: frozenset[str]
    refs: tuple[PageRef, ...]
    hits: tuple[Hit, ...]
    rank: int | None
    recall: float
    reciprocal_rank: float

    @property
    def found(self) -> bool:
        """Whether any relevant page made the top k."""
        return self.rank is not None

    @property
    def expected(self) -> str:
        """Expected pages as a person wrote them, for a failure line."""
        return ", ".join(ref.citation for ref in self.refs)


@dataclass(frozen=True)
class RunReport:
    """Every question's result, plus the identity of what produced them.

    The identity fields are not bookkeeping. A metric is a number about a
    specific encoder querying a specific collection; two runs under different
    encoders are not comparable, and nothing in the number itself says so. Every
    other stage in this pipeline records the identity of what it produced
    alongside the artifact - an eval that skipped that would be the one place
    where a stale comparison passes silently.
    """

    goldset_name: str
    k: int
    model_id: str
    model_revision: str
    collection: str
    corpus_pages: int
    created_at: str
    results: tuple[QuestionResult, ...]

    def __len__(self) -> int:
        return len(self.results)

    @property
    def hit_rate(self) -> float:
        """Share of questions with at least one relevant page in the top k."""
        return self._mean(1.0 if r.found else 0.0 for r in self.results)

    @property
    def mean_recall(self) -> float:
        """Mean recall@k. Below hit_rate exactly when a multi-page question
        found some of its pages but not all."""
        return self._mean(r.recall for r in self.results)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: how near the top the first right answer lands."""
        return self._mean(r.reciprocal_rank for r in self.results)

    @property
    def top1_rate(self) -> float:
        """Share of questions whose very top hit is relevant.

        The metric that still discriminates when the corpus is smaller than k.
        It is also the one closest to what a user experiences: the first page is
        the one a person sees first and the one a VLM weighs most heavily.
        """
        return self._mean(1.0 if r.rank == 1 else 0.0 for r in self.results)

    @property
    def saturated(self) -> bool:
        """True when k reaches the whole corpus, so recall@k cannot fail.

        With 3 indexed pages and k=5, every query returns everything and
        hit_rate and mean_recall are 1.0 by arithmetic rather than by merit.
        Saying so is the point: a metric that cannot fail measures nothing, and
        a vacuous 1.000 left unlabelled will eventually be read as a pass.
        """
        return self.corpus_pages <= self.k

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        """Questions where nothing relevant was retrieved - the work queue."""
        return tuple(r for r in self.results if not r.found)

    def _mean(self, values) -> float:
        """Mean over questions, 0.0 for an empty run rather than ZeroDivisionError."""
        total = list(values)
        return sum(total) / len(total) if total else 0.0


def _score(question, hits: tuple[Hit, ...], k: int) -> QuestionResult:
    """Score one question's hit list against its known answer pages.

    Hits arrive already ranked by the index; this never reorders them. Scoring a
    list it had sorted itself would measure the harness, not the retriever.
    """
    retrieved = [hit.page_id for hit in hits]
    return QuestionResult(
        qid=question.qid,
        question=question.question,
        relevant=question.relevant,
        refs=question.refs,
        hits=hits,
        rank=metrics.rank_of_first_relevant(retrieved, question.relevant, k),
        recall=metrics.recall_at_k(retrieved, question.relevant, k),
        reciprocal_rank=metrics.reciprocal_rank(retrieved, question.relevant, k),
    )


def run_retrieval(
    goldset: GoldSet,
    retriever=None,
    k: int | None = None,
    on_question=None,
) -> RunReport:
    """Query every question in the gold set and score the results.

    Pass a retriever to reuse one; otherwise one is opened and closed here. The
    distinction matters because it holds an exclusive lock on the Qdrant
    directory - a caller that already has one open would deadlock against its
    own process, and one opened here must be released even if a query raises.

    k defaults to EVAL_TOP_K and is passed explicitly to search, so the serving
    default in RETRIEVE_TOP_K cannot quietly change what is being measured.

    on_question, if given, is called with each result as it lands. Queries are
    ~50ms each, but encoding dominates and a few hundred questions is long
    enough that silence looks like a hang.
    """
    k = k or config.EVAL_TOP_K

    contract = index_manifest.read_manifest()
    if contract is None:
        raise RuntimeError("nothing indexed yet; run index_document first")

    owned = retriever is None
    if owned:
        from multirag.retrieve import Retriever

        retriever = Retriever.open()

    try:
        corpus_pages = retriever.page_count
        results = []
        for question in goldset.questions:
            hits = tuple(retriever.search(question.question, limit=k))
            result = _score(question, hits, k)
            results.append(result)
            if on_question is not None:
                on_question(result)
    finally:
        if owned:
            retriever.close()

    return RunReport(
        goldset_name=goldset.name,
        k=k,
        model_id=contract["model_id"],
        model_revision=contract["model_revision"],
        collection=contract["collection"],
        corpus_pages=corpus_pages,
        created_at=datetime.now(timezone.utc).isoformat(),
        results=tuple(results),
    )
