from experiments import dataset_adapters as dataset_adapters_module
from experiments.dataset_adapters import (
    InferenceRecord,
    TaskType,
    adapt_2wikimultihopqa,
    adapt_bioasq,
    adapt_hotpotqa,
    adapt_medqa,
    adapt_medhop,
    adapt_multihoprag,
    adapt_realmedqa,
    build_passage_corpus,
    build_global_corpus_passages,
    get_dataset_corpus_profile,
    infer_hop_count_from_raw,
)


def test_adapt_medhop_maps_supports_and_candidates():
    raw = {
        "id": "medhop_1",
        "query": "Which protein mediates the interaction between drug A and drug B?",
        "answer": "Protein X",
        "candidates": ["Protein Y", "Protein X", "Protein Z"],
        "supports": [
            "Drug A activates Protein X in macrophages.",
            "Protein X is inhibited by Drug B in vivo.",
        ],
    }

    inf, gold = adapt_medhop(raw)

    assert inf.id == "medhop_1"
    assert inf.dataset == "medhop"
    assert inf.question == raw["query"]
    assert inf.contexts == raw["supports"]
    assert inf.task_type == TaskType.MCQ.value
    assert inf.options == {"A": "Protein Y", "B": "Protein X", "C": "Protein Z"}

    assert gold.id == "medhop_1"
    assert gold.short_answer == "protein x"
    assert gold.long_answer is None
    assert gold.aliases is None


def test_adapt_medqa_handles_letter_answer_idx():
    raw = {
        "question": "Which action is correct?",
        "options": {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
        },
        "answer_idx": "B",
        "answer": "Option B",
    }

    inf, gold = adapt_medqa(raw, q_id="medqa_1")

    assert inf.id == "medqa_1"
    assert inf.dataset == "medqa"
    assert inf.task_type == TaskType.MCQ.value
    assert inf.contexts == []
    assert inf.options == raw["options"]
    assert gold.short_answer == "option b"


def test_adapt_multihoprag_maps_query_answer_and_evidence():
    raw = {
        "query": "Did the Reuters piece come before the AP piece?",
        "answer": "yes",
        "question_type": "temporal_query",
        "evidence_list": [
            {
                "title": "Reuters sample",
                "source": "Reuters",
                "category": "business",
                "published_at": "2023-10-01T12:00:00+00:00",
                "fact": "Reuters reported the event first.",
            },
            {
                "title": "AP sample",
                "source": "Associated Press",
                "category": "business",
                "published_at": "2023-10-02T12:00:00+00:00",
                "fact": "AP reported the event later.",
            },
        ],
    }

    inf, gold = adapt_multihoprag(raw, q_id="mhr_1")

    assert inf.id == "mhr_1"
    assert inf.dataset == "multihoprag"
    assert inf.question == raw["query"]
    assert inf.task_type == TaskType.BINARY.value
    assert inf.options == {"A": "yes", "B": "no"}
    assert len(inf.contexts) == 2
    assert "Title: Reuters sample" in inf.contexts[0]
    assert "Source: Reuters" in inf.contexts[0]
    assert "Fact: Reuters reported the event first." in inf.contexts[0]

    assert gold.id == "mhr_1"
    assert gold.short_answer == "yes"
    assert gold.long_answer is None
    assert gold.aliases is None


def test_adapt_bioasq_factoid_preserves_aliases():
    raw = {
        "id": "bioasq_1",
        "body": "What drug is also known as acetylsalicylic acid?",
        "type": "factoid",
        "exact_answer": [["Aspirin", "acetylsalicylic acid"]],
        "ideal_answer": "Aspirin is also called acetylsalicylic acid.",
        "snippets": [{"text": "Aspirin, or acetylsalicylic acid, is commonly used."}],
    }

    inf, gold = adapt_bioasq(raw)

    assert inf.task_type == TaskType.FREE_TEXT.value
    assert gold.short_answer == "aspirin"
    assert gold.aliases == ["acetylsalicylic acid"]


def test_adapt_2wiki_does_not_infer_aliases_from_evidences():
    raw = {
        "_id": "2wiki_1",
        "question": "Who is the mother of the director of Polish-Russian War?",
        "answer": "Małgorzata Braunek",
        "type": "compositional",
        "context": [["Polish-Russian War", ["Sentence one."]]],
        "evidences": [
            ["Polish-Russian War", "director", "Xawery Żuławski"],
            ["Xawery Żuławski", "mother", "Małgorzata Braunek"],
        ],
    }

    _, gold = adapt_2wikimultihopqa(raw)
    assert gold.aliases is None


def test_bioasq_profile_marks_question_contexts_as_gold_evidence():
    profile = get_dataset_corpus_profile("bioasq")

    assert profile["question_context_role"] == "gold_evidence"
    assert profile["requires_shared_corpus_for_fair_retrieval"] is True


def test_adapt_realmedqa_uses_question_only_and_recommendation_as_gold():
    raw = {
        "Question": "How should beta-blockers be titrated in heart failure?",
        "Recommendation": "Introduce beta-blockers in a start low, go slow manner and monitor after each titration.",
        "Generator": "LLM",
        "Plausible": "Completely",
        "Answered": "Completely",
    }

    inf, gold = adapt_realmedqa(raw, q_id="realmedqa_1")

    assert inf.id == "realmedqa_1"
    assert inf.dataset == "realmedqa"
    assert inf.question == raw["Question"]
    assert inf.contexts == []
    assert inf.task_type == TaskType.FREE_TEXT.value

    assert gold.short_answer == raw["Recommendation"].lower()
    assert gold.long_answer == raw["Recommendation"]


def test_realmedqa_profile_requires_shared_corpus():
    profile = get_dataset_corpus_profile("realmedqa")

    assert profile["question_context_role"] == "no_context"
    assert profile["requires_shared_corpus_for_fair_retrieval"] is True


def test_multihoprag_uses_shared_corpus_for_fair_retrieval():
    profile = get_dataset_corpus_profile("multihoprag")
    passages = build_global_corpus_passages("multihoprag")

    assert profile["question_context_role"] == "gold_evidence"
    assert profile["requires_shared_corpus_for_fair_retrieval"] is True
    assert passages is not None
    assert len(passages) > 0


def test_bioasq_shared_corpus_absent_without_local_abstract_corpus(monkeypatch):
    monkeypatch.setattr(dataset_adapters_module, "_load_optional_json_records", lambda candidate_paths: None)
    passages = build_global_corpus_passages("bioasq")
    assert passages is None


def test_realmedqa_shared_corpus_uses_verified_ideal_subset(monkeypatch):
    monkeypatch.setattr(
        dataset_adapters_module,
        "_load_realmedqa_records",
        lambda candidate_paths: [
            {
                "Question": "Q1",
                "Recommendation": "R1",
                "_normalized_plausible": "completely",
                "_normalized_answered": "completely",
                "_fallback_id": "realmedqa_a",
            },
            {
                "Question": "Q2",
                "Recommendation": "R2",
                "_normalized_plausible": "partially",
                "_normalized_answered": "completely",
                "_fallback_id": "realmedqa_b",
            },
        ],
    )
    passages = build_global_corpus_passages("realmedqa")

    assert passages is not None
    assert len(passages) == 1
    assert passages[0].dataset == "realmedqa"
    assert passages[0].text == "Recommendation:\nR1"


def test_infer_hop_count_from_raw_uses_question_decomposition_when_present():
    raw = {
        "question_decomposition": [
            {"q": "step 1"},
            {"q": "step 2"},
            {"q": "step 3"},
        ]
    }

    assert infer_hop_count_from_raw("musique", "3hop__example", raw) == 3


def test_infer_hop_count_from_raw_uses_dataset_specific_evidence_fields():
    multihoprag_raw = {
        "evidence_list": [
            {"fact": "A"},
            {"fact": "B"},
            {"fact": "C"},
        ]
    }
    twowiki_raw = {
        "evidences": [
            ["A", "rel", "B"],
            ["B", "rel", "C"],
        ]
    }
    hotpot_raw = {
        "supporting_facts": [
            ["Title A", 0],
            ["Title B", 1],
        ]
    }

    assert infer_hop_count_from_raw("multihoprag", "mhr_1", multihoprag_raw) == 3
    assert infer_hop_count_from_raw("2wikimultihopqa", "2wiki_1", twowiki_raw) == 2
    assert infer_hop_count_from_raw("hotpotqa", "hotpot_1", hotpot_raw) == 2


def test_infer_hop_count_from_raw_falls_back_to_id_prefix():
    assert infer_hop_count_from_raw("musique", "4hop1__abc", {}) == 4


def test_build_passage_corpus_preserves_question_local_duplicates_when_requested():
    records = [
        InferenceRecord(
            id="q1",
            dataset="musique",
            question="Q1",
            contexts=["Shared passage", "Local passage 1"],
            task_type=TaskType.FREE_TEXT.value,
        ),
        InferenceRecord(
            id="q2",
            dataset="musique",
            question="Q2",
            contexts=["Shared passage", "Local passage 2"],
            task_type=TaskType.FREE_TEXT.value,
        ),
    ]

    deduped = build_passage_corpus(records)
    question_scoped = build_passage_corpus(records, dedupe_across_questions=False)

    assert len(deduped) == 3
    assert len(question_scoped) == 4
    assert [p.question_id for p in question_scoped if p.text == "Shared passage"] == ["q1", "q2"]


def test_build_passage_corpus_preserves_structured_titles_with_periods():
    raw = {
        "_id": "hp1",
        "question": "Which film starred James Bond?",
        "answer": "dr. no",
        "type": "bridge",
        "context": [
            ["Dr. No", ["James Bond appears in the film."]],
        ],
    }

    inf, _ = adapt_hotpotqa(raw)
    passages = build_passage_corpus([inf], dedupe_across_questions=False)

    assert len(passages) == 1
    assert passages[0].text.startswith("Dr. No. James Bond")
    assert passages[0].source_title == "Dr. No"
