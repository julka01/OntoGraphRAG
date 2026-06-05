from experiments.answer_formatting import (
    build_answer_instructions,
    normalize_answer_to_contract,
)


def test_pubmedqa_instructions_include_maybe():
    text = build_answer_instructions("pubmedqa", "binary")
    assert "yes, no, or maybe" in text
    assert "exactly one word" in text.lower()
    assert "study conclusion" in text.lower()


def test_mcq_instructions_include_options():
    text = build_answer_instructions(
        "medhop",
        "mcq",
        options={"A": "Protein X", "B": "Protein Y"},
    )
    assert "multiple-choice" in text.lower()
    assert "A. Protein X" in text
    assert "B. Protein Y" in text


def test_multihoprag_instructions_make_abstention_last_resort():
    text = build_answer_instructions("multihoprag", "free_text")
    assert "Insufficient Information" in text
    assert "shortest correct entity" in text
    assert "last resort" in text.lower()
    assert "best-supported answer span" in text.lower()


def test_2wiki_free_text_instructions_are_short_answer_only():
    text = build_answer_instructions("2wikimultihopqa", "free_text")
    assert "short-answer multi-hop qa task" in text.lower()
    assert "shortest correct entity" in text.lower()
    assert "do not write an explanatory sentence" in text.lower()


def test_realmedqa_instructions_expect_concise_recommendation():
    text = build_answer_instructions("realmedqa", "free_text")
    assert "clinical recommendation qa" in text.lower()
    assert "1 to 3 sentences" in text.lower()
    assert "insufficient information" in text.lower()
    assert "last resort" in text.lower()


def test_realmedqa_contract_truncates_to_concise_recommendation():
    text = (
        "Recommend dialectical behavior therapy. "
        "Do not exclude people with BPD from services. "
        "Offer crisis planning. "
        "Additional implementation notes are optional."
    )
    assert (
        normalize_answer_to_contract("realmedqa", "free_text", text)
        == "Recommend dialectical behavior therapy. Do not exclude people with BPD from services. Offer crisis planning."
    )


def test_realmedqa_contract_normalizes_insufficient_information():
    assert (
        normalize_answer_to_contract(
            "realmedqa",
            "free_text",
            "Insufficient information to make a grounded recommendation from the retrieved text.",
        )
        == "Insufficient Information."
    )


def test_pubmedqa_answer_contract_normalizes_leading_label():
    assert (
        normalize_answer_to_contract(
            "pubmedqa",
            "binary",
            "Yes, the abstract overall supports the claim.",
        )
        == "yes"
    )
    assert (
        normalize_answer_to_contract(
            "pubmedqa",
            "binary",
            "maybe. the evidence is mixed across subgroups.",
        )
        == "maybe"
    )


def test_bioasq_binary_contract_normalizes_leading_label():
    assert (
        normalize_answer_to_contract(
            "bioasq",
            "binary",
            "No, the evidence does not support that conclusion.",
        )
        == "no"
    )


def test_generic_binary_contract_normalizes_leading_label():
    assert (
        normalize_answer_to_contract(
            "hotpotqa",
            "binary",
            "No, only Prince has been inducted into the Rock and Roll Hall of Fame.",
            question="Were both Prince and Patty Jenkins have been inducted into the Rock and Roll Hall of Fame?",
        )
        == "no"
    )


def test_free_text_answers_are_left_unchanged():
    text = "Lost and Delirious"
    assert normalize_answer_to_contract("musique", "free_text", text) == text


def test_bioasq_factoid_contract_extracts_subject_phrase():
    assert (
        normalize_answer_to_contract(
            "bioasq",
            "free_text",
            "Interleukin-6 is a cytokine involved in inflammation.",
            question="What is interleukin-6?",
        )
        == "Interleukin-6"
    )


def test_bioasq_factoid_contract_extracts_numeric_span():
    assert (
        normalize_answer_to_contract(
            "bioasq",
            "free_text",
            "The prevalence was 42% in the treated group.",
            question="How many patients responded to treatment?",
        )
        == "42%"
    )


def test_short_answer_wrapper_is_removed_for_multihop_datasets():
    assert (
        normalize_answer_to_contract(
            "2wikimultihopqa",
            "free_text",
            "Final answer: Lost and Delirious",
        )
        == "Lost and Delirious"
    )


def test_multihop_yes_no_answers_are_collapsed_to_label_only():
    assert (
        normalize_answer_to_contract(
            "hotpotqa",
            "free_text",
            "No. The evidence does not indicate that Patty Jenkins was inducted.",
            question="Were both Prince and Patty Jenkins inducted into the Rock and Roll Hall of Fame?",
        )
        == "no"
    )


def test_multihop_date_answers_extract_short_date_span():
    assert (
        normalize_answer_to_contract(
            "hotpotqa",
            "free_text",
            "Stoffel Vandoorne won the Monaco GP2 Series round on May 22 and 23, 2015.",
            question="When did Belgium racer Stoffel Vandoorne win the Monaco GP2 Series round?",
        )
        == "May 22 and 23, 2015"
    )


def test_multihop_club_answers_strip_common_suffixes():
    assert (
        normalize_answer_to_contract(
            "hotpotqa",
            "free_text",
            "Gainsborough Trinity F.C.",
            question="Which football club did Michael Whitham briefly manage in Lincolnshire, England?",
        )
        == "Gainsborough Trinity"
    )


def test_generic_fallback_keeps_chat_like_answer_intact():
    text = "The study suggests a modest benefit, but confidence is limited."
    assert normalize_answer_to_contract("", "free_text", text) == text
