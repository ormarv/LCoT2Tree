# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

import random
from lighteval.metrics.dynamic_metrics import *
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language




def multilingual_extractive_match_metric_at_k(
    language: Language = Language.ENGLISH,
    gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    n: int = 8,
    precision: int = 6,
    timeout_seconds: int = 5,
) -> SampleLevelMetric:

    @timeout(2)
    def add_to_specifics_with_timeout(
        formatted_doc: Doc, extracted_predictions: list[list[str]], extracted_golds: list[list[str]]
    ) -> None:
        if formatted_doc.specific is None:
            formatted_doc.specific = {}

        formatted_doc.specific["extracted_predictions"] = [
            str(pred) for preds in extracted_predictions for pred in preds
        ]
        formatted_doc.specific["extracted_golds"] = [str(gold) for golds in extracted_golds for gold in golds]

    def sample_level_fn(golds: list[str], predictions: list[str], formatted_doc: Doc) -> list:
        gold_extraction_regexes = get_extraction_regexes(formatted_doc, gold_extraction_target, language)
        pred_extraction_regexes = get_extraction_regexes(formatted_doc, pred_extraction_target, language)

        extracted_predictions = [
            extract_target_from_pred(pred, pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
            for pred in predictions
        ]
        extracted_golds = [
            extract_target_from_pred(gold, gold_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds)
            for gold in golds
        ]

        # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            logger.warning(f"We did not manage to extract a gold in the correct format. Gold: {golds}")
            extracted_golds = [[gold] for gold in golds]

        if all(len(p) == 0 for p in extracted_predictions):
            logger.warning(
                f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Pred: {predictions}"
            )

        # We have to use timeout because the sypmy to str conversion can be very slow
        try:
            add_to_specifics_with_timeout(formatted_doc, extracted_predictions, extracted_golds)
        except Exception:  # noqa: E722
            logger.warning("Timeout when adding extracted predictions and golds to specific")

        binary_list =  [
                (
                    1
                    if any(
                        compare_gold_target(gold, pred, precision, timeout_seconds=timeout_seconds)
                        for gold in extracted_golds
                    )
                    else 0
                )
                for pred in extracted_predictions
        ]
        value = ''.join(str(i) for i in binary_list)
        value = int(value, 2)
        return value

    return SampleLevelMetric(
        metric_name="extractive_match@1:"+str(n),
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )



# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

MATH_QUERY_TEMPLATE_R1 = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question.split('')}

Hint: to solve the problem, you may need to figure out how many 8-digit numbers,using each of the digits 1 through 8 exactly once,are divisibleby 1111.
""".strip()

MATH_QUERY_TEMPLATE_R2 = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. To solve the problem, you may need to figure out how many 8-digit numbers,using each of the digits 1 through 8 exactly once,are divisibleby 1111. Think step by step before answering.

{Question}
""".strip()

MATH_QUERY_TEMPLATE_R3 = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}

Hint: to solve the problem, you may need to figure out how many times does the sine curve cross the log curve? The log function is a slowly increasing function in the interval where LHS is going from -1 to 1.
""".strip()

MATH_QUERY_TEMPLATE_R4 = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. 
To solve the problem, you may need to figure out how many times does the sine curve cross the log curve? The log function is a slowly increasing function in the interval where LHS is going from -1 to 1.
Think step by step before answering.

{Question}
""".strip()


# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


GPQA_QUERY_TEMPLATE_R1 = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}

Hint: to solve the problem, you may need to identify the final product when cyclobutyl(cyclopropyl)methanol reacts with phosphoricacid in water.
""".strip()

GPQA_QUERY_TEMPLATE_R2 = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.
To solve the problem, you may need to identify the final product when cyclobutyl(cyclopropyl)methanol reacts with phosphoricacid in water.
Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

GPQA_QUERY_TEMPLATE_R3 = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}

Hint: to solve the problem, you may need to figuer out how can cyclopropane ring opening be utilized as a strain-relieving mechanism to form an allylic carbocation.
""".strip()

GPQA_QUERY_TEMPLATE_R4 = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. To solve the problem, you may need to figuer out how can cyclopropane ring opening be utilized as a strain-relieving mechanism to form an allylic carbocation. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

latex_gold_multi_metric = multilingual_extractive_match_metric_at_k(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    n=10,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
)

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gold_multi_metric = multilingual_extractive_match_metric_at_k(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    n=10,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
)

gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)

math_unrelated_condition = [
    # 'To solve the problem, you may need to figure out how many 8-digit numbers,using each of the digits 1 through 8 exactly once,are divisibleby 1111.',
    # 'To solve the problem, you may need to figure out how many times does the sine curve cross the log curve? The log function is a slowly increasing function in the interval where LHS is going from -1 to 1.',
    'To solve the problem, you may need to find the sum of all values P(1) for polynomials P(z) with certain conditions.',
    'To solve the problem, you may need to find the range ofis this fraction: \[ \frac{x^3 + y^3 +z^3 -x^{-3}- y^{-3}-z^{-3}}{x+y+z-x^{-1}-y^{-1} - z^{-1}} \]',
    'To solve the problem, you may need to find out what percent of the area of a regular hexagonwas removed when it\'s truncated to form a regulardodecagon.'
]

gpqa_unrelated_condition = [
    'To solve the problem, you may need to figure out which compound has the most electronically deshielded hydrogen.',
    'To solve the problem, you may need to name one product formed from these conditions.',
    'To solve the problem, you may need start with the material (R)-(+)-Limonene, which is a cyclic monoterpene.',

]

def math_prompt_r1_fn(line, task_name: str = None):
    seq = line["problem"].split(".") 
    problem = f"{'.'.join(seq[:-1])}. {math_unrelated_condition[0]} {seq[-1]}"
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=problem),
        choices=[line["solution"]],
        gold_index=0,
    )

def math_prompt_r2_fn(line, task_name: str = None):
    seq = line["problem"].split(".") 
    problem = f"{'.'.join(seq[:-1])}. {math_unrelated_condition[1]} {seq[-1]}"
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=problem),
        choices=[line["solution"]],
        gold_index=0,
    )

def math_prompt_r3_fn(line, task_name: str = None):
    seq = line["problem"].split(".") 
    problem = f"{'.'.join(seq[:-1])}. {math_unrelated_condition[2]} {seq[-1]}"
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=problem),
        choices=[line["solution"]],
        gold_index=0,
    )
def math_prompt_r4_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE_R4.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def math_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def jingyou_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )

def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )
    

def gpqa_prompt_r1_fn(line, task_name: str = None):
    seq = line["Question"].split(".") 
    problem = f"{'.'.join(seq[:-1])}. {gpqa_unrelated_condition[0]} {seq[-1]}"
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=problem
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )
    


def gpqa_prompt_r2_fn(line, task_name: str = None):
    seq = line["Question"].split(".") 
    problem = f"{'.'.join(seq[:-1])}. {gpqa_unrelated_condition[1]} {seq[-1]}"

    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=problem
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )
    


def gpqa_prompt_r3_fn(line, task_name: str = None):
    seq = line["Question"].split(".") 
    problem = f"{'.'.join(seq[:-1])}. {gpqa_unrelated_condition[2]} {seq[-1]}"

    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=problem
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )
    
    

def gpqa_prompt_r4_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE_R4.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )
    

def create_task_config(name, suite, prompt_function, hf_repo, hf_subset, hf_avail_splits, evaluation_splits, metric):
    return LightevalTaskConfig(
        name=name,
        suite=suite,
        prompt_function=prompt_function,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=hf_avail_splits,
        evaluation_splits=evaluation_splits,
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        metric=[metric],
        version=1,
    )

aime24 = create_task_config("aime24", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_c1 = create_task_config("aime24_c1", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/checklist_l1.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_c5 = create_task_config("aime24_c5", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/checklist_l5.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_c10 = create_task_config("aime24_c10", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/checklist_l10.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_b1 = create_task_config("aime24_b1", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/textbugger_l1.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_b5 = create_task_config("aime24_b5", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/textbugger_l5.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_b10 = create_task_config("aime24_b10", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/textbugger_l10.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_f1 = create_task_config("aime24_f1", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/textfooler_l1.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_f5 = create_task_config("aime24_f5", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/textfooler_l5.json", "default", ["train"], ["train"], expr_gold_metric)
aime24_f10 = create_task_config("aime24_f10", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24/textfooler_l10.json", "default", ["train"], ["train"], expr_gold_metric)


jingyou = create_task_config("jingyou", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/Jingyou_filter_v1_trans_ds_300.jsonl", "default", ["train"], ["train"], expr_gold_metric)
jingyou_c1 = create_task_config("jingyou_c1", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/checklist_l1.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_c5 = create_task_config("jingyou_c5", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/checklist_l5.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_c10 = create_task_config("jingyou_c10", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/checklist_l10.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_b1 = create_task_config("jingyou_b1", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/textbugger_l1.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_b5 = create_task_config("jingyou_b5", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/textbugger_l5.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_b10 = create_task_config("jingyou_b10", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/textbugger_l10.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_f1 = create_task_config("jingyou_f1", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/textfooler_l1.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_f5 = create_task_config("jingyou_f5", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/textfooler_l5.json", "default", ["train"], ["train"], expr_gold_metric)
jingyou_f10 = create_task_config("jingyou_f10", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/jingyou/textfooler_l10.json", "default", ["train"], ["train"], expr_gold_metric)


math_500 = create_task_config("math_500", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_metric)
math_500_c1 = create_task_config("math_500_c1", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/checklist_l1.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_c5 = create_task_config("math_500_c5", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/checklist_l5.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_c10 = create_task_config("math_500_c10", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/checklist_l10.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_b1 = create_task_config("math_500_b1", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/textbugger_l1.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_b5 = create_task_config("math_500_b5", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/textbugger_l5.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_b10 = create_task_config("math_500_b10", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/textbugger_l10.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_f1 = create_task_config("math_500_f1", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/textfooler_l1.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_f5 = create_task_config("math_500_f5", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/textfooler_l5.json", "default", ["train"], ["train"], latex_gold_metric)
math_500_f10 = create_task_config("math_500_f10", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500/textfooler_l10.json", "default", ["train"], ["train"], latex_gold_metric)



math_500_r1 = create_task_config("math_500_r1", ["custom"], math_prompt_r1_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_metric)
math_500_r2 = create_task_config("math_500_r2", ["custom"], math_prompt_r2_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_metric)
math_500_r3 = create_task_config("math_500_r3", ["custom"], math_prompt_r3_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_metric)
math_500_r4 = create_task_config("math_500_r4", ["custom"], math_prompt_r4_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_metric)


gpqa_diamond = create_task_config("gpqa:diamond", ["custom"], gpqa_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_diamond.jsonl", "default", ["train"], ["train"], gpqa_metric)

gpqa_diamond_r1 = create_task_config("gpqa:diamond_r1", ["custom"], gpqa_prompt_r1_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_diamond.jsonl", "default", ["train"], ["train"], gpqa_metric)
gpqa_diamond_r2 = create_task_config("gpqa:diamond_r2", ["custom"], gpqa_prompt_r2_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_diamond.jsonl", "default", ["train"], ["train"], gpqa_metric)
gpqa_diamond_r3 = create_task_config("gpqa:diamond_r3", ["custom"], gpqa_prompt_r3_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_diamond.jsonl", "default", ["train"], ["train"], gpqa_metric)
gpqa_diamond_r4 = create_task_config("gpqa:diamond_r4", ["custom"], gpqa_prompt_r4_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_diamond.jsonl", "default", ["train"], ["train"], gpqa_metric)


aime24_multi = create_task_config("aime24_multi", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24.json", "default", ["train"], ["train"], expr_gold_multi_metric)
math_500_multi = create_task_config("math_500_multi", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_multi_metric)
jingyou_multi = create_task_config("jingyou_multi", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/Jingyou_filter_v1_trans_ds_1k.jsonl", "default", ["train"], ["train"], expr_gold_multi_metric)



# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime24_multi)
TASKS_TABLE.append(aime24_c1)
TASKS_TABLE.append(aime24_c5)
TASKS_TABLE.append(aime24_c10)
TASKS_TABLE.append(aime24_b1)
TASKS_TABLE.append(aime24_b5)
TASKS_TABLE.append(aime24_b10)
TASKS_TABLE.append(aime24_f1)
TASKS_TABLE.append(aime24_f5)
TASKS_TABLE.append(aime24_f10)
# TASKS_TABLE.append(aime25)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(math_500_r1)
TASKS_TABLE.append(math_500_r2)
TASKS_TABLE.append(math_500_r3)
TASKS_TABLE.append(math_500_r4)
TASKS_TABLE.append(math_500_multi)
TASKS_TABLE.append(math_500_c1)
TASKS_TABLE.append(math_500_c5)
TASKS_TABLE.append(math_500_c10)
TASKS_TABLE.append(math_500_b1)
TASKS_TABLE.append(math_500_b5)
TASKS_TABLE.append(math_500_b10)
TASKS_TABLE.append(math_500_f1)
TASKS_TABLE.append(math_500_f5)
TASKS_TABLE.append(math_500_f10)
TASKS_TABLE.append(gpqa_diamond)
TASKS_TABLE.append(gpqa_diamond_r1)
TASKS_TABLE.append(gpqa_diamond_r2)
TASKS_TABLE.append(gpqa_diamond_r3)
TASKS_TABLE.append(gpqa_diamond_r4)
TASKS_TABLE.append(jingyou)
TASKS_TABLE.append(jingyou_multi)
TASKS_TABLE.append(jingyou_c1)
TASKS_TABLE.append(jingyou_c5)
TASKS_TABLE.append(jingyou_c10)
TASKS_TABLE.append(jingyou_b1)
TASKS_TABLE.append(jingyou_b5)
TASKS_TABLE.append(jingyou_b10)
TASKS_TABLE.append(jingyou_f1)
TASKS_TABLE.append(jingyou_f5)
TASKS_TABLE.append(jingyou_f10)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
