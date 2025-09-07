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
from lighteval.tasks.extended.lcb.main import *




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


# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

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


gpqa_multi_metric = multilingual_extractive_match_metric_at_k(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
    n=10,
)


gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


def codegen_multi_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    generated_code_snippets = [[extract_code(pred) for pred in predictions]]  # noqa: F841
    evaluation_sample = {  # noqa: F841
        "inputs": formatted_doc.specific["inputs"],
        "outputs": formatted_doc.specific["outputs"],
        "fn_name": formatted_doc.specific["fn_name"],
    }
    evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

    metrics, _ = codegen_metrics(
        evaluation_sample,
        generated_code_snippets,
        k_list=[1],  # Only run for Pass@1
        num_process_evaluate=8,
    )
    return metrics["correct"]


lcb_codegen_multi_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",  # This is the way of informing the number of generations currently
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=codegen_multi_metric,
    corpus_level_fn=np.mean,
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
    random.seed(42)
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



def mmlupro_prompt_fn(line, task_name: str = None):

    choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

    query = f'The following are multiple choice questions (with answers) about {line["category"]}. Think step by step and then finish your answer with "The final answer is: X" where X is the correct letter choice.'
    query += "Question:\n" + line["question"] + "\n"
    options = line["options"]
    query += "Options:\n"
    for i, opt in enumerate(options):
        query += "{}. {}\n".format(choices[i], opt)
    query += "Answer: Let's think step by step."

    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        gold_index=int(line["answer_index"]),
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
jingyou = create_task_config("jingyou", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/Jingyou_filter_v1_trans_ds_300.jsonl", "default", ["train"], ["train"], expr_gold_metric)
math_500 = create_task_config("math_500", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_metric)
gpqa_diamond = create_task_config("gpqa:diamond", ["custom"], gpqa_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_diamond.jsonl", "default", ["train"], ["train"], gpqa_metric)
mmlu_pro = create_task_config("mmlu_pro", ["custom"], mmlupro_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/mmlu_pro_3k.jsonl", "default", ["train"], ["train"], gpqa_metric)

aime24_multi = create_task_config("aime24_multi", ["custom"], aime_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/aime24.json", "default", ["train"], ["train"], expr_gold_multi_metric)

math_l5_multi = create_task_config("math_l5_multi", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math_level5.jsonl", "default", ["train"], ["train"], latex_gold_multi_metric)
math_500_multi = create_task_config("math_500_multi", ["custom"], math_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/math500.jsonl", "default", ["train"], ["train"], latex_gold_multi_metric)
jingyou_multi = create_task_config("jingyou_multi", ["custom"], jingyou_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/Jingyou_filter_v1_trans_ds_1k.jsonl", "default", ["train"], ["train"], expr_gold_multi_metric)
gpqa_diamond_multi = create_task_config("gpqa:diamond_multi", ["custom"], gpqa_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_diamond.jsonl", "default", ["train"], ["train"], gpqa_multi_metric)
gpqa_main_multi = create_task_config("gpqa:main_multi", ["custom"], gpqa_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/gpqa_main.jsonl", "default", ["train"], ["train"], gpqa_multi_metric)
mmlu_pro_multi = create_task_config("mmlu_pro_multi", ["custom"], mmlupro_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/mmlu_pro_3k.jsonl", "default", ["train"], ["train"], gpqa_multi_metric)
lcb_multi_v5 = create_task_config("lcb_v5", ["custom"], lcb_codegeneration_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/lcb_v5.jsonl", "default", ["train"], ["train"], lcb_codegen_multi_metric)
lcb_multi_v6 = create_task_config("lcb_v6", ["custom"], lcb_codegeneration_prompt_fn, "/mmu_nlp_hdd/jianggangwei/datasets/lcb_v6.jsonl", "default", ["train"], ["train"], lcb_codegen_multi_metric)




# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime24_multi)
# TASKS_TABLE.append(aime25)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(math_l5_multi)
TASKS_TABLE.append(math_500_multi)
TASKS_TABLE.append(jingyou)
TASKS_TABLE.append(jingyou_multi)
TASKS_TABLE.append(gpqa_diamond)
TASKS_TABLE.append(gpqa_diamond_multi)
TASKS_TABLE.append(gpqa_main_multi)
TASKS_TABLE.append(mmlu_pro)
TASKS_TABLE.append(mmlu_pro_multi)
TASKS_TABLE.append(lcb_multi_v5)
TASKS_TABLE.append(lcb_multi_v6)



# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
