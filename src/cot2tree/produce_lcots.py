#!/usr/bin/env python3
from typing import Dict, List, Tuple
def produce_lcots(lrm, train_samples:List[Tuple[str,str]],eval_samples:List[Tuple[str,str]], test_samples:Dict[str,List[Tuple[str,str]]], save_dict:str):
    train_lcots = [lrm.run(question) for question,_ in train_samples]
    eval_lcots = [lrm.run(question) for question,_ in eval_samples]
    test_lcots = {}
    for subject, samples in list(test_samples.items()):
        test_lcots[subject] = [lrm.run(question) for question,_ in samples]
    with open(save_dict+"/train.txt","w+") as f:
        print(train_lcots, file=f)
    with open(save_dict+"/eval.txt","w+") as f:
        print(eval_lcots, file=f)
    with open(save_dict+"/test.txt","w+") as f:
        print(test_lcots, file=f)
    return train_lcots, eval_lcots, test_lcots

DATASET_PATH = "cais/mmlu"
list_subsets = ['abstract_algebra', 'all', 'anatomy', 'astronomy', 'auxiliary_train', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
questions = {}
choices = {}
answers = {}
expected_answers = {}
subjects = {}
idx = 0
for subset in list_subsets:
    if subset!='all':
        mmlu = load_dataset(DATASET_PATH,subset)
        for line in mmlu[:45]:
            questions[idx] = mmlu["question"]
            choices[idx] = mmlu["choices"]
            expected_answers[idx] = mmlu["answer"]
            subjects[idx] = subset
            idx += 1

# Now we query each of our three LRM clients for each question

for idx, question in questions:
    list_answers = []
    # here, add the answers by each LLM
