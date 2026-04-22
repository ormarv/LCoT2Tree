#!/usr/bin/env python3
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple
import numpy as np

def eval_dataset_to_list(dataset:Dataset, nb_samples_per_subj:int)->List[Tuple[str,str]]:
    samples_by_subject = {}
    for sample in dataset:
        subject = sample['subject']
        if subject not in samples_by_subject:
            samples_by_subject[subject] = []
        samples_by_subject[subject].append((sample['question']+"\nPossible answers: "+"\n".join(sample["choices"]), sample['choices'][sample['answer']]))
    indices = [np.random.randint(0,len(samples), nb_samples_per_subj).tolist() for _,samples in samples_by_subject.items()]
    list_samples = []
    #print([subject for subject in samples_by_subject])
    for i, (_, samples) in enumerate(samples_by_subject.items()):
        list_samples.extend(np.array(samples)[indices[i]].tolist())
    return list_samples


def load_MMLU(nb_samples_per_subj:int, seed:int=42):
    part_names = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    np.random.seed(seed)
    dataset = load_dataset("/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/datasets--cais--mmlu/")
    print(dataset)
    train_split = dataset["train"]
    all_train_samples = np.array([(sample['question']+"\nPossible answers: "+"\n".join(sample["choices"]), sample['choices'][sample['answer']]) for sample in train_split])
    indices = np.random.randint(0, len(all_train_samples), nb_samples_per_subj*len(part_names))
    train_samples = all_train_samples[indices].tolist()
    eval_split = dataset["validation"]
    eval_samples = eval_dataset_to_list(eval_split, nb_samples_per_subj)
    test_split = dataset["test"]
    # now we get the list of samples by subject for the test split
    samples_by_subject = {}
    for sample in test_split:
        subject = sample['subject']
        if subject not in samples_by_subject:
            samples_by_subject[subject] = []
        samples_by_subject[subject].append((sample['question']+"\nPossible answers: "+"\n".join(sample["choices"]), sample['choices'][sample['answer']]))
    indices = [np.random.randint(0,len(samples), nb_samples_per_subj).tolist() for _,samples in samples_by_subject.items()]
    #print(len(list(samples_by_subject.items())[0]))
    test_samples = {item[0]:np.array(item[1])[indices[i]].tolist() for i, item in enumerate(samples_by_subject.items())}
    return train_samples, eval_samples, test_samples



train_samples, eval_samples, test_samples = load_MMLU(3)
print(len(train_samples))
#print(train_samples)
print(len(eval_samples))
print([len(subject) for subject in test_samples])
print(len(test_samples['public_relations']))
print(train_samples[0])
