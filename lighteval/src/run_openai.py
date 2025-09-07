import argparse
import sys
sys.path.append("/mmu_nlp_hdd/jianggangwei/LcotRobust/lighteval/src")
from lighteval.main_endpoint import openai


def main():
    parser = argparse.ArgumentParser(description='Example command line argument parser.')

    parser.add_argument(
        '--model_args',
        type=str,
        help="Model arguments in the form key1=value1,key2=value2,... or path to yaml config file (see examples/model_configs/transformers_model.yaml)",
        required=True
    )
    parser.add_argument(
        '--tasks',
        type=str,
        help="Comma-separated list of tasks to evaluate on.",
        required=True
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        help="Use system prompt for evaluation.",
        default=None
    )
    parser.add_argument(
        '--dataset_loading_processes',
        type=int,
        help="Number of processes to use for dataset loading.",
        default=1
    )
    parser.add_argument(
        '--custom_tasks',
        type=str,
        help="Path to custom tasks directory.",
        default=None
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        help="Cache directory for datasets and models.",
        default='CACHE_DIR'  # 使用实际的 CACHE_DIR 变量值
    )
    parser.add_argument(
        '--num_fewshot_seeds',
        type=int,
        help="Number of seeds to use for few-shot evaluation.",
        default=1
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Output directory for evaluation results.",
        default="results"
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help="Push results to the huggingface hub.",
        default=False
    )
    parser.add_argument(
        '--push_to_tensorboard',
        action='store_true',
        help="Push results to tensorboard.",
        default=False
    )
    parser.add_argument(
        '--public_run',
        action='store_true',
        help="Push results and details to a public repo.",
        default=False
    )
    parser.add_argument(
        '--results_org',
        type=str,
        help="Organization to push results to.",
        default=None
    )
    parser.add_argument(
        '--save_details',
        action='store_true',
        help="Save detailed, sample per sample, results.",
        default=False
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        help="Maximum number of samples to evaluate on.",
        default=None
    )
    parser.add_argument(
        '--job_id',
        type=int,
        help="Optional job id for future reference.",
        default=0
    )

    args = parser.parse_args()

    openai(**vars(args))

    # args.model_args, args.tasks, args.use_chat_template, args.system_prompt, args.dataset_loading_processes, args.custom_tasks, args.cache_dir, args.num_fewshot_seeds, args.load_responses_from_details_date_id, args.output_dir, args.push_to_hub, args.push_to_tensorboard, args.public_run, args.results_org, args.save_details, args.max_samples, args.job_id

if __name__ == '__main__':
    main()