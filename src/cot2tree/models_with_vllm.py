#!/usr/bin/env python3
from vllm import LLM, SamplingParams

def run_gguf_inference(model_path, tokenizer):
    # Sample prompts.
    prompts = [
        "How many helicopters can a human eat in one sitting?",
        "What's the future of AI?",
    ]
    prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    # Create an LLM.
    llm = LLM(model=model_path, tokenizer=tokenizer)

    outputs = llm.chat(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    tokenizer = "models--unsloth--DeepSeek-R1-Distill-Llama-70B-GGUF/snapshots/732dd974083ea5877d7b6d788b36fe7c2e5eab36/"
    model = "models--unsloth--DeepSeek-R1-Distill-Llama-70B-GGUF/snapshots/732dd974083ea5877d7b6d788b36fe7c2e5eab36/DeepSeek-R1-Distill-Llama-70B-Q5_K_M.gguf"
    run_gguf_inference(model, tokenizer)
