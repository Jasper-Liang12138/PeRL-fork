
import os
import random
import re

from datasets import load_dataset

# from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb
# simple count down dataset


def format_reward_func(completions, **kwargs):
    """
    Format reward function. Checks if model output matches format: <think>...</think><answer>...</answer>

    Args:
        completions (list[str]): Generated outputs
    Returns:
        list[float]: Reward scores
    """
    # Initialize reward list
    rewards = []
    # Iterate over generated outputs
    for completion in completions:
        try:
            # Add <think> tag at start to facilitate regex matching
            completion = "<think>" + completion

            if random.random() < 0.1:  # Write output to file with 10% probability
                # Create output directory if not exists
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)  # Write generated output

            # Regex to match <think> and <answer> tags
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)  # Regex match

            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)  # Reward 0 for incorrect format
            else:
                rewards.append(1.0)  # Reward 1 for correct format
        except Exception:
            rewards.append(0.0)  # Reward 0 on exceptions

    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """
    Equation reward function. Checks if result is correct and numbers are used per requirements
    (each number used once, using only provided numbers).

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Provided numbers

    Returns:
        list[float]: Reward scores
    """
    # Initialize reward list
    rewards = []
    # Iterate over outputs, targets and provided numbers
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # Add <think> tag at start to facilitate regex matching
            completion = "<think>" + completion
            # Regex to match <answer> tag
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)  # Reward 0 if <answer> tag not found
                continue
            equation = match.group(1).strip()  # Extract content inside <answer>
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # Check if all numbers are used and used only once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue

            # Only allow digits, basic operators, parentheses and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)  # Reward 0 if equation contains invalid characters
                continue

            # Compute equation result
            result = eval(equation, {"__builtins__": None}, {})
            # Check if result matches ground truth (error < 1e-5)
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)  # Reward 1 if correct

                # Write successful outputs to file with 10% probability
                if random.random() < 0.10:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)
            else:
                rewards.append(0.0)  # Reward 0 if incorrect result
        except Exception:
            rewards.append(0.0)  # Reward 0 if evaluation fails

    return rewards

def thought_len_reward_func(completions, **kwargs):
    """
    "Thought length" reward function. Checks if <think> tag content length is greater than 1000.

    Args:
        completions (list[str]): Generated outputs
    Returns:
        list[float]: Reward scores
    """
    # Initialize reward list
    rewards = []
    # Iterate over generated outputs
    for completion in completions:
        try:
            # Add <think> tag at start to facilitate regex matching
            completion = "<think>" + completion
            # Regex to match <think> tag
            match = re.search(r"<think>(.*?)</think>", completion)
            # If <think> tag is found
            if match:
                thought_process = match.group(1).strip()  # Extract inside <think> tag
                thought_length = len(thought_process)  # Compute length
                if thought_length > 1000:
                    rewards.append(1.0)  # Reward 1 if length > 1000
                else:
                    rewards.append(0.0)  # Otherwise reward 0
            else:
                rewards.append(0.0)  # Reward 0 if tag not found
                continue
        except Exception:
            rewards.append(0.0)  # Reward 0 on exceptions

    return rewards


def load_count_down_dataset(
    dataset_name_or_path: str, 
    example_numbers: int = None,
    tokenizer = None
):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # Load dataset from Hugging Face Hub
    dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset = load_dataset(dataset_id, split="train")
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42)

    if example_numbers is not None and len(dataset) > example_numbers:
        dataset = dataset.select(range(example_numbers))

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }]
        return {
                "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": target
            }

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "reward_functions": [format_reward_func, equation_reward_func, thought_len_reward_func]
    }