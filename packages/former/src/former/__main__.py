from typing import Literal
from former.util import load_config
from former.processor import get_processor
import click


config = load_config()


@click.command()
@click.option("--audio-path", type=str, required=False)
@click.option("--audio-list", type=str, required=False)
@click.option("--mode", type=str, default="longform")
def run(audio_path, audio_list, mode):
    """
    Run the model with the given configuration.

    Args:
        config_path (str, optional): Path to the YAML config file. If None, uses kwargs.
        **kwargs: Configuration parameters that override the YAML config.
    """
    # Load base config from file if provided

    # Print the configuration
    print(f"Model Checkpoint: {config.model_checkpoint}")
    print(f"Device: {config.device}")
    print(f"Total Duration in a Batch (in second): {config.total_batch_duration}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"Left Context Size: {config.left_context_size}")
    print(f"Right Context Size: {config.right_context_size}")

    assert config.model_checkpoint is not None, "You must specify the path to the model"

    processor = get_processor()

    if mode == "longform":
        result = processor.transcribe(audio_path, mode="longform")
    elif mode == "batch":
        result = processor.transcribe(audio_list, mode="batch")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    print(result)


if __name__ == "__main__":
    run()
