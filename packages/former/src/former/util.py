import os
import torch
import yaml
import pathlib
from hydra import initialize, compose
from former.model.utils.init_model import init_model
from former.model.utils.checkpoint import load_checkpoint
from former.model.utils.file_utils import read_symbol_table
from pydub import AudioSegment


DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    None: None,
}


@torch.no_grad()
def init(model_checkpoint, device):

    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(config_path, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config, config_path)
    model.eval()
    load_checkpoint(model, checkpoint_path)

    model.encoder = model.encoder.to(device)
    model.ctc = model.ctc.to(device)
    # print('the number of encoder params: {:,d}'.format(num_params))

    symbol_table = read_symbol_table(symbol_table_path)
    char_dict = {v: k for k, v in symbol_table.items()}

    return model, char_dict


def load_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # set bit depth to 16bit
    audio = audio.set_channels(1)  # set to mono
    audio = torch.as_tensor(
        audio.get_array_of_samples(), dtype=torch.float32
    ).unsqueeze(0)
    return audio


def load_config(**kwargs):
    """Load configuration using Hydra."""
    # Register the config class
    # cs = ConfigStore.instance()
    # cs.store(name="config", node=Config)

    # Initialize Hydra and load config
    with initialize(config_path="./conf", version_base=None):
        conf = compose(config_name="config")

    # Override with any provided kwargs
    for key, value in kwargs.items():
        setattr(conf, key, value)

    # Set up device - keep as string
    device_str = (
        "cuda" if torch.cuda.is_available() and conf.device == "cuda" else "cpu"
    )
    conf.device = device_str

    # Convert model checkpoint path to absolute path
    conf.model_checkpoint = str(
        pathlib.Path(__file__).parent.parent.parent / conf.model_checkpoint
    )

    return conf
