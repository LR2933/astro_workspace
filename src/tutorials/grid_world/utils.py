import pickle
from datetime import datetime
from pathlib import Path


def save_model(q_values, project_name: str):
    """保存 Q-learning 模型。"""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / f"{project_name}_{timestamp}.pkl").open("wb") as f:
        pickle.dump(dict(q_values), f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_dir


def load_model(model_path: str | Path):
    """加载 Q-learning 模型。"""

    with Path(model_path).open("rb") as f:
        return pickle.load(f)


def encode_obs(obs) -> tuple[int, int, int, int]:
    """转换obs的格式: dict -> tuple"""
    return (
        int(obs["agent"][0]),
        int(obs["agent"][1]),
        int(obs["target"][0]),
        int(obs["target"][1]),
    )
