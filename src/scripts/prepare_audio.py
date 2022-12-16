import hydra
import pyrootutils
from omegaconf import DictConfig

from src import utils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    dm = hydra.utils.instantiate(cfg.datamodule.audio)
    dm.prepare_data()
    dm.setup()


if __name__ == "__main__":
    main()
