import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging

from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()

    tasks = [instantiate(task_cfg) for task_cfg in cfg.task_registry.values()]
    
    data_provider = instantiate(cfg.data.provider)

    logger.info("Generating Raw Data...")
    data_provider.generate_raw_data()
    
    logger.info("Generating Splits...")
    data_provider.generate_split_data(tasks=tasks)

if __name__ == "__main__":
    main()