import hydra
from omegaconf import ListConfig, DictConfig

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(FLAGS):
    if not FLAGS.no_train:
        dataset_train = hydra.utils.instantiate(FLAGS.train_dataset)
    else:
        dataset_train = None
    
    if isinstance(FLAGS.validate_dataset, DictConfig):
        FLAGS.validate_dataset = ListConfig([FLAGS.validate_dataset])
    dataset_validate = [hydra.utils.instantiate(cfg) for cfg in FLAGS.validate_dataset]

if __name__ == "__main__":
    main()