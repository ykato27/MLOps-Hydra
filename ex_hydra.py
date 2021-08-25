import hydra
from omegaconf import DictConfig
import lightgbm as lgb

@hydra.main(config_name='config') 
def my_ex(cfg: DictConfig) -> None:
    print(cfg)
    model = get_model(cfg.params)
    return model

def get_model(params):
    return lgb.LGBMClassifier(**params)
 
if __name__ == "__main__":
    my_ex()