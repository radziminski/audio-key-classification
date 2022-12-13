import hydra


# Function that takes a config that for some reason was not instantiated recursively (since it thad "target"
# instead of "_target_")
# TODO: investigate if this can be done in less hacky way
def instantiate_delayed(config):
    target = config["target"]
    del config["target"]
    instantiated = hydra.utils.instantiate(config, _target_=target)

    return instantiated
