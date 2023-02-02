from detectron2.config import CfgNode

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _CC

    return _CC.clone()
