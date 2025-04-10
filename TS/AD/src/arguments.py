from omegaconf import OmegaConf
import argparse

def parser():
    parser = argparse.ArgumentParser(description='koreabank')
    parser.add_argument('--model_name', type=str, help='model name for experiments')
    parser.add_argument('--default_cfg', type=str, help='configuration for default setting')
    parser.add_argument('--model_cfg', type=str, help='configuration for model')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()

    # load configs
    cfg = OmegaConf.load(args.default_cfg)
    cfg = OmegaConf.merge(cfg, {'MODEL':{'modelname' : args.model_name}})
    
    if args.model_cfg:
        model_cfg = OmegaConf.load(args.model_cfg)
        modelname = args.model_name

        # Check if the modelname is in the model_config
        if modelname in model_cfg:
            model_setting_conf = OmegaConf.create(model_cfg[modelname])
            # Merge the specific model config with the default config
            cfg = OmegaConf.merge(cfg, {'MODELSETTING' : model_setting_conf})
        else:
            print(f"Model '{modelname}' not found in the model_config.")
            return cfg
            
    # update cfg
    if args.opts:
        for opt in args.opts:
            if "=" in opt:
                k, v = opt.split("=", 1)
                try:
                    OmegaConf.update(cfg, k, eval(v), merge=True)
                except:
                    OmegaConf.update(cfg, k, v, merge=True)
                    
    return cfg  