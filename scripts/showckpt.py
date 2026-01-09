import torch

try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False


def inspect_checkpoint(ckpt_path):
    """Load and inspect the structure of a PyTorch checkpoint file."""
    print(f"Loading checkpoint: {ckpt_path}\n")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print("=" * 60)
    print("Checkpoint Top-Level Keys:")
    print("=" * 60)
    if isinstance(ckpt, dict):
        for key in ckpt.keys():
            print(f"  - {key}")
    else:
        print(f"  Checkpoint is not a dict, type: {type(ckpt)}")
    
    print("\n" + "=" * 60)
    print("Detailed Structure:")
    print("=" * 60)
    
    if isinstance(ckpt, dict):
        for key, value in ckpt.items():
            print(f"\n[{key}]")
            
            # Handle OmegaConf DictConfig
            if HAS_OMEGACONF and isinstance(value, DictConfig):
                print("  OmegaConf DictConfig:")
                try:
                    config_dict = OmegaConf.to_container(value, resolve=True)
                    for k, v in config_dict.items():
                        if isinstance(v, dict):
                            print(f"    {k}: <dict with {len(v)} keys>")
                        else:
                            print(f"    {k}: {v}")
                except Exception as e:
                    print(f"    (Error converting: {e})")
            
            # Handle state_dict
            elif key == "state_dict" and isinstance(value, dict):
                print(f"  Number of parameters: {len(value)}")
                print("  Parameter keys and shapes:")
                for param_name, param_value in value.items():
                    if torch.is_tensor(param_value):
                        print(f"    {param_name}: {param_value.shape}")
                    else:
                        print(f"    {param_name}: {type(param_value).__name__}")
            
            # Handle regular dict
            elif isinstance(value, dict):
                print(f"  Dict with {len(value)} keys:")
                for k, v in list(value.items())[:10]:
                    print(f"    {k}: {type(v).__name__}")
                if len(value) > 10:
                    print(f"    ... and {len(value) - 10} more keys")
            
            # Handle list
            elif isinstance(value, list):
                print(f"  List with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0]).__name__}")
            
            # Handle tensor
            elif torch.is_tensor(value):
                print(f"  Tensor shape: {value.shape}")
            
            # Handle other types
            else:
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  Type: {type(value).__name__}, Value: {val_str}")


if __name__ == "__main__":
    ckpt_path = "/home_zfs/wanghanyi/mRNAHyena/checkpoint/mRNAUMAP/weights.ckpt"
    inspect_checkpoint(ckpt_path)