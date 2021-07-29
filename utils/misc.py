from pathlib import Path


def find_snapshot(opt, interactive=True):
    ckpt_path = Path(opt.ckpt)
    if ckpt_path.exists():
        return ckpt_path
    else:
        print(f"Cannot find checkpoints from `opt.ckpt={opt.ckpt}`.")

    if interactive:
        # Import readline module to improve the experience of input
        # noinspection PyUnresolvedReferences
        import readline

        while True:
            inputs = input("Please input a checkpoint file ([q] to skip):")
            if inputs in ["q", "Q"]:
                break
            ckpt_path = Path(inputs)
            if ckpt_path.exists():
                return ckpt_path
            else:
                print("Not found!")

    return None
