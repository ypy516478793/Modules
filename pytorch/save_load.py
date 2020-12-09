from natsort import natsorted
import torch
import os

## Save/Load from state_dict
def save_state_dict(cp_path, model):
    torch.save(model.state_dict(), cp_path)

def load_state_dict(cp_dir, load_model, model):
    if load_model:
        cp_list = [i for i in os.listdir(cp_dir) if i.endswith("pt")]
        cp_list = natsorted(cp_list)
        assert len(cp_list) > 0, "No model listed in {:s}".format(cp_dir)
        if isinstance(load_model, bool):
            cp_file = cp_list[-1]
        else:
            cp_file = load_model
        start_epoch = int("".join(filter(str.isdigit, cp_file)))
        cp_path = os.path.join(cp_dir, cp_file)
        model.load_state_dict(torch.load(cp_path))   ## load from state_dict
        print("Model loaded from {}".format(cp_path))
    else:
        start_epoch = 0
    return start_epoch



## Save/Load entire model
def save_model(cp_path, model):
    torch.save(model, cp_path)
    
def load_model(cp_dir, load_model):
    if load_model:
        cp_list = [i for i in os.listdir(cp_dir) if i.endswith("pt")]
        cp_list = natsorted(cp_list)
        assert len(cp_list) > 0, "No model listed in {:s}".format(cp_dir)
        if isinstance(load_model, bool):
            cp_file = cp_list[-1]
        else:
            cp_file = load_model
        start_epoch = int("".join(filter(str.isdigit, cp_file)))
        cp_path = os.path.join(cp_dir, cp_file)
        model = torch.load(cp_path)   ## load entire model
        print("Model loaded from {}".format(cp_path))
    else:
        start_epoch = 0
        model = None
    return start_epoch, model