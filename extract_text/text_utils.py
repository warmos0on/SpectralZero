try:
    import extract_text.text_encoder as encoder
except:
    import text_encoder as encoder
import torch
import json
import numpy as np


def load_text_encoder(pt_loc):
    pretrained_dict = torch.jit.load(pt_loc, map_location="cuda").state_dict()
    embed_dim = pretrained_dict["text_projection"].shape[1]
    context_length = pretrained_dict["positional_embedding"].shape[0]
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 12

    text_encoder = encoder.Text_encoder(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers
    )

    model_state = text_encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state and 'visual' not in k.split('.')}

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]

    model_state.update(pretrained_dict)
    text_encoder.load_state_dict(model_state)
    return text_encoder.cuda(), embed_dim


def get_all_nlp(pt_loc, *args):
    text_encoder, embedding = load_text_encoder(pt_loc)
    text_encoder.eval()
    container = {}

    with torch.no_grad():
        for index, data in enumerate(args):
            text_projection = text_encoder(data)
            text_projection = text_projection.detach()
            container.update({str(index): text_projection})
    return container, embedding


def get_attrbute(att_loc, label_name):
    # 获取数据集的类别描述
    att_final = {}
    with open(att_loc, "r") as f:
        att = json.load(f)
        for labels in label_name:
            try:
                current_att = att[labels]
                att_final[labels] = current_att
            except:
                continue
    f.close()
    return att_final


def get_seen_unseen_class(label_value, keys=False):
    # key为ture时返回seen unseen label-classes对
    seen, unseen = label_value["seen"], label_value["unseen"]
    # key为false时返回seen unseen classes列表
    seen_label = np.array(list(seen.values()))
    unseen_label = np.array(list(unseen.values()))
    return seen_label, unseen_label


def get_att_des(att, only_name=False, seen_label=None, unseen_label=None, test=False):
    final_att = []
    for key, value in seen_label.items():
        current_att = att.get(key)
        attr_string = ", ".join(f"'{k}':'{v}'" for k, v in current_att.items())
        full_sentence = f"A hyperspectral image of '{key}, {attr_string}." if not only_name else f"{key}"
        final_att.append(full_sentence)
    for key, value in unseen_label.items():
        current_att = att.get(key)
        attr_string = ", ".join(f"'{k}':'{v}'" for k, v in current_att.items())
        full_sentence = f"A hyperspectral image of '{key}, {attr_string}." if not only_name else f"{key}"
        final_att.append(full_sentence)
    return final_att


