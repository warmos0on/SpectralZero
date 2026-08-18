try:
    import extract_text.clip as clip
    import extract_text.text_utils as utils
except:
    import clip
    import text_utils as utils
import torch
import numpy as np


def extract(label_value, args):
    """
    获取文本embedding
    :param label_value: seen和unseen label-classes对
    :param args: 参数
    :return: unseen seen all的label-classes对, text_projection, 新的args
    """
    seen_label_name, unseen_label_name, _ = label_value["seen"], label_value["unseen"], label_value["all"]
    att_name = args["attribute_dir"] + r"{}_class_attributes.json".format(args["dataset"])
    # 获取每一类的描述对
    all_att = utils.get_attrbute(att_name, label_value["all"])
    train_att = utils.get_att_des(all_att, seen_label=seen_label_name, unseen_label=unseen_label_name, only_name=args["only_name"])
    train_result = torch.cat([clip.tokenize(per_att).cuda() for per_att in train_att])

    text_embedding, embed_dim = utils.get_all_nlp(args["pretrain_loc"], train_result)
    args.update({"embed_dim": embed_dim}) 
    return text_embedding["0"], args, train_att

