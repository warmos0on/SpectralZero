"""Phase 2 快速校验：新旧文本属性在 CLIP 文本空间中的类原型可分性。

不训练模型，只对比两个属性词表生成的类文本 embedding 两两余弦相似度。
相似度越低（越接近 0）说明类原型越可分，训练/测试对齐越有益。
"""
import sys
import json
import numpy as np
import torch

from extract_text import clip
from extract_text import text_utils as tutils


ALL_CLASSES = [
    "Corn-notill", "Soybean-mintill", "Woods", "Grass-trees", "Corn-mintill",
    "Grass-pasture", "Buildings-Grass-Trees-Drives", "Hay-windrowed",
    "Soybean-notill", "Wheat", "Alfalfa", "Corn", "Grass-pasture-mowed",
    "Oats", "Soybean-clean", "Stone-Steel-Towers",
]

GROUPS = {
    1: ["Corn-notill", "Hay-windrowed", "Woods"],
    2: ["Grass-trees", "Buildings-Grass-Trees-Drives", "Soybean-mintill"],
    3: ["Grass-pasture-mowed", "Stone-Steel-Towers", "Corn-mintill"],
    4: ["Corn", "Alfalfa", "Grass-pasture"],
    5: ["Wheat", "Oats", "Soybean-notill"],
    6: ["Soybean-clean"],
}


def build_sentences(att_file, unseen_classes):
    with open(att_file, "r", encoding="utf-8") as f:
        att = json.load(f)
    seen = {c: i for i, c in enumerate(ALL_CLASSES) if c not in unseen_classes}
    unseen = {c: i for i, c in enumerate(unseen_classes)}
    return tutils.get_att_des(att, only_name=False, seen_label=seen, unseen_label=unseen)


def main():
    att_files = {
        "v1_natural": "./attributes/Indian_class_attributes_v1_natural.json",
        "physical": "./attributes/Indian_class_attributes_physical.json",
    }
    enc, _ = tutils.load_text_encoder("./extract_text/ViT-L-14.pt")
    enc.eval()

    out = []
    out.append("=== CLIP 文本原型可分性校验（token 数 + 组内两两余弦相似度）===")
    reprs = {}
    for name, path in att_files.items():
        reprs[name] = {}
        for gi, unseen in GROUPS.items():
            sentences = build_sentences(path, unseen)
            tokens = torch.cat([clip.tokenize(s).cuda() for s in sentences])
            max_tok = int((tokens != 0).sum(dim=1).max())
            longest = sorted(
                zip(sentences, (tokens != 0).sum(dim=1).tolist()),
                key=lambda x: -x[1],
            )[:3]
            for s, n in longest:
                print(f"  [len={n}] {s}")
            with torch.no_grad():
                emb = enc(tokens)
            emb = emb / emb.norm(dim=1, keepdim=True)
            # 训练侧：全部 16 类原型；测试侧：该组 unseen 原型（排在末尾）
            n_seen = len(ALL_CLASSES) - len(unseen)
            all_sim = emb @ emb.t()
            unseen_emb = emb[n_seen:]
            u_sim = unseen_emb @ unseen_emb.t()
            triu_all = all_sim[np.triu_indices(all_sim.shape[0], k=1)]
            triu_u = u_sim[np.triu_indices(u_sim.shape[0], k=1)]
            reprs[name][gi] = {
                "max_tok": max_tok,
                "all_mean": float(triu_all.mean()),
                "unseen_mean": float(triu_u.mean()) if triu_u.numel() else float("nan"),
                "unseen_min": float(triu_u.min()) if triu_u.numel() else float("nan"),
            }
            u_mean = float(triu_u.mean()) if triu_u.numel() else float("nan")
            u_min = float(triu_u.min()) if triu_u.numel() else float("nan")
            out.append(
                f"[{name}] G{gi} unseen={unseen} max_tokens={max_tok} "
                f"pairs={len(triu_u)} unseen_sim_mean={u_mean:.4f} "
                f"min={u_min:.4f} all16_sim_mean={triu_all.mean():.4f}"
            )

    out.append("\n=== 对比（越低越可分） ===")
    for gi in sorted(GROUPS):
        nat = reprs["v1_natural"][gi]
        phy = reprs["physical"][gi]
        d_unseen = phy["unseen_mean"] - nat["unseen_mean"]
        d_all = phy["all_mean"] - nat["all_mean"]
        out.append(
            f"G{gi}: unseen_sim natural={nat['unseen_mean']:.4f} -> physical={phy['unseen_mean']:.4f} "
            f"(delta={d_unseen:+.4f}) | all16 sim delta={d_all:+.4f}"
        )

    txt = "\n".join(out)
    print(txt)
    with open("./result_IP/文本原型可分性校验.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
