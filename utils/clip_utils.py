import torch


def get_clip_text_dict(old_state_dict, transformer_width=512, embed_dim=768):
    new_dict = {}
    # Iterate through the parameter dict
    for old_key, value in old_state_dict.items():
        if old_key.startswith('positional_embedding'):
            new_dict["positional_embedding"] = value

        elif 'text_projection' in old_key:
            new_dict["text_projection"] = value

        elif 'token_embedding.weight' in old_key:
            new_dict["token_embedding.weight"] = value

        elif old_key == 'ln_final.weight':
            new_dict["ln_final.weight"] = value

        elif old_key == 'ln_final.bias':
            new_dict["ln_final.bias"] = value

        elif old_key.startswith('transformer'):
            new_dict[old_key] = value

    return new_dict
