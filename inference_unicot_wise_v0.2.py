import torch
import numpy as np
from tqdm import tqdm
import os
import json
import copy
import argparse
from PIL import Image
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache

from data.data_utils import add_special_tokens, pil_img2rgb

import torch.distributed as dist


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# Due to limited training resource, we use more light-weight vit transformation settings
from data.transforms import ImageTransform
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(518, 224, 14)
vit_transform_und = ImageTransform(980, 378, 14)


def apply_scale(width, height, scale):

    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))
    
    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = _make_divisible(new_width, 16)
    new_height = _make_divisible(new_height, 16)

    return new_width, new_height


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    
    return generation_input


SYSTEM_PROMPT = "You are a talented and imaginative digital artist, skilled in turning user prompts into stunning visual artworks. Your role is to understand the user's intentions and bring their ideas on your canvas. During this process, you will use the following abilities to improve your crafts:  \n\n- **Prompt Translation**: If user prompts are ambiguous or underspecified, rewrite them into clear, detailed, and visually rich descriptions and enclose them with `<think></think>`.  \n\n- **Subtask Breakdown**: If user prompts are complex, break the prompts into simpler subtasks, enclose each subtask with `<subtask></subtask>` and accomplish them on your canvas iteratively.  \n\n- **Self Reflection**: As your canvas changes, critically evaluate its present appearance for factual correctness, logical soundness, alignment with user intention, and wrap your analysis in <eval>...</eval>. Then provide specific, actionable editing instructions in <sugg>...</sugg> and apply them in your next step."


@torch.inference_mode()
def generate_think_and_breakdown(
    prompt, resolution=1024, max_length=2048, device=None
):
    h, w = resolution, resolution

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]
    
    # system prompt
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[SYSTEM_PROMPT],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)  
        
    ### load use prompt 
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    ### generate think
    generation_input = model.prepare_start_tokens(newlens, new_rope, new_token_ids)
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.generate_text(
            past_key_values=copy.deepcopy(past_key_values),
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
    print("="*30, "original think", "="*30)
    print(think_output) 

    ### load think 
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[think_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)
    
    ### generate breakdown
    generation_input = model.prepare_start_tokens(newlens, new_rope, new_token_ids)
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.generate_text(
            past_key_values=copy.deepcopy(past_key_values),
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        breakdown_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
    print("="*30, "original breakdown", "="*30)
    print(breakdown_output) 
    
    return think_output, breakdown_output


def generate_evaluation_and_editing(
    image, prompt, think_text, resolution=1024, max_length=2048, device=None, max_image_size=1024, min_image_size=512
):
# whether the image is edited
    modify_flag = True
    # set output size
    w, h = image.size
    scale = min(max_image_size / max(w, h), 1.0)
    scale = max(scale, min_image_size / min(w, h))
    w, h = apply_scale(w, h, scale)

    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]
    
    # add question text
    text_list = [SYSTEM_PROMPT, prompt, think_text]
    for text_element in text_list:
        generation_input, newlens, new_rope = gen_model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[text_element],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    # evaluate generated image and generate refined prompt
    generation_input, newlens, new_rope = gen_model.prepare_vit_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=[image],
        transforms=vit_transform_und, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)

    generation_input = gen_model.prepare_start_tokens(newlens, new_rope, new_token_ids)
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_text(
            past_key_values=copy.deepcopy(past_key_values),
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        eval_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[eval_output],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  

    generation_input = gen_model.prepare_start_tokens(newlens, new_rope, new_token_ids)
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_text(
            past_key_values=copy.deepcopy(past_key_values),
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        refine_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1] 
    print("="*30, "Refinment", "="*30)
    print(refine_output) 

    if ("Everything is good" in refine_output) or ("Everything is well" in refine_output) or ("No editing" in refine_output) or ("no editing" in refine_output):
        modify_flag = False
    return eval_output, refine_output, modify_flag


@torch.inference_mode()
def generate_image(prompt, num_timesteps=50, cfg_scale=4.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, resolution=1024, device=None):
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(resolution, resolution)], 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0]
    cfg_new_rope = [0]

    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope, 
        image_sizes=[(resolution, resolution)],
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, resolution//8, resolution//8)
    image = vae_model.decode(latent.to(device))
    tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)

    return tmpimage


def editing_image(
    image, prompt, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None, device=None,
):
    # set output size
    if img_size is None:
        w, h = image.size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = apply_scale(w, h, scale)
    else:
        h, w = img_size
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = apply_scale(w, h, scale)
    print(f"Image size: H-{h} W-{w}")


    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens, new_rope = [0], [0]

    # add VAE
    generation_input, newlens, new_rope = gen_model.prepare_vae_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=[image],
        transforms=vae_transform, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)

    # add ViT
    generation_input, newlens, new_rope = gen_model.prepare_vit_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=[image],
        transforms=vit_transform, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)

    # cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
    
    # cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
    
    # origin
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)  
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h//16, w//16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h//8, w//8)
    tmpimage = vae_model.decode(latent)
    tmpimage = ((tmpimage * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)

    return tmpimage


def split_prompts(prompts, split_key="subtask"):
    split_begin = f"<{split_key}>"
    split_end = f"</{split_key}>"

    breakdown_prompt_list = prompts.split(split_begin)
    breakdown_prompt_list = [item.split(split_end)[0].strip() for item in breakdown_prompt_list if split_end in item]
    return breakdown_prompt_list


def preprocess_prompt(prompt, split_key="subtask"):
    split_begin = f"<{split_key}>"
    split_end = f"</{split_key}>"

    prompt = prompt.split(split_begin)[-1]
    prompt = prompt.split(split_end)[0].strip()
    prompt = prompt.replace("\n", " ").replace("\r", " ").strip()
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using CausalFusion model.")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument('--model_path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--model_fine_tuned_path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--group_num', type=int, default=1,
                        help='Total number of groups (must be a positive integer)')
    parser.add_argument('--group_id', type=int, default=0,
                        help='ID of the current group (0-based index)')
    parser.add_argument('--data_path', type=str, default="./sampled",
                        help='dataset path')
    parser.add_argument('--output_dir', type=str, default="output",
                        help='results path')
    parser.add_argument('--mode', type=str, default="pure_breakdown",
                        help='pure_breakdown or pure_self_relfection')

    args = parser.parse_args()

    metadata = json.load(open(args.data_path, "r"))
    splited_metadata = metadata[args.group_id::args.group_num] # obtain one sample every `group_num` samples and start from idx `group_id``
    
    outdir = args.output_dir

    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = f"cuda:{0}"
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())}, # CFFF A100 has 80GB VRAM
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print(device_map)
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    if torch.cuda.device_count() == 1: # should not be used for this setting, will draw OOM error
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Load ckpt
    # Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(args.model_fine_tuned_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload")

    vae_model = vae_model.cuda().eval()
    gen_model = model = model.eval()

    #hyper-parameter config only used for the r0 generationS
    gen_inference_hyper=dict(
        cfg_scale=args.cfg_scale,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
    )

    edit_inference_hyper=dict(
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0., 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
    )
    outpath = outdir
    os.makedirs(outpath, exist_ok=True)
    sample_path = outpath

    if args.mode == "pure_breakdown":
        for dataItem in tqdm(splited_metadata):
            prompt = dataItem["Prompt"]
            prompt_id = dataItem["prompt_id"]

            if os.path.exists(os.path.join(sample_path, f"{prompt_id}.png")):
                continue

            # generate think prompt and breakdown prompt
            think_prompt, breakdown_prompt = generate_think_and_breakdown(
                            prompt=prompt,
                            resolution=args.resolution,
                            device=None)
            think_prompt = think_prompt.strip()
            breakdown_prompt = breakdown_prompt.strip()
            with open(os.path.join(sample_path, f"{prompt_id}_think.txt"), "w") as f:
                f.write(think_prompt)
            with open(os.path.join(sample_path, f"{prompt_id}_breakdown.txt"), "w") as f:
                f.write(breakdown_prompt)

            # preprocess breakdown prompt
            breakdown_prompt_list = split_prompts(breakdown_prompt, split_key="subtask")
            print("Total {} sub-tasks.".format(len(breakdown_prompt_list)))

            # initialize image
            i = 0
            tmpimage = generate_image(
                prompt=breakdown_prompt_list[0],
                resolution=args.resolution,
                device=device,
                **gen_inference_hyper)
            
            tmpimage.save(os.path.join(sample_path, f"{prompt_id}_tmp_{i}.png"))
            tmpimage = pil_img2rgb(tmpimage)

            for breakdown_prompt in breakdown_prompt_list[1:]:
                tmpimage = pil_img2rgb(tmpimage)
                tmpimage = editing_image(
                    image=tmpimage,
                    prompt=breakdown_prompt,
                    device=device,
                    **edit_inference_hyper)
                i += 1
                tmpimage.save(os.path.join(sample_path, f"{prompt_id}_tmp_{i}.png"))
            tmpimage.save(os.path.join(sample_path, f"{prompt_id}.png"))
        print("Done.")
    elif args.mode == "pure_self_relfection":
        for dataItem in tqdm(splited_metadata):
            prompt = dataItem["Prompt"]
            prompt_id = dataItem["prompt_id"]

            if os.path.exists(os.path.join(sample_path, f"{prompt_id}.png")):
                continue

            # generate think prompt and breakdown prompt
            think_prompt, _ = generate_think_and_breakdown(
                            prompt=prompt,
                            resolution=args.resolution,
                            device=None)
            think_prompt = think_prompt.strip()

            with open(os.path.join(sample_path, f"{prompt_id}_think.txt"), "w") as f:
                f.write(think_prompt)

            # preprocess breakdown prompt
            think_user_prompt = preprocess_prompt(think_prompt, split_key="think")
            print("User rephrased prompt: ", think_user_prompt) 

            # initialize image
            i = 0
            tmpimage = generate_image(
                prompt=think_user_prompt,
                resolution=args.resolution,
                device=device,
                **gen_inference_hyper)
            
            tmpimage.save(os.path.join(sample_path, f"{prompt_id}_tmp_{i}.png"))
            tmpimage = pil_img2rgb(tmpimage)

            # start self reflection
            modify_format = "<modify round {}>: {} \n"
            modify_text = ""
            modify_flag = True
            # lanunch a new round of generation untill every thing is good
            while i < 20:
                ### evaluate current image and generate refined prompt
                eval_text, refine_text, modify_flag = generate_evaluation_and_editing(
                    image=tmpimage,
                    prompt=prompt,
                    think_text=think_prompt,
                    resolution=args.resolution,
                    device=device,
                )

                modify_text += modify_format.format(i, eval_text + refine_text)
                
                ##### start editing
                if not modify_flag:
                    print("Everything is good now.")
                    break
                
                i += 1
                editing_prompt = preprocess_prompt(refine_text, split_key="sugg")
                print(f"Round {i} editing prompt: ", editing_prompt)

                tmpimage = pil_img2rgb(tmpimage)
                tmpimage = editing_image(
                    image=tmpimage,
                    prompt=editing_prompt,
                    device=device,
                    **edit_inference_hyper)
                tmpimage.save(os.path.join(sample_path, f"{prompt_id}_tmp_{i}.png"))

            tmpimage.save(os.path.join(sample_path, f"{prompt_id}.png"))
            with open(os.path.join(sample_path, f"{prompt_id}_refine.txt"), "w") as f:
                f.write(modify_text)

        print("Done.")
    else:
        print("Unsupported mode: ", args.mode)
