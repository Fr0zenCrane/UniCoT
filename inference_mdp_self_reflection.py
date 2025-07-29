import hashlib
import torch
import numpy as np
import random
import os
import json
import copy
import argparse
from tqdm import tqdm, trange
from PIL import Image
from safetensors.torch import load_file
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


from data.transforms import ImageTransform
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 378, 14)


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


SYSTEM_PROMPT = "You are an intelligent assistant designed to answer questions accurately and improve through self-evaluation; for each question, follow this three-step process: (1) Initial Answer Generation: generate a complete, thoughtful answer using your current knowledge, and if generating an image, first think through the planning process enclosed in <think>...</think> tags before outputting the image; (2) Self-Reflection: critically evaluate your response for factual correctness, logical soundness, and alignment with the question's intent, and enclose this reflection in <sr>...</sr> tags; (3) Correction and Regeneration: if issues are found, write a brief modification suggestion enclosed in <sugg>...</sugg> and regenerate an improved answer accordingly."


@torch.inference_mode()
def generate_image_with_think(
    prompt, num_timesteps=50, cfg_scale=4.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=4.0, resolution=1024,
    max_length=2048, simple_think=False, device=None
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
        
    ##########  cfg
    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    ##########  cfg    
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
        
    ########## think
    tmp_past_key_values = copy.deepcopy(past_key_values)
    tmp_newlens = copy.deepcopy(newlens)
    tmp_new_rope = copy.deepcopy(new_rope)
    tmp_generation_input, tmp_newlens, tmp_new_rope = model.prepare_prompts(
        curr_kvlens=tmp_newlens,
        curr_rope=tmp_new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        tmp_past_key_values = model.forward_cache_update_text(tmp_past_key_values, **tmp_generation_input)      
    tmp_generation_input = model.prepare_start_tokens(tmp_newlens, tmp_new_rope, new_token_ids)
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.generate_text(
            past_key_values=tmp_past_key_values,
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
    print("="*30, "original think", "="*30)
    print(think_output) 
    if simple_think:
        think_output_list = think_output.split("</think>")
        if think_output_list[1] != "":
            think_output = think_output_list[1].strip()
        print("="*30, "processed think", "="*30)
        print(think_output) 

    ########## think    
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

    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    ########## generate image
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_scale, 
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type="global",
            cfg_text_past_key_values=None,
            cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
            cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
            cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
            cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
            **generation_input,
        )
    
    latent0 = unpacked_latent[0]
    latent0 = latent0.reshape(1, h//16, w//16, 2, 2, 16)
    latent0 = torch.einsum("nhwpqc->nchpwq", latent0)
    latent0 = latent0.reshape(1, 16, h//8, w//8)
    image = vae_model.decode(latent0.to("cuda"))
    tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmpimage = Image.fromarray(tmpimage)
    
    return tmpimage, think_output

@torch.inference_mode()
def refine_image_with_cot(
    image, prompt, think_text, num_timesteps=50, 
    cfg_text_scale=4.0, cfg_img_scale=2.0,
    cfg_interval=[0, 1.0], cfg_renorm_min=0., 
    cfg_type="serial_text_img", cfg_renorm_type="text_channel", 
    timestep_shift=3.0, max_image_size=1024, min_image_size=512, img_size=None,
    max_length=2048, simple_think=False, device=None
):  
    # whether the image is edited
    modify_flag = False
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
            cfg_img_past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    # evaluate generated image and generate refined prompt
    # generate evaluation
    for _ in range(3):
        tmp_past_key_values = copy.deepcopy(past_key_values)
        tmp_newlens = copy.deepcopy(newlens)
        tmp_new_rope = copy.deepcopy(new_rope)

        # add question image (vit)
        tmp_generation_input, tmp_newlens, tmp_new_rope = gen_model.prepare_vit_images(
            curr_kvlens=tmp_newlens,
            curr_rope=tmp_new_rope, 
            images=[image],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            tmp_past_key_values = gen_model.forward_cache_update_vit(tmp_past_key_values, **tmp_generation_input)

        tmp_generation_input = gen_model.prepare_start_tokens(tmp_newlens, tmp_new_rope, new_token_ids)
        tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
        tmp_past_key_values_eval = copy.deepcopy(tmp_past_key_values)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_text(
                past_key_values=tmp_past_key_values_eval,
                max_length=max_length,
                do_sample=True,
                temperature=0.3,
                end_token_id=new_token_ids['eos_token_id'],
                **tmp_generation_input,
                )
            output = tokenizer.decode(unpacked_latent[:,0])
            eval_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        del tmp_past_key_values_eval

        print("="*30, "Evaluation", "="*30)
        print(eval_output)

        tmp_generation_input, tmp_newlens, tmp_new_rope = gen_model.prepare_prompts(
            curr_kvlens=tmp_newlens,
            curr_rope=tmp_new_rope, 
            prompts=[eval_output],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            tmp_past_key_values = gen_model.forward_cache_update_text(tmp_past_key_values, **tmp_generation_input)  

        tmp_generation_input = gen_model.prepare_start_tokens(tmp_newlens, tmp_new_rope, new_token_ids)
        tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
        tmp_past_key_values_refin = copy.deepcopy(tmp_past_key_values)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_text(
                past_key_values=tmp_past_key_values_refin,
                max_length=max_length,
                do_sample=True,
                temperature=0.3,
                end_token_id=new_token_ids['eos_token_id'],
                **tmp_generation_input,
                )
            output = tokenizer.decode(unpacked_latent[:,0])
            refine_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1] 
        del tmp_past_key_values_refin
            
        print("="*30, "Refinment", "="*30)
        print(refine_output) 


        if ("Everything is good" not in refine_output) and ("Everything is well" not in refine_output) and ("No editing" not in refine_output) and ("no editing" not in refine_output):
            modify_flag = True
            break
    
    if not modify_flag:
        print("No modification needed.")
        return image, refine_output, False

    # rewrite the refine insturction
    enhance_prompt = "Rewrite the text within the <sugg>...</sugg> tags to make it more specific and detailed as an editing instruction. Ensure that only the revised instruction is enclosed within <sugg>...</sugg> tags."
    tmp_generation_input, tmp_newlens, tmp_new_rope = gen_model.prepare_prompts(
        curr_kvlens=tmp_newlens,
        curr_rope=tmp_new_rope, 
        prompts=[enhance_prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        tmp_past_key_values = gen_model.forward_cache_update_text(tmp_past_key_values, **tmp_generation_input)  

    tmp_generation_input = gen_model.prepare_start_tokens(tmp_newlens, tmp_new_rope, new_token_ids)
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_text(
            past_key_values=tmp_past_key_values,
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        enhance_refine_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1] 
    del tmp_past_key_values

    print("="*30, "Enhanced Refinment", "="*30)
    print(enhance_refine_output) 

    # updata original cache
    # add org image
    # add VAE
    generation_input, newlens, new_rope = gen_model.prepare_vae_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=[image],
        transforms=vae_transform, 
        new_token_ids=new_token_ids,
        #timestep=0.0,
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

    ##########  cfg_text
    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)], 
    )
    generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
    
    ##########  cfg_img
    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    
    # prepare text list
    text_list = [SYSTEM_PROMPT, prompt, think_text, enhance_refine_output]
    for text_element in text_list:
        generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope, 
            prompts=[text_element],
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

    ##########  origin
    # add cot_output
    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[enhance_refine_output],
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
    
    return tmpimage, refine_output+"\n<enhance>\n"+enhance_refine_output, modify_flag


def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""

    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = PIL.Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using CausalFusion model.")
    parser.add_argument("--cfg_text_scale", type=float, default=3)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model_path', type=str, default='Fr0zenCrane/UniCoT-7B-MoT')
    parser.add_argument('--group_num', type=int, default=1,
                        help='Total number of groups (must be a positive integer)')
    parser.add_argument('--group_id', type=int, default=0,
                        help='ID of the current group (0-based index)')
    parser.add_argument('--data_path', type=str, default="./eval/gen/wise/final_data.json",
                        help='wise prompt path')
    parser.add_argument('--outdir', type=str, default="./results",
                        help='output image results')
    args = parser.parse_args()

    metadata = [line.strip() for line in open(args.data_path, mode='r')]

    splited_metadata = metadata[args.group_id::args.group_num] # obtain one sample every `group_num` samples and start from idx `group_id``
    selected_idx = list(range(args.group_id, len(metadata), args.group_num))
    assert len(selected_idx) == len(splited_metadata)

    outdir = args.outdir if args.outdir else "./results"
    os.makedirs(outdir, exist_ok=True)

    seed = 42
    if seed is not None:
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

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)

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
        checkpoint=os.path.join(args.model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload")

    vae_model = vae_model.cuda().eval()
    gen_model = model = model.eval()

    # hyper-parameter config used for self-reflection
    cfg_text_scale = args.cfg_text_scale
    cfg_img_scale = args.cfg_img_scale
    cfg_interval = [0., 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    #hyper-parameter config only used for the r0 generation
    gen_inference_hyper=dict(
        cfg_scale=4.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
    )

    for idx, dataItem in tqdm(zip(selected_idx, splited_metadata), total=len(selected_idx)):
        prompt = dataItem
        prompt_id = str(idx)
        save_name = "_".join(" ".split(prompt))
        save_name = save_name[:480]
        save_name = f"{prompt_id}_{save_name}"

        if os.path.exists(os.path.join(outdir, f"{prompt_id}.png")):
            continue
        image, think_text = generate_image_with_think(
                        prompt=prompt,
                        resolution=1024,
                        device=None,
                        **gen_inference_hyper)
        image = pil_img2rgb(image)
        think_text = think_text.strip()

        i = 1
        modify_format = "<modify round {}>: {} \n"
        modify_text = ""

        sample = refine_image_with_cot(
                            image=image,
                            prompt=prompt,
                            think_text=think_text,
                            cfg_text_scale=cfg_text_scale, 
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval, 
                            cfg_renorm_min=cfg_renorm_min,
                            timestep_shift=timestep_shift, 
                            num_timesteps=num_timesteps,
                            max_length=8192, 
                            device=device,
                        )
        if sample[2]:
            # lanunch a new round of generation untill every thing is good
            while sample[2] and i <= 20:
                modify_text += modify_format.format(i, sample[1])
                i += 1
                sample = refine_image_with_cot(
                                    image=sample[0],
                                    prompt=prompt,
                                    think_text=think_text,
                                    cfg_text_scale=cfg_text_scale, 
                                    cfg_img_scale=cfg_img_scale,
                                    cfg_interval=cfg_interval, 
                                    cfg_renorm_min=cfg_renorm_min,
                                    timestep_shift=timestep_shift, 
                                    num_timesteps=num_timesteps,
                                    max_length=8192, 
                                    device=device,
                                )
            sample[0].save(os.path.join(outdir, f"{save_name}.png"))

            with open(os.path.join(outdir, f"{save_name}_refine.txt"), "w") as f:
                f.write(modify_text)
        else:
            image.save(os.path.join(outdir, f"{save_name}.png"))
        image.save(os.path.join(outdir, f"{save_name}_ori.png"))
        with open(os.path.join(outdir, f"{save_name}_think.txt"), "w") as f:
            f.write(think_text)

    print("Done.")
