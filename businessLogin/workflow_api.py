import math
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


# how to set global variables for batch size
GbatchSize = 7

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    KSamplerAdvanced,
    LoraLoader,
    EmptyLatentImage,
    VAEDecode,
    NODE_CLASS_MAPPINGS,
    CLIPSetLastLayer,
    ControlNetApplyAdvanced,
    LatentUpscaleBy,
    VAELoader,
)


def Cooking(batchsize: int, startindex: int):
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimplewithnoiseselect = NODE_CLASS_MAPPINGS[
            "CheckpointLoaderSimpleWithNoiseSelect"
        ]()
        checkpointloadersimplewithnoiseselect_1 = (
            checkpointloadersimplewithnoiseselect.load_checkpoint(
                ckpt_name="aamKizukihassakuSfwNsfw_v10.safetensors",
                beta_schedule="sqrt_linear (AnimateDiff)",
                use_custom_scale_factor=False,
                scale_factor=0.18000000000000002,
            )
        )

        vaeloader = VAELoader()
        vaeloader_2 = vaeloader.load_vae(vae_name="cleanvae_v10.safetensors")

        controlnetloaderadvanced = NODE_CLASS_MAPPINGS["ControlNetLoaderAdvanced"]()
        controlnetloaderadvanced_70 = controlnetloaderadvanced.load_controlnet(
            control_net_name="control_v11p_sd15_lineart.pth"
        )

        ade_animatediffuniformcontextoptions = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffUniformContextOptions"
        ]()
        ade_animatediffuniformcontextoptions_94 = (
            ade_animatediffuniformcontextoptions.create_options(
                context_length=16, context_stride=1, context_overlap=4,closed_loop="true",
                context_schedule="uniform", fuse_method="flat"
            )
        )

        controlnetloaderadvanced_97 = controlnetloaderadvanced.load_controlnet(
            control_net_name="control_v11p_sd15_softedge.pth"
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_151 = emptylatentimage.generate(
            width=912, height=512, batch_size=batchsize
        )

        loadimagesfromdirectory = NODE_CLASS_MAPPINGS["LoadImagesFromDirectory"]()
        loadimagesfromdirectory_172 = loadimagesfromdirectory.load_images(
            directory="/home/yori/Downloads/outputLineArt",
            image_load_cap=batchsize,
            start_index=startindex,
        )

        loadimagesfromdirectory_174 = loadimagesfromdirectory.load_images(
            directory="/home/yori/Downloads/outputSoftedge",
            image_load_cap=batchsize,
            start_index=startindex,
        )

        text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
        text_multiline_229 = text_multiline.text_multiline(
            text="ugly, deformed, bad lighting, blurry, text, clouds, watermark, extra hands, bad quality, deformed hands, deformed fingers, nostalgic, drawing, painting, bad anatomy, worst quality, blurry, blurred, normal quality, bad focus, tripod, three legs, weird legs, short legs, bag, handbag, 3 hands, 4 hands, three hands\n\n(embedding:BadDream:1) boy, man, male,\n(embedding:ng_deepnegative_v1_75t:1), \n(embedding:epiCNegative:1), \n(embedding:bad-picture-chill-75v:1), \n(embedding:AS-YoungV2-neg:1), \n(embedding:ERA09NEGV2:1) "
        )

        floatconstant = NODE_CLASS_MAPPINGS["FloatConstant"]()
        floatconstant_391 = floatconstant.get_value(value=0.65)

        cr_lora_stack = NODE_CLASS_MAPPINGS["CR LoRA Stack"]()
        cr_lora_stack_393 = cr_lora_stack.lora_stacker(
            switch_1="On",
            lora_name_1="None",
            model_weight_1=1.5,
            clip_weight_1=1,
            switch_2="On",
            lora_name_2="None",
            model_weight_2=1,
            clip_weight_2=1,
            switch_3="On",
            lora_name_3="None",
            model_weight_3=1,
            clip_weight_3=1,
        )

        cr_lora_stack_346 = cr_lora_stack.lora_stacker(
            switch_1="On",
            lora_name_1="None",
            model_weight_1=1,
            clip_weight_1=1,
            switch_2="On",
            lora_name_2="None",
            model_weight_2=0.75,
            clip_weight_2=1,
            switch_3="On",
            lora_name_3="None",
            model_weight_3=1,
            clip_weight_3=1,
            lora_stack=get_value_at_index(cr_lora_stack_393, 0),
        )

        cr_apply_lora_stack = NODE_CLASS_MAPPINGS["CR Apply LoRA Stack"]()
        cr_apply_lora_stack_347 = cr_apply_lora_stack.apply_lora_stack(
            model=get_value_at_index(checkpointloadersimplewithnoiseselect_1, 0),
            clip=get_value_at_index(checkpointloadersimplewithnoiseselect_1, 1),
            lora_stack=get_value_at_index(cr_lora_stack_346, 0),
        )

        int_literal = NODE_CLASS_MAPPINGS["Int Literal"]()
        int_literal_436 = int_literal.get_int(int=3)

        ade_animatediffmodelsettingssimple = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffModelSettingsSimple"
        ]()
        ade_animatediffmodelsettingssimple_415 = (
            ade_animatediffmodelsettingssimple.get_motion_model_settings(
                motion_pe_stretch=get_value_at_index(int_literal_436, 0),
                min_motion_scale=100,
                max_motion_scale=100,
            )
        )

        ade_animatediffloaderwithcontext = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffLoaderWithContext"
        ]()
        ade_animatediffloaderwithcontext_93 = (
            ade_animatediffloaderwithcontext.load_mm_and_inject_params(
                model_name="temporaldiff-v1-animatediff.ckpt",
                beta_schedule="sqrt_linear (AnimateDiff)",
                motion_scale=1,
                apply_v2_models_properly=False,
                model=get_value_at_index(cr_apply_lora_stack_347, 0),
                context_options=get_value_at_index(
                    ade_animatediffuniformcontextoptions_94, 0
                ),
                motion_model_settings=get_value_at_index(
                    ade_animatediffmodelsettingssimple_415, 0
                ),
            )
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_226 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-24, clip=get_value_at_index(cr_apply_lora_stack_347, 1)
        )

        loraloader = LoraLoader()
        loraloader_373 = loraloader.load_lora(
            lora_name="pytorch_lora_weights.safetensors",
            strength_model=get_value_at_index(floatconstant_391, 0),
            strength_clip=0.1,
            model=get_value_at_index(ade_animatediffloaderwithcontext_93, 0),
            clip=get_value_at_index(clipsetlastlayer_226, 0),
        )

        smz_cliptextencode = NODE_CLASS_MAPPINGS["smZ CLIPTextEncode"]()
        smz_cliptextencode_230 = smz_cliptextencode.encode(
            text=get_value_at_index(text_multiline_229, 0),
            parser="comfy",
            mean_normalization=True,
            multi_conditioning=False,
            use_old_emphasis_implementation=False,
            with_SDXL=False,
            ascore=6,
            width=1024,
            height=1024,
            crop_w=0,
            crop_h=0,
            target_width=1024,
            target_height=1024,
            text_g="",
            text_l="",
            smZ_steps=1,
            clip=get_value_at_index(loraloader_373, 1),
        )

        text_multiline_345 = text_multiline.text_multiline(
            text="(masterpiece, top quality, best quality, official art, beautiful and aesthetic), Extreme detailed, Sexy ,1girl, brown hair, black dress, dancing"
        )

        smz_cliptextencode_344 = smz_cliptextencode.encode(
            text=get_value_at_index(text_multiline_345, 0),
            parser="comfy",
            mean_normalization=True,
            multi_conditioning=True,
            use_old_emphasis_implementation=False,
            with_SDXL=False,
            ascore=6,
            width=1024,
            height=1024,
            crop_w=0,
            crop_h=0,
            target_width=1024,
            target_height=1024,
            text_g="",
            text_l="",
            smZ_steps=1,
            clip=get_value_at_index(loraloader_373, 1),
        )

        text_string = NODE_CLASS_MAPPINGS["Text String"]()
        text_string_353 = text_string.text_string(
            text="Batch_", text_b="", text_c="", text_d=""
        )

        int_literal_362 = int_literal.get_int(int=1)

        cfg_literal = NODE_CLASS_MAPPINGS["Cfg Literal"]()
        cfg_literal_382 = cfg_literal.get_float(float=4)

        int_literal_383 = int_literal.get_int(int=10)

        string_to_text = NODE_CLASS_MAPPINGS["String to Text"]()
        string_to_text_398 = string_to_text.string_to_text(
            string="/home/yori/Downloads/output2"
        )

        cfg_literal_417 = cfg_literal.get_float(float=1.2000000000000002)

        int_literal_421 = int_literal.get_int(int=6)

        int_literal_422 = int_literal.get_int(int=5)

        cfg_literal_430 = cfg_literal.get_float(float=3)

        floatconstant_439 = floatconstant.get_value(value=0.8)

        floatconstant_440 = floatconstant.get_value(value=0)

        floatconstant_441 = floatconstant.get_value(value=0.8)

        floatconstant_443 = floatconstant.get_value(value=1)

        floatconstant_444 = floatconstant.get_value(value=0)

        floatconstant_445 = floatconstant.get_value(value=0.9500000000000001)

        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        tonemapnoisewithrescalecfg = NODE_CLASS_MAPPINGS["TonemapNoiseWithRescaleCFG"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        ksampleradvanced = KSamplerAdvanced()
        latentupscaleby = LatentUpscaleBy()
        bnk_injectnoise = NODE_CLASS_MAPPINGS["BNK_InjectNoise"]()
        vaedecode = VAEDecode()
        vhs_splitimages = NODE_CLASS_MAPPINGS["VHS_SplitImages"]()
        cr_integer_to_string = NODE_CLASS_MAPPINGS["CR Integer To String"]()
        text_concatenate = NODE_CLASS_MAPPINGS["Text Concatenate"]()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for q in range(1):
            modelsamplingdiscrete_369 = modelsamplingdiscrete.patch(
                sampling="lcm", zsnr=False, model=get_value_at_index(loraloader_373, 0)
            )

            tonemapnoisewithrescalecfg_427 = tonemapnoisewithrescalecfg.patch(
                tonemap_multiplier=get_value_at_index(cfg_literal_430, 0),
                rescale_multiplier=1,
                model=get_value_at_index(modelsamplingdiscrete_369, 0),
            )

            controlnetapplyadvanced_72 = controlnetapplyadvanced.apply_controlnet(
                strength=get_value_at_index(floatconstant_439, 0),
                start_percent=get_value_at_index(floatconstant_440, 0),
                end_percent=get_value_at_index(floatconstant_441, 0),
                positive=get_value_at_index(smz_cliptextencode_344, 0),
                negative=get_value_at_index(smz_cliptextencode_230, 0),
                control_net=get_value_at_index(controlnetloaderadvanced_70, 0),
                image=get_value_at_index(loadimagesfromdirectory_172, 0),
            )

            controlnetapplyadvanced_99 = controlnetapplyadvanced.apply_controlnet(
                strength=get_value_at_index(floatconstant_443, 0),
                start_percent=get_value_at_index(floatconstant_444, 0),
                end_percent=get_value_at_index(floatconstant_445, 0),
                positive=get_value_at_index(controlnetapplyadvanced_72, 0),
                negative=get_value_at_index(controlnetapplyadvanced_72, 1),
                control_net=get_value_at_index(controlnetloaderadvanced_97, 0),
                image=get_value_at_index(loadimagesfromdirectory_174, 0),
            )

            ksampleradvanced_399 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=get_value_at_index(int_literal_383, 0),
                cfg=get_value_at_index(cfg_literal_382, 0),
                sampler_name="dpmpp_3m_sde_gpu",
                scheduler="normal",
                start_at_step=0,
                end_at_step=get_value_at_index(int_literal_421, 0),
                return_with_leftover_noise="disable",
                model=get_value_at_index(tonemapnoisewithrescalecfg_427, 0),
                positive=get_value_at_index(controlnetapplyadvanced_99, 0),
                negative=get_value_at_index(controlnetapplyadvanced_99, 1),
                latent_image=get_value_at_index(emptylatentimage_151, 0),
            )

            latentupscaleby_408 = latentupscaleby.upscale(
                upscale_method="nearest-exact",
                scale_by=get_value_at_index(cfg_literal_417, 0),
                samples=get_value_at_index(ksampleradvanced_399, 0),
            )

            bnk_injectnoise_410 = bnk_injectnoise.inject_noise(
                strength=1, latents=get_value_at_index(latentupscaleby_408, 0)
            )

            ksampleradvanced_407 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=get_value_at_index(int_literal_383, 0),
                cfg=get_value_at_index(cfg_literal_382, 0),
                sampler_name="ddim",
                scheduler="karras",
                start_at_step=get_value_at_index(int_literal_422, 0),
                end_at_step=269,
                return_with_leftover_noise="disable",
                model=get_value_at_index(tonemapnoisewithrescalecfg_427, 0),
                positive=get_value_at_index(controlnetapplyadvanced_99, 0),
                negative=get_value_at_index(controlnetapplyadvanced_99, 1),
                latent_image=get_value_at_index(bnk_injectnoise_410, 0),
            )

            vaedecode_10 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_407, 0),
                vae=get_value_at_index(vaeloader_2, 0),
            )

            vhs_splitimages_367 = vhs_splitimages.split_images(
                split_index=5, images=get_value_at_index(loadimagesfromdirectory_172, 0)
            )

            vhs_splitimages_366 = vhs_splitimages.split_images(
                split_index=5, images=get_value_at_index(loadimagesfromdirectory_174, 0)
            )

            vhs_splitimages_368 = vhs_splitimages.split_images(
                split_index=5, images=get_value_at_index(vaedecode_10, 0)
            )

            cr_integer_to_string_361 = cr_integer_to_string.convert(
                int_=get_value_at_index(int_literal_362, 0)
            )

            text_concatenate_354 = text_concatenate.text_concatenate(
                delimiter="", clean_whitespace="true"
            )

            image_save_334 = image_save.was_save_images(
                output_path=get_value_at_index(string_to_text_398, 0),
                filename_prefix=get_value_at_index(text_concatenate_354, 0),
                filename_delimiter="_",
                filename_number_padding=4,
                filename_number_start="false",
                extension="png",
                dpi=100,
                quality=100,
                optimize_image="false",
                lossless_webp="false",
                overwrite_mode="false",
                show_history="true",
                show_history_by_prefix="true",
                embed_workflow="true",
                show_previews="true",
                images=get_value_at_index(vaedecode_10, 0),
            )

def CountNumberOfImages(directory: str) -> int:
    count = 0
    for _ in os.listdir(directory):
        count += 1
    return count


def main():
    directory = "/home/yori/Downloads/outputLineArt"
    count = CountNumberOfImages(directory)
    print(count)

    b = count/GbatchSize
    rem = count%GbatchSize

    for i in range(0, math.floor(b)):
        Cooking(GbatchSize, i*GbatchSize)

    if rem != 0:
        Cooking(rem, math.floor(b)*GbatchSize)




if __name__ == "__main__":
    main()
