#!/bin/bash

# This file will be sourced in init.sh

# https://raw.githubusercontent.com/ai-dock/comfyui/main/config/provisioning/default.sh

# Packages are installed after nodes so we can fix them...

PYTHON_PACKAGES=(
    #"opencv-python==4.7.0.72"
)

NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/Gourieff/comfyui-reactor-node"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/crystian/ComfyUI-Crystools"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/taabata/LCM_Inpaint_Outpaint_Comfy"
    "https://github.com/palant/image-resize-comfyui"
    "https://github.com/BadCafeCode/masquerade-nodes-comfyui"
    "https://github.com/storyicon/comfyui_segment_anything"
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"
    "https://github.com/bronkula/comfyui-fitsize"
    "https://github.com/pythongosssss/ComfyUI-WD14-Tagger"
    "https://github.com/SLAPaper/ComfyUI-Image-Selector"
    "https://github.com/mav-rik/facerestore_cf"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG"
    "https://github.com/AuroBit/ComfyUI-OOTDiffusion"
    "https://github.com/FlyingFireCo/tiled_ksampler"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/melMass/comfy_mtb"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/MrForExample/ComfyUI-AnimateAnyone-Evolved"
)

CHECKPOINT_MODELS=(
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    #"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
)

LORA_MODELS=(
    #"https://civitai.com/api/download/models/16576"
)

VAE_MODELS=(
    "https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
)

ESRGAN_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/4x_NMKD-Siax_200k.pth"
)

CONTROLNET_MODELS=(
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors"
    "https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_depth_fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    DISK_GB_AVAILABLE=$(($(df --output=avail -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_USED=$(($(df --output=used -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_ALLOCATED=$(($DISK_GB_AVAILABLE + $DISK_GB_USED))
    provisioning_print_header
    provisioning_get_nodes
    provisioning_install_python_packages
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/ckpt" \
        "${CHECKPOINT_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/lora" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/vae" \
        "${VAE_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/esrgan" \
        "${ESRGAN_MODELS[@]}"
    build_ai_hub_models_configuration
    provisioning_print_end
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                    micromamba -n comfyui run ${PIP_INSTALL} -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                micromamba -n comfyui run ${PIP_INSTALL} -r "${requirements}"
            fi
        fi
    done
}

function provisioning_install_python_packages() {
    if [ ${#PYTHON_PACKAGES[@]} -gt 0 ]; then
        micromamba -n comfyui run ${PIP_INSTALL} ${PYTHON_PACKAGES[*]}
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    dir="$1"
    mkdir -p "$dir"
    shift
    if [[ $DISK_GB_ALLOCATED -ge $DISK_GB_REQUIRED ]]; then
        arr=("$@")
    else
        printf "WARNING: Low disk space allocation - Only the first model will be downloaded!\n"
        arr=("$1")
    fi
    
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
}

function build_ai_hub_models_configuration() {
    printf "Start AiHub configuring ..."
    sudo apt-get install -y axel
    comfy_path="/opt/ComfyUI"
    mkdir -p $comfy_path/models
    mkdir -p $comfy_path/models/checkpoints
    mkdir -p $comfy_path/models/clip_vision
    mkdir -p $comfy_path/models/ipadapter
    mkdir -p $comfy_path/models/loras
    mkdir -p $comfy_path/models/style_models
    mkdir -p $comfy_path/models/ipadapter   
    mkdir -p $comfy_path/models/upscale_models

    cd $comfy_path/custom_nodes/comfyui-reactor-node && pip install -r requirements.txt && python install.py && cd ../../..

    axel -n 8 -o $comfy_path/models/checkpoints/sd15_real.safetensors "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    axel -n 8 -o $comfy_path/models/loras/detail_tweaker_xl.safetensors "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor"
    
    wget -O $comfy_path/models/clip_vision/clip_vision_xl.safetensors  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" &
    wget -O $comfy_path/models/ipadapter/ip_adapter_plus_sdxl.safetensors "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
    wget -O $comfy_path/models/style_models/coadapter-style-sd15v1.pth "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/coadapter-style-sd15v1.pth" &
    wget -O  $comfy_path/custom_nodes/ComfyUI-BRIA_AI-RMBG/RMBG-1.4/model.pth https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth
    wget -O $comfy_path/models/controlnet/diffusers_xl_canny_full.safetensors "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors" &
    wget -O $comfy_path/models/controlnet/diffusers_xl_zoe_depth.safetensors "https://huggingface.co/diffusers/controlnet-zoe-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors"
    wget -O $comfy_path/models/controlnet/diffusion_xl_depth_fp16.safetensors "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors" &
    wget -O $comfy_path/models/controlnet/sdxl_segmentation_ade20k_controlnet.safetensors "https://huggingface.co/abovzv/sdxl_segmentation_controlnet_ade20k/resolve/main/sdxl_segmentation_ade20k_controlnet.safetensors" 
    wget -O $comfy_path/models/ipadapter/ip-adapter-plus_sd15.safetensors "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors" &
    wget -O $comfy_path/models/clip_vision/clip_vision.safetensors  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
    wget -O $comfy_path/models/controlnet/control_sd15_inpaint_depth_hand_fp16.safetensors  "https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/control_sd15_inpaint_depth_hand_fp16.safetensors" &
    wget -O $comfy_path/models/controlnet/control_v11p_sd15_seg.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/diffusion_pytorch_model.safetensors"
    wget -O $comfy_path/models/controlnet/control_v11p_sd15_canny.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.safetensors" &
    wget -O $comfy_path/models/upscale_models/4x-UltraSharp.pth "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
    
    wget -O $comfy_path/models/controlnet/control_v11p_sd15_seg.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/diffusion_pytorch_model.safetensors"
    wget -O $comfy_path/models/controlnet/control_v11p_sd15_canny.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.safetensors" &
    wget -O $comfy_path/models/controlnet/control_v11f1e_sd15_tile.bin "https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/diffusion_pytorch_model.bin"
    wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/denoising_unet.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/denoising_unet.pth" &
    wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/motion_module.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/motion_module.pth"
    wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/pose_guider.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/pose_guider.pth" &
    wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/reference_unet.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/reference_unet.pth"
    wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin" &
    wget -O $comfy_path/models/vae/diffusion_pytorch_model.bin "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"
    wget -O $comfy_path/models/clip_vision/pytorch_model.bin "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin"
}

provisioning_start
