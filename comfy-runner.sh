# bash <(curl -s https://raw.githubusercontent.com/MAkC8/ai-hub-comfy-init/main/comfy-runner.sh)

comfy_path="/root/ComfyUI"
installation_completed="$comfy_path/installation_completed.txt"

if [ -f $installation_completed ]; then
    sudo lsof -t -i :8188 | xargs kill -9
    cd $comfy_path && python main.py --listen 0.0.0.0 &
    echo "Installation completed."
else
    sudo apt update && sudo apt install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update && sudo apt install -y software-properties-common
    sudo apt update && sudo apt install -y python3.9 python3.9-venv python3.9-distutils
    curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.9
    sudo ln -s /usr/bin/python3.9 /usr/bin/python
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    sudo apt-get update
    sudo apt-get install -y build-essential cmake libgl1-mesa-glx python3-dev axel lsof unzip
    pip install --upgrade pip setuptools wheel
    dpkg -L python3-dev | grep Python.h
    sudo apt-get install -y python3.9-dev
    export CFLAGS="-I/usr/include/python3.9"
    pip install insightface

    rm -rf $comfy_path

    git clone https://github.com/comfyanonymous/ComfyUI.git $comfy_path
    cd $comfy_path && pip install -r requirements.txt && cd ..

    mkdir -p $comfy_path/models
    mkdir -p $comfy_path/models/checkpoints
    mkdir -p $comfy_path/models/clip_vision
    mkdir -p $comfy_path/models/ipadapter
    mkdir -p $comfy_path/models/loras
    mkdir -p $comfy_path/models/style_models
    mkdir -p $comfy_path/models/ipadapter   
    mkdir -p $comfy_path/models/upscale_models

    git clone https://github.com/ltdrdata/ComfyUI-Manager $comfy_path/custom_nodes/ComfyUI-Manager
    cd $comfy_path/custom_nodes/ComfyUI-Manager && pip install -r requirements.txt && cd ../../..
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux $comfy_path/custom_nodes/comfyui_controlnet_aux
    cd $comfy_path/custom_nodes/comfyui_controlnet_aux && pip install -r requirements.txt && cd ../../..
    git clone https://github.com/Gourieff/comfyui-reactor-node $comfy_path/custom_nodes/comfyui-reactor-node
    cd $comfy_path/custom_nodes/comfyui-reactor-node && pip install -r requirements.txt && python install.py && cd ../../..
    git clone https://github.com/mav-rik/facerestore_cf.git $comfy_path/custom_nodes/facerestore_cf
    cd $comfy_path/custom_nodes/facerestore_cf && pip install -r requirements.txt && cd ../../..
    git clone https://github.com/crystian/ComfyUI-Crystools.git $comfy_path/custom_nodes/ComfyUI-Crystools
    cd $comfy_path/custom_nodes/ComfyUI-Crystools && pip install -r requirements.txt && cd ../../..
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git $comfy_path/custom_nodes/ComfyUI_IPAdapter_plus
    git clone https://github.com/palant/image-resize-comfyui.git $comfy_path/custom_nodes/image-resize-comfyui
    git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git $comfy_path/custom_nodes/masquerade-nodes-comfyui
    git clone https://github.com/storyicon/comfyui_segment_anything.git $comfy_path/custom_nodes/comfyui_segment_anything
    cd $comfy_path/custom_nodes/comfyui_segment_anything && pip install -r requirements.txt && cd ../../..
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive $comfy_path/custom_nodes/ComfyUI_UltimateSDUpscale
    git clone https://github.com/bronkula/comfyui-fitsize.git $comfy_path/custom_nodes/comfyui-fitsize
    git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG.git $comfy_path/custom_nodes/ComfyUI-BRIA_AI-RMBG
    git clone https://github.com/pythongosssss/ComfyUI-WD14-Tagger $comfy_path/custom_nodes/ComfyUI-WD14-Tagger
    cd $comfy_path/custom_nodes/ComfyUI-WD14-Tagger && pip install -r requirements.txt && cd ../../..
    git clone https://github.com/FlyingFireCo/tiled_ksampler.git $comfy_path/custom_nodes/tiled_ksampler
    wget -O $comfy_path/custom_nodes/ComfyUI-BRIA_AI-RMBG/RMBG-1.4/model.pth https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth
    git clone https://github.com/thecooltechguy/ComfyUI-Stable-Video-Diffusion $comfy_path/custom_nodes/ComfyUI-Image-Selector

    ### VHS & VFI
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git $comfy_path/custom_nodes/ComfyUI-VideoHelperSuite
    cd $comfy_path/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt && cd ../../..
    git clone  https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git $comfy_path/custom_nodes/ComfyUI-Frame-Interpolation
    cd $comfy_path/custom_nodes/ComfyUI-Frame-Interpolation && python install.py && cd ../../..
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git $comfy_path/custom_nodes/ComfyUI_Custom_Scripts

    ### CatVton
    cd $comfy_path
    wget -O "./ComfyUI-CatVTON.zip" "https://github.com/MAkC8/ai-hub-comfy-init/blob/main/ComfyUI-CatVTON.zip?raw=true"
    rm -rf ./ComfyUI-CatVTON $comfy_path/custom_nodes/ComfyUI-CatVTON 
    unzip ./ComfyUI-CatVTON.zip
    mv ./ComfyUI-CatVTON $comfy_path/custom_nodes/
    rm ./ComfyUI-CatVTON.zip

    pip install diffusers accelerate
    mkdir -p $comfy_path/custom_nodes/ComfyUI-CatVTON/main-git
    cd $comfy_path/custom_nodes/ComfyUI-CatVTON && python -m venv .venv && source .venv/bin/activate && cd $comfy_path/..
    git clone https://github.com/Zheng-Chong/CatVTON.git $comfy_path/custom_nodes/ComfyUI-CatVTON/main-git/CatVTON
    cd $comfy_path/custom_nodes/ComfyUI-CatVTON/main-git/CatVTON && pip install -r requirements.txt && cd $comfy_path/..
    deactivate
    git clone https://github.com/facebookresearch/detectron2.git $comfy_path/custom_nodes/ComfyUI-CatVTON/main-git/detectron2
    cd $comfy_path/custom_nodes/ComfyUI-CatVTON/main-git/detectron2 && python -m pip install . && cd projects/DensePose && pip install . && cd $comfy_path/..
    rm -rf $comfy_path/custom_nodes/ComfyUI-CatVTON/main-git
    cd $comfy_path/..
    ###---------
    git clone https://github.com/cubiq/ComfyUI_InstantID.git $comfy_path/custom_nodes/ComfyUI_InstantID
    cd $comfy_path/custom_nodes/ComfyUI_InstantID && pip install -r requirements.txt && cd ../../..
    
    axel -n 8 -o $comfy_path/models/checkpoints/sd15_real.safetensors "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=pruned&fp=fp16"

    sudo lsof -t -i :8188 | xargs kill -9
    cd $comfy_path && python main.py --listen 0.0.0.0 &

    wget -O $comfy_path/models/checkpoints/JuggernautXL.safetensors "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    wget -O $comfy_path/models/clip_vision/clip_vision_xl.safetensors  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"

    mkdir -p $comfy_path/models/instantid && wget -O $comfy_path/models/instantid/ip-adapter.bin "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin"
    wget -O $comfy_path/models/controlnet/instant_id_control_net.safetensors "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors"

    mkdir -p $comfy_path/models/insightface/models/ && wget -O $comfy_path/models/insightface/models/antelopev2.zip https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip
    unzip $comfy_path/models/insightface/models/antelopev2.zip
    mv ./antelopev2 $comfy_path/models/insightface/models/
    rm $comfy_path/models/insightface/models/antelopev2.zip

    mkdir -p $comfy_path/models/ipadapter && wget -O $comfy_path/models/ipadapter/ip_adapter_plus_sdxl.safetensors "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"

    wget -O $comfy_path/models/style_models/coadapter-style-sd15v1.pth "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/coadapter-style-sd15v1.pth" &
    mkdir -p $comfy_path/models/ipadapter && wget -O $comfy_path/models/ipadapter/ip-adapter-plus_sd15.safetensors "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"
    wget -O $comfy_path/models/clip_vision/clip_vision.safetensors  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
    wget -O $comfy_path/models/controlnet/control_sd15_inpaint_depth_hand_fp16.safetensors  "https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/control_sd15_inpaint_depth_hand_fp16.safetensors" &
    wget -O $comfy_path/models/controlnet/control_v11p_sd15_seg.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/diffusion_pytorch_model.safetensors"
    wget -O $comfy_path/models/controlnet/control_v11p_sd15_canny.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.safetensors" &
    wget -O $comfy_path/models/controlnet/control_v11p_sd15_lineart.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/diffusion_pytorch_model.safetensors"
    wget -O $comfy_path/models/controlnet/control_v11f1e_sd15_tile.bin "https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/diffusion_pytorch_model.bin" &
    wget -O $comfy_path/models/upscale_models/4x-UltraSharp.pth "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"

    wget -O $comfy_path/models/controlnet/control_v11p_sd15_openpose.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors"
    wget -O $comfy_path/models/controlnet/OpenPoseXL2.safetensors "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors"
    axel -n 8 -o $comfy_path/models/loras/detail_tweaker_xl.safetensors "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor"
    wget -O $comfy_path/models/controlnet/diffusers_xl_canny_full.safetensors "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors"
    wget -O $comfy_path/models/controlnet/diffusers_xl_zoe_depth.safetensors "https://huggingface.co/diffusers/controlnet-zoe-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" &
    wget -O $comfy_path/models/controlnet/diffusion_xl_depth_fp16.safetensors "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors"
    wget -O $comfy_path/models/controlnet/sdxl_segmentation_ade20k_controlnet.safetensors "https://huggingface.co/abovzv/sdxl_segmentation_controlnet_ade20k/resolve/main/sdxl_segmentation_ade20k_controlnet.safetensors" &

    ### AnimateAnyone
    # git clone https://github.com/MrForExample/ComfyUI-AnimateAnyone-Evolved.git $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved
    # cd $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved && pip install -r requirements.txt && cd ../../..
    # wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/denoising_unet.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/denoising_unet.pth" &
    # wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/motion_module.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/motion_module.pth"
    # wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/pose_guider.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/pose_guider.pth" &
    # wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/reference_unet.pth "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/reference_unet.pth"
    # wget -O $comfy_path/custom_nodes/ComfyUI-AnimateAnyone-Evolved/pretrained_weights/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin" &
    # wget -O $comfy_path/models/vae/diffusion_pytorch_model.bin "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"
    # wget -O $comfy_path/models/clip_vision/pytorch_model.bin "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin"

    # ### AnimatedDiff
    # git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git $comfy_path/custom_nodes/ComfyUI-AnimateDiff-Evolved
    # mkdir $comfy_path/models/animatediff_models
    # wget -O $comfy_path/models/animatediff_models/v3_sd15_mm.ckpt "https://huggingface.co/guoyww/animatediff/resolve/cd71ae134a27ec6008b968d6419952b0c0494cf2/v3_sd15_mm.ckpt"
    # wget -O $comfy_path/models/animatediff_models/mm_sd_v15_v2.ckpt "https://huggingface.co/guoyww/animatediff/resolve/cd71ae134a27ec6008b968d6419952b0c0494cf2/mm_sd_v15_v2.ckpt" &
    # wget -O $comfy_path/models/animatediff_models/mm-Stabilized_high.pth "https://huggingface.co/manshoety/AD_Stabilized_Motion/resolve/main/mm-Stabilized_high.pth"
    # wget -O $comfy_path/models/animatediff_models/mm-p_0.75.pth "https://huggingface.co/manshoety/beta_testing_models/resolve/main/mm-p_0.75.pth" &
    # wget -O $comfy_path/models/animatediff_models/temporaldiff-v1-animatediff.safetensors "https://huggingface.co/CiaraRowles/TemporalDiff/blob/main/temporaldiff-v1-animatediff.safetensors"
    # wget -O $comfy_path/models/animatediff_motion_lora/v2_lora_PanLeft.ckpt "https://huggingface.co/guoyww/animatediff/resolve/cd71ae134a27ec6008b968d6419952b0c0494cf2/v2_lora_PanLeft.ckpt" &
    # wget -O $comfy_path/models/animatediff_motion_lora/v2_lora_PanRight.ckpt "https://huggingface.co/guoyww/animatediff/resolve/cd71ae134a27ec6008b968d6419952b0c0494cf2/v2_lora_PanRight.ckpt"
    # wget -O $comfy_path/models/animatediff_motion_lora/v2_lora_TiltDown.ckpt "https://huggingface.co/guoyww/animatediff/resolve/cd71ae134a27ec6008b968d6419952b0c0494cf2/v2_lora_TiltDown.ckpt" &
    # wget -O $comfy_path/models/animatediff_motion_lora/v2_lora_ZoomOut.safetensors "https://huggingface.co/guoyww/animatediff-motion-lora-zoom-out/resolve/main/diffusion_pytorch_model.safetensors"
    # wget -O $comfy_path/models/controlnet/v3_sd15_sparsectrl_rgb.ckpt "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_rgb.ckpt"

    # wget -O $comfy_path/models/animatediff_models/AnimateLCM_sd15_t2v.ckpt "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt"
    # wget -O $comfy_path/models/animatediff_motion_lora/AnimateLCM_sd15_t2v_lora.safetensors "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors"
    # wget -O $comfy_path/models/clip_vision/laion2B.safetensors "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/model.safetensors"
    # axel -n 8 -o $comfy_path/models/animatediff_motion_lora/pxlpshr_shatter_400.safetensors "https://civitai.com/api/download/models/350715?type=Model&format=SafeTensor&token=dde65e385e7f360d4a29055d34bbac25"
    # axel -n 8 -o $comfy_path/models/checkpoints/photon_lcm.safetensors "https://civitai.com/api/download/models/344398?type=Model&format=SafeTensor&size=pruned&fp=fp16&token=dde65e385e7f360d4a29055d34bbac25"

    # ====> AnimateLCM
    # mv $comfy_path/models/svd/svd.safetensors $comfy_path/models/checkpoints/svd.safetensors
    # wget -O $comfy_path/models/ultralytics/yolov8x.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8x.pt
    # wget -O $comfy_path/models/animatediff_models/animatediff_lightning_8step_diffusers.safetensors https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_8step_diffusers.safetensors

    # wget -O $comfy_path/models/animatediff_models/AnimateLCM_sd15_t2v.ckpt "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt"
    # wget -O $comfy_path/models/animatediff_motion_lora/AnimateLCM_sd15_t2v_lora.safetensors "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors"
    # cp $comfy_path/models/animatediff_motion_lora/AnimateLCM_sd15_t2v_lora.safetensors $comfy_path/models/loras/AnimateLCM_sd15_t2v_lora.safetensors
    # wget -O $comfy_path/models/clip_vision/laion2B.safetensors "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/model.safetensors"
    # axel -n 8 -o $comfy_path/models/animatediff_motion_lora/pxlpshr_shatter_400.safetensors "https://civitai.com/api/download/models/350715?type=Model&format=SafeTensor&token=dde65e385e7f360d4a29055d34bbac25"
    # axel -n 8 -o $comfy_path/models/checkpoints/photon_lcm.safetensors "https://civitai.com/api/download/models/344398?type=Model&format=SafeTensor&size=pruned&fp=fp16&token=dde65e385e7f360d4a29055d34bbac25"

    # wget -O $comfy_path/models/animatediff_models/AnimateLCM-SVD-xt-1.1.safetensors "https://huggingface.co/wangfuyun/AnimateLCM-SVD-xt/resolve/main/AnimateLCM-SVD-xt-1.1.safetensors"
    # wget -O $comfy_path/models/animatediff_models/AnimateLCM_sd15_i2v.ckpt "https://huggingface.co/wangfuyun/AnimateLCM-I2V/resolve/main/AnimateLCM_sd15_i2v.ckpt"
    # wget -O $comfy_path/models/animatediff_motion_lora/AnimateLCM_sd15_i2v_lora.safetensors "https://huggingface.co/wangfuyun/AnimateLCM-I2V/resolve/main/AnimateLCM_sd15_i2v_lora.safetensors"
    # cp $comfy_path/models/animatediff_motion_lora/AnimateLCM_sd15_i2v_lora.safetensors $comfy_path/models/loras/AnimateLCM_sd15_i2v_lora.safetensors
    # mv $comfy_path/models/animatediff_models/AnimateLCM-SVD-xt-1.1.safetensors $comfy_path/models/checkpoints/AnimateLCM-SVD-xt-1.1.safetensors

    touch $installation_completed
    echo "Installation completed"
fi
