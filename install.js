module.exports = async (kernel) => {
  let cmd = "uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu"
  if (kernel.platform === 'darwin') {
    if (kernel.arch === "arm64") {
      cmd = "uv pip install torch torchaudio torchvision"
    } else {
      cmd = "uv pip install torch==2.1.2 torchaudio==2.1.2"
    }
  } else {
    if (kernel.gpu === 'nvidia') {
      if (kernel.gpu_model && / 50.+/.test(kernel.gpu_model)) {
        cmd = "uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
      } else {
        cmd = "uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu124"
      }
    } else if (kernel.gpu === 'amd') {
      cmd = "uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2"
    } 
  }
//module.exports = {
//  "cmds": {
//    "nvidia": "pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu124",
//    "amd": "pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2",
//    "default": "pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0"
//  },
//  "requires": [{
//    "type": "conda",
//    "name": "ffmpeg",
//    "args": "-c conda-forge"
//  }],
  return {
    "run": [{
      "method": "shell.run",
      "params": {
        "message": "git clone https://github.com/cocktailpeanut/Moore-AnimateAnyone app"
      }
    },
    {
      "method": "shell.run",
      "params": {
        "path": "app",
        "venv": "env",
        "message": [
          cmd,
//          "{{(gpu === 'nvidia' ? self.cmds.nvidia : (gpu === 'amd' ? self.cmds.amd : self.cmds.default))}}",
          "uv pip install {{platform === 'darwin' ? 'eva-decord' : 'decord'}}",
          "python -m pip install pip==24.0",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // AA
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/denoising_unet.pth?download=true",
        "dir": "app/pretrained_weights",
      }
    },
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/motion_module.pth?download=true",
        "dir": "app/pretrained_weights",
      }
    },
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/pose_guider.pth?download=true",
        "dir": "app/pretrained_weights",
      }
    },
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/reference_unet.pth?download=true",
        "dir": "app/pretrained_weights",
      }
    },

    // sd1.5
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/cocktailpeanut/stable-diffusion-v1-5/raw/main/feature_extractor/preprocessor_config.json",
        "path": "app/pretrained_weights/stable-diffusion-v1-5/feature_extractor/preprocessor_config.json",
      }
    }, {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/cocktailpeanut/stable-diffusion-v1-5/raw/main/model_index.json",
        "path": "app/pretrained_weights/stable-diffusion-v1-5/model_index.json",
      }
    }, {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/cocktailpeanut/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin?download=true",
        "dir": "app/pretrained_weights/stable-diffusion-v1-5/unet",
      }
    }, {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/cocktailpeanut/stable-diffusion-v1-5/raw/main/unet/config.json",
        "path": "app/pretrained_weights/stable-diffusion-v1-5/unet/config.json",
      }
    }, {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/cocktailpeanut/stable-diffusion-v1-5/raw/main/v1-inference.yaml",
        "path": "app/pretrained_weights/stable-diffusion-v1-5/v1-inference.yaml",
      }
    },
    // vae
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json",
        "path": "app/pretrained_weights/sd-vae-ft-mse/config.json",
      }
    },
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin?download=true",
        "dir": "app/pretrained_weights/sd-vae-ft-mse",
      }
    },
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors?download=true",
        "dir": "app/pretrained_weights/sd-vae-ft-mse",
      }
    },

    // image encoder
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin?download=true",
        "dir": "app/pretrained_weights/image_encoder",
      }
    },
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/raw/main/image_encoder/config.json",
        "path": "app/pretrained_weights/image_encoder/config.json",
      }
    },

    // DwPose
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/cocktailpeanut/dwp/resolve/main/dw-ll_ucoco_384.onnx?download=true",
        "dir": "app/pretrained_weights/DWPose",
      }
    },
    {
      "method": "fs.download",
      "params": {
        "uri": "https://huggingface.co/cocktailpeanut/dwp/resolve/main/yolox_l.onnx?download=true",
        "dir": "app/pretrained_weights/DWPose",
      }
    },
    {
      "method": "notify",
      "params": {
        "html": "Click the 'start' tab to get started!"
      }
    }]
  }
}
