module.exports = {
  "cmds": {
    "nvidia": "pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118",
    "amd": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6",
    "default": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu"
  },
  "requires": [{
    "type": "conda",
    "name": "ffmpeg",
    "args": "-c conda-forge"
  }],
  "run": [{
    "method": "shell.run",
    "params": {
      "message": "git clone https://github.com/cocktailpeanut/Moore-AnimateAnyone app"
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
      "uri": "https://huggingface.co/runwayml/stable-diffusion-v1-5/raw/main/feature_extractor/preprocessor_config.json",
      "dir": "app/pretrained_weights/stable-diffusion-v1-5/feature_extractor",
    }
  }, {
    "method": "fs.download",
    "params": {
      "uri": "https://huggingface.co/runwayml/stable-diffusion-v1-5/raw/main/model_index.json",
      "dir": "app/pretrained_weights/stable-diffusion-v1-5",
    }
  }, {
    "method": "fs.download",
    "params": {
      "uri": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin?download=true",
      "dir": "app/pretrained_weights/stable-diffusion-v1-5/unet",
    }
  }, {
    "method": "fs.download",
    "params": {
      "uri": "https://huggingface.co/runwayml/stable-diffusion-v1-5/raw/main/unet/config.json",
      "dir": "app/pretrained_weights/stable-diffusion-v1-5/unet",
    }
  }, {
    "method": "fs.download",
    "params": {
      "uri": "https://huggingface.co/runwayml/stable-diffusion-v1-5/raw/main/v1-inference.yaml",
      "dir": "app/pretrained_weights/stable-diffusion-v1-5",
    }
  },
  // vae
  {
    "method": "fs.download",
    "params": {
      "uri": "https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json",
      "dir": "app/pretrained_weights/sd-vae-ft-mse",
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
      "dir": "app/pretrained_weights/image_encoder",
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
    "method": "shell.run",
    "params": {
      "path": "app",
      "venv": "env",
      "message": [
        "{{(gpu === 'nvidia' ? self.cmds.nvidia : (gpu === 'amd' ? self.cmds.amd : self.cmds.default))}}",
        "pip install {{platform === 'darwin' ? 'eva-decord' : 'decord'}}",
        "pip install -r requirements.txt"
      ]
    }
  }, {
    "method": "notify",
    "params": {
      "html": "Click the 'start' tab to get started!"
    }
  }]
}
