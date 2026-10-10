======================================================================
### ODTSR_code
  29: 1. Based on Qwen-Image, we train a **single-step** SR model using LoRA, with the model parameters reaching 20B.
  30: 2. With our proposed Noise-hybrid Visual Stream and Fidelity-aware Adversarial Training, the SR process can be jointly controlled by prompts as well as a Fidelity Weight 
  31: 3. English and Chinese prompts are supported, and the model demonstrates strong performance in text images, fine-grained textures and face images.
  53: Qualitative results of controllable SR with prompt and adjustable Fidelity Weight (denoted as $f$) on Div2k-val
  54: dataset. As $f$ decreases from 1 to 0, detail generation and prompt adherence gradually strengthen.
  85: 5. Download base model to your disk: [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image/tree/main)
  87: 6. (For training) Download base model to your disk: [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main)
  88: 7. (For inference) Download the trained ODTSR model weight: [huggingface](https://huggingface.co/double8fun/ODTSR/tree/main)
  94: We now supports tile-based processing (tile size: 512×512), enabling input of arbitrary resolutions and SR at any scale factor.
  96: Please replace ``experiments/qwen_one_step_gan/${EXP_DATE}/checkpoints/net_gen_iter_10001.pth`` with the trained ODTSR model weight.
  97: ```
  98: sh examples/qwen_image/test_gan.sh
  99: ```
  106: ```
  107: sh examples/qwen_image/test_gradio.sh
  108: ```
======================================================================
### PixRestore_code
  39: (2) To capture both degradation and semantic cues, we use a vision encoder to extract multi-layer dense features from the LQ image, and use an adaptive layer router to pr
  58: ## 🍭 Quick Inference
  59: #### Step 1: Download the pretrained models
  60: Download the PixRestore model from [`obox`](https://sbox.myoas.com/l/Ba05f09268ea6081a)(pwd: PixRestore824), [HuggingFace](https://huggingface.co/VCLab-PolyU/PixRestore),
  62: DINOv2 is downloaded from Meta's official torch hub on first use. For offline
  63: machines, clone [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
  64: and pass a local repository and/or checkpoint:
  67: --dinov2-repository /path/to/dinov2 \
  68: --dinov2-checkpoint /path/to/dinov2_vits14_pretrain.pth
  69: ```
  81: --config configs/train.yaml \
  82: -c /path/to/checkpoint \
  83: -i /path/to/image_or_folder \
  115: For DINO-GAN finetuning, use the GAN config and an optional pretrained model:
  119: --manifest data/source_a.jsonl data/source_b.jsonl \
  120: --pretrained /path/to/checkpoint \
  121: --output-dir outputs/gan_experiment
  124: Run both self-contained CPU checks before a full training job. They create
  125: temporary paired images and do not download DINO weights:
======================================================================
### TADSR_code
  7: <a href='https://zty557.github.io/TADSR_HomePage/'><img src='https://img.shields.io/badge/Project page-TADSR-1bb41b.svg'></a> &nbsp;&nbsp;
  8: <a href=''><img src='https://img.shields.io/badge/Space-huggingface-ffd700.svg'></a> &nbsp;&nbsp;
  9: </div>
  30: - [x] Release training and inference code
  31: - [x] Release Checkpoints
  53: **Step 1: Download Checkpoints**
  55: Download the [[TADSR](https://huggingface.co/zty557/TADSR)] checkpoints and place them in the directories `preset/weights`.
  59: Place low-quality images in `preset/datasets/test_datasets/`.
  60: You can download `RealSR`, `DrealSR` and `RealLR200` from [[SeeSR](https://drive.google.com/drive/folders/1L2VsQYQRKhWJxe6yWZU9FgBWSgBCk6mz)],
  61: Thanks for their awesome works.
  79: - Download the training datasets `LSDIR`.
  80: - Following [[SeeSR](https://github.com/cswry/SeeSR)], you can generate the LR-HR pairs for training using.
======================================================================
### TSD_SR_code
  32: - **[2025.03]** Training code is released.
  33: - **[2025.01]** Release the TSD-SR, including the inference codes and pretrained models.
  34: - **[2024.12]** This repo is created.
  55: ## 🚀 <a name="start"></a>Quick Start
  56: #### Step 1: Download the pretrained models
  57: - Download the pretrained SD3 models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main).
  58: - Download the TSD-SR lora weights and prompt embeddings from [GoogleDrive](https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI?usp=drive_link) or [O
  60: You can put the models weights into `checkpoint/tsdsr`.
  61: You can put the prompt embbedings into `dataset/default`.
  68: python test/test_tsdsr.py \
  69: --pretrained_model_name_or_path /path/to/your/sd3 \
  70: -i imgs/test \
  71: -o outputs/test \
  72: --lora_dir checkpoint/tsdsr \
  73: --embedding_dir dataset/default
  76: ## 🌈 <a name="start"></a>Evaluation
  77: #### Step 1: Download the StableSR test datasets
  78: - **Download StableSR testsets** (DrealSRVal_crop128, RealSRVal_crop128, DIV2K_V2_val) from [GoogleDrive](https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9
  79: - **Unzip** them into `imgs/StableSR_testsets/`, the data folder should be like this:
  92: ```
  93: #### Step 2: Download the pretrained models
  94: - Download the TSD-SR lora weights `checkpoint/tsdsr-mse` from [GoogleDrive](https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI?usp=drive_link) or [
  99: python test/test_tsdsr.py \
  100: --pretrained_model_name_or_path /path/to/your/sd3 \
  101: -i imgs/StableSR_testsets/DrealSRVal_crop128/test_LR \
  102: -o outputs/DrealSR \
  103: --lora_dir checkpoint/tsdsr-mse \
  104: --embedding_dir dataset/default \
  115: ## 🔥 <a name="start"></a>Train
  116: #### Step 1: Download the training data
  117: - Download the [LSDIR](https://huggingface.co/ofsoundof/LSDIR), [FLICKR2K](https://www.kaggle.com/datasets/daehoyang/flickr2k), [DIV2K](https://data.vision.ee.ethz.ch/cvl
  120: - **Generate degraded images**: We employ the same degradation pipeline as [SeeSR](https://github.com/cswry/SeeSR/tree/main). More details can be found at [here](https://
  121: - **Prompt texts**: You may use either tag-based prompts (generated via [DAPE](https://github.com/cswry/SeeSR/blob/main/README.md#-quick-inference), download from [here](
  162: ```
  163: #### Step 4: Download the teacher lora and null prompts embeddings
  164: - Download the [teacher models](https://drive.google.com/file/d/1do8pfdm_oNUhJKxTlC_x7LqY7NlE0-Q7/view?usp=sharing) and put them in `checkpoint/teacher/`.
  165: - Download the [null prompts embeddings](https://drive.google.com/drive/folders/1_kSod1CCq_xwdwDnLYUhr7iaT70eFPBD?usp=sharing) and put them in `dataset/null/`.
  167: We derive the teacher LoRA weights through fine-tuning of the pretrained SD3 model with HR data, optimizing via Diffusion loss. This is designed to enhance the teacher mo
  168: The null prompts embeddings are used to compute cfg when training.
  172: export MODEL_NAME="/path/to/your/sd3_model";
  173: export TEACHER_MODEL_NAME="checkpoint/teacher/";
  174: export CHECKPOINT_PATH="checkpoint/tsdsr";
  175: export HF_ENDPOINT="https://hf-mirror.com";
======================================================================
### VARestorer_code
  15: 🤗
  16: [<a href="https://huggingface.co/EvanEternal/VARestorer" target="_blank">Huggingface</a>]
  17: </h3>
  50: | **1 step** | **0.23 s** | **~10&times; faster** | **27.3 M params** |
  51: | :---: | :---: | :---: | :---: |
  86: ### 🗂️ 2. Download Checkpoints
  88: Please download our pretrained [checkpoint](https://huggingface.co/EvanEternal/VARestorer/tree/main/weights) (or from [Google Drive](https://drive.google.com/file/d/1Nkwl
  90: ```
  91: |-- weights
  92: |--|-- flan-t5-xl
  93: |--|-- general_swinir_v1.ckpt
  94: |--|-- infinity_2b_reg.pth
  95: |--|-- infinity_vae_d32reg.pth
  96: |--|-- varestorer.pth
  97: ...
  114: Download the LSDIR dataset. The split file is provided at `data/LSDIR/splits/1.000_000084991.jsonl`; before training, update the LSDIR paths inside this JSONL file to poi
  123: DATA_PATH=/path/to/LSDIR/splits \
  124: PRETRAINED_CHECKPOINT_PATH=weights/infinity_2b_reg.pth \
  125: SWINIR_PATH=weights/general_swinir_v1.ckpt \
  126: NPROC_PER_NODE=8 \
  130: Training logs are written to `local_output/`, checkpoints to `checkpoints/`.
  [weights 目录] ['.gitkeep']