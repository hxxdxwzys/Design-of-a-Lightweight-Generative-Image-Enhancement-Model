# -*- coding: utf-8 -*-
"""一致性对比: runner_14b forward vs vosr 参考管线 (分进程)
用法: python verify_runner_14b.py runner|ref
"""
import sys
import torch

if sys.argv[1] == 'runner':
    sys.path.insert(0, r'D:/models/final_compliant_14b/model_dir')
    import runner as R
    r = R.Runner(r'D:/models/final_compliant_14b/model_dir')
    chain = r.model.cuda().eval()
    torch.manual_seed(7)
    x = (torch.rand(1, 3, 512, 512) * 2 - 1).half().cuda()
    z = torch.randn(1, 16, 64, 64).half().cuda()
    import torch.nn.functional as F
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=chain.dinov2_size,
                            mode='bicubic').clamp(0., 255.)
        xn = (raw / 255.0 - chain.mean) / chain.std
        v = chain.venc.prepare_tokens_with_masks(xn, None)
        zfeat = None
        for i, blk in enumerate(chain.venc.blocks):
            v = blk(v)
            if i in chain.layer_list:
                zfeat = v[:, 1:].detach()
        lat = chain.vae.encode(x).mode()
        lat = (lat - chain.lat_mean) * chain.lat_std_inv
        u = chain.diT(torch.cat([lat, z], 1), torch.ones(1, device='cuda'),
                      torch.zeros(1, device='cuda'), [zfeat])
        z1 = z - u
        out = chain.vae.decode(z1 / chain.lat_std_inv + chain.lat_mean)
    torch.save({'x': x.cpu(), 'z': z.cpu(), 'out': out.cpu()}, r'D:/tmp_timing/consist_runner.pt')
    print('RUNNER_SAVED')
else:
    sys.argv = ['verify_runner_14b']  # 防 argparse 读入脚本参数
    sys.path.insert(0, r'D:/models/finals/VOSR')
    sys.path.insert(0, r'D:/models/finals/VOSR/models')
    sys.path.insert(0, r'D:/tmp_timing')
    import argparse
    from vosr_lora_infer_14b import build_model
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from inference_vosr_onestep import load_config_with_cli, load_dinov2
    from tiled_vae import encode_latent, decode_latent
    BASE = r'C:/models/finals/preset/ckpts/VOSR_1.4B_os'
    LORA = r'D:/tmp_timing/p100_ckpts/p512_14b_m0s1_lora_s4000.pt'
    QWEN = r'C:/models/finals/preset/ckpts/Qwen-Image-vae-2d'
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', '-c', default=BASE)
    ap.add_argument('--config', default='')
    args = load_config_with_cli(BASE, ap)
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)
    lora_path = torch.load(LORA, map_location='cpu', weights_only=False)
    dev = 'cuda'
    m = build_model(args, lora_path, lora_scale=0.93).to(dev).half().eval()
    vae = AutoencoderKLQwenImage2D.from_pretrained(QWEN).to(dev).half().eval()
    venc = load_dinov2(args, dev).half().eval()
    venc.head = torch.nn.Identity()
    args.vae_tile_size = 0
    args.posterior_mode = True
    torch.manual_seed(7)
    x = (torch.rand(1, 3, 512, 512) * 2 - 1).half().cuda()
    z = torch.randn(1, 16, 64, 64).half().cuda()
    import torch.nn.functional as F
    with torch.no_grad():
        raw = F.interpolate((0.5 * x + 0.5) * 255.0, size=args.dinov2_size,
                            mode='bicubic').clamp(0., 255.)
        MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float16).view(1, 3, 1, 1).cuda()
        STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float16).view(1, 3, 1, 1).cuda()
        xn = (raw / 255.0 - MEAN) / STD
        feats, x_norm = venc.forward_with_features(xn)
        zz = [v for k, v in feats.items() if k.startswith('layer_')]
        zz[-1] = x_norm
        zz = [zz[i] for i in args.layer_dinov2b_list]
        lat, lm, lsi = encode_latent(vae, x, args, dev)
        u = m(torch.cat([lat, z], 1), torch.ones(1, device=dev),
              torch.zeros(1, device=dev), zz)
        z1 = z - u
        out = decode_latent(vae, z1, args, lm, lsi)
    torch.save({'x': x.cpu(), 'z': z.cpu(), 'out': out.cpu()}, r'D:/tmp_timing/consist_ref.pt')
    print('REF_SAVED')
