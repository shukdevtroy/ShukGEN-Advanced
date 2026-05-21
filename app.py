"""
ShukGEN Advanced — Premium Gradio Application
Identity-Preserving VAE + HD Super Resolution + 8 Style Variants
"""

import os
import io
import math
import random
import base64
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import timm

import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# STYLE METADATA
# ─────────────────────────────────────────────────────────────────────────────
STYLE_NAMES = [
    "Youthful",
    "Aged / Mature",
    "Dramatic Light",
    "Soft Glow",
    "Intense / Bold",
    "Warm Golden Hour",
    "Cool / Moody",
    "Sketch / Artistic",
]

STYLE_EMOJIS = ["✨", "🎞️", "🎭", "🌸", "🔥", "🌅", "🌊", "🎨"]

STYLE_COLORS = [
    "#27ae60", "#c0392b", "#8e44ad", "#f39c12",
    "#e74c3c", "#d35400", "#2980b9", "#16a085"
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (exact copy from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def norm(ch):
    for g in [8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            norm(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            norm(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
    def forward(self, x): return x + self.net(x)


class SE(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        mid = max(ch // ratio, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.SiLU(),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.net(x).view(x.size(0), x.size(1), 1, 1)


class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=8):
        super().__init__()
        while ch % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.norm = norm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), norm(out_ch), nn.SiLU())
        self.res  = ResBlock(out_ch)
        self.se   = SE(out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x    = self.se(self.res(self.conv(x)))
        skip = x
        return self.down(x), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up_conv = nn.Conv2d(in_ch, in_ch * 4, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.conv    = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.norm_l  = norm(out_ch)
        self.act     = nn.SiLU()
        self.res     = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.shuffle(self.up_conv(x))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.act(self.norm_l(self.conv(x))))


class FaceVAE(nn.Module):
    def __init__(self, latent_dim=512, base_filters=48):
        super().__init__()
        f            = base_filters
        self.f       = f
        self.latent  = latent_dim
        self.pool_sz = 4
        flat         = f * 16 * self.pool_sz * self.pool_sz

        # ── Encoder ──────────────────────────────────────────────────────────
        self.inc   = nn.Sequential(nn.Conv2d(3, f, 3, padding=1, bias=False), norm(f), nn.SiLU())
        self.down1 = DownBlock(f,    f*2)
        self.down2 = DownBlock(f*2,  f*4)
        self.down3 = DownBlock(f*4,  f*8)
        self.down4 = DownBlock(f*8,  f*16)
        self.btn   = nn.Sequential(ResBlock(f*16), AttnBlock(f*16), ResBlock(f*16), SE(f*16))
        self.pool  = nn.AdaptiveAvgPool2d(self.pool_sz)
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.fc_dec = nn.Linear(latent_dim, flat)

        # Up-blocks: up1 = first (right after bottleneck), up4 = last (near output)
        self.up1 = UpBlock(f*16, f*16, f*8)   # uses s4  →  f*8  output
        self.up2 = UpBlock(f*8,  f*8,  f*4)   # uses s3  →  f*4  output

        # Mid-decoder attention at f*4 resolution (192 channels when base_filters=48)
        self.attn_d = AttnBlock(f*4)

        self.up3 = UpBlock(f*4,  f*4,  f*2)   # uses s2  →  f*2  output
        self.up4 = UpBlock(f*2,  f*2,  f)     # uses s1  →  f    output

        # Output head
        self.outc = nn.Sequential(
            norm(f), nn.SiLU(),
            nn.Conv2d(f, 3, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x1 = self.inc(x)
        x2, s1 = self.down1(x1)
        x3, s2 = self.down2(x2)
        x4, s3 = self.down3(x3)
        x5, s4 = self.down4(x4)
        b  = self.btn(x5)
        p  = self.pool(b).flatten(1)
        return self.fc_mu(p), self.fc_logvar(p), (s1, s2, s3, s4)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, skips):
        s1, s2, s3, s4 = skips
        f  = self.f
        h  = self.fc_dec(z).view(z.size(0), f*16, self.pool_sz, self.pool_sz)
        h  = F.interpolate(h, size=s4.shape[2:], mode='bilinear', align_corners=False)
        h  = self.up1(h, s4)    # f*16 → f*8
        h  = self.up2(h, s3)    # f*8  → f*4
        h  = self.attn_d(h)     # mid-decoder attention at f*4 (192 ch)
        h  = self.up3(h, s2)    # f*4  → f*2
        h  = self.up4(h, s1)    # f*2  → f
        return self.outc(h)

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skips), mu, logvar

    def reconstruct(self, x):
        mu, _, skips = self.encode(x)
        return self.decode(mu, skips)


class SRResBlock(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(num_parameters=ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
    def forward(self, x): return x + self.net(x)


class SRHead(nn.Module):
    def __init__(self, n_blocks=10, ch=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, ch, 9, padding=4),
            nn.PReLU(num_parameters=ch)
        )
        self.body     = nn.Sequential(*[SRResBlock(ch) for _ in range(n_blocks)])
        self.body_end = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(num_parameters=ch)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(ch, 3, 9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        body = self.body_end(body) + head
        up   = self.upsample(body)
        return self.tail(up)


# ─────────────────────────────────────────────────────────────────────────────
# STYLE TRANSFORM BANK  (exact copy from notebook)
# ─────────────────────────────────────────────────────────────────────────────

class StyleTransformBank:

    @staticmethod
    def _np(img):
        return np.array(img, dtype=np.float32) / 255.0

    @staticmethod
    def _pil(arr):
        return Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))

    @staticmethod
    def _gamma(arr, g):
        return np.power(arr.clip(1e-6, 1.0), g)

    @staticmethod
    def _rgb_to_hsv(rgb):
        r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v    = maxc
        s    = np.where(maxc != 0, (maxc - minc) / (maxc + 1e-10), 0.0)
        d    = maxc - minc + 1e-10
        h    = np.zeros_like(r)
        m    = maxc != minc
        mr   = m & (maxc == r)
        mg   = m & (maxc == g)
        mb   = m & (maxc == b)
        h[mr] = (60 * ((g[mr] - b[mr]) / d[mr])) % 360
        h[mg] = 60 * ((b[mg] - r[mg]) / d[mg]) + 120
        h[mb] = 60 * ((r[mb] - g[mb]) / d[mb]) + 240
        return np.stack([h/360, s, v], axis=-1)

    @staticmethod
    def _hsv_to_rgb(hsv):
        h, s, v = hsv[...,0]*360, hsv[...,1], hsv[...,2]
        hi = (h / 60).astype(int) % 6
        f  = (h / 60) - np.floor(h / 60)
        p  = v * (1 - s)
        q  = v * (1 - f * s)
        t  = v * (1 - (1 - f) * s)
        rgb = np.zeros(hsv.shape, dtype=np.float32)
        for i, (rv, gv, bv) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
            m = hi == i
            rgb[...,0][m] = rv[m]; rgb[...,1][m] = gv[m]; rgb[...,2][m] = bv[m]
        return rgb

    @classmethod
    def _saturate(cls, arr, factor):
        hsv = cls._rgb_to_hsv(arr)
        hsv[...,1] = (hsv[...,1] * factor).clip(0, 1)
        return cls._hsv_to_rgb(hsv)

    @staticmethod
    def _vignette(arr, strength=0.5):
        H, W = arr.shape[:2]
        y = np.linspace(-1, 1, H)[:, None]
        x = np.linspace(-1, 1, W)[None, :]
        mask = 1.0 - strength * (x**2 + y**2)
        return arr * mask[..., None].clip(0, 1)

    @staticmethod
    def _grain(arr, amount=0.04, seed=42):
        rng = np.random.default_rng(seed=seed)
        return (arr + rng.normal(0, amount, arr.shape).astype(np.float32)).clip(0, 1)

    @staticmethod
    def _blur(arr, sigma):
        return np.stack(
            [gaussian_filter(arr[...,c], sigma) for c in range(3)], axis=-1
        ).astype(np.float32)

    @classmethod
    def style_0_youthful(cls, img):
        arr = cls._np(img)
        arr = (arr * 1.22).clip(0, 1)
        arr = cls._saturate(arr, 1.55)
        blr  = cls._blur(arr, sigma=3.0)
        lum  = arr.mean(axis=-1, keepdims=True)
        mask = np.exp(-((lum - 0.6)**2) / (2*0.2**2))
        arr  = arr * (1 - 0.45*mask) + blr * (0.45*mask)
        arr[...,0] = (arr[...,0] * 1.06).clip(0,1)
        pil = cls._pil(arr)
        return ImageEnhance.Sharpness(pil).enhance(0.5)

    @classmethod
    def style_1_aged_mature(cls, img):
        arr = cls._np(img)
        arr = cls._saturate(arr, 0.25)
        grey = arr.mean(axis=-1, keepdims=True)
        sepia = np.concatenate([grey * 1.12, grey * 0.88, grey * 0.65], axis=-1)
        arr = arr * 0.35 + sepia * 0.65
        arr = cls._gamma(arr, 1.55)
        arr = arr * 0.78 + 0.10
        rng = np.random.default_rng(seed=7)
        tex = rng.normal(0, 0.03, arr.shape[:2]).astype(np.float32)
        tex = gaussian_filter(tex, 0.7)[..., None]
        arr = (arr + tex * 0.55).clip(0, 1)
        return cls._pil(arr)

    @classmethod
    def style_2_dramatic_light(cls, img):
        arr = cls._np(img)
        H, W = arr.shape[:2]
        x       = np.linspace(0, 1, W)[None, :]
        shadow  = 1.0 / (1.0 + np.exp(-14 * (x - 0.52)))
        shadow3 = shadow[..., None]
        light   = cls._gamma(arr, 0.60) * 1.35
        dark    = cls._gamma(arr, 2.50) * 0.18
        arr  = light * (1 - shadow3) + dark * shadow3
        arr  = (arr - 0.5) * 3.0 + 0.5
        arr  = cls._saturate(arr, 0.80)
        arr  = cls._vignette(arr, strength=0.75)
        return cls._pil(arr)

    @classmethod
    def style_3_soft_glow(cls, img):
        arr = cls._np(img)
        hi    = np.clip(arr - 0.45, 0, 1)
        bloom = cls._blur(hi, sigma=14)
        arr   = (arr + bloom * 2.2).clip(0, 1)
        soft  = cls._blur(arr, sigma=1.8)
        arr   = arr * 0.68 + soft * 0.32
        arr[...,0] = (arr[...,0] * 1.18).clip(0,1)
        arr[...,1] = (arr[...,1] * 1.06).clip(0,1)
        arr[...,2] = (arr[...,2] * 0.82).clip(0,1)
        arr   = arr * 0.82 + 0.15
        arr   = cls._saturate(arr, 1.35)
        return cls._pil(arr)

    @classmethod
    def style_4_intense_bold(cls, img):
        arr = cls._np(img)
        arr = cls._gamma(arr, 0.68)
        arr = cls._saturate(arr, 2.6)
        arr = (arr - 0.5) * 1.9 + 0.5
        lum  = arr.mean(axis=-1, keepdims=True)
        mid  = np.exp(-((lum - 0.5)**2) / (2*0.25**2))
        arr  = (arr + mid * 0.07).clip(0, 1)
        pil = cls._pil(arr)
        pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
        return pil.filter(ImageFilter.UnsharpMask(radius=1, percent=90, threshold=1))

    @classmethod
    def style_5_warm_golden(cls, img):
        arr = cls._np(img)
        r = (arr[...,0] * 1.40 + 0.10).clip(0, 1)
        g = (arr[...,1] * 1.12 + 0.05).clip(0, 1)
        b = (arr[...,2] * 0.48 - 0.03).clip(0, 1)
        arr  = np.stack([r, g, b], axis=-1)
        dark = np.exp(-arr.mean(axis=-1, keepdims=True) / 0.22)
        arr  = (arr + dark * 0.22 * np.array([0.95, 0.62, 0.08])).clip(0,1)
        hi   = np.clip(arr - 0.60, 0, 1)
        arr  = (arr + cls._blur(hi, 10) * 1.4).clip(0, 1)
        arr  = cls._saturate(arr, 1.25)
        return cls._pil(arr)

    @classmethod
    def style_6_cool_moody(cls, img):
        arr = cls._np(img)
        arr[...,0] = (arr[...,0] * 0.68 - 0.05).clip(0, 1)
        arr[...,1] = (arr[...,1] * 0.87 + 0.02).clip(0, 1)
        arr[...,2] = (arr[...,2] * 1.35 + 0.08).clip(0, 1)
        arr  = cls._gamma(arr, 1.30)
        lum  = arr.mean(axis=-1, keepdims=True)
        mid  = np.exp(-((lum - 0.5)**2) / (2*0.18**2))
        hsv  = cls._rgb_to_hsv(arr)
        hsv[...,1] = (hsv[...,1] * (1 - mid[...,0] * 0.55)).clip(0, 1)
        arr  = cls._hsv_to_rgb(hsv)
        arr  = (arr - 0.5) * 1.45 + 0.5
        arr  = cls._vignette(arr, strength=0.60)
        arr  = cls._grain(arr, amount=0.038, seed=42)
        return cls._pil(arr)

    @classmethod
    def style_7_sketch_artistic(cls, img):
        arr = cls._np(img)
        steps    = 5
        arr_post = np.floor(arr * steps) / steps
        lum      = arr.mean(axis=-1)
        edges    = np.abs(lum - gaussian_filter(lum, sigma=1.2)) * 10.0
        edges    = edges.clip(0, 1)[..., None]
        arr_e    = arr_post * (1 - edges * 0.88)
        arr_e    = cls._saturate(arr_e, 1.9)
        canvas   = np.array([0.95, 0.92, 0.85], dtype=np.float32)
        arr_e    = arr_e * 0.86 + canvas * 0.14
        rng      = np.random.default_rng(seed=99)
        grain    = rng.uniform(-0.03, 0.03, arr.shape).astype(np.float32)
        arr_e    = (arr_e + grain).clip(0, 1)
        pil      = cls._pil(arr_e)
        return pil.filter(ImageFilter.UnsharpMask(radius=1, percent=130, threshold=2))

    TRANSFORMS = [
        style_0_youthful.__func__,
        style_1_aged_mature.__func__,
        style_2_dramatic_light.__func__,
        style_3_soft_glow.__func__,
        style_4_intense_bold.__func__,
        style_5_warm_golden.__func__,
        style_6_cool_moody.__func__,
        style_7_sketch_artistic.__func__,
    ]

    @classmethod
    def apply(cls, img, style_idx):
        return cls.TRANSFORMS[style_idx](cls, img)

    @classmethod
    def apply_all(cls, img):
        return [cls.apply(img, i) for i in range(len(cls.TRANSFORMS))]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

_vae = None
_sr  = None
_cfg = None

def load_model(path: str):
    global _vae, _sr, _cfg
    ckpt = torch.load(path, map_location=DEVICE)
    c    = ckpt['config']
    vae  = FaceVAE(c['latent_dim'], c['base_filters']).to(DEVICE)
    sr   = SRHead(n_blocks=10, ch=64).to(DEVICE)
    vae.load_state_dict(ckpt['vae_state'])
    sr.load_state_dict(ckpt['sr_state'])
    vae.eval(); sr.eval()
    _vae = vae; _sr = sr; _cfg = c
    return (
        f"✅ Model loaded from: {os.path.basename(path)}\n"
        f"   FaceVAE  : {sum(p.numel() for p in vae.parameters())/1e6:.2f}M params\n"
        f"   SRHead   : {sum(p.numel() for p in sr.parameters())/1e6:.2f}M params\n"
        f"   Pipeline : {c['image_size']}×{c['image_size']} → {c['sr_size']}×{c['sr_size']} HD\n"
        f"   Device   : {DEVICE}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(pil_img: Image.Image, size: int) -> torch.Tensor:
    t = TF.to_tensor(TF.resize(pil_img.convert("RGB"), (size, size)))
    return TF.normalize(t, [0.5]*3, [0.5]*3).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).cpu().clamp(-1, 1)
    return transforms.ToPILImage()((t + 1.0) / 2.0)


def apply_with_strength(hd_pil, style_idx, alpha):
    styled  = StyleTransformBank.apply(hd_pil, style_idx)
    rec_arr = np.array(hd_pil,  dtype=np.float32)
    sty_arr = np.array(styled,  dtype=np.float32)
    blended = rec_arr * (1 - alpha) + sty_arr * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(image, style_strength):
    if _vae is None or _sr is None:
        raise gr.Error("⚠️ Please load a model first using the Model Path field above.")
    if image is None:
        raise gr.Error("⚠️ Please upload an image.")

    lr_size = _cfg.get("image_size", 256)
    hr_size = _cfg.get("sr_size",    512)

    pil_input = Image.fromarray(image).convert("RGB")
    x_lr = preprocess(pil_input, lr_size)

    with torch.no_grad():
        recon_256_tensor = _vae.reconstruct(x_lr)
        recon_512_tensor = _sr(recon_256_tensor)

    recon_256_pil = tensor_to_pil(recon_256_tensor)
    recon_512_pil = tensor_to_pil(recon_512_tensor)

    # Apply strength-blended styles
    alpha = style_strength / 100.0
    styled_pils = [
        apply_with_strength(recon_512_pil, i, alpha)
        for i in range(len(STYLE_NAMES))
    ]

    orig_hr = pil_input.resize((hr_size, hr_size), Image.LANCZOS)

    return (
        np.array(orig_hr),
        np.array(recon_256_pil),
        np.array(recon_512_pil),
        *[np.array(s) for s in styled_pils],
    )


def latent_walk(image, style_a_idx, style_b_idx, steps):
    if _vae is None or _sr is None:
        raise gr.Error("⚠️ Please load a model first.")
    if image is None:
        raise gr.Error("⚠️ Please upload an image.")

    lr_size = _cfg.get("image_size", 256)
    pil_input = Image.fromarray(image).convert("RGB")
    x_lr = preprocess(pil_input, lr_size)

    steps = int(steps)
    walk_frames = []

    with torch.no_grad():
        mu, logvar, skips = _vae.encode(x_lr)
        direction = F.normalize(torch.randn_like(mu), dim=-1)

        for i, t_val in enumerate(torch.linspace(-2.5, 2.5, steps)):
            z_walk  = mu + t_val.item() * direction
            rec_256 = _vae.decode(z_walk, skips)
            rec_512 = _sr(rec_256)
            rec_pil = tensor_to_pil(rec_512)
            alpha   = i / (steps - 1)
            sa  = np.array(StyleTransformBank.apply(rec_pil, style_a_idx), dtype=np.float32)
            sb  = np.array(StyleTransformBank.apply(rec_pil, style_b_idx), dtype=np.float32)
            blended = Image.fromarray((sa*(1-alpha) + sb*alpha).clip(0,255).astype(np.uint8))
            walk_frames.append(np.array(blended))

    return walk_frames


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — Warm Ivory · Premium Editorial Light Theme
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

/* ── Design tokens ───────────────────────────────────────────────── */
:root {
    --cream:       #faf7f2;
    --parchment:   #f3ede3;
    --warm-white:  #fefcf8;
    --sand:        #e8dece;
    --sand-dark:   #d4c8b8;
    --ink:         #2c2520;
    --ink-mid:     #5a4f47;
    --ink-light:   #8c7f75;
    --ink-faint:   #b8ada5;
    --terracotta:  #c4714a;
    --terracotta-l:#e8956d;
    --sage:        #6b8f71;
    --sage-l:      #9ab89f;
    --gold:        #b8943a;
    --gold-l:      #d4b05a;
    --dusty-rose:  #c47a7a;
    --slate-blue:  #6a7d96;
    --radius:      10px;
    --radius-lg:   16px;
    --shadow-soft: 0 2px 16px rgba(44,37,32,0.08);
    --shadow-card: 0 4px 24px rgba(44,37,32,0.10);
    --border:      #ddd5c8;
    --border-light:#e8e0d6;
}

/* ── Base ────────────────────────────────────────────────────────── */
body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--ink) !important;
}

/* Subtle linen texture overlay */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

/* ── Hero ────────────────────────────────────────────────────────── */
.shuk-hero {
    text-align: center;
    padding: 56px 24px 40px;
    background: linear-gradient(160deg, var(--warm-white) 0%, var(--parchment) 100%);
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

/* Warm glow blob behind title */
.shuk-hero::before {
    content: '';
    position: absolute;
    top: -40px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 260px;
    background: radial-gradient(ellipse, rgba(196,113,74,0.10) 0%, rgba(184,148,58,0.06) 50%, transparent 80%);
    pointer-events: none;
    animation: warmPulse 6s ease-in-out infinite;
}

@keyframes warmPulse {
    0%, 100% { opacity: 0.7; transform: translateX(-50%) scale(1); }
    50%       { opacity: 1;   transform: translateX(-50%) scale(1.08); }
}

.shuk-hero h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: clamp(2.6rem, 5.5vw, 4rem) !important;
    font-weight: 400 !important;
    font-style: italic;
    color: var(--ink) !important;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0 0 10px !important;
}

.shuk-hero h1 span {
    color: var(--terracotta) !important;
}

.shuk-hero .tagline {
    font-size: 0.95rem;
    color: var(--ink-light);
    letter-spacing: 0.02em;
    font-weight: 400;
    margin-top: 4px;
}

.shuk-hero .badge-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 20px;
}

.badge {
    background: var(--warm-white);
    border: 1px solid var(--sand-dark);
    border-radius: 100px;
    padding: 4px 13px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: var(--ink-mid);
    letter-spacing: 0.04em;
    box-shadow: 0 1px 4px rgba(44,37,32,0.06);
}

/* ── Pipeline bar ────────────────────────────────────────────────── */
.pipeline-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 4px;
    padding: 18px 0 0;
}
.pipe-step {
    background: var(--warm-white);
    border: 1px solid var(--sand-dark);
    border-radius: 7px;
    padding: 5px 12px;
    font-size: 0.70rem;
    font-family: 'DM Mono', monospace;
    color: var(--ink-mid);
    white-space: nowrap;
    box-shadow: 0 1px 3px rgba(44,37,32,0.06);
}
.pipe-arrow { color: var(--ink-faint); font-size: 1rem; margin: 0 1px; }

/* ── Section labels ──────────────────────────────────────────────── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-light);
}

/* ── Buttons ─────────────────────────────────────────────────────── */
button.primary, .gr-button-primary {
    background: var(--terracotta) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 11px 26px !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 3px 14px rgba(196,113,74,0.35) !important;
    letter-spacing: 0.01em;
}
button.primary:hover {
    background: var(--terracotta-l) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(196,113,74,0.45) !important;
}
button.primary:active { transform: translateY(0) !important; }

button.secondary, .gr-button-secondary {
    background: var(--warm-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--ink-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: border-color 0.2s, color 0.2s, box-shadow 0.2s !important;
}
button.secondary:hover {
    border-color: var(--terracotta) !important;
    color: var(--terracotta) !important;
    box-shadow: 0 2px 10px rgba(196,113,74,0.15) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────── */
input[type="text"], textarea, .gr-textbox input {
    background: var(--warm-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.83rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 3px rgba(196,113,74,0.12) !important;
    outline: none !important;
}

/* ── Textbox status ──────────────────────────────────────────────── */
.gr-textbox textarea {
    background: var(--warm-white) !important;
    color: var(--sage) !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.7 !important;
    border-radius: var(--radius) !important;
}

/* ── Sliders ─────────────────────────────────────────────────────── */
input[type="range"] { accent-color: var(--terracotta) !important; }

/* ── Block / card overrides ──────────────────────────────────────── */
.block, .gr-block {
    background: var(--warm-white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-soft) !important;
}

/* ── Image component ─────────────────────────────────────────────── */
.gr-image, [data-testid="image"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--parchment) !important;
    overflow: hidden;
}

/* ── Tab nav ─────────────────────────────────────────────────────── */
.tab-nav {
    border-bottom: 1px solid var(--border) !important;
    background: var(--parchment) !important;
}
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--ink-light) !important;
    border: none !important;
    background: transparent !important;
    padding: 11px 20px !important;
    transition: color 0.2s !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected, .tab-nav button[aria-selected="true"] {
    color: var(--terracotta) !important;
    border-bottom: 2px solid var(--terracotta) !important;
    background: rgba(196,113,74,0.04) !important;
    font-weight: 600 !important;
}
.tab-nav button:hover { color: var(--ink-mid) !important; }

/* ── Labels ──────────────────────────────────────────────────────── */
label, .gr-label, span.svelte-1gfkn6j {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--ink-mid) !important;
    letter-spacing: 0.01em;
}

/* ── Dropdown ────────────────────────────────────────────────────── */
.gr-dropdown select, select {
    background: var(--warm-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--parchment); }
::-webkit-scrollbar-thumb { background: var(--sand-dark); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--terracotta); }

/* ── Divider ─────────────────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid var(--border-light);
    margin: 8px 0;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 28px;
    color: #3ba729 !important;
    font-size: 0.88rem;
    border-top: 1px solid var(--border-light);
    margin-top: 40px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.03em;
    background: var(--parchment);
}
.footer span { color: var(--terracotta); font-style: italic; }

/* ── Gallery ─────────────────────────────────────────────────────── */
.gr-gallery {
    background: var(--parchment) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Accordion ───────────────────────────────────────────────────── */
details, .gr-accordion {
    background: var(--warm-white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
}

.load-model-btn{
    margin-top: 24px !important;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="yellow",
            neutral_hue="stone",
            font=["DM Sans", "sans-serif"],
        ).set(
            body_background_fill="#faf7f2",
            body_text_color="#2c2520",
            block_background_fill="#fefcf8",
            block_border_color="#ddd5c8",
            block_label_text_color="#8c7f75",
            input_background_fill="#fefcf8",
            input_border_color="#ddd5c8",
            button_primary_background_fill="#c4714a",
            button_primary_text_color="#ffffff",
            button_secondary_background_fill="#fefcf8",
            button_secondary_border_color="#ddd5c8",
            button_secondary_text_color="#5a4f47",
        ),
        title="ShukGEN Advanced",
    ) as demo:

        # ── HERO ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="shuk-hero">
            <h1><span>Shuk</span>GEN Advanced</h1>
            <p class="tagline">Identity-Preserving Face Reconstruction &nbsp;·&nbsp; HD Super Resolution &nbsp;·&nbsp; 8 Style Variants</p>
            <div class="badge-row">
                <span class="badge">FaceVAE</span>
                <span class="badge">SRHead 512px</span>
                <span class="badge">EMA Weights</span>
                <span class="badge">Identity Loss</span>
                <span class="badge">FFT Sharpness</span>
                <span class="badge">8 Style Transforms</span>
            </div>
            <div class="pipeline-bar">
                <span class="pipe-step">Input 256×256</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-step">FaceVAE Encode</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-step">z · latent 512d</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-step">FaceVAE Decode</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-step">SRHead 512×512</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-step">StyleTransformBank</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-step">8 HD Variants ✦</span>
            </div>
        </div>
        """)

        # ── MODEL LOADER ──────────────────────────────────────────────────────
        gr.HTML('<div class="section-label" style="margin: 24px 0 10px;">Model Configuration</div>')
        with gr.Row():
            with gr.Column(scale=4):
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="e.g.  shukgen_v4_final.pth   or   checkpoints_shukgen_v4/best.pth",
                    value="shukgen_v4_final.pth",
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("Load Model", variant="primary", elem_classes="load-model-btn")

        model_status = gr.Textbox(
            label="Status",
            interactive=False,
            placeholder="Load a model to get started...",
            lines=4,
        )
        load_btn.click(fn=load_model, inputs=model_path, outputs=model_status)

        gr.HTML('<hr class="divider" style="margin: 28px 0;">')

        # ── TABS ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── TAB 1: Main Inference ─────────────────────────────────────────
            with gr.TabItem("  Reconstruct & Style  "):

                gr.HTML('<div class="section-label" style="margin-top:20px;">Input & Controls</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(
                            label="Upload Face Image",
                            type="numpy",
                            height=310,
                            image_mode="RGB",
                        )
                        style_strength = gr.Slider(
                            minimum=0, maximum=100, value=100, step=1,
                            label="Style Strength (%)",
                            info="0 = pure HD reconstruction   ·   100 = full style applied",
                        )
                        run_btn = gr.Button("Generate All Outputs  ✦", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.HTML('<div class="section-label">Core Outputs</div>')
                        with gr.Row():
                            out_orig = gr.Image(label="Original  (HR Reference)", height=200)
                            out_256  = gr.Image(label="VAE Recon  ·  256×256",    height=200)
                            out_512  = gr.Image(label="HD Recon ✦  ·  512×512",   height=200)

                gr.HTML('<div class="section-label" style="margin-top:28px;">8 Style Variants  ·  HD 512×512</div>')

                style_label_colors = [
                    ("✦  Youthful",          "#6b8f71"),
                    ("⌛  Aged / Mature",     "#8c6a4a"),
                    ("◑  Dramatic Light",    "#6a5a8c"),
                    ("◌  Soft Glow",         "#b8943a"),
                    ("◈  Intense / Bold",    "#c4714a"),
                    ("☀  Warm Golden Hour",  "#b87c3a"),
                    ("◐  Cool / Moody",      "#4a7a8c"),
                    ("✏  Sketch / Artistic", "#4a7a5a"),
                ]

                style_outputs = []
                for row in range(2):
                    with gr.Row():
                        for col in range(4):
                            idx = row * 4 + col
                            label, color = style_label_colors[idx]
                            img = gr.Image(
                                label=label,
                                height=175,
                                elem_id=f"style_{idx}",
                            )
                            style_outputs.append(img)

                run_btn.click(
                    fn=run_inference,
                    inputs=[input_img, style_strength],
                    outputs=[out_orig, out_256, out_512] + style_outputs,
                )

            # ── TAB 2: Latent Walk ────────────────────────────────────────────
            with gr.TabItem("  Latent Space Walk  "):

                gr.HTML("""
                <div style="padding: 16px 0 8px; color: #8c7f75; font-size: 0.88rem; line-height: 1.7;
                            font-family: 'DM Sans', sans-serif; border-bottom: 1px solid #e8e0d6; margin-bottom: 20px;">
                    Explore the latent space of your image. The model encodes the face to a 512-d vector,
                    then walks along a random direction while gradually blending between two chosen styles.
                </div>
                """)

                with gr.Row():
                    walk_input = gr.Image(label="Input Image", type="numpy", height=270)
                    with gr.Column():
                        style_a = gr.Dropdown(
                            choices=[(f"{STYLE_EMOJIS[i]}  {n}", i) for i, n in enumerate(STYLE_NAMES)],
                            value=2, label="Style A  (start)",
                        )
                        style_b = gr.Dropdown(
                            choices=[(f"{STYLE_EMOJIS[i]}  {n}", i) for i, n in enumerate(STYLE_NAMES)],
                            value=6, label="Style B  (end)",
                        )
                        walk_steps = gr.Slider(4, 12, value=8, step=1, label="Walk Steps")
                        walk_btn = gr.Button("Generate Latent Walk", variant="primary")

                gr.HTML('<div class="section-label" style="margin-top:22px;">Walk Frames  ·  Style A → Style B</div>')
                walk_gallery = gr.Gallery(
                    label="",
                    columns=8,
                    height=220,
                )

                walk_btn.click(
                    fn=latent_walk,
                    inputs=[walk_input, style_a, style_b, walk_steps],
                    outputs=walk_gallery,
                )

            # ── TAB 3: Style Strength Explorer ───────────────────────────────
            with gr.TabItem("  Style Strength  "):

                gr.HTML("""
                <div style="padding: 16px 0 8px; color: #8c7f75; font-size: 0.88rem; line-height: 1.7;
                            font-family: 'DM Sans', sans-serif; border-bottom: 1px solid #e8e0d6; margin-bottom: 20px;">
                    Choose a single style and preview how it blends from pure HD reconstruction (0%) to full effect (100%)
                    across five evenly-spaced steps.
                </div>
                """)

                with gr.Row():
                    exp_input = gr.Image(label="Input Image", type="numpy", height=270)
                    with gr.Column():
                        exp_style = gr.Dropdown(
                            choices=[(f"{STYLE_EMOJIS[i]}  {n}", i) for i, n in enumerate(STYLE_NAMES)],
                            value=0, label="Choose Style",
                        )
                        exp_btn = gr.Button("Explore Strength", variant="primary")

                gr.HTML('<div class="section-label" style="margin-top:22px;">Blend Steps  ·  0% → 25% → 50% → 75% → 100%</div>')

                exp_outs = []
                with gr.Row():
                    for a in [0, 25, 50, 75, 100]:
                        img = gr.Image(label=f"α = {a}%", height=195)
                        exp_outs.append(img)

                def explore_strength(image, style_idx):
                    if _vae is None or _sr is None:
                        raise gr.Error("Load model first.")
                    if image is None:
                        raise gr.Error("Upload an image first.")
                    lr_size = _cfg.get("image_size", 256)
                    pil_input = Image.fromarray(image).convert("RGB")
                    x_lr = preprocess(pil_input, lr_size)
                    with torch.no_grad():
                        recon_256 = _vae.reconstruct(x_lr)
                        recon_512 = _sr(recon_256)
                    hd_pil = tensor_to_pil(recon_512)
                    results = []
                    for a in [0.0, 0.25, 0.50, 0.75, 1.0]:
                        results.append(np.array(apply_with_strength(hd_pil, int(style_idx), a)))
                    return results

                exp_btn.click(fn=explore_strength, inputs=[exp_input, exp_style], outputs=exp_outs)

            # ── TAB 4: Architecture ───────────────────────────────────────────
            with gr.TabItem("  Architecture  "):
                gr.HTML("""
                <div style="padding: 28px 0; max-width: 820px; margin: 0 auto; line-height: 1.8;
                            font-family: 'DM Sans', sans-serif;">

                    <h2 style="font-family: 'DM Serif Display', serif; font-size: 1.7rem; font-weight: 400;
                                font-style: italic; color: #2c2520; margin: 0 0 6px;">
                        ShukGEN Advanced Architecture
                    </h2>
                    <p style="color: #8c7f75; font-size: 0.88rem; margin: 0 0 28px;">
                        Identity-Preserving VAE · Super Resolution · Deterministic Style Transforms
                    </p>

                    <table style="width:100%; border-collapse:collapse; font-size:0.84rem;">
                        <thead>
                        <tr style="border-bottom: 2px solid #ddd5c8;">
                            <th style="text-align:left; padding:10px 14px; color:#5a4f47;
                                       font-family:'DM Sans',sans-serif; font-weight:600;
                                       font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;">
                                Component
                            </th>
                            <th style="text-align:left; padding:10px 14px; color:#5a4f47;
                                       font-family:'DM Sans',sans-serif; font-weight:600;
                                       font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;">
                                What It Does
                            </th>
                        </tr>
                        </thead>
                        <tbody>
                        <tr style="border-bottom:1px solid #e8e0d6; background:#fefcf8;">
                            <td style="padding:11px 14px; color:#c4714a; font-family:'DM Mono',monospace; font-size:0.82rem; white-space:nowrap;">FaceVAE</td>
                            <td style="padding:11px 14px; color:#5a4f47;">U-Net VAE with multi-head attention bottleneck. Encodes face → 512-d latent → reconstructs at 256×256.</td>
                        </tr>
                        <tr style="border-bottom:1px solid #e8e0d6;">
                            <td style="padding:11px 14px; color:#c4714a; font-family:'DM Mono',monospace; font-size:0.82rem;">Identity Loss</td>
                            <td style="padding:11px 14px; color:#5a4f47;">InceptionResNetV2 (timm) measures cosine similarity of face embeddings — preserves <em>who</em> the person is.</td>
                        </tr>
                        <tr style="border-bottom:1px solid #e8e0d6; background:#fefcf8;">
                            <td style="padding:11px 14px; color:#c4714a; font-family:'DM Mono',monospace; font-size:0.82rem;">FFT Loss</td>
                            <td style="padding:11px 14px; color:#5a4f47;">Penalises missing high-frequency detail — hair strands, skin pores, sharp edges.</td>
                        </tr>
                        <tr style="border-bottom:1px solid #e8e0d6;">
                            <td style="padding:11px 14px; color:#c4714a; font-family:'DM Mono',monospace; font-size:0.82rem;">SSIM Loss</td>
                            <td style="padding:11px 14px; color:#5a4f47;">Structural Similarity Index — evaluates quality the way human eyes do.</td>
                        </tr>
                        <tr style="border-bottom:1px solid #e8e0d6; background:#fefcf8;">
                            <td style="padding:11px 14px; color:#c4714a; font-family:'DM Mono',monospace; font-size:0.82rem;">SRHead</td>
                            <td style="padding:11px 14px; color:#5a4f47;">SRRESNET-style: 10 residual blocks + PixelShuffle 2× — 256×256 → 512×512 HD, no checkerboard artifacts.</td>
                        </tr>
                        <tr style="border-bottom:1px solid #e8e0d6;">
                            <td style="padding:11px 14px; color:#c4714a; font-family:'DM Mono',monospace; font-size:0.82rem;">EMA Weights</td>
                            <td style="padding:11px 14px; color:#5a4f47;">Exponential Moving Average (decay 0.999) — inference uses averaged weights for smoother, sharper outputs.</td>
                        </tr>
                        <tr style="background:#fefcf8;">
                            <td style="padding:11px 14px; color:#c4714a; font-family:'DM Mono',monospace; font-size:0.82rem;">StyleTransformBank</td>
                            <td style="padding:11px 14px; color:#5a4f47;">8 deterministic pixel-space transforms on HD output: Youthful, Aged, Dramatic Light, Soft Glow, Intense, Golden Hour, Cool/Moody, Sketch.</td>
                        </tr>
                        </tbody>
                    </table>

                    <h3 style="font-family:'DM Serif Display',serif; font-size:1.1rem; font-weight:400;
                                font-style:italic; margin:32px 0 14px; color:#2c2520;">
                        Training Losses
                    </h3>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">
                        <div style="background:#fefcf8; border:1px solid #ddd5c8; border-radius:10px;
                                    padding:18px; border-left: 3px solid #c4714a;">
                            <div style="color:#c4714a; font-weight:600; font-size:0.80rem;
                                        letter-spacing:0.06em; text-transform:uppercase; margin-bottom:10px;">
                                VAE Losses  ·  256×256
                            </div>
                            <div style="color:#5a4f47; font-size:0.84rem; line-height:2;">
                                L1 Pixel &nbsp;·&nbsp; Perceptual VGG (4 layers)<br>
                                SSIM &nbsp;·&nbsp; Identity (FaceNet) &nbsp;·&nbsp; KL Divergence
                            </div>
                        </div>
                        <div style="background:#fefcf8; border:1px solid #ddd5c8; border-radius:10px;
                                    padding:18px; border-left: 3px solid #6b8f71;">
                            <div style="color:#6b8f71; font-weight:600; font-size:0.80rem;
                                        letter-spacing:0.06em; text-transform:uppercase; margin-bottom:10px;">
                                SR Losses  ·  512×512
                            </div>
                            <div style="color:#5a4f47; font-size:0.84rem; line-height:2;">
                                L1 Pixel &nbsp;·&nbsp; FFT Sharpness &nbsp;·&nbsp; Perceptual VGG
                            </div>
                        </div>
                    </div>
                </div>
                """)

        # ── FOOTER ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer">
            <span>ShukGEN Advanced</span>
            &nbsp;·&nbsp; Identity-Preserving HD Face Reconstruction
            &nbsp;·&nbsp; VAE + SRHead + StyleTransformBank
            &nbsp;·&nbsp; PyTorch &amp; Gradio
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=True
    )