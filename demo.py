#! /usr/bin/env python

import concurrent.futures as cf
import enum

import skrample.common
import skrample.sampling as structured
import skrample.scheduling as scheduling
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers.models.clip import CLIPTextModel, CLIPTokenizer

from diffusion_eet import deet_upcast as deet


@enum.unique
class DeetMode(enum.StrEnum):
    SAMPLE = enum.auto()
    "deet(sample, sample⁻¹)"
    PREDICTION = enum.auto()
    "deet(prediction, prediction⁻¹)"
    UNPREDICT = enum.auto()
    "deet(prediction, sample)"
    RENOISE = enum.auto()
    "deet(output, sample)"


with torch.inference_mode():
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float16
    url: str = "Lykon/dreamshaper-8"
    seed = torch.Generator("cpu").manual_seed(0)
    steps: int = 20
    cfg: float = 8
    prompt: str = "masterpiece, fantasy artwork, tabby kitten, rainbow flowers"
    negative: str = "blurry, noisy, cropped"

    schedule: scheduling.SkrampleSchedule = scheduling.Karras(scheduling.Scaled())
    sampler: structured.SkrampleSampler = structured.DPM(order=2)
    predictor: skrample.common.Predictor = skrample.common.predict_epsilon

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(url, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        url, subfolder="text_encoder", device_map=device, torch_dtype=dtype
    )
    model: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(  # type: ignore
        url, subfolder="unet", device_map=device, torch_dtype=dtype
    )
    model.fuse_qkv_projections()
    image_encoder: AutoencoderKL = AutoencoderKL.from_pretrained(  # type: ignore
        url, subfolder="vae", device_map=device, torch_dtype=dtype
    )

    text_embeds: torch.Tensor = text_encoder(
        tokenizer(
            [prompt, negative],
            padding="max_length",
            return_tensors="pt",
        ).input_ids.to(device=device)
    ).last_hidden_state

    scheudle_np = schedule.schedule(steps)

    @torch.inference_mode()
    def sample_deet(mode: DeetMode, deviation: float) -> Image.Image:
        sample: torch.Tensor = torch.randn([1, 4, 80, 80], generator=seed.clone_state()).to(dtype=dtype, device=device)
        previous: list[structured.SKSamples[torch.Tensor]] = []
        for n, (timestep, sigma) in enumerate(tqdm(scheudle_np.tolist())):
            if mode == DeetMode.SAMPLE and previous:
                sample = deet(sample, previous[-1].sample, deviation)

            output = model(sample.repeat((2, 1, 1, 1)), timestep, text_embeds).sample

            if mode == DeetMode.RENOISE:
                output = deet(output, sample, deviation)

            conditioned, unconditioned = output.chunk(2)
            guided: torch.Tensor = conditioned + (cfg - 1) * (conditioned - unconditioned)
            prediction = predictor(sample, guided, sigma, schedule.sigma_transform)

            if mode == DeetMode.PREDICTION and previous:
                prediction = deet(prediction, previous[-1].prediction, deviation)
            elif mode == DeetMode.UNPREDICT:
                prediction = deet(prediction, sample, deviation)

            sampler_output = sampler.sample(
                sample=sample,
                prediction=prediction,
                step=n,
                sigma_schedule=scheudle_np[:, 1],
                sigma_transform=schedule.sigma_transform,
                previous=tuple(previous),
            )

            previous.append(sampler_output)
            sample = sampler_output.final

        image: torch.Tensor = image_encoder.decode(sample / image_encoder.config.scaling_factor).sample[0]  # type: ignore
        srgb = Image.fromarray(
            ((image + 1) * (255 / 2)).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
        )
        return srgb

    with cf.ThreadPoolExecutor(4) as pool:
        devs: tuple[float, ...] = (2, 1.1, 1, 0.9, 0.5, 0.1, 0, -0.1, -0.5, -0.9, -1, -1.1, -2)
        # devs: tuple[float, ...] = (0, 0.5, 1, 1.5, 2)
        modes: tuple[DeetMode, ...] = (DeetMode.SAMPLE, DeetMode.RENOISE, DeetMode.PREDICTION, DeetMode.UNPREDICT)
        deets: tuple[tuple[DeetMode, float], ...] = tuple((m, d) for d in devs for m in modes)

        images: list[cf.Future[Image.Image]] = [pool.submit(sample_deet, mode, dev) for (mode, dev) in deets]

        width, height = images[0].result().size
        font_size = width // 15
        stroke_width = max(1, font_size // 10)
        pad = font_size // 4
        fill = (255, 255, 255)
        stroke_fill = (0, 0, 0)

        canvas = Image.new("RGB", (width * len(modes), height * len(devs)), (0, 0, 0))
        for (m, d), image in zip(deets, tqdm(images, desc="Total"), strict=True):
            x, y = (modes.index(m) * width, devs.index(d) * height)
            canvas.paste(image.result(), (x, y))

            # X labels
            if y == 0:
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (x + pad, y),
                    m,
                    fill=fill,
                    stroke_fill=stroke_fill,
                    font_size=font_size,
                    stroke_width=stroke_width,
                )

            # Y labels
            if x == 0:
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (pad, y + height - font_size - pad),
                    f"deviation: {d}",
                    fill=fill,
                    stroke_fill=stroke_fill,
                    font_size=font_size,
                    stroke_width=stroke_width,
                )

        canvas.save("demo.png")
