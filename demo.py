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

from diffusion_eet import DEET, DEETMode


@enum.unique
class DEETModeP(enum.StrEnum):
    PREDICTION = enum.auto()
    "deet(prediction, prediction⁻¹)"
    UNPREDICT = enum.auto()
    "deet(prediction, sample)"


with torch.inference_mode():
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float16
    url: str = "Lykon/dreamshaper-8"
    seed = torch.Generator("cpu").manual_seed(0)
    steps: int = 20
    cfg: float = 8
    prompt: str = "masterpiece, fantasy artwork, tabby kitten, rainbow flowers"
    negative: str = "blurry, noisy, cropped"

    use_context_manager: bool = True

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
    def sample_deet(mode: DEETMode | DEETModeP, config: DEET) -> Image.Image:
        sample: torch.Tensor = torch.randn([1, 4, 80, 80], generator=seed.clone_state()).to(dtype=dtype, device=device)
        previous: list[structured.SKSamples[torch.Tensor]] = []

        prev_output: torch.Tensor | None = None
        prev_sample: torch.Tensor | None = None
        prev_prediction: torch.Tensor | None = None

        if use_context_manager and mode in DEETMode:
            ctx = config.hook_module(model, mode=mode)  # type: ignore
            ctx.__enter__()
            mode = ""  # type: ignore

        for n, (timestep, sigma) in enumerate(tqdm(scheudle_np.tolist())):
            if mode == DEETMode.INPUT and prev_sample is not None:
                new_sample = config(sample, prev_sample)
                prev_sample, sample = sample, new_sample
            else:
                prev_sample = sample

            output = model(sample.repeat((2, 1, 1, 1)), timestep, text_embeds).sample

            if mode == DEETMode.OUTPUT and prev_output is not None:
                new_output = config(output, prev_output)
                prev_output, output = output, new_output
            elif mode == DEETMode.BACKWARD:
                output = config(output, sample)
            else:
                prev_output = output

            conditioned, unconditioned = output.chunk(2)
            guided: torch.Tensor = conditioned + (cfg - 1) * (conditioned - unconditioned)
            prediction = predictor(sample, guided, sigma, schedule.sigma_transform)

            if mode == DEETModeP.PREDICTION and prev_prediction is not None:
                new_prediction = config(prediction, prev_prediction)
                prev_prediction, prediction = prediction, new_prediction
            elif mode == DEETModeP.UNPREDICT:
                prediction = config(prediction, sample)
            else:
                prev_prediction = prediction

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

        if use_context_manager and mode in DEETMode:
            ctx.__exit__()  # type: ignore

        image: torch.Tensor = image_encoder.decode(sample / image_encoder.config.scaling_factor).sample[0]  # type: ignore
        srgb = Image.fromarray(
            ((image + 1) * (255 / 2)).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
        )
        return srgb

    with cf.ThreadPoolExecutor(1 if use_context_manager else 4) as pool:
        configs: tuple[DEET, ...] = (
            DEET(0, 1, False),
            DEET(1.5, 1, False),
            DEET(2, 2, True),
            DEET(3, 0.5, False),
            DEET(-2, -1, False),
        )
        modes: tuple[DEETMode | DEETModeP, ...] = tuple(DEETMode) + tuple(DEETModeP)
        deets: tuple[tuple[DEETMode | DEETModeP, DEET], ...] = tuple((m, d) for d in configs for m in modes)

        images: list[cf.Future[Image.Image]] = [pool.submit(sample_deet, mode, dev) for (mode, dev) in deets]

        width, height = images[0].result().size
        font_size = width // 15
        stroke_width = max(1, font_size // 10)
        pad = font_size // 4
        fill = (255, 255, 255)
        stroke_fill = (0, 0, 0)

        canvas = Image.new("RGB", (width * len(modes), height * len(configs)), (0, 0, 0))
        for (m, d), image in zip(deets, tqdm(images, desc="Total"), strict=True):
            x, y = (modes.index(m) * width, configs.index(d) * height)
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
                    f"DEET({d.deviation}, {d.power}, {d.invert})",
                    fill=fill,
                    stroke_fill=stroke_fill,
                    font_size=font_size,
                    stroke_width=stroke_width,
                )

        canvas.save("demo.png")
