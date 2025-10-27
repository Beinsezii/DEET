import contextlib
import dataclasses
import enum
import inspect
import math
from collections.abc import Generator, MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


def spowf[T: torch.Tensor | float](x: T, f: float) -> T:
    """Computes x^f in absolute then re-applies the sign to stabilize chaotic inputs.
    More computationally expensive than plain `math.pow`"""
    return abs(x) ** f * (-1 * (x < 0) | 1)  # type: ignore


def deet(
    low_noise: torch.Tensor,
    high_noise: torch.Tensor,
    deviation: float = 1,
    power: float = 1,
    invert: bool = False,
    equalize: bool = False,
) -> torch.Tensor:
    if deviation > 0:
        # noise values < deviation will get masked
        mask = low_noise.std() * (math.sqrt(abs(deviation) + 1) - 1) - abs(low_noise)
    elif deviation < 0:
        # noise values > deviation will get masked
        mask = abs(low_noise) - low_noise.std() / (math.sqrt(abs(deviation) + 1) - 1)
    else:
        return low_noise

    mask = spowf(mask, power).clamp(0, 1)

    deeted = low_noise * (1 - mask) + high_noise * mask * (invert * -1 | 1)

    if equalize:
        deeted = deeted * (low_noise.std() / deeted.std())

    return deeted


@enum.unique
class DEETMode(enum.StrEnum):
    INPUT = enum.auto()
    "deet(input, input⁻¹)"
    OUTPUT = enum.auto()
    "deet(output, output⁻¹)"
    BACKWARD = enum.auto()
    "deet(output, input)"


@dataclasses.dataclass(frozen=True)
class DEET:
    deviation: float = 1
    power: float = 1
    invert: bool = False
    equalize: bool = False
    compute_scale: torch.dtype | None = torch.float64

    def __call__(self, low_noise: torch.Tensor, high_noise: torch.Tensor) -> torch.Tensor:
        if self.compute_scale is None:
            return deet(
                low_noise,
                high_noise,
                self.deviation,
                self.power,
                self.invert,
                self.equalize,
            )
        else:
            return deet(
                low_noise.to(dtype=self.compute_scale),
                high_noise.to(dtype=self.compute_scale),
                self.deviation,
                self.power,
                self.invert,
                self.equalize,
            ).to(dtype=low_noise.dtype)

    @contextlib.contextmanager
    def hook_module(
        self,
        module: torch.nn.Module,
        mode: DEETMode,
        enable: bool = True,
        residual_deet: bool = False,
        fwd_arg: int | str = 0,
    ) -> Generator[None, None, None]:
        if not enable:
            yield
            return

        deet_high_noise: torch.Tensor | None = None
        handles: list[RemovableHandle] = []

        if isinstance(fwd_arg, int):
            arg_index, arg_key = fwd_arg, list(inspect.signature(module.forward).parameters.keys())[fwd_arg]
        else:
            arg_key, arg_index = fwd_arg, list(inspect.signature(module.forward).parameters.keys()).index(fwd_arg)

        def deet_input(_: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            nonlocal deet_high_noise

            if len(args) > arg_index:
                sample: torch.Tensor = args[arg_index]
            else:
                sample = kwargs[arg_key]

            if deet_high_noise is not None:
                new_sample = self(sample, deet_high_noise)

                # CoW for hygeine
                if len(args) > arg_index:
                    args = (*args[:arg_index], new_sample, *args[arg_index + 1 :])
                else:
                    kwargs = kwargs | {arg_key: new_sample}

                if residual_deet:
                    sample = new_sample

            deet_high_noise = sample

            return args, kwargs

        def deet_output_backward[T](_: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: T) -> T:
            nonlocal deet_high_noise

            if isinstance(output, torch.Tensor):
                output_tensor: torch.Tensor = output
            elif isinstance(output, MutableMapping):
                output_key, output_tensor = next((k, v) for k, v in output.items() if isinstance(v, torch.Tensor))
            elif isinstance(output, MutableSequence):
                output_index, output_tensor = next((n, o) for n, o in enumerate(output) if isinstance(o, torch.Tensor))
            else:
                msg = f"Output type {type(output).__name__} not supported by DEET hooks"
                raise NotImplementedError(msg)

            if mode == DEETMode.OUTPUT:
                if deet_high_noise is not None:
                    new_output_tensor = self(output_tensor, deet_high_noise)

                    # Not CoW to keep the output typing
                    if isinstance(output, torch.Tensor):
                        output = new_output_tensor  # type: ignore
                    elif isinstance(output, MutableMapping):
                        output[output_key] = new_output_tensor  # type: ignore # Guaranteed bound
                    elif isinstance(output, MutableSequence):
                        output[output_index] = new_output_tensor  # type: ignore # Guaranteed bound
                    else:
                        raise NotImplementedError

                    if residual_deet:
                        output_tensor = new_output_tensor

                deet_high_noise = output_tensor

            elif mode == DEETMode.BACKWARD:
                if len(args) > arg_index:
                    sample: torch.Tensor = args[arg_index]
                else:
                    sample = kwargs[arg_key]

                new_output_tensor = self(output_tensor.clone(), sample.clone())

                if isinstance(output, torch.Tensor):
                    output = new_output_tensor  # type: ignore
                elif isinstance(output, MutableMapping):
                    output[output_key] = new_output_tensor  # type: ignore # Guaranteed bound
                elif isinstance(output, MutableSequence):
                    output[output_index] = new_output_tensor  # type: ignore # Guaranteed bound
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            return output

        try:
            if mode == DEETMode.INPUT:
                handles.append(module.register_forward_pre_hook(deet_input, with_kwargs=True, prepend=True))
            elif mode in (DEETMode.OUTPUT, DEETMode.BACKWARD):
                handles.append(module.register_forward_hook(deet_output_backward, with_kwargs=True))
            yield
        finally:
            for h in handles:
                h.remove()
