from typing import TYPE_CHECKING, Any

from comfy.patcher_extension import WrappersMP
from comfy_api.latest import ComfyExtension, io

from .src import diffusion_eet as deet

if TYPE_CHECKING:
    import torch

    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor


class DEET(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DEET",
            display_name="DEET",
            category="model_patches",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(id="mode", options=deet.DEETMode),
                io.Float.Input("deviation", default=1.5, min=-1e32, max=1e32, step=0.1, round=0.01),
                io.Float.Input("power", default=1, min=-1e32, max=1e32, step=0.1, round=0.01),
                io.Boolean.Input("invert", default=False),
                io.Boolean.Input("equalize", default=False),
                io.Boolean.Input("residual", default=False),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model: "ModelPatcher",
        mode: deet.DEETMode,
        deviation: float,
        power: float,
        invert: bool,
        equalize: bool,
        residual: bool,
    ) -> io.NodeOutput:  # type: ignore
        cloned = model.clone()
        assert model.clone_has_same_weights(cloned)

        deet_struct = deet.DEET(deviation=deviation, power=power, invert=invert, equalize=equalize)
        x_prev: torch.Tensor | None = None
        o_prev: torch.Tensor | None = None

        def wrapper(ex: "WrapperExecutor", x: "torch.Tensor", *args: Any, **kwargs: Any) -> "torch.Tensor":  # noqa: ANN401
            nonlocal x_prev, o_prev

            if mode == deet.DEETMode.INPUT and x_prev is not None:
                new_x = deet_struct(x, x_prev)
                x, x_prev = new_x, x
                if residual:
                    x_prev = x
            else:
                x_prev = x

            output = ex.original(x, *args, **kwargs)

            if mode == deet.DEETMode.OUTPUT and o_prev is not None:
                new_output = deet_struct(output, o_prev)
                output, o_prev = new_output, output
                if residual:
                    o_prev = output
            elif mode == deet.DEETMode.BACKWARD:
                output = deet_struct(output, x)
            else:
                o_prev = output

            return output

        cloned.add_wrapper(WrappersMP.APPLY_MODEL, wrapper)
        return io.NodeOutput(cloned)


class DEETExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DEET]


async def comfy_entrypoint() -> DEETExtension:
    return DEETExtension()
