import torch.nn as nn

### convert models
def convert_linear_to_te(linear: nn.Linear):
    from transformer_engine.pytorch import Linear as TElinear

    te_linear = TElinear(
        in_features  = linear.in_features,
        out_features = linear.out_features,
        bias         = linear.bias is not None,
        params_dtype = linear.weight.dtype,
        device       = linear.weight.device
    )

    # 复制权重
    te_linear.weight.data.copy_(linear.weight.data)
    if linear.bias is not None:
        te_linear.bias.data.copy_(linear.bias.data)

    return te_linear

def convert_linear_to_dw(linear: nn.Linear, precision="precision"):
    import dw

    return dw.modules.FcLayer(linear, precision=precision)

def convert_dw_modules(module):
    import dw
    dw_mappings = {
        # nn.Conv2d: dw.ConvLayer,
        nn.Conv3d: dw.Conv3dLayer,
        # nn.LayerNorm: dw.LayernormLayer,
        # nn.GroupNorm: dw.GroupnormLayer,
        nn.SiLU: dw.SiLULayer,
        nn.GELU: dw.GeluLayer,
        # nn.Dropout: dw.DropoutLayer,
    }

    for k, v in dw_mappings.items():
        if isinstance(module, k):
            return v(module)
    
    return module


def replace_modules(model, arch="NV", precision="baseline"):
    for name, module in model.named_children():
        # if name == "lm_head": continue
        if name == "time_embedding" or name == "time_projection":
            continue
        # if name.isdigit():
        #     if int(name) >= 1 and int(name) <= 34:
        #         continue

        # print(f"Replacing module: {name} of module {module}. arch: {arch}")
        replace_modules(module, arch, precision)    # 递归替换&寻找

        if isinstance(module, nn.Linear):
            if arch == "NV" and precision != "baseline":
                setattr(model, name, convert_linear_to_te(module))
            elif arch == "DW":
                setattr(model, name, convert_linear_to_dw(module, precision))
                # setattr(model, name, module)
        else:
            if arch == "DW":
                setattr(model, name, convert_dw_modules(module))