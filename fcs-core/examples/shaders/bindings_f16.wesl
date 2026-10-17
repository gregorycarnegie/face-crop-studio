enable f16;

@group(0) @binding(0) var<storage, read> input_tensor: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
