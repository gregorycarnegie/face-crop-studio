// Both compilers use these NCHW kernels. The host validates all shapes and buffer lengths.
device_fn! {
    fn byte(value: f32) -> u32 {
        let v = value.max(0.0).min(255.0);
        let n = v as u32;
        let f = v - n as f32;
        n + u32::from(f > 0.5 || (f == 0.5 && n & 1 != 0))
    }
}
device_fn! {
    fn component(pixel: u32, channel: u32) -> f32 { ((pixel >> (channel * 8)) & 255) as f32 }
}

cuda_kernel! {
    fn fcs_pixel(pixels: *mut u32, len: u32, exposure: f32, brightness: f32,
        contrast: f32, saturation: f32, flags: u32,
    ) {
        let i = index();
        if i >= len { return; }
        unsafe {
            let pixel = *pixels.add(i as usize);
            let mut rgb = [component(pixel, 0), component(pixel, 1), component(pixel, 2)];
            let mut c = 0;
            while c < 3 {
                let v = rgb.get_unchecked_mut(c);
                if flags & 1 != 0 { *v *= exposure; }
                if flags & 2 != 0 { *v += brightness; }
                if flags & 4 != 0 { *v = ((*v * (1.0 / 255.0) - 0.5) * contrast + 0.5) * 255.0; }
                c += 1;
            }
            let gray = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114;
            let mut result = pixel & 0xff000000;
            c = 0;
            while c < 3 {
                let v = *rgb.get_unchecked(c);
                let v = if flags & 8 != 0 { gray * (1.0 - saturation) + v * saturation } else { v };
                result |= byte(v) << (c * 8);
                c += 1;
            }
            *pixels.add(i as usize) = result;
        }
    }
}

cuda_kernel! {
    fn fcs_blur(input: *const u32, output: *mut u32, weights: *const f32,
        width: u32, height: u32, radius: u32, direction: u32,
    ) {
        let i = index();
        if width == 0 || i >= width * height { return; }
        unsafe {
            let mut result = *input.add(i as usize) & 0xff000000;
            let mut c = 0;
            while c < 3 {
                let mut sum = 0.0;
                let mut offset = -(radius as i32);
                while offset <= radius as i32 {
                    let x = (i % width) as i32 + if direction == 0 { offset } else { 0 };
                    let y = (i / width) as i32 + if direction == 1 { offset } else { 0 };
                    let at = y.max(0).min(height as i32 - 1) as u32 * width + x.max(0).min(width as i32 - 1) as u32;
                    sum = fma(component(*input.add(at as usize), c), *weights.add((offset + radius as i32) as usize), sum);
                    offset += 1;
                }
                result |= byte(sum) << (c * 8);
                c += 1;
            }
            *output.add(i as usize) = result;
        }
    }
}

cuda_kernel! {
    fn fcs_bilateral(input: *const u32, output: *mut u32, width: u32, height: u32,
        radius: u32, sigma_space: f32, sigma_color: f32, amount: f32,
    ) {
        let i = index();
        if width == 0 || i >= width * height { return; }
        unsafe {
            let center = *input.add(i as usize);
            let mut accum = [0.0; 3];
            let mut weight_sum = 0.0;
            let mut dy = -(radius as i32);
            while dy <= radius as i32 {
                let mut dx = -(radius as i32);
                while dx <= radius as i32 {
                    let x = ((i % width) as i32 + dx).max(0).min(width as i32 - 1) as u32;
                    let y = ((i / width) as i32 + dy).max(0).min(height as i32 - 1) as u32;
                    let sample = *input.add((y * width + x) as usize);
                    let dr = component(sample, 0) - component(center, 0);
                    let dg = component(sample, 1) - component(center, 1);
                    let db = component(sample, 2) - component(center, 2);
                    let weight = exp(-(dx * dx + dy * dy) as f32 / (2.0 * sigma_space * sigma_space).max(1e-4))
                        * exp(-(dr * dr + dg * dg + db * db) / (2.0 * sigma_color * sigma_color).max(1e-4));
                    let mut c = 0;
                    while c < 3 {
                        let a = accum.get_unchecked_mut(c);
                        *a = fma(component(sample, c as u32), weight, *a);
                        c += 1;
                    }
                    weight_sum += weight;
                    dx += 1;
                }
                dy += 1;
            }
            let mut result = center & 0xff000000;
            let mut c = 0;
            while c < 3 {
                let v = component(center, c);
                let filtered = if weight_sum > 1e-5 { *accum.get_unchecked(c as usize) / weight_sum } else { v };
                result |= byte(v * (1.0 - amount) + filtered * amount) << (c * 8);
                c += 1;
            }
            *output.add(i as usize) = result;
        }
    }
}

cuda_kernel! {
    fn fcs_background(sharp: *const u32, blurred: *const u32, output: *mut u32,
        width: u32, height: u32, mask_size: f32,
    ) {
        let i = index();
        if width == 0 || i >= width * height { return; }
        let x = (i % width) as f32;
        let y = (i / width) as f32;
        let dx = (x - width as f32 * 0.5) / (width as f32 * 0.5 * mask_size);
        let dy = (y - height as f32 * 0.5) / (height as f32 * 0.5 * mask_size);
        let blend = ((sqrt(dx * dx + dy * dy) - 0.9) / 0.2).max(0.0).min(1.0);
        unsafe {
            let a = *sharp.add(i as usize);
            let b = *blurred.add(i as usize);
            let mut result = 0;
            let mut c = 0;
            while c < 4 {
                result |= byte(component(a, c) * (1.0 - blend) + component(b, c) * blend) << (c * 8);
                c += 1;
            }
            *output.add(i as usize) = result;
        }
    }
}

cuda_kernel! {
    fn fcs_histogram(pixels: *const u32, histogram: *mut u32, len: u32) {
        let i = index();
        if i >= len { return; }
        unsafe {
            let p = *pixels.add(i as usize);
            let mut c = 0;
            while c < 3 {
                atomic_increment(histogram.add((c * 256 + ((p >> (c * 8)) & 255)) as usize));
                c += 1;
            }
        }
    }
}
cuda_kernel! {
    fn fcs_lut(histogram: *const u32, lut: *mut u32, len: u32) {
        let c = index();
        if c >= 3 { return; }
        unsafe {
            let mut first = 0;
            let mut n = 0;
            while n < 256 {
                let h = *histogram.add((c * 256 + n) as usize);
                if h > 0 { first = h; break; }
                n += 1;
            }
            let denominator = len.saturating_sub(first).max(1) as f32;
            let mut cumulative = 0u32;
            n = 0;
            while n < 256 {
                cumulative += *histogram.add((c * 256 + n) as usize);
                *lut.add((c * 256 + n) as usize) = byte(cumulative.saturating_sub(first) as f32 / denominator * 255.0);
                n += 1;
            }
        }
    }
}
cuda_kernel! {
    fn fcs_equalize(pixels: *mut u32, lut: *const u32, len: u32) {
        let i = index();
        if i >= len { return; }
        unsafe {
            let p = *pixels.add(i as usize);
            let mut result = p & 0xff000000;
            let mut c = 0;
            while c < 3 {
                result |= *lut.add((c * 256 + ((p >> (c * 8)) & 255)) as usize) << (c * 8);
                c += 1;
            }
            *pixels.add(i as usize) = result;
        }
    }
}

device_fn! {
    unsafe fn inside(px: f32, py: f32, points: *const f32, count: u32) -> bool {
        let mut result = false;
        let mut i = 0;
        let mut j = count - 1;
        unsafe {
            while i < count {
                let ix = *points.add((i * 2) as usize);
                let iy = *points.add((i * 2 + 1) as usize);
                let jx = *points.add((j * 2) as usize);
                let jy = *points.add((j * 2 + 1) as usize);
                // Half-open edges must agree despite backend division/FMA rounding.
                let epsilon = 0.0000005 * px.abs().max(ix.abs()).max(jx.abs()).max(1.0);
                if (iy > py) != (jy > py) && px < fma(jx - ix, (py - iy) / (jy - iy), ix) - epsilon { result = !result; }
                j = i;
                i += 1;
            }
        }
        result
    }
}
device_fn! {
    unsafe fn distance(px: f32, py: f32, points: *const f32, count: u32) -> f32 {
        let mut d = f32::MAX;
        let mut s = 1.0;
        let mut i = 0;
        let mut j = count - 1;
        unsafe {
            while i < count {
                let ix = *points.add((i * 2) as usize);
                let iy = *points.add((i * 2 + 1) as usize);
                let ex = *points.add((j * 2) as usize) - ix;
                let jy = *points.add((j * 2 + 1) as usize);
                let ey = jy - iy;
                let wx = px - ix;
                let wy = py - iy;
                let t = ((wx * ex + wy * ey) / (ex * ex + ey * ey)).max(0.0).min(1.0);
                let bx = wx - ex * t;
                let by = wy - ey * t;
                d = d.min(bx * bx + by * by);
                let c0 = py >= iy;
                let c1 = py < jy;
                let c2 = ex * wy > ey * wx;
                if (c0 && c1 && c2) || (!c0 && !c1 && !c2) { s = -s; }
                j = i;
                i += 1;
            }
        }
        s * sqrt(d)
    }
}
cuda_kernel! {
    fn fcs_shape(pixels: *mut u32, points: *const f32, width: u32, height: u32,
        count: u32, softness: f32, intensity: f32, color: u32,
    ) {
        let i = index();
        if width == 0 || count < 3 || i >= width * height { return; }
        let x = (i % width) as f32;
        let y = (i / width) as f32;
        unsafe {
            let mut coverage = 0.0;
            let mut sample = 0;
            while sample < 4 {
                if inside(x + 0.25 + (sample & 1) as f32 * 0.5, y + 0.25 + (sample >> 1) as f32 * 0.5, points, count) { coverage += 0.25; }
                sample += 1;
            }
            if softness > 0.0 {
                let radius = (width.min(height) as f32 * 0.5 * softness).max(1.0);
                coverage *= (0.5 - 0.5 * distance(x + 0.5, y + 0.5, points, count) / radius).max(0.0).min(1.0);
            }
            let pixel = *pixels.add(i as usize);
            let mut result = byte(component(pixel, 3) * coverage) << 24;
            let mut c = 0;
            while c < 3 {
                let v = component(pixel, c);
                result |= byte(v + (1.0 - coverage) * intensity * (component(color, c) - v)) << (c * 8);
                c += 1;
            }
            *pixels.add(i as usize) = result;
        }
    }
}
cuda_kernel! {
    fn fcs_conv(
        input: *const f32, weights: *const f32, bias: *const f32, output: *mut f32,
        height: u32, width: u32, out_h: u32, out_w: u32, in_per_group: u32,
        out_channels: u32, kernel: u32, stride: u32, pad: u32, groups: u32, relu: u32,
    ) {
        let i = index();
        if groups == 0 { return; }
        let out_per_group = out_channels / groups;
        let plane = out_h * out_w;
        if out_h == 0 || out_w == 0 || plane == 0 || out_per_group == 0 { return; }
        if i >= out_channels * out_h * out_w { return; }
        let x = i % out_w;
        let y = i / out_w % out_h;
        let channel = i / plane;
        let first = channel / out_per_group * in_per_group;
        // SAFETY: the host checks the topology and allocations; each thread owns output[i].
        unsafe {
            let mut sum = *bias.add(channel as usize);
            let mut c = 0;
            while c < in_per_group {
                let mut ky = 0;
                while ky < kernel {
                    let iy = y * stride + ky;
                    let mut kx = 0;
                    while kx < kernel {
                        let ix = x * stride + kx;
                        if iy >= pad && iy - pad < height && ix >= pad && ix - pad < width {
                            let source = ((first + c) * height + iy - pad) * width + ix - pad;
                            let weight = ((channel * in_per_group + c) * kernel + ky) * kernel + kx;
                            sum += *input.add(source as usize) * *weights.add(weight as usize);
                        }
                        kx += 1;
                    }
                    ky += 1;
                }
                c += 1;
            }
            if relu != 0 && sum < 0.0 { sum = 0.0; }
            *output.add(i as usize) = sum;
        }
    }
}

// Dedicated paths avoid the generic convolution's group and kernel loops.
cuda_kernel! {
    fn fcs_pointwise(input: *const f32, weights: *const f32, bias: *const f32, output: *mut f32,
        height: u32, width: u32, out_h: u32, out_w: u32, in_per_group: u32,
        out_channels: u32, _kernel: u32, stride: u32, pad: u32, _groups: u32, relu: u32,
    ) {
        let i = index();
        let plane = out_h * out_w;
        if out_h == 0 || plane == 0 || out_w == 0 || i >= out_channels * plane { return; }
        let channel = i / plane;
        let y = i / out_w % out_h * stride;
        let x = i % out_w * stride;
        unsafe {
            let mut sum = *bias.add(channel as usize);
            if y >= pad && y - pad < height && x >= pad && x - pad < width {
                let mut c = 0;
                while c < in_per_group {
                    sum += *input.add(((c * height + y - pad) * width + x - pad) as usize)
                        * *weights.add((channel * in_per_group + c) as usize);
                    c += 1;
                }
            }
            *output.add(i as usize) = if relu != 0 { sum.max(0.0) } else { sum };
        }
    }
}

cuda_kernel! {
    fn fcs_depthwise(input: *const f32, weights: *const f32, bias: *const f32, output: *mut f32,
        height: u32, width: u32, out_h: u32, out_w: u32, _in_per_group: u32,
        out_channels: u32, kernel: u32, stride: u32, pad: u32, _groups: u32, relu: u32,
    ) {
        let i = index();
        let plane = out_h * out_w;
        if out_h == 0 || plane == 0 || out_w == 0 || i >= out_channels * plane { return; }
        let channel = i / plane;
        let y = i / out_w % out_h * stride;
        let x = i % out_w * stride;
        unsafe {
            let mut sum = *bias.add(channel as usize);
            let mut ky = 0;
            while ky < kernel {
                let mut kx = 0;
                while kx < kernel {
                    let iy = y + ky;
                    let ix = x + kx;
                    if iy >= pad && iy - pad < height && ix >= pad && ix - pad < width {
                        sum += *input.add(((channel * height + iy - pad) * width + ix - pad) as usize)
                            * *weights.add(((channel * kernel + ky) * kernel + kx) as usize);
                    }
                    kx += 1;
                }
                ky += 1;
            }
            *output.add(i as usize) = if relu != 0 { sum.max(0.0) } else { sum };
        }
    }
}

cuda_kernel! {
    fn fcs_add(a: *const f32, b: *const f32, output: *mut f32, len: u32) {
        let i = index() as usize;
        if i < len as usize {
            // SAFETY: three equal-sized buffers; each thread owns output[i].
            unsafe { *output.add(i) = *a.add(i) + *b.add(i); }
        }
    }
}

cuda_kernel! {
    fn fcs_upsample(input: *const f32, output: *mut f32, channels: u32, height: u32, width: u32) {
        let i = index();
        let out_w = width * 2;
        let out_h = height * 2;
        let plane = out_h * out_w;
        if out_h == 0 || out_w == 0 || plane == 0 { return; }
        if i < channels * out_h * out_w {
            let source = ((i / plane) * height + i / out_w % out_h / 2) * width + i % out_w / 2;
            // SAFETY: the output has twice the spatial dimensions of the input.
            unsafe { *output.add(i as usize) = *input.add(source as usize); }
        }
    }
}

