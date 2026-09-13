//! Charcoal surfaces and orange accents shared with the Face Crop Studio website.

use egui::{
    Color32, Context, CornerRadius, FontDefinitions, FontFamily, Margin, Shadow, Stroke, Visuals,
};

// ── Palette (from HTML CSS vars) ──────────────────────────────────────────────

pub struct P;
impl P {
    pub const BG: Color32 = Color32::from_rgb(0x0b, 0x0b, 0x10);
    pub const BG1: Color32 = Color32::from_rgb(0x10, 0x10, 0x17);
    pub const BG2: Color32 = Color32::from_rgb(0x18, 0x18, 0x22);
    pub const SURFACE: Color32 = Color32::from_rgb(0x18, 0x18, 0x22);
    pub const SURFACE2: Color32 = Color32::from_rgb(0x26, 0x26, 0x32);
    pub const RULE: Color32 = Color32::from_rgb(0x38, 0x38, 0x46);
    pub const RULE2: Color32 = Color32::from_rgb(0x7c, 0x7c, 0x8f);
    pub const INK: Color32 = Color32::from_rgb(0xf4, 0xf3, 0xf7);
    pub const INK2: Color32 = Color32::from_rgb(0xc6, 0xc5, 0xd0);
    pub const INK3: Color32 = Color32::from_rgb(0xab, 0xaa, 0xba);
    pub const ACCENT: Color32 = Color32::from_rgb(0xf5, 0x69, 0x24);
    pub const PEACH: Color32 = Color32::from_rgb(0xff, 0xa0, 0x6b);
    pub const PEACH_DEEP: Color32 = Self::ACCENT;
    pub const CYAN: Color32 = Color32::from_rgb(0x8f, 0xcd, 0xff);
    pub const LIME: Color32 = Self::INK2;
    pub const ROSE: Color32 = Color32::from_rgb(0xff, 0xb5, 0xae);
    // Batch outcomes. Kept clear of the orange accents (PEACH marks "needs a look") so
    // success and failure read apart at a glance.
    pub const GREEN: Color32 = Color32::from_rgb(0x6f, 0xd6, 0x8a);
    pub const RED: Color32 = Color32::from_rgb(0xff, 0x5c, 0x5c);

    // Semi-transparent helpers
    pub fn rule_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0x38, 0x38, 0x46, a)
    }
    pub fn peach_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0xf5, 0x69, 0x24, a)
    }
    pub fn cyan_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0x8f, 0xcd, 0xff, a)
    }
    pub fn lime_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0xc6, 0xc5, 0xd0, a)
    }
    pub fn rose_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0xff, 0xb5, 0xae, a)
    }
    pub fn white_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, a)
    }
    pub fn black_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, a)
    }
}

// ── Font families ─────────────────────────────────────────────────────────────

/// Apply global theme + fonts to the egui context.
pub fn apply(ctx: &Context) {
    // Select the dark style before installing it: OS light mode must not leave
    // native inputs on a different palette from our painted panels.
    ctx.set_theme(egui::Theme::Dark);
    setup_fonts(ctx);

    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.interact_size.y = 30.0;
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.window_margin = Margin::same(12);
    style.spacing.indent = 12.0;
    style.spacing.scroll.bar_width = 8.0;
    style.visuals = build_visuals();
    ctx.set_global_style(style);
}

fn setup_fonts(ctx: &Context) {
    let mut fonts = FontDefinitions::default();

    // Prefer the platform's regular-weight UI font over the bundled light face.
    for path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts
                .font_data
                .insert("studio".into(), egui::FontData::from_owned(bytes).into());
            fonts
                .families
                .entry(FontFamily::Proportional)
                .or_default()
                .insert(0, "studio".into());
            break;
        }
    }

    // Add Hack (embedded monospace) as a fallback for the proportional family.
    // Ubuntu-Light only covers Latin/Cyrillic; Hack adds arrows, box-drawing,
    // and geometric shapes so characters like → ─ □ ↶ ↷ render correctly.
    // No bundled font has ✓ (U+2713); use ✔ (U+2714, from the emoji fonts).
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .push("Hack".into());

    // CJK fallback: the bundled fonts have no CJK glyphs, so Chinese/Japanese/
    // Korean file names render as boxes. Load a system font if one exists.
    let cjk_candidates = [
        "C:/Windows/Fonts/msyh.ttc",          // Windows: Microsoft YaHei
        "/System/Library/Fonts/PingFang.ttc", // macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", // Linux (Noto)
    ];
    for path in cjk_candidates {
        if let Ok(bytes) = std::fs::read(path) {
            fonts
                .font_data
                .insert("cjk".into(), egui::FontData::from_owned(bytes).into());
            for family in [FontFamily::Proportional, FontFamily::Monospace] {
                fonts.families.entry(family).or_default().push("cjk".into());
            }
            break;
        }
    }

    // Register a named "mono" family used for badges, chips, labels
    fonts.families.insert(
        FontFamily::Name("mono".into()),
        fonts.families[&FontFamily::Monospace].clone(),
    );

    ctx.set_fonts(fonts);
}

fn build_visuals() -> Visuals {
    let mut v = Visuals::dark();
    v.override_text_color = Some(P::INK);
    v.weak_text_color = Some(P::INK3);
    v.text_edit_bg_color = Some(P::BG1);
    v.hyperlink_color = P::CYAN;
    v.panel_fill = P::SURFACE;
    v.window_fill = P::BG;
    v.extreme_bg_color = P::BG;
    v.faint_bg_color = P::BG1;

    let rule_stroke = Stroke::new(1.0, P::RULE);
    let rule2_stroke = Stroke::new(1.0, P::RULE2);
    let cyan_stroke = Stroke::new(1.5, P::CYAN);

    v.widgets.noninteractive.bg_fill = P::SURFACE;
    v.widgets.noninteractive.bg_stroke = rule_stroke;
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, P::INK2);

    v.widgets.inactive.bg_fill = P::SURFACE2;
    v.widgets.inactive.weak_bg_fill = P::SURFACE2;
    v.widgets.inactive.bg_stroke = rule2_stroke;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, P::INK2);

    v.widgets.hovered.bg_fill = P::RULE;
    v.widgets.hovered.weak_bg_fill = P::RULE;
    v.widgets.hovered.bg_stroke = rule2_stroke;
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, P::INK);

    v.widgets.active.bg_fill = P::SURFACE2;
    v.widgets.active.weak_bg_fill = P::SURFACE2;
    v.widgets.active.bg_stroke = cyan_stroke;
    v.widgets.active.fg_stroke = Stroke::new(1.0, P::INK);

    v.widgets.open.bg_fill = P::SURFACE2;
    v.widgets.open.weak_bg_fill = P::SURFACE2;
    v.widgets.open.bg_stroke = rule2_stroke;
    for widget in [
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        widget.corner_radius = CornerRadius::same(6);
    }

    v.selection.bg_fill = P::ACCENT;
    v.selection.stroke = Stroke::new(2.0, P::BG);
    v.warn_fg_color = P::PEACH;
    v.error_fg_color = P::ROSE;

    v.window_corner_radius = CornerRadius::same(10);
    v.menu_corner_radius = CornerRadius::same(8);
    v.window_stroke = Stroke::new(1.0, P::white_alpha(13));

    v.window_shadow = Shadow {
        offset: [0, 8],
        blur: 40,
        spread: 2,
        color: P::black_alpha(180),
    };
    v.popup_shadow = Shadow {
        offset: [0, 4],
        blur: 20,
        spread: 1,
        color: P::black_alpha(160),
    };

    v
}

// ── Convenience color accessors ───────────────────────────────────────────────

/// Badge colour for a file status string.
pub fn badge_color(status: &str) -> (Color32, Color32) {
    match status {
        "ok" | "done" => (P::lime_alpha(30), P::LIME),
        "run" | "running" => (P::peach_alpha(35), P::PEACH),
        "err" | "error" => (P::rose_alpha(35), P::ROSE),
        _ => (P::white_alpha(10), P::INK3), // skip / queued / —
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contrast(a: Color32, b: Color32) -> f32 {
        let luminance = |c: Color32| {
            let linear = |v: u8| {
                let v = v as f32 / 255.0;
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * linear(c.r()) + 0.7152 * linear(c.g()) + 0.0722 * linear(c.b())
        };
        let (a, b) = (luminance(a), luminance(b));
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    #[test]
    fn studio_theme_keeps_text_and_controls_legible() {
        for bg in [P::BG, P::BG1, P::SURFACE, P::SURFACE2] {
            for fg in [P::INK, P::INK2, P::INK3, P::PEACH, P::CYAN, P::ROSE] {
                assert!(contrast(fg, bg) >= 4.5, "text {fg:?} on {bg:?}");
            }
            assert!(contrast(P::RULE2, bg) >= 3.0, "control outline on {bg:?}");
        }
        assert!(contrast(P::BG, P::ACCENT) >= 4.5);
        let ctx = Context::default();
        ctx.set_theme(egui::Theme::Light);
        apply(&ctx);
        let style = ctx.global_style();
        assert!(style.visuals.dark_mode);
        assert_eq!(style.visuals.text_edit_bg_color, Some(P::BG1));
        assert_eq!(style.visuals.widgets.inactive.weak_bg_fill, P::SURFACE2);
    }
}
