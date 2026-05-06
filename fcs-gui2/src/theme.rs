//! Visual theme matching the Face Crop Studio HTML mockup palette.

use egui::{
    Color32, Context, CornerRadius, FontDefinitions, FontFamily, Margin, Shadow, Stroke, Visuals,
};

// ── Palette (from HTML CSS vars) ──────────────────────────────────────────────

pub struct P;
impl P {
    pub const BG: Color32 = Color32::from_rgb(0x0a, 0x0f, 0x1c);
    pub const BG1: Color32 = Color32::from_rgb(0x10, 0x17, 0x29);
    pub const BG2: Color32 = Color32::from_rgb(0x15, 0x20, 0x3a);
    pub const SURFACE: Color32 = Color32::from_rgb(0x0f, 0x16, 0x26);
    pub const SURFACE2: Color32 = Color32::from_rgb(0x13, 0x1c, 0x30);
    pub const RULE: Color32 = Color32::from_rgb(0x1d, 0x27, 0x42);
    pub const RULE2: Color32 = Color32::from_rgb(0x2a, 0x36, 0x5a);
    pub const INK: Color32 = Color32::from_rgb(0xf5, 0xf4, 0xee);
    pub const INK2: Color32 = Color32::from_rgb(0xa4, 0xad, 0xc7);
    pub const INK3: Color32 = Color32::from_rgb(0x6c, 0x74, 0x91);
    pub const PEACH: Color32 = Color32::from_rgb(0xff, 0xb8, 0x9a);
    pub const PEACH_DEEP: Color32 = Color32::from_rgb(0xff, 0x85, 0x56);
    pub const CYAN: Color32 = Color32::from_rgb(0x7b, 0xe0, 0xd6);
    pub const LIME: Color32 = Color32::from_rgb(0xd8, 0xf0, 0x6a);
    pub const ROSE: Color32 = Color32::from_rgb(0xff, 0x6b, 0x8b);

    // Semi-transparent helpers
    pub fn rule_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0x1d, 0x27, 0x42, a)
    }
    pub fn peach_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0xff, 0xb8, 0x9a, a)
    }
    pub fn cyan_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0x7b, 0xe0, 0xd6, a)
    }
    pub fn lime_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0xd8, 0xf0, 0x6a, a)
    }
    pub fn rose_alpha(a: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(0xff, 0x6b, 0x8b, a)
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
    setup_fonts(ctx);

    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 5.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.window_margin = Margin::same(0);
    style.spacing.indent = 12.0;
    style.spacing.scroll.bar_width = 8.0;
    style.visuals = build_visuals();
    ctx.set_global_style(style);
}

fn setup_fonts(ctx: &Context) {
    let mut fonts = FontDefinitions::default();

    // Add Hack (embedded monospace) as a fallback for the proportional family.
    // Ubuntu-Light only covers Latin/Cyrillic; Hack adds arrows, box-drawing,
    // and geometric shapes so characters like → ─ □ ↶ ↷ render correctly.
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .push("Hack".into());

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

    v.widgets.inactive.bg_fill = P::white_alpha(5);
    v.widgets.inactive.bg_stroke = rule_stroke;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, P::INK2);

    v.widgets.hovered.bg_fill = P::white_alpha(15);
    v.widgets.hovered.bg_stroke = rule2_stroke;
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, P::INK);

    v.widgets.active.bg_fill = P::white_alpha(20);
    v.widgets.active.bg_stroke = cyan_stroke;
    v.widgets.active.fg_stroke = Stroke::new(1.0, P::INK);

    v.widgets.open.bg_fill = P::SURFACE2;
    v.widgets.open.bg_stroke = rule2_stroke;

    v.selection.bg_fill = P::cyan_alpha(60);
    v.selection.stroke = Stroke::new(1.0, P::CYAN);

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
