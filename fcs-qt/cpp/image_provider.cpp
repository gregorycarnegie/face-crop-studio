// QML's Image only loads URLs. This provider serves image://fcs/<key> from the Rust image store
// (src/images.rs), so photos and previews reach the screen without touching the disk.

#include <QtGui/QImage>
#include <QtQml/QQmlEngine>
#include <QtQuick/QQuickImageProvider>

#include <climits>
#include <cstddef>
#include <cstdint>

extern "C" {
using FcsAlloc = std::uint8_t *(*)(void *ctx, std::uint32_t width, std::uint32_t height,
                                   std::uint32_t channels, std::size_t *stride);
bool fcs_image_request(const char *id, std::size_t id_len, void *ctx, FcsAlloc alloc);
}

namespace {

// Rust sizes the image through this, then copies its pixels into the returned buffer.
extern "C" std::uint8_t *allocate(void *ctx, std::uint32_t width, std::uint32_t height,
                                  std::uint32_t channels, std::size_t *stride)
{
    auto *image = static_cast<QImage *>(ctx);
    *image = QImage(int(width), int(height),
                    channels == 4 ? QImage::Format_RGBA8888 : QImage::Format_RGB888);
    if (image->isNull())
        return nullptr;
    *stride = std::size_t(image->bytesPerLine());
    return image->bits();
}

class FcsImageProvider final : public QQuickImageProvider
{
public:
    FcsImageProvider() : QQuickImageProvider(QQuickImageProvider::Image) { }

    QImage requestImage(const QString &id, QSize *size, const QSize &requestedSize) override
    {
        const QByteArray key = id.toUtf8();
        QImage image;
        if (!fcs_image_request(key.constData(), std::size_t(key.size()), &image, &allocate))
            return {};
        if (size)
            *size = image.size();
        // Honour sourceSize: shrink to fit, never enlarge; a zero side is unbounded.
        const QSize bound(requestedSize.width() > 0 ? requestedSize.width() : INT_MAX,
                          requestedSize.height() > 0 ? requestedSize.height() : INT_MAX);
        if (image.width() > bound.width() || image.height() > bound.height())
            image = image.scaled(bound, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        return image;
    }
};

} // namespace

// The engine owns the provider from here on.
extern "C" void fcs_install_image_provider(QQmlEngine *engine)
{
    engine->addImageProvider(QStringLiteral("fcs"), new FcsImageProvider);
}
