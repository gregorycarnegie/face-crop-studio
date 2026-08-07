# Privacy Policy — Face Crop Studio

Effective date: 7 August 2026

Face Crop Studio is a desktop application for detecting faces in photographs and
cropping them. It is designed to run entirely on your own computer.

## Summary

Face Crop Studio does not collect, transmit, or share any personal information.
It has no accounts, no analytics, no advertising, and no network connection of
any kind. Everything you open, process, and save stays on your device.

## Information the app accesses

**Your images and documents.** The app reads the image files and folders you
choose, and the spreadsheet or database files (CSV, XLSX, SQLite) you optionally
supply to drive batch renaming. It writes output images only to the location you
select. These files are read and written directly on your device and are never
uploaded anywhere.

**Your camera.** If you use the webcam capture feature, the app opens the camera
you select in order to show a live preview and detect faces in it. Camera frames
are held in memory for that preview and are discarded as you go; a frame is only
written to disk if you explicitly capture and save it. The camera is opened only
while you are using that feature, and never in the background. This is why the
app declares the webcam capability.

**Face detection.** Face detection runs locally, on your device, using a model
bundled inside the application. No image, face, biometric template, or
derivative of your photographs is sent off the device. The app does not
recognise or identify individuals — it locates faces in an image so it can crop
them, and does not build or store any face database.

## Information stored on your device

**Settings.** Your preferences (output format, crop options, recent choices) are
saved locally in the application's own settings file so they persist between
sessions. You can reset them by deleting that file.

**Optional timing logs.** The app has an optional, off-by-default diagnostic
setting that records processing timings to the local log output to help
troubleshoot performance. Despite being labelled "telemetry" in the interface,
this information is only written locally and is never transmitted to us or to
anyone else.

**Metadata written into output images.** Depending on your metadata setting, the
app may embed processing details into the images it produces, which can include
the file path of the source image. A file path can contain personal information
such as your user name. If you intend to share the output images, set the
metadata option to "strip" to prevent this. This data is embedded in your own
files on your own device; it is not sent to us.

## Information we collect

None. We do not operate a server for this application, and the application
contains no code that makes network requests. We receive no data from your use
of the app, and therefore have nothing to store, sell, or disclose.

The app's About screen contains a link to our website. Following that link opens
the page in your own web browser, which is a separate action governed by your
browser and by that website; the app itself sends nothing.

## Sharing with third parties

We do not share your information with third parties, because we do not collect
any. The application contains no third-party analytics, advertising, tracking,
or crash-reporting components.

If you installed the app from the Microsoft Store, Microsoft may collect
information about the installation itself under its own privacy policy. That is
between you and Microsoft, and is outside this application.

## Your control

Because all data stays on your device, you remain in full control of it. You can
delete any output image, settings file, or log at any time using your operating
system. Uninstalling the application removes the application and its settings;
images you saved to your own folders are left untouched so you do not lose your
work. You can decline camera access at any time through Windows privacy
settings, in which case the webcam feature simply does not function and the rest
of the app is unaffected.

## Children

Face Crop Studio is a general-purpose photo tool and is not directed at
children. It does not knowingly collect information from anyone, of any age.

## Security

Because your data never leaves your device, it is protected by the security of
your own computer and user account. We recommend keeping your operating system
up to date and using the access controls and disk encryption your system
provides.

## Changes to this policy

If this policy changes, the updated version will accompany the corresponding
release of the application, with a revised effective date above.

## Contact

Face Crop Studio is open source. Questions or concerns about this policy, or
about how the application handles your data, can be raised on the project's
public issue tracker, which is monitored:

https://github.com/gregorycarnegie/face-crop-studio/issues

Project website: https://facecropstudio.com/

The source code is public, so every claim made in this policy can be verified
independently rather than taken on trust.
