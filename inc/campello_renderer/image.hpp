#pragma once

namespace systems::leal::campello_renderer {

    enum class ImageFormat {
        uint8,
        uint16,
        float32
    };

    struct Image {

        uint32_t width;
        uint32_t height;
        uint32_t components;
        ImageFormat format;
        uint8_t *data;

        Image(
            uint32_t width,
            uint32_t height,
            uint32_t components,
            ImageFormat format,
            uint8_t *data) {

            this->width = width;
            this->height = height;
            this->components = components;
            this->format = format;
            this->data = data;                
        }

        ~Image() {
            if (data != nullptr) {
                delete data;
            }
        }

    };
}