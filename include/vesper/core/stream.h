#pragma once

#include <vesper/core/device.h>
#include <memory>

namespace vesper {

class StreamImpl; // Forward declaration for PIMPL or backend-specific handle

class Stream {
public:
    // Creates a new stream on the specified device.
    explicit Stream(Device device);
    ~Stream();

    // Disable copy
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Enable move
    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;

    // Synchronize the host with this stream.
    // Blocks the CPU until all operations on this stream are complete.
    void synchronize() const;

    // Returns the raw backend handle (cudaStream_t or hipStream_t) as void*
    void* raw_handle() const;

    Device device() const { return device_; }

    // Returns the default stream for the given device.
    static Stream& default_stream(Device device);

    // Sets the current stream for the calling thread.
    // Does not take ownership of the stream object, only the handle.
    // Users should ensure the stream object outlives its usage as current if owning handle.
    static void set_current(const Stream& stream);

    // Returns the current stream for the calling thread on the given device.
    static Stream& current(Device device);

private:
    Device device_;
    void* handle_ = nullptr; // cudaStream_t/hipStream_t cast to void*
    bool own_handle_ = true; // Does this object own the handle? (Default stream doesn't)

    // Private constructor for wrapping existing handles (e.g. default stream)
    Stream(Device device, void* handle, bool own);
};

} // namespace vesper
