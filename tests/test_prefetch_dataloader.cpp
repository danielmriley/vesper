#include <vesper/data/prefetch_dataloader.h>
#include <vesper/data/dataset.h>
#include <vesper/core/factories.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <csignal>
#include <unistd.h>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#else
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#endif

using vesper::data::Dataset;
using vesper::data::Sample;
using vesper::data::TensorDataset;
using vesper::data::PrefetchDataLoader;

// Watchdog: if a case deadlocks (e.g. the num_workers==0 regression), SIGALRM
// fires and fails the run loudly instead of hanging the test suite forever.
static void on_timeout(int) {
    const char msg[] = "FAIL: PrefetchDataLoader timed out (deadlock regression)\n";
    ssize_t n = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)n;
    _exit(1);
}

// A dataset whose samples always throw, to exercise worker error propagation.
class ThrowingDataset : public Dataset {
public:
    explicit ThrowingDataset(size_t n) : n_(n) {}
    Sample get_item(size_t) override {
        throw std::runtime_error("bad sample");
    }
    size_t size() const override { return n_; }
private:
    size_t n_;
};

static std::shared_ptr<TensorDataset> make_dataset(int n) {
    auto x = vesper::zeros({n, 5}, vesper::DType::Float32, TEST_DEVICE);
    auto y = vesper::zeros({n, 1}, vesper::DType::Float32, TEST_DEVICE);
    return std::make_shared<TensorDataset>(x, y);
}

// Drain a full epoch, returning the batch count and total sample count.
static int count_batches(PrefetchDataLoader& loader, size_t& total_samples) {
    loader.start_epoch();
    int batches = 0;
    total_samples = 0;
    while (auto batch = loader.next()) {
        ++batches;
        total_samples += static_cast<size_t>(batch->first.shape()[0]);
    }
    return batches;
}

int main() {
    std::signal(SIGALRM, on_timeout);
    alarm(10);  // watchdog covering the whole run

    const int N = 10;
    const size_t batch_size = 4;        // 10 samples / 4 -> 3 batches (4, 4, 2)
    const int expected_batches = 3;

    // Case 1: num_workers == 0 must return all batches synchronously, no hang.
    {
        auto ds = make_dataset(N);
        PrefetchDataLoader loader(ds, batch_size, /*shuffle=*/false, /*num_workers=*/0);
        size_t total = 0;
        int batches = count_batches(loader, total);
        if (batches != expected_batches || total != static_cast<size_t>(N)) {
            std::cerr << "Case 1 (num_workers=0) failed: batches=" << batches
                      << " total=" << total << "\n";
            return 1;
        }
        std::cout << "Case 1 (synchronous, num_workers=0) passed\n";
    }

    // Case 2: batch_size == 0 must throw from the constructor.
    {
        auto ds = make_dataset(N);
        bool threw = false;
        try {
            PrefetchDataLoader loader(ds, /*batch_size=*/0);
        } catch (const std::exception&) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "Case 2 (batch_size=0) failed: no exception thrown\n";
            return 1;
        }
        std::cout << "Case 2 (batch_size=0 throws) passed\n";
    }

    // Case 3: num_workers == 2 must yield the same number of batches.
    {
        auto ds = make_dataset(N);
        PrefetchDataLoader loader(ds, batch_size, /*shuffle=*/false, /*num_workers=*/2);
        size_t total = 0;
        int batches = count_batches(loader, total);
        if (batches != expected_batches || total != static_cast<size_t>(N)) {
            std::cerr << "Case 3 (num_workers=2) failed: batches=" << batches
                      << " total=" << total << "\n";
            return 1;
        }
        std::cout << "Case 3 (num_workers=2) passed\n";
    }

    // Case 4: a worker exception must surface from next(), not std::terminate.
    {
        auto ds = std::make_shared<ThrowingDataset>(N);
        PrefetchDataLoader loader(ds, batch_size, /*shuffle=*/false, /*num_workers=*/1);
        loader.start_epoch();
        bool threw = false;
        try {
            (void)loader.next();
        } catch (const std::exception&) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "Case 4 (worker exception) failed: no exception surfaced\n";
            return 1;
        }
        std::cout << "Case 4 (worker exception propagation) passed\n";
    }

    std::cout << "All PrefetchDataLoader tests passed!\n";
    return 0;
}
