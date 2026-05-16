#ifndef FAST_REPLAY_RING_BUFFER_HPP
#define FAST_REPLAY_RING_BUFFER_HPP

#include <atomic>
#include <cstddef>
#include <vector>

namespace fastreplay {

// SPSC ring buffer using std::atomic for index
// synchronization. Avoids OS-level primitives
// (std::mutex) on the hot data path (push/pop).
//
// NOTE: std::atomic is NOT guaranteed to be
// lock-free on all platforms. Check
// std::atomic<T>::is_lock_free() if needed.
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity)
        : capacity_(capacity)
		, buf_(capacity + 1) // +1 for full/empty distinguish
		, head_(0) // set init head/tail to 0
        , tail_(0) {}

    ~RingBuffer() = default;

    // SPSC queue semantics: non-copyable, non-movable
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    int* data() { return buf_.data(); }

    std::size_t head() const {
        return head_.load(std::memory_order_relaxed);
    }

    void advance_head(std::size_t n) {
        std::size_t cur = head_.load(
            std::memory_order_relaxed);
        head_.store(
            (cur + n) % (capacity_ + 1),
            std::memory_order_release);
    }

    // Producer-side: push one element.
    // Returns false if the buffer is full.
    bool push(int value) {
        std::size_t cur_tail = tail_.load(
            std::memory_order_relaxed);
        std::size_t next_tail =
            (cur_tail + 1) % (capacity_ + 1);
        // Acquire: see consumer's latest head_
        if (next_tail == head_.load(
                std::memory_order_acquire)) {
            return false; // full
        }
        buf_[cur_tail] = value;
        // Release: make buf_ write visible
        // before advancing tail
        tail_.store(
            next_tail,
            std::memory_order_release);
        return true;
    }

    // Consumer-side: pop one element.
    // Returns false if empty.
    bool pop(int* out_val) {
        std::size_t cur_head = head_.load(
            std::memory_order_relaxed);
        // Acquire: see producer's latest tail_
        if (cur_head == tail_.load(
                std::memory_order_acquire)) {
            return false; // empty
        }
        if (out_val != nullptr) {
            *out_val = buf_[cur_head];
        }
        // Release: ensure buf_ read completes
        // before advancing head
        head_.store(
            (cur_head + 1) % (capacity_ + 1),
            std::memory_order_release);
        return true;
    }

    // Approximate size (relaxed ordering).
    // In concurrent use, may be slightly stale.
    std::size_t size() const {
        std::size_t t = tail_.load(
            std::memory_order_relaxed);
        std::size_t h = head_.load(
            std::memory_order_relaxed);
        return (t - h + capacity_ + 1)
            % (capacity_ + 1);
    }

    bool empty() const {
        return head_.load(
            std::memory_order_relaxed)
            == tail_.load(
                std::memory_order_relaxed);
    }

    std::size_t capacity() const {
        return capacity_;
    }

private:
    std::size_t capacity_;
    std::vector<int> buf_;
    std::atomic<std::size_t> head_;
    std::atomic<std::size_t> tail_;
};

} // namespace fastreplay

#endif // FAST_REPLAY_RING_BUFFER_HPP