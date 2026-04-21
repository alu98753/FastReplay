#ifndef FAST_REPLAY_RING_BUFFER_HPP
#define FAST_REPLAY_RING_BUFFER_HPP

#include <cstddef>
#include <mutex>
#include <vector>

namespace fastreplay {

//interface placeholder
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
  
    // Push a val into buffer. Return true on success, false if full.
    bool push(int value){
		std::lock_guard<std::mutex> lock(mtx_);
		std::size_t next_tail = (tail_ + 1) % (capacity_ + 1);
		if(next_tail == head_){
			return false; // full
		}
		buf_[tail_] = value;
		tail_ = next_tail;
		return true;
    }

	// Pop a value from buffer. (C++11 pass-by-reference)
	// Return true on success, false if empty
	bool pop(int& out_val){
		std::lock_guard<std::mutex> lock(mtx_);
		if(head_ == tail_){
			return false; // empty
		}
		out_val = buf_[head_];
		head_ = (head_ + 1) % (capacity_ + 1);
		return true;
	}

	std::size_t size() const {
		std::lock_guard<std::mutex> lock(mtx_);
		return (tail_ - head_ + capacity_ + 1) % (capacity_ + 1);
	}

	bool empty() const {
		std::lock_guard<std::mutex> lock(mtx_);
		return head_ == tail_;
	}

	std::size_t capacity() const {
		return capacity_;
	}

    /*Week2: push/pop
      Week3: buffer_protocol
      Week4: atomic-based synchronization
    */

private:
    std::size_t capacity_; // user-facing capacity actual store size is capacity +1
    std::vector<int> buf_; // internal storage (size = capacity + 1)
    std::size_t head_; // read index
    std::size_t tail_; // write idx
    mutable std::mutex mtx_; // week2 mutex-based sync
};

} 

#endif // FAST_REPLAY_RING_BUFFER_HPP