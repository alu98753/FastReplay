#ifndef FAST_REPLAY_RING_BUFFER_HPP
#define FAST_REPLAY_RING_BUFFER_HPP

#include <cstddef>

namespace fastreplay {

//interface placeholder
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity);
    ~RingBuffer();

    /*Week2: push/pop
      Week3: buffer_protocol
      Week4: atomic-based synchronization
    */

    std::size_t capacity() const;

private:
    std::size_t capacity_;
};

} 

#endif // FAST_REPLAY_RING_BUFFER_HPP