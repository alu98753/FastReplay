#include <gtest/gtest.h>
#include "fastreplay/ring_buffer.hpp"

#include <thread>
#include <vector>

// Using C++11

// w1: only test compile at include
TEST(SmokeTest, HeaderCompiles) {
    EXPECT_EQ(1, 1);
}

// ---- Week 2: RingBuffer unit tests ----

TEST(RingBufferTest, ConstructorSetsCapacity) {
    fastreplay::RingBuffer rb(8);
    EXPECT_EQ(rb.capacity(), 8);
}

TEST(RingBufferTest, InitiallyEmpty) {
    fastreplay::RingBuffer rb(4);
    EXPECT_EQ(rb.size(), 0);
    EXPECT_TRUE(rb.empty());
}

TEST(RingBufferTest, PushAndPop) {
    fastreplay::RingBuffer rb(4);
    EXPECT_TRUE(rb.push(42));
    EXPECT_EQ(rb.size(), 1);

    int val = 0;
    EXPECT_TRUE(rb.pop(&val)); // C++11: pop 回傳 bool，值放在 val
    EXPECT_EQ(val, 42);
    EXPECT_EQ(rb.size(), 0);
}

TEST(RingBufferTest, FIFO_Order) {
    fastreplay::RingBuffer rb(4);
    rb.push(10);
    rb.push(20);
    rb.push(30);

    int v = 0;
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 10);
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 20);
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 30);
}

TEST(RingBufferTest, PopFromEmptyReturnsFalse) {
    fastreplay::RingBuffer rb(4);
    int v = 0;
    EXPECT_FALSE(rb.pop(&v)); // C++11: 空 buffer 時回傳 false，v 不被寫入
}

TEST(RingBufferTest, PushToFullReturnsFalse) {
    // 內部實際 array 大小 = capacity + 1，可用空間 = capacity
    fastreplay::RingBuffer rb(3);
    EXPECT_TRUE(rb.push(1));
    EXPECT_TRUE(rb.push(2));
    EXPECT_TRUE(rb.push(3));
    EXPECT_FALSE(rb.push(4)); // full
}

TEST(RingBufferTest, WrapAround) {
    fastreplay::RingBuffer rb(3);

    // 填滿 3 格
    rb.push(1);
    rb.push(2);
    rb.push(3);

    // pop 2 格，騰出空間
    int v = 0;
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 1);
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 2);

    // push 再填 2 格（此時 tail 會 wrap around）
    EXPECT_TRUE(rb.push(4));
    EXPECT_TRUE(rb.push(5));

    // 驗證 FIFO 順序
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 3);
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 4);
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 5);
    EXPECT_TRUE(rb.empty());
}

TEST(RingBufferTest, SizeTracksCorrectly) {
    fastreplay::RingBuffer rb(4);
    EXPECT_EQ(rb.size(), 0);

    rb.push(1);
    EXPECT_EQ(rb.size(), 1);

    rb.push(2);
    EXPECT_EQ(rb.size(), 2);

    int v = 0;
    rb.pop(&v);
    EXPECT_EQ(rb.size(), 1);

    rb.pop(&v);
    EXPECT_EQ(rb.size(), 0);
}

TEST(RingBufferTest, AdvanceHeadCorrectness) {
    // advance_head across wrap boundary
    fastreplay::RingBuffer rb(4); // phys = 5
    for (int i = 0; i < 4; ++i) {
        rb.push(i);
    }
    // head=0, advance by 3 -> head=3
    rb.advance_head(3);
    EXPECT_EQ(rb.head(), 3);
    EXPECT_EQ(rb.size(), 1);

    // pop remaining, push more to set up wrap
    int v = 0;
    rb.pop(&v);
    EXPECT_EQ(v, 3);
    for (int i = 10; i < 14; ++i) {
        rb.push(i);
    }
    // head=4, tail wraps. advance across boundary
    rb.advance_head(2);
    std::size_t h = rb.head();
    // (4+2) % 5 = 1
    EXPECT_EQ(h, 1);
}

TEST(RingBufferTest, MultiWrapCycle) {
    // 3 full fill-drain cycles
    fastreplay::RingBuffer rb(3);
    int base = 0;
    for (int cycle = 0; cycle < 3; ++cycle) {
        // fill
        for (int i = 0; i < 3; ++i) {
            EXPECT_TRUE(rb.push(base + i));
        }
        EXPECT_EQ(rb.size(), 3);
        EXPECT_FALSE(rb.push(999)); // full

        // drain & verify FIFO
        for (int i = 0; i < 3; ++i) {
            int v = -1;
            EXPECT_TRUE(rb.pop(&v));
            EXPECT_EQ(v, base + i);
        }
        EXPECT_TRUE(rb.empty());
        base += 10;
    }
}

TEST(RingBufferTest, CapacityOneEdgeCase) {
    // capacity=1, phys=2, indices {0,1}
    fastreplay::RingBuffer rb(1);
    EXPECT_EQ(rb.capacity(), 1);
    EXPECT_TRUE(rb.empty());

    EXPECT_TRUE(rb.push(42));
    EXPECT_FALSE(rb.push(99)); // full
    EXPECT_EQ(rb.size(), 1);

    int v = 0;
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 42);
    EXPECT_TRUE(rb.empty());

    // second cycle
    EXPECT_TRUE(rb.push(77));
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 77);
    EXPECT_TRUE(rb.empty());
}

TEST(RingBufferTest, PopNullptrDiscards) {
    // pop with nullptr discards value
    fastreplay::RingBuffer rb(4);
    rb.push(10);
    rb.push(20);
    rb.push(30);
    EXPECT_EQ(rb.size(), 3);

    // discard first element
    EXPECT_TRUE(rb.pop(nullptr));
    EXPECT_EQ(rb.size(), 2);

    // next pop should yield 20
    int v = 0;
    EXPECT_TRUE(rb.pop(&v));
    EXPECT_EQ(v, 20);
}

// ---- Concurrent stress test ----

TEST(RingBufferTest, ConcurrentSPSC) {
    // two-thread push/pop stress test
    const int N = 100000;
    fastreplay::RingBuffer rb(1024);

    // Producer thread: push 0..N-1
    std::thread producer([&rb]() {
        for (int i = 0; i < N; ++i) {
            while (!rb.push(i)) {
                // spin until space available
            }
        }
    });

    // Consumer thread: pop N values
    std::vector<int> received;
    received.reserve(N);
    std::thread consumer([&rb, &received]() {
        for (int i = 0; i < N; ++i) {
            int val = 0;
            while (!rb.pop(&val)) {
                // spin until data available
            }
            received.push_back(val);
        }
    });

    producer.join();
    consumer.join();

    // Verify FIFO ordering preserved
    ASSERT_EQ(
        static_cast<int>(received.size()), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(received[i], i);
    }
    EXPECT_TRUE(rb.empty());
}

// ---- sample_indices tests (Issue #28) ----

TEST(SampleIndicesTest, BasicSampling) {
    // head=0, buffer has 5 elements in slots 0..4
    fastreplay::RingBuffer rb(8);
    for (int i = 0; i < 5; ++i) {
        rb.push(i * 10);
    }
    EXPECT_EQ(rb.size(), 5);

    auto indices = rb.sample_indices(3);
    EXPECT_EQ(indices.size(), 3);

    // All indices must be in valid physical range [0, 5)
    for (auto idx : indices) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 5);
    }
}

TEST(SampleIndicesTest, HeadNonZero) {
    // This is the critical test: after pops, head != 0.
    // sample_indices must return indices in the VALID
    // range, not the stale physical range [0, size).
    fastreplay::RingBuffer rb(5);
    for (int v : {10, 20, 30, 40, 50}) {
        rb.push(v);
    }
    // Pop 2 → head moves to 2, valid data at slots 2,3,4
    int discard;
    rb.pop(&discard);
    rb.pop(&discard);
    EXPECT_EQ(rb.size(), 3);
    EXPECT_EQ(rb.head(), 2);

    auto indices = rb.sample_indices(100);
    for (auto idx : indices) {
        // Valid physical slots are {2, 3, 4}
        EXPECT_GE(idx, 2);
        EXPECT_LE(idx, 4);
    }

    // Verify sampled data is never stale (10 or 20)
    for (auto idx : indices) {
        int val = rb.data()[idx];
        EXPECT_TRUE(val == 30 || val == 40 || val == 50)
            << "Got stale value " << val << " at idx " << idx;
    }
}

TEST(SampleIndicesTest, WrapAround) {
    // Set up buffer so valid data wraps around the end.
    // capacity=4, phys=5, slots {0,1,2,3,4}
    fastreplay::RingBuffer rb(4);
    // Fill and pop to advance head past midpoint
    for (int i = 0; i < 4; ++i) rb.push(i);
    int v;
    rb.pop(&v); // head=1
    rb.pop(&v); // head=2
    rb.pop(&v); // head=3
    rb.push(10); // tail wraps to 0 → slot 4=10
    rb.push(11); // slot 0=11
    // Now: head=3, tail=1, valid slots = {3, 4, 0}
    EXPECT_EQ(rb.size(), 3);

    auto indices = rb.sample_indices(200);
    for (auto idx : indices) {
        EXPECT_TRUE(idx == 0 || idx == 3 || idx == 4)
            << "Invalid index " << idx
            << " (expected 0, 3, or 4)";
    }
}

TEST(SampleIndicesTest, EmptyBufferThrows) {
    fastreplay::RingBuffer rb(4);
    EXPECT_THROW(rb.sample_indices(1), std::runtime_error);
}

TEST(SampleIndicesTest, BatchSizeLargerThanBuffer) {
    // RL samples WITH replacement, so batch_size > size is valid
    fastreplay::RingBuffer rb(4);
    rb.push(1);
    rb.push(2);
    EXPECT_EQ(rb.size(), 2);
    // batch_size=10 > size=2: allowed (with replacement)
    auto indices = rb.sample_indices(10);
    EXPECT_EQ(indices.size(), 10);
    for (auto idx : indices) {
        EXPECT_TRUE(idx == 0 || idx == 1);
    }
}