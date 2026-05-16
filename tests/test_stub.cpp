#include <gtest/gtest.h>
#include "fastreplay/ring_buffer.hpp"

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

// ---- Issue #11 pre-migration regression tests ----

TEST(RingBufferTest, AdvanceHeadCorrectness) {
    // Gap 1: advance_head across wrap boundary
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
    // Gap 2: 3 full fill-drain cycles
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
    // Gap 3: capacity=1, phys=2, indices {0,1}
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
    // Gap 4: pop with nullptr discards value
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