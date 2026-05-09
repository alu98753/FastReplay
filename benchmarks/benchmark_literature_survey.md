# FastReplay Benchmark 文獻調查 (Literature Survey)

> 本文件基於 `lib/` 目錄下所有 6 個對標框架的原始碼、論文及技術文件，詳細分析各自的 Benchmark 方法論 (methodology)、指標 (metrics) 與測試條件 (test conditions)，並評估其對 FastReplay 的適用性。

---

## 一、底層佇列庫（純資料結構層）

### 1. SPSCQueue (Rigtorp)
**來源**：[lib/SPSCQueue/](file:///home/clu98753cs13/course/NSD/FastReplay/lib/SPSCQueue/)

#### Benchmark 方法論
SPSCQueue 的 Benchmark 分為兩個獨立的測試，設計極為精簡且聚焦：

**測試 A：吞吐量測試 (Throughput Benchmark)**
- **方法**：一個生產者執行緒連續 `emplace` 10,000,000 個 `int`，一個消費者執行緒連續 `pop`
- **測量方式**：從第一個 `emplace` 到最後一個 `pop` 被消費完的總時間
- **指標**：**ops/ms**（每毫秒操作次數）
- **資料型態**：`int`（4 bytes），刻意使用最小元素以隔離「佇列邏輯開銷」

**測試 B：延遲測試 (Latency / Round-Trip Time Benchmark)**
- **方法**：兩個佇列 q1, q2 構成一個「乒乓 (Ping-Pong)」通道。Thread A 寫入 q1 → Thread B 從 q1 讀取並寫入 q2 → Thread A 從 q2 讀取
- **測量方式**：完成 10,000,000 次完整往返的總時間 / 迭代次數
- **指標**：**ns RTT**（Round-Trip Time，奈秒級往返延遲）

#### 關鍵技術細節
- **CPU 綁核 (CPU Pinning)**：使用 `pthread_setaffinity_np` 將兩個執行緒分別綁定到指定的 CPU 核心，排除作業系統排程 (scheduling) 造成的測量雜訊
- **計時器**：使用 `std::chrono::steady_clock`（單調時鐘，不受系統時間調整影響）
- **對照組**：直接在同一份程式碼中比較 `boost::lockfree::spsc_queue` 與 `folly::ProducerConsumerQueue`

#### 報告結果範例（AMD Ryzen 9 3900X）

| Queue | Throughput (ops/ms) | Latency RTT (ns) |
|---|---:|---:|
| SPSCQueue | 362,723 | 133 |
| boost::lockfree::spsc | 209,877 | 222 |
| folly::ProducerConsumerQueue | 148,818 | 147 |

> [!IMPORTANT]
> **對 FastReplay 的適用性**：這是與 FastReplay 最直接對標的 Benchmark 模式。FastReplay 的 Atomic 版本應該能夠使用幾乎完全相同的測試架構，直接對比 Mutex 版本 vs Atomic 版本的 ops/ms 與 ns RTT。唯一需要調整的是將 `int` 替換為更接近 RL 場景的資料大小（如 `float[4]` 模擬 CartPole obs）。

---

### 2. MPMCQueue (Rigtorp)
**來源**：[lib/MPMCQueue/](file:///home/clu98753cs13/course/NSD/FastReplay/lib/MPMCQueue/)

#### Benchmark 方法論
MPMCQueue 的 README 中明確標註 Benchmark 尚在 TODO 列表中（`[ ] Add benchmarks`），但其**測試方法論**值得參考：

- **正確性驗證**：單執行緒功能測試（包含建構式/解構式調用正確性）
- **並發模糊測試 (Fuzz Test)**：多執行緒下驗證所有元素均能正確入列/出列

#### 關鍵技術細節
- **Ticket-based 實作**：使用 `turn` 變數控制讀寫順序，避免 CAS (Compare-And-Swap) 迴圈的效能退化
- **False Sharing 防護**：使用 `std::hardware_destructive_interference_size`（C++17）

> [!NOTE]
> **對 FastReplay 的適用性**：MPMCQueue 的架構過於複雜（多生產者多消費者），不直接適用。但其「多執行緒模糊測試」的正確性驗證思路，可以作為 FastReplay 並發壓力測試的參考模板。

---

## 二、RL Replay Buffer 專用庫

### 3. cpprb (Yamada)
**來源**：[lib/cpprb/](file:///home/clu98753cs13/course/NSD/FastReplay/lib/cpprb/)

#### Benchmark 方法論
cpprb 並沒有發表正式的學術論文。其效能宣稱主要來自 GitLab 文件中的「Comparison」頁面（目前已無法存取）。但從其原始碼結構中，可以識別出以下測試策略：

- **單元測試**：[test/](file:///home/clu98753cs13/course/NSD/FastReplay/lib/cpprb/test/) 包含大量 Python 測試（`test_PyReplayBuffer.py` 有 16KB），覆蓋功能正確性
- **Segment Tree 效能**：[test/segmenttree_bench.cpp](file:///home/clu98753cs13/course/NSD/FastReplay/lib/cpprb/test/segmenttree_bench.cpp) (3.6KB) 專門測試 Prioritized Experience Replay 中 Segment Tree 的操作效能
- **多進程測試**：`test_mp.py` (16KB) 測試 Python multiprocessing 環境下的 Buffer 行為

#### 推測的關鍵指標（基於原始碼分析）
- **add/sample 時間**：每次 `add()` 與 `sample()` 的平均耗時
- **Segment Tree 操作時間**：用於量化 PER (Prioritized Experience Replay) 的額外開銷

> [!TIP]
> **對 FastReplay 的適用性**：cpprb 的 `segmenttree_bench.cpp` 是一個非常好的 C++ 微觀測試範本。FastReplay 可以參考其結構撰寫 C++ 端的 push/sample 效能測試。然而，cpprb 使用 Mutex（鎖式設計），這正是 FastReplay 試圖超越的對照組。

---

### 4. Reverb (DeepMind)
**來源**：[lib/reverb/](file:///home/clu98753cs13/course/NSD/FastReplay/lib/reverb/) ｜ 論文：[reverb.txt](file:///home/clu98753cs13/course/NSD/FastReplay/benchmarks/literature/reverb.txt)

#### Benchmark 方法論（論文 Section 5）
Reverb 的效能評估是所有框架中**最嚴謹且最具系統工程深度**的，設計了兩組獨立的壓力測試：

**測試 A：插入效能 (Inserting Benchmark, Section 5.1)**
- **方法**：1 到 200 個分散式 client 同時向單一 Reverb server 插入資料
- **資料特徵**：使用隨機 `float32` 張量（刻意排除壓縮效果），payload 跨越四個數量級：400B, 4KB, 40KB, 400KB
- **Chunk/Sequence 長度**：設為 1（確保 Items 不共享資料，隔離純 I/O 效能）

**測試 B：取樣效能 (Sampling Benchmark, Section 5.2)**
- **方法**：與插入測試相同架構，但所有 client 同時取樣

#### 核心指標

| 指標 | 定義 | 數值範例 |
|---|---|---|
| **BPS (Bytes Per Second)** | 伺服器每秒處理的總位元組數 | 最高 **11 GB/s**（受網路頻寬限制） |
| **QPS (Queries Per Second)** | 伺服器每秒處理的 Item 操作數 | 插入 ~60K；取樣 ~600K |
| **SPI (Sample to Insert Ratio)** | 每插入一筆資料被取樣的次數 | 用於控制梯度更新頻率 |

#### 關鍵發現（直接引用論文）
1. **Mutex 是插入端的主要瓶頸**：取樣 QPS 比插入 QPS 高出 **10 倍**，原因是取樣端已實作了降低 Table mutex contention 的優化，但插入端尚未實作
2. **Sharding 驗證**：為了驗證「Mutex 是瓶頸」的假說，作者將負載分散到同一伺服器上的多個 Table，結果插入 QPS 提升了 **~200%**
3. **BPS 天花板**：11 GB/s 的上限在不同 payload 大小下均一致，強烈暗示 (indicate) 限制來自網路頻寬而非 Reverb 本身

> [!IMPORTANT]
> **對 FastReplay 的適用性**：
> - **BPS 指標**：FastReplay 是 In-process（非分散式），因此 BPS 的天花板不是網路，而是**記憶體頻寬 (Memory Bandwidth)**。這個指標需要重新定義為「記憶體搬運速度」
> - **Mutex Contention 的證明方法**：Reverb 透過 Sharding 實驗來「證明」瓶頸來自 Mutex。FastReplay 可以採用類似的邏輯：直接對比 Mutex 版本 vs Atomic 版本的 QPS，若 Atomic 版本 QPS 顯著提升，即可歸因於 Lock 競爭的消除
> - **跨 Payload 大小的測試**：必須模仿 Reverb，測試不同資料維度（CartPole 的 4 floats vs Atari 的 84×84 image）

---

## 三、RL 訓練框架層

### 5. EnvPool (Sea AI Lab / Weng et al., 2022)
**來源**：[lib/envpool/](file:///home/clu98753cs13/course/NSD/FastReplay/lib/envpool/) ｜ 論文：[envpool.txt](file:///home/clu98753cs13/course/NSD/FastReplay/benchmarks/literature/envpool.txt)

#### Benchmark 方法論（論文 Section 4）
EnvPool 的評估分為兩個層次，與我們先前的 Micro/Macro 分層完美吻合：

**Part 1：純環境模擬 (Pure Environment Simulation, Section 4.1)**
- **方法**：使用隨機取樣的動作 (randomly sampled actions) 作為輸入，純粹測量環境引擎的模擬速度
- **硬體配置**：跨越三種規模——Laptop (12 CPU cores)、Workstation (32 cores)、DGX-A100 (256 cores)
- **環境類型**：Atari Pong（3D 影像，高記憶體佔用）、MuJoCo Ant（1D 向量，低記憶體佔用）
- **測量方式**：50K 次迭代的平均 FPS，Atari 的 frameskip=4、MuJoCo sub-step=5
- **指標**：**FPS (Frames Per Second)**
- **對照組**：For-loop、Subprocess、Sample-Factory、EnvPool (sync/async/numa+async)

**Part 2：端到端訓練 (End-to-End Agent Training, Section 4.2)**
- **方法**：將 EnvPool 整合進 CleanRL、rl_games、Acme 等現有訓練框架
- **指標**：
  1. **Episodic Return vs Frames**（學習曲線，證明 sample efficiency 不受影響）
  2. **Episodic Return vs Runtime (minutes)**（證明 wall-clock 加速）
- **時間分解剖析 (Time Decomposition Profiling)**：EnvPool 將每次迭代的總時間拆解為四個組件：
  1. **Environment Step Time**：`env.step(act)` 的耗時
  2. **Inference Time**：計算動作、log probability、value 的耗時
  3. **Training Time**：前向與反向傳播的耗時
  4. **Other Time**：GPU↔CPU 資料搬移、指標寫入等

#### 關鍵發現
- 在 Laptop + Atari + N=8 的配置下，環境步進時間從 ~73% 降至 ~10%
- End-to-end 訓練時間從 200 分鐘降至 73 分鐘

#### 實際 Benchmark 腳本分析
[benchmark/test_envpool.py](file:///home/clu98753cs13/course/NSD/FastReplay/lib/envpool/benchmark/test_envpool.py) 的實作非常簡潔：
```python
t = time.time()
for _ in tqdm.trange(args.total_step):
    info = env.recv()[-1]
    env.send(action, info["env_id"])
duration = time.time() - t
fps = total_step * batch_size / duration * frame_skip
```
- 使用 `time.time()` 進行 wall-clock 計時
- FPS 計算公式：`(總步數 × batch_size × frame_skip) / 總秒數`

> [!IMPORTANT]
> **對 FastReplay 的適用性**：
> - **時間分解方法論**是 FastReplay 宏觀測試中最重要的參考。FastReplay 不是環境引擎，而是 Buffer，因此我們的時間分解應調整為：
>   1. `Buffer_Push_Time`：`buffer.push(transition)` 的耗時
>   2. `Buffer_Sample_Time`：`buffer.sample(batch_size)` 的耗時
>   3. `NN_Forward_Time`：推論耗時
>   4. `NN_Backward_Time`：訓練耗時
>   5. `Other_Time`：環境步進 + GPU↔CPU 搬移
> - **跨硬體測試**：EnvPool 在 Laptop / Workstation / DGX-A100 三個環境下測試。FastReplay 至少需要在 WSL (CPU-only) 與 RTX 4090 上測試

---

### 6. Tianshou (Weng et al., 2022)
**來源**：[lib/tianshou/](file:///home/clu98753cs13/course/NSD/FastReplay/lib/tianshou/) ｜ 論文：[tianshou.txt](file:///home/clu98753cs13/course/NSD/FastReplay/benchmarks/literature/tianshou.txt)

#### Benchmark 方法論（論文 Section 2 & Benchmark 網站）
Tianshou 的 Benchmark **不聚焦於底層效能**，而是聚焦於**演算法正確性與收斂品質**：

- **環境**：OpenAI Gym MuJoCo 任務套件（9 個環境）
- **演算法**：8 個經典演算法（PPO, SAC, DDPG, TD3 等）
- **指標**：**Median Performance**（各環境正規化分數的中位數，與 reference implementations 比較）
- **統計方法**：每個實驗使用 **10 個隨機種子**
- **目標**：證明 Tianshou 的實作品質，而非底層資料結構的速度

#### 與 Buffer 效能相關的觀察
Tianshou 論文中**完全沒有**針對 ReplayBuffer 的效能數據。其架構圖（Figure 1）顯示 Buffer 位於「Encapsulation Layer」，被視為基礎設施而非研究亮點。

> [!NOTE]
> **對 FastReplay 的適用性**：
> - Tianshou 本身不提供 Buffer 效能數據，但**這正是 FastReplay 的切入點**：證明替換 Tianshou 的 Python ReplayBuffer 為 FastReplay 的 C++ Buffer 後，能獲得可量化的效能增益
> - Tianshou 的 `Batch` 物件使用 SoA 設計，與 FastReplay 的 `DictBuffer` 架構高度吻合，使整合成本極低

---

## 四、指標對照表：各框架的 Benchmark 設計總覽

| 框架 | 層級 | 主要指標 | 資料型態 | 對照組 | 計時工具 | CPU Pinning |
|---|---|---|---|---|---|---|
| **SPSCQueue** | Queue | ops/ms, ns RTT | `int` | boost, folly | `steady_clock` | ✅ |
| **MPMCQueue** | Queue | *(TODO)* | `int` | *(TODO)* | — | ✅ |
| **cpprb** | Buffer | add/sample time | RL Transition | Python baseline | *(推測 `time.time`)* | ❌ |
| **Reverb** | Buffer+Transport | **BPS**, **QPS** | Random float32 (400B–400KB) | 自身不同 client 數 | *(未公開)* | ❌ |
| **EnvPool** | Env Engine | **FPS** | Atari/MuJoCo frames | Subprocess, Sample-Factory | `time.time` | ✅ (NUMA) |
| **Tianshou** | Framework | Median Performance | RL episodes | Other RL libs | N/A | ❌ |

---

## 五、FastReplay 應該採用的指標與原因

基於以上調查，FastReplay 的 Benchmark 應建立以下指標體系：

### Micro-benchmark 層（不需 GPU，WSL 可執行）

| 指標 | 來源框架 | 定義 | 為什麼要測 |
|---|---|---|---|
| **ops/ms** | SPSCQueue | 每毫秒完成的 push+pop 操作數 | 直接對比 Mutex vs Atomic 的佇列邏輯開銷 |
| **ns RTT** | SPSCQueue | 單次往返延遲 | 量化 memory barrier 的具體開銷 |
| **BPS (MB/s)** | Reverb | 每秒搬運的位元組數 | 識別是否達到記憶體頻寬天花板 |
| **跨 Payload 測試** | Reverb | 在不同資料大小下重複測試 | 區分「佇列邏輯開銷」vs「記憶體搬運開銷」 |

### Macro-benchmark 層（需 GPU，RTX 4090 執行）

| 指標 | 來源框架 | 定義 | 為什麼要測 |
|---|---|---|---|
| **Buffer Time Proportion** | EnvPool (改良) | `(Buffer_Push + Buffer_Sample) / Total_Time` | 證明 Buffer 在高速 GPU 下佔據顯著比例 |
| **SPS (Steps Per Second)** | EnvPool | 每秒完成的完整訓練步數 | End-to-end 效能的最終評判標準 |
| **時間分解 (Time Decomposition)** | EnvPool | 將每次迭代拆解為 4-5 個組件 | 精確定位效能增益來自何處 |
| **Episodic Return vs Runtime** | EnvPool + Tianshou | 學習曲線 vs wall-clock 時間 | 證明加速不犧牲 sample efficiency |
