# BÁO CÁO PHÂN TÍCH KỸ THUẬT VÀ CHIẾN LƯỢC GPU FINOPS
## Dự Án Tối Ưu Hóa Chi Phí Hạ Tầng GPU — NimbusAI
**Học viên:** Đậu Quang Ý (MSHV: 2A202602661)  
**Khóa học:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25  
**Vị trí giả định:** Lead GPU FinOps Engineer, NimbusAI  
**Thời gian hoàn thành:** Tháng 10/2026  

---

## 1. Tổng Quan Hiệu Quả: Baseline vs. Optimized

Trong vai trò Lead FinOps Engineer tại NimbusAI, tôi đã tiến hành kiểm toán toàn diện telemetry 11 GPU, log 2.400 requests mô hình ngôn ngữ lớn (LLM), cùng 8 workloads huấn luyện và inference. Kết quả tối ưu hóa tổng thể như sau:

| Chỉ số tài chính & hiệu năng | Baseline (Chưa tối ưu) | Optimized (Sau tối ưu) | Cải thiện / Tiết kiệm |
|---|---|---|---|
| **Tổng chi phí hàng tháng** | **$27,133 / tháng** | **$14,626 / tháng** | **-$12,507 / tháng (46.1%)** |
| **Chi phí Inference / ngày** | $48.87 / ngày | $8.48 / ngày | -$40.39 / ngày (82.6%) |
| **Đơn giá phục vụ token** | **$6.488 / 1M-token** | **$1.126 / 1M-token** | **Giảm 5.76× (82.6%)** |
| **Chi phí mua GPU (Purchasing)** | $25,667 / tháng | $15,627 / tháng | -$10,040 / tháng (39.1%) |
| **Lãng phí GPU treo (Idle waste)**| $600 / tháng ($20/ngày) | $0 / tháng | Tiết kiệm 100% idle waste |
| **Hiệu suất phục vụ năng lượng** | 1.89 kWh/ngày (Non-reasoning) | Cắt giảm 14.9 kWh/ngày | Tiết kiệm 50% reasoning energy |

> **Nhận định cốt lõi:** Thay vì cố gắng cắt giảm số giờ thuê máy móc thô sơ, phương pháp tiếp cận FinOps hiện đại tập trung vào việc chuẩn hóa chi phí theo đơn vị giá trị kinh doanh: **`$/1M-token`** giảm từ **$6.488** xuống chỉ còn **$1.126**.

---

## 2. Phân Tích Đóng Góp Của Từng Đòn Bẩy (FinOps Levers)

Khoản ngân sách tiết kiệm **$12,507 / tháng** được phân rã thành 4 đòn bẩy chính:

```
+-----------------------------------------------------------------------------+
|                     PHÂN BỔ TIẾT KIỆM THEO ĐÒN BẨY                          |
+-----------------------------------------------------------------------------+
| 1. Purchasing (Spot & Reserved Commitments)    : $10,040 / tháng  (80.3%)   |
| 2. Inference Levers (Cascade, Cache, Batch)   : $1,212 / tháng   ( 9.7%)   |
| 3. Right-sizing "GPU-Util Lies"                : $655 / tháng     ( 5.2%)   |
| 4. Kill Idle GPUs (Reaping qua đêm)            : $600 / tháng     ( 4.8%)   |
+-----------------------------------------------------------------------------+
| TỔNG TIẾT KIỆM DỰ KIẾN                         : $12,507 / tháng  (100%)    |
+-----------------------------------------------------------------------------+
```

### Tại sao Purchasing Optimization đóng góp lớn nhất (80.3%)?
1. **Khối lượng GPU-giờ khổng lồ:** Huấn luyện (training) và tinh chỉnh (fine-tuning) các mô hình nền tảng tiêu tốn cụm 8× H100 chạy hàng trăm giờ mỗi tháng. Trên hóa đơn đám mây, chi phí compute thô của training chiếm tới 94.6% tổng ngân sách ban đầu ($25,667 / $27,133).
2. **Chiết khấu sâu từ nhà cung cấp:** 
   - Hợp đồng **Reserved 3-Year** mang lại mức chiết khấu **45%** cho các dịch vụ 24/7 ổn định (chỉ cần duty cycle vượt điểm hòa vốn $1 - 0.45 = 55\%$, tương đương 13.2 giờ/ngày).
   - **Spot Instance + Checkpointing** giảm giá từ **$2.50/hr** xuống **$1.50/hr** (chiết khấu 40%) cho H100 với tỷ lệ gián đoạn thực tế chỉ ~3–5% và checkpoint overhead chỉ 3%.

---

## 3. Bản Chất Của "GPU-Util Lie" Và Tác Động Tài Chính

### Hiện tượng phát hiện trong Telemetry
Khi rà soát cụm 11 GPU thông qua hàm `metrics.flag_util_lies()` (ngưỡng: `gpu_util_pct >= 90%` nhưng `MFU < 30%`), hệ thống phát hiện 2 node bất thường nghiêm trọng:
- **`gpu-h100-4`**: `gpu_util_pct = 98.2%`, nhưng **MFU chỉ đạt 0.194** (19.4%) và **MBU đạt 0.207**.
- **`gpu-a10g-1`**: `gpu_util_pct = 96.9%`, nhưng **MFU chỉ đạt 0.268** (26.8%) và **MBU đạt 0.302**.

### Cơ chế phần cứng (Hardware Root Cause)
1. **`nvidia-smi` đo gì?** Công cụ đo thời gian của xung nhịp (clock time activity) — chỉ cần trong 1 giây có ít nhất 1 Streaming Multiprocessor (SM) đang có kernel hoạt động (dù chỉ là 1 warp chạy lệnh nạp bộ nhớ scalar), `nvidia-smi` sẽ ghi nhận util đạt 100%.
2. **Memory Wall & LLM Decode Bottleneck:**
   - Trong pha **Prefill** (xử lý prompt đầu vào), tính toán là ma trận nhân ma trận lớn (GEMM), có chỉ số Arithmetic Intensity rất cao (~455 FLOP/byte), nằm ở vùng **Compute-Bound** trên mô hình Roofline.
   - Trong pha **Decode** (sinh từng token tiếp theo), thuật toán là ma trận nhân vector (GEMV), Arithmetic Intensity tụt xuống chỉ còn **1–2 FLOP/byte**, hoàn toàn nằm ở vùng **Memory-Bound**.
   - Với batch size nhỏ, Tensor Cores của H100 (đỉnh 990 TFLOPs FP16) phải chờ nạp ma trận trọng số từ HBM3 memory bandwidth (3.35 TB/s). Các bóng bán dẫn Tensor Cores rảnh rỗi (stalled) tới hơn 75% thời gian để đợi dữ liệu từ bộ nhớ.
3. **Tác động tài chính:**
   - NimbusAI đang trả nguyên giá On-demand đắt đỏ **$2.50 / GPU-giờ** cho H100 để nhận lại hiệu năng tính toán chưa bằng 1 con chip giá $1.00/giờ.
   - Việc hạ cấp (`Right-sizing`) `gpu-h100-4` sang **A100 ($1.79/hr)** giúp tiết kiệm ngay **$0.71/giờ ($511/tháng)** mà thông lượng token sinh ra không hề suy giảm, do băng thông đạt được (achieved BW = 0.69 TB/s) hoàn toàn nằm trong khả năng của A100 (peak BW = 2.00 TB/s).

---

## 4. Chi Tiết 5 Phần Mở Rộng "Your Turn" Đã Triển Khai

Cả 5 extensions đều được triển khai với thuật toán định lượng, đo lường thực nghiệm trên tập dữ liệu và có unit test kiểm chứng độc lập trong [`test_extensions.py`](file:///c:/Users/iDEAPAD/Downloads/Day25/K4-L3-DAY25-DauQuangY-2A202602661-GpuFinOps/tests/test_extensions.py):

### D.1 — Extension 1: Chính sách mua linh hoạt đa yếu tố (`recommend_tier`)
- **Triển khai:** Nâng cấp `pricing.recommend_tier` xem xét đồng thời: (1) Tỷ lệ gián đoạn theo từng dòng GPU (H100/H200 có spot interruption 3%, trong khi A10G là 12%); (2) Thời hạn dự án (dự án ngắn hạn < 1 năm không mua hợp đồng Reserved 3 năm mà đối chiếu với Reserved 1 năm 30% discount).
- **Kết quả đo lường:** Chi phí theo chính sách an toàn đa yếu tố là **$17,679/tháng** (tiết kiệm 31.1%) so với chính sách ngây thơ **$15,627/tháng**. Sự chênh lệch phản ánh rủi ro không bị "khóa cứng vốn" (capital lock-in) 3 năm vào các dự án thử nghiệm chỉ kéo dài 1–2 tuần.

### D.2 — Extension 2: Right-sizing theo MBU & Đơn Giá VRAM (`$/GB-VRAM`)
- **Triển khai:** Xây dựng ma trận phân tích `$/GB-VRAM-hr` cho toàn bộ 7 dòng GPU trong catalog:
  - H100: $0.0312/GB-hr | A100: $0.0224/GB-hr | A10G: $0.0417/GB-hr | MI300X: $0.0102/GB-hr.
  - Phân tích achieved memory bandwidth của các instance memory-bound (MBU < 0.35) và tìm node thay thế có giá rẻ hơn nhưng đủ băng thông.
- **Kết quả đo lường:** Đề xuất 4 khuyến nghị right-sizing khả thi (chuyển `gpu-h100-4` và `gpu-h100-5` sang A100; chuyển `gpu-a100-0` và `gpu-a100-1` sang A10G). **Tổng tiềm năng tiết kiệm: $2,160.00 / tháng**.

### D.3 — Extension 3: Kinh Tế Học Của Prompt Caching (`cache_is_worth_it`)
- **Triển khai:** Hàm `cache_is_worth_it` và `cache_break_even_reads` tính toán ngưỡng hòa vốn số lần đọc:
  $$N_{\text{break-even}} = \frac{\text{write\_cost}}{\text{price\_in} \times (1 - \text{read\_discount})} = \frac{1.25 \times P_{\text{in}}}{P_{\text{in}} \times 0.90} \approx 1.39 \text{ lần đọc}$$
- **Kết quả đo lường:** Với hệ thống chatbot/RAG của NimbusAI, các prefix chỉ dẫn hệ thống (system prompt) có tần suất đọc lặp lại trung bình là **4.5 lần** (> 1.39). Caching chắc chắn sinh lời, mang lại tỷ lệ giảm giá hữu hiệu tới 90% cho 1,703,990 cached tokens trong tập dữ liệu.

### D.4 — Extension 4: Ngân Sách Năng Lượng & Chi Phí Reasoning
- **Triển khai:** Tách bạch hành vi của traffic `is_reasoning=1` vs `is_reasoning=0` dựa trên hệ số tiêu thụ năng lượng $80\times$ của chuỗi suy luận (CoT tokens).
- **Kết quả đo lường:**
  - Reasoning chỉ chiếm **8.4% tổng số requests** (201/2,400) nhưng chiếm **35.6% lượng token đầu ra**.
  - Reasoning tiêu tốn **94.0% tổng năng lượng inference** (29.79 kWh/ngày so với 1.89 kWh/ngày của non-reasoning)!
  - **Chính sách đề xuất:** Áp dụng Semantic Router: chỉ kích hoạt reasoning khi độ phức tạp câu hỏi cao (classifier confidence < 0.85). Giảm 50% reasoning không cần thiết giúp tiết kiệm **$20.95/tháng** và **447 kWh điện/tháng**.

### D.5 — Extension 5: Lịch Trình Nhận Thức Carbon Đa Vùng (Carbon-Aware Scheduling)
- **Triển khai:** Hàm `compare_all_regions` phân tích 4,227 kWh điện hàng tháng của các tác vụ interruptible qua 5 khu vực điện toán đám mây.
- **Kết quả đo lường:**
  - Chạy tại `us-east-1` (lưới điện than/khí): phát thải **1,606.3 kg CO2e / tháng**.
  - Chuyển sang `europe-north1` (lưới điện thủy điện Na Uy): phát thải chỉ còn **126.8 kg CO2e / tháng**.
  - **Mức giảm phát thải:** Cắt giảm **1,479.5 kg CO2e / tháng (giảm 92.1%)**, đồng thời tiền điện rẻ hơn từ $0.12/kWh xuống $0.09/kWh.

---

## 5. Khuyến Nghị Dành Cho Lãnh Đạo NimbusAI (Top 3 Actions)

Nếu được trao quyền FinOps Lead tại NimbusAI, 3 hành động đầu tiên tôi sẽ thực hiện ngay trong 30 ngày tới:

1. **Hành động 1 (Tuần 1 — Khóa ngay 80% lãng phí): Chuyển đổi chính sách mua GPU**
   - Ký kết hợp đồng Reserved 3 năm cho 3 dịch vụ inference cốt lõi (`job-infer-chat`, `job-infer-rag`, `job-infer-search` có duty cycle 75–100%).
   - Bật cờ Spot Instance kèm lưu checkpoint tự động (S3/GCS bucket) cho 5 jobs huấn luyện/batch.
   - *Kết quả tức thì:* Giảm ngay **$10,040 / tháng** vào hóa đơn điện toán đám mây tiếp theo.

2. **Hành động 2 (Tuần 2 — Tối ưu hóa Inference Engine): Triển khai Cascade & Prompt Caching Gateway**
   - Đặt bộ định tuyến thông minh (LiteLLM Proxy hoặc vLLM router): 60% tác vụ hội thoại cơ bản chuyển từ model Large ($15/1M out) sang Small ($0.40/1M out).
   - Bật Context Caching cho toàn bộ tài liệu RAG dài > 1,024 tokens.
   - *Kết quả:* Giảm chi phí token từ **$6.49** xuống **$1.13 / 1M-token**, sẵn sàng mở rộng quy mô người dùng gấp 5 lần mà không lo đội ngân sách.

3. **Hành động 3 (Tuần 3–4 — Thiết lập Văn hóa Trách nhiệm): Kích hoạt Chargeback theo chuẩn FOCUS**
   - Với tỷ lệ gắn nhãn (tag coverage) đạt **92%** (vượt ngưỡng 80%), chuyển ngay từ chế độ Showback sang **Chargeback**.
   - Xuất dữ liệu chi phí định kỳ qua `outputs/focus_export.csv` để đối soát ngân sách trực tiếp trừ vào P&L của từng phòng ban (`assistant`, `search`, `eval`, `rag`), buộc các kỹ sư phải chủ động tắt GPU chạy không (idle reaper).

---

## 6. Phụ Lục: Trả Lời Câu Hỏi Kiểm Tra Hiểu Biết (Oral Check)

### 1. "GPU-Util 98% có nghĩa là GPU đang làm việc hiệu quả không? Tại sao?"
> **Trả lời:** Hoàn toàn **KHÔNG**. `nvidia-smi` chỉ đo thời gian mà Streaming Multiprocessor (SM) có xung nhịp hoạt động, không đo mật độ tính toán thực tế. Khi chạy LLM Decode với batch size nhỏ, ma trận tính toán nhỏ khiến Tensor Cores phải chờ nạp trọng số từ bộ nhớ HBM (memory stall), chỉ số Model FLOPs Utilization (MFU) chỉ đạt ~20% dù GPU-Util báo 98%. Bạn đang trả tiền cho cả cỗ máy H100 nhưng chỉ tận dụng được 1/5 sức mạnh tính toán.

### 2. "Tại sao cần ≥ 80% tag coverage mới dám chargeback?"
> **Trả lời:** Showback là hiển thị chi phí để nâng cao nhận thức, trong khi Chargeback là thực sự khấu trừ tiền từ ngân sách hoạt động của các nhóm. Nếu tỷ lệ tag coverage dưới 80%, có tới >20% chi phí không rõ chủ sở hữu (untagged/noise). Nếu áp đặt phân bổ hoặc chargeback trên dữ liệu thiếu sót, sẽ tạo ra xung đột gay gắt giữa các bộ phận kỹ thuật và làm méo mó bức tranh kinh tế thực tế.

### 3. "Nếu công ty bạn có 70% workload interruptible, bạn sẽ tối ưu purchasing như thế nào?"
> **Trả lời:** 
> - Với 70% workload có thể gián đoạn (thường là offline batch processing, training, nightly evaluation), ta sẽ đưa toàn bộ lên **Spot Instances** kết hợp với cơ chế tự động lưu checkpoint vào object storage phân tán (overhead ghi checkpoint chỉ ~3%). 
> - Thiết lập cơ chế đa vùng (multi-region / multi-AZ spot bidding): nếu một loại GPU bị thu hồi, scheduler tự động tái khởi động trên loại GPU tương đương hoặc ở vùng khác.
> - 30% còn lại là production inference phục vụ người dùng 24/7 sẽ dùng **1-Year hoặc 3-Year Reserved / Savings Plans** để tối ưu chiết khấu. Chiến lược này giúp cắt giảm 50–70% tổng hóa đơn mua phần cứng.

### 4. "Đo bằng $/GPU-hr vs $/1M-token — khi nào con số này cho kết quả trái ngược nhau?"
> **Trả lời:** Hai chỉ số sẽ trái ngược nhau khi có sự chênh lệch lớn về **hiệu quả tối ưu phần mềm** (vLLM PagedAttention, TensorRT-LLM, batching, quantization FP8/FP4).
> - *Ví dụ:* Cụm máy A thuê H100 với giá rẻ $2.00/GPU-hr nhưng chạy script Python cơ bản, batch=1, không cache, chỉ phục vụ được 100,000 token/giờ $\rightarrow$ Chi phí là **$20.00 / 1M-token**.
> - Cụm máy B thuê H100 giá $2.50/GPU-hr (đắt hơn theo giờ), nhưng áp dụng Continuous Batching, KV-cache sharing và FP8, phục vụ tới 2,500,000 token/giờ $\rightarrow$ Chi phí chỉ là **$1.00 / 1M-token**.
> - Đo theo `$/GPU-hr` tưởng cụm A rẻ hơn, nhưng thực tế đo theo đơn vị giá trị doanh thu `$/1M-token` thì cụm B rẻ hơn gấp **20 lần**.

### 5. "Tại sao LLM decode là memory-bound còn prefill là compute-bound?"
> **Trả lời:**
> - Trong pha **Prefill**, mô hình tiếp nhận toàn bộ chuỗi prompt (ví dụ 2,048 tokens) cùng lúc. Phép toán là ma trận nhân ma trận $W \times X$ (GEMM), khối lượng tính toán là $O(B \times S \times D^2)$ với $S$ lớn. Tỷ số FLOPs trên mỗi byte nạp từ bộ nhớ (Arithmetic Intensity) rất cao (~455 FLOP/byte trên H100), vượt xa điểm ridge point của phần cứng (~295 FLOP/byte), nên chip bị nghẽn bởi tốc độ nhân ma trận $\rightarrow$ **Compute-bound**.
> - Trong pha **Decode**, mô hình sinh từng token một cách tuần tự. Ở mỗi bước, toàn bộ ma trận trọng số hàng chục tỷ tham số phải được nạp lại từ bộ nhớ HBM qua bus chỉ để nhân với đúng một token mới (ma trận nhân vector GEMV). Arithmetic Intensity tụt xuống chỉ còn **1–2 FLOP/byte**, thấp hơn rất nhiều so với ridge point, do đó tốc độ sinh token bị khống chế hoàn toàn bởi băng thông đọc bộ nhớ (Memory Bandwidth) $\rightarrow$ **Memory-bound**.
