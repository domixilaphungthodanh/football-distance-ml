# Học Machine Learning qua xG (Expected Goals)

[Mở notebook trên Google Colab](https://colab.research.google.com/github/domixilaphungthodanh/football-distance-ml/blob/main/xg/notebooks/learn_xg.ipynb) rồi chọn **Runtime → Run all** để thực hành trong trình duyệt. Cell đầu tự lấy mã nguồn từ GitHub; lần đầu tải dữ liệu 64 trận có thể mất vài phút. Nếu Colab báo thiếu thư viện, chạy `!pip install -r /content/football-distance-ml/requirements.txt` trong một cell rồi chạy lại notebook.

![StatsBomb](https://raw.githubusercontent.com/hudl/open-data/master/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png)

**Nguồn dữ liệu: [StatsBomb Open Data](https://github.com/hudl/open-data)**, các trận FIFA World Cup nam 2022. StatsBomb công bố [dữ liệu giải đấu](https://statsbomb.com/news/statsbomb-release-free-2022-world-cup-data/) và yêu cầu ghi nguồn, dùng logo khi chia sẻ phân tích. Repository này chỉ lưu mã và hướng dẫn; dữ liệu JSON tải về, bảng trích xuất, mô hình và báo cáo được giữ trên máy.

## xG có nghĩa là gì?

Một cú sút có `xG = 0.20` nghĩa là mô hình ước tính **20% khả năng thành bàn** trong các tình huống tương tự. xG là xác suất *trước khi biết kết quả cú sút*, không phải lời khẳng định cú sút này chắc chắn ghi bàn. Tổng xG của đội là tổng xác suất các cú sút, không phải số bàn thắng chắc chắn sẽ có.

Dự án này chỉ xét cú sút trong trận World Cup 2022, bỏ sút luân lưu và phạt đền vì phạt đền có tình huống cố định khác biệt. Mỗi dòng dữ liệu là một cú sút; nhãn `goal = 1` nếu cú sút thành bàn, ngược lại `0`. Các đầu vào: tọa độ sút `x, y`, khoảng cách và góc nhìn khung thành suy ra từ tọa độ, bộ phận cơ thể, kiểu sút, có bị gây áp lực và sút một chạm. Tọa độ StatsBomb dùng sân 120 × 80, cầu môn ở `(120, 40)`. Các trường kết quả cú sút, điểm `statsbomb_xg` có sẵn, vị trí bóng sau cú sút và khung hình cầu thủ **không được dùng làm đầu vào**.

## Làm theo từng bước

Mở **PowerShell** và dán lần lượt các lệnh sau. File `run-xg.cmd` chạy trực tiếp từ PowerShell, kể cả khi máy chặn file `.ps1`:

```powershell
cd E:\football-distance
.\xg\run-xg.cmd prepare
.\xg\run-xg.cmd test
.\xg\run-xg.cmd train
.\xg\run-xg.cmd predict
.\xg\run-xg.cmd notebook
```

Trên máy Windows của bạn hiện nay, `prepare` và `test` chạy được; `train`, `predict` và cell huấn luyện trong notebook vẫn bị chính sách Application Control chặn scikit-learn. File `.cmd` chỉ giúp gọi lệnh từ PowerShell, không thay đổi chính sách Windows. Để chạy trọn bài học ngay, dùng liên kết Colab ở đầu trang.

Muốn **đọc mã Python ngay trong PowerShell**, dùng `Get-Content`. Muốn sửa mã, dùng VS Code:

```powershell
Get-Content .\xg\src\project.py -TotalCount 80
code .\xg\src\project.py
```

`prepare` tạo bảng cú sút; `train` huấn luyện và lưu mô hình; `predict` chạy ví dụ trong `xg/examples/shot.json`; `notebook` mở bài học. Bạn cũng có thể chạy từng lệnh Python trực tiếp từ PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe xg\src\project.py prepare
.\.venv\Scripts\python.exe xg\src\project.py train
.\.venv\Scripts\python.exe -m notebook xg\notebooks\learn_xg.ipynb
.\.venv\Scripts\python.exe xg\src\project.py predict --input xg\examples\shot.json
.\.venv\Scripts\python.exe -m unittest discover -s xg\tests -v
```

Nếu chưa có `.venv`, chạy `python -m venv .venv` trước. Các lệnh gọi trực tiếp `python.exe`, nên PowerShell không cần cho phép chạy file `.ps1`. Chọn kernel **Football ML (.venv)** trong Jupyter. Nếu kernel chưa có, chạy:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name football-ml --display-name "Football ML (.venv)"
```

`prepare` tải danh sách trận và sự kiện của 64 trận lần đầu; có thể mất vài phút. Lần sau dữ liệu được đọc từ `xg/data/raw/`. Chạy notebook bằng **Restart Kernel & Run All** sau khi chuẩn bị dữ liệu. Notebook gọi cùng hàm với CLI nên kết quả thống nhất.

1. **Đọc dữ liệu:** `prepare` lọc sự kiện Shot, chuyển thành bảng và kiểm tra đủ 64 trận. Xem `xg/data/processed/shots_2022.csv`; mỗi dòng là một cú sút.
2. **Hiểu nhãn và đặc điểm:** Notebook đếm bàn thắng, xem vị trí sút, rồi minh họa phép tính khoảng cách và góc. Đây là dữ liệu để mô hình học mối liên hệ giữa hoàn cảnh sút và bàn thắng.
3. **Chia dữ liệu:** giữ khoảng 20% **trận** làm test với seed 42. Toàn bộ cú sút của một trận đi cùng tập, tránh đánh giá quá lạc quan do các cú sút cùng trận có chung bối cảnh.
4. **Học trên tập train:** so sánh mô hình đoán tỷ lệ bàn chung (`baseline`), hồi quy logistic và rừng ngẫu nhiên. `GroupKFold` chia 5 lượt theo trận; mỗi lượt chỉ học bước điền thiếu, chuẩn hóa và mã hóa từ phần train của lượt đó.
5. **Chọn và đánh giá:** chọn theo **Brier score** cross-validation thấp nhất. Brier là trung bình `(xG - goal)²`, càng thấp càng tốt. Chỉ sau khi chọn xong mới đo trên test. Xem thêm log loss, ROC AUC và biểu đồ hiệu chỉnh. Điểm xG của StatsBomb, nếu có, được báo cáo riêng để tham khảo; mô hình học tập không dùng nó làm đầu vào.
6. **Dùng mô hình:** `predict` đọc JSON như `xg/examples/shot.json`, kiểm tra kiểu dữ liệu, trả về `xg` và phần trăm. Nếu tình huống ngoài phạm vi dữ liệu train, kết quả kèm cảnh báo.

Kết quả thật sau khi chạy nằm ở `reports/xg/results.json`, `model_comparison.csv`, `test_predictions.csv`, `evaluation.png`. Mô hình lưu tại `models/xg_world_cup_2022.joblib`. Các file này chỉ có trên máy của bạn. File `joblib` nên chỉ tải từ nguồn bạn tin tưởng.

### Kết quả đã kiểm chứng

[Lượt chạy GitHub Actions](https://github.com/domixilaphungthodanh/football-distance-ml/actions/workflows/xg.yml) đã chạy chuẩn bị dữ liệu, kiểm thử, huấn luyện, dự đoán mẫu và toàn bộ notebook. Dữ liệu có **64 trận, 1.430 cú sút, 152 bàn thắng**. Chia theo trận tạo **51 trận/1.119 cú sút train** và **13 trận/311 cú sút test**. Hồi quy logistic có Brier score cross-validation thấp nhất trong ba mô hình thử nghiệm. Trên test, Brier là **0,0761**, log loss **0,2682**, ROC AUC **0,8110**; baseline Brier **0,0949**. Điểm xG do StatsBomb cung cấp có Brier **0,0691** trên cùng test để tham khảo. Mô hình học tập dùng ít đặc điểm hơn và không dùng điểm StatsBomb làm đầu vào. Kết quả có thể thay đổi nếu dữ liệu nguồn hoặc phiên bản thư viện thay đổi.

Nếu Windows báo `Application Control policy has blocked this file` khi nạp scikit-learn, chính sách ký mã trên máy đang chặn thư viện. Phần `prepare` vẫn chạy được, còn `train`, `predict` và cell huấn luyện trong notebook cần môi trường cho phép scikit-learn. Trên máy đã triển khai, lỗi này được xác nhận trong nhật ký Code Integrity; lượt chạy GitHub Actions ở trên kiểm chứng toàn bộ quy trình trong môi trường Linux.

## Giới hạn

Đây là mô hình học tập từ một giải đấu và một số đặc điểm đơn giản. Góc sút là góc hình học trên tọa độ sân StatsBomb, không tính vị trí thủ môn hoặc hậu vệ. Một cú sút bị chặn vẫn có nhãn `0`. Kết quả trên test phản ánh khả năng áp dụng cho **trận khác cùng giải**, chưa chứng minh cho các giải khác. Brier hoặc AUC tốt không bảo đảm xác suất đã được hiệu chỉnh tốt ở mọi khoảng xG; hãy xem biểu đồ hiệu chỉnh và số cú sút trong từng khoảng.
