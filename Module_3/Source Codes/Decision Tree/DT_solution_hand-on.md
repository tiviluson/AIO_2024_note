## Bài toán Decision Tree - Giải tay

### Dữ liệu ban đầu:

Ta có 10 dòng dữ liệu đầu tiên để huấn luyện cây quyết định.

**Dữ liệu:**
| Outlook | Temperature | Humidity | Wind | PlayTennis |
|-----------|-------------|----------|--------|------------|
| Sunny | Hot | High | Weak | No |
| Sunny | Hot | High | Strong | No |
| Overcast | Hot | High | Weak | Yes |
| Rain | Mild | High | Weak | Yes |
| Rain | Cool | Normal | Weak | Yes |
| Rain | Cool | Normal | Strong | No |
| Overcast | Cool | Normal | Strong | Yes |
| Sunny | Mild | High | Weak | No |
| Sunny | Cool | Normal | Weak | Yes |
| Rain | Mild | Normal | Weak | Yes |

### Bước 1: Tính Entropy tổng

Tổng số dòng: 10 (4 dòng **No**, 6 dòng **Yes**).

$$
Entropy(S) = - \left( \frac{4}{10} \log_2 \frac{4}{10} \right) - \left( \frac{6}{10} \log_2 \frac{6}{10} \right) = 0.971
$$

### Bước 2: Tính Information Gain cho từng thuộc tính

#### Thuộc tính **Outlook**:

| Outlook  | Số lượng | Số lượng Yes | Số lượng No | Entropy |
| -------- | -------- | ------------ | ----------- | ------- |
| Overcast | 2        | 2            | 0           | 0.000   |
| Rain     | 4        | 3            | 1           | 0.811   |
| Sunny    | 4        | 1            | 3           | 0.811   |

Tính Information Gain:

$$
IG(Outlook) = 0.971 - \left( \frac{2}{10} \times 0.000 + \frac{4}{10} \times 0.811 + \frac{4}{10} \times 0.811 \right) = 0.322
$$

#### Thuộc tính **Temperature**:

| Temperature | Số lượng | Số lượng Yes | Số lượng No | Entropy |
| ----------- | -------- | ------------ | ----------- | ------- |
| Cool        | 4        | 3            | 1           | 0.811   |
| Hot         | 3        | 1            | 2           | 0.918   |
| Mild        | 4        | 2            | 2           | 0.918   |

Tính Information Gain:

$$
IG(Temperature) = 0.971 - \left( \frac{3}{10} \times 0.811 + \frac{3}{10} \times 0.918 + \frac{4}{10} \times 0.918 \right) = 0.0955
$$

#### Thuộc tính **Humidity**:

| Humidity | Số lượng | Số lượng Yes | Số lượng No | Entropy |
| -------- | -------- | ------------ | ----------- | ------- |
| High     | 5        | 2            | 3           | 0.971   |
| Normal   | 5        | 4            | 1           | 0.722   |

Tính Information Gain:

$$
IG(Humidity) = 0.971 - \left( \frac{5}{10} \times 0.971 + \frac{5}{10} \times 0.722 \right) = 0.1245
$$

#### Thuộc tính **Wind**:

| Wind   | Số lượng | Số lượng Yes | Số lượng No | Entropy |
| ------ | -------- | ------------ | ----------- | ------- |
| Strong | 4        | 1            | 3           | 0.918   |
| Weak   | 6        | 5            | 1           | 0.863   |

Tính Information Gain:

$$
IG(Wind) = 0.971 - \left( \frac{4}{10} \times 0.918 + \frac{6}{10} \times 0.863 \right) = 0.0913
$$

### Bước 3: Chọn thuộc tính phân chia tốt nhất

- **Outlook** có Information Gain cao nhất (0.322), vì vậy ta chọn thuộc tính này để phân chia đầu tiên.

### Bước 4: Phân chia nhánh theo Outlook

#### Nhánh **Outlook = Overcast**:

- Tất cả đều là **Yes**, dừng lại ở đây.

#### Nhánh **Outlook = Rain**:

Tính entropy cho các giá trị của thuộc tính **Wind**:
| Wind | Số lượng | Số lượng Yes | Số lượng No | Entropy |
|--------|----------|--------------|-------------|---------|
| Strong | 1 | 0 | 1 | 0.000 |
| Weak | 3 | 3 | 0 | 0.000 |

Information Gain cho Wind:

$$
IG(Wind) = 0.811 - \left( \frac{1}{4} \times 0.000 + \frac{3}{4} \times 0.000 \right) = 0.811
$$

- Chọn thuộc tính **Wind** để phân chia tiếp. Các nhánh con sẽ là:
  - **Wind = Strong**: Kết quả là **No**.
  - **Wind = Weak**: Kết quả là **Yes**.

#### Nhánh **Outlook = Sunny**:

Tính entropy cho các giá trị của thuộc tính **Temperature**:
| Temperature | Số lượng | Số lượng Yes | Số lượng No | Entropy |
|-------------|----------|--------------|-------------|---------|
| Cool | 1 | 1 | 0 | 0.000 |
| Hot | 2 | 0 | 2 | 0.000 |
| Mild | 1 | 0 | 1 | 0.000 |

Information Gain cho Temperature:

$$
IG(Temperature) = 0.811 - \left( \frac{1}{4} \times 0.000 + \frac{2}{4} \times 0.000 + \frac{1}{4} \times 0.000 \right) = 0.811
$$

- Chọn thuộc tính **Temperature** để phân chia tiếp. Các nhánh con sẽ là:
  - **Temperature = Cool**: Kết quả là **Yes**.
  - **Temperature = Hot**: Kết quả là **No**.
  - **Temperature = Mild**: Kết quả là **No**.

### Cây quyết định hoàn chỉnh:

```
Outlook
├── Overcast: Yes
├── Rain
│   ├── Wind = Strong: No
│   └── Wind = Weak: Yes
└── Sunny
    ├── Temperature = Cool: Yes
    ├── Temperature = Hot: No
    └── Temperature = Mild: No
```

### Bước 5: Dự đoán và Đánh giá

Sử dụng cây quyết định để dự đoán 4 dòng dữ liệu cuối cùng (dữ liệu kiểm tra):

| Thực tế | Dự đoán |
| ------- | ------- |
| Yes     | No      |
| Yes     | Yes     |
| Yes     | Yes     |
| No      | No      |

**Độ chính xác**:

$$
Accuracy = \frac{3}{4} = 0.75 = 75\%
$$
