# comfyui_my_img_util

ComfyUI 用の実験用のノードです。

# Simple Image Rotate

角度を指定して画像を回転できます。画像がはみ出た部分は画像サイズが変更されます。

生まれた隙間は、透明画像として出力されます。

mask には、隙間部分を白色として出力します。

![image](https://github.com/user-attachments/assets/6569dee8-f8e6-4bf8-8ee2-6b2fdfbb4100)

# Auto Image Selector

入力のあった画像のうち、有効な画像でなおかつ、ランクの数字が小さいものを優先して返します。

例えば、バイパスされていた場合や、条件によっては生成されない画像があった場合に、代替画像を用意しておく、といった使い方が出来ます。

ランクは小さいものが優先されます。画像の入力が無いものは選択されません。ランクは最小が1で、最も優先されます。同じランクがあった場合は、上に接続されたものが優先されます。

![image](https://github.com/user-attachments/assets/eb658c74-02eb-4064-bce2-c5852e80358f)
