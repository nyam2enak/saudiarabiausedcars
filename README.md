# saudiarabiausedcars
Menganalisis dan memodelkan data mobil bekas yang dijual di Arab Saudi untuk memahami faktor-faktor yang memengaruhi harga jual dan membantu dalam prediksi harga secara otomatis.

Tujuannya :

1. Memprediksi harga mobil bekas berdasarkan fitur seperti merek, tahun, jarak tempuh, ukuran mesin, dan opsi kendaraan.
2. Mengidentifikasi fitur paling berpengaruh terhadap harga (misalnya: apakah merek atau tahun lebih penting
3. Membantu pembeli dan penjual dalam membuat keputusan harga yang lebih akurat
4. Mengembangkan model machine learning yang bisa digunakan dalam aplikasi jual-beli mobil atau sistem rekomendasi.

Fitur dan Target pada Dataset : 
-	Type: Tipe atau model mobil
-	Region: Wilayah atau regional mobil dijual
-	Make: Brand atau merk mobil
-	Gear_Type: Manual atau Otomatis
-	Origin: Asal mobil 
-	Options: Kategori aksesoris yang dipaketkan dalam penjualan mobil.
-	Year: Tahun produksi mobil
-	Engine_Size: Kapasitas mesin mobil
-	Mileage: Jarak tempuh yang sudah dilewati oleh mobil
-	Negotiable: Status mobil bisa ditawar atau tidak.
-   Price : harga mobil yang dijadikan target untuk prediksi.


Untuk menjalankan file notebook (*.ipynb) :
1. buka Visual studio code, dan buka file NoviantoChris_capstone3_Revisi2.ipyb
2. pastikan file data_saudi_used_cars.csv berada pada satu folder yang sama dengan file notebooknya
3. pilih kernel python untuk menjalakan program notebooknys
4. klik tombol run all untuk melihat hasil eksekusi setiap cell kode

Untuk menjalankan file predict.py :
1. buka Visual Studio Code dan buka file predict.py
2. pastikan file xgboost_model_final.sav berada satu folder dengan file programnya
3. Pada window terminal ketik streamlit run predict.py
4. progam akan terbuka di browser
