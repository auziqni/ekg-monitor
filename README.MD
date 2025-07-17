# EKG monitor python

ekg monitoring application with ESP32 and ADS1115

## 🛠️ Developer Guide (How to Build the App)

Langkah-langkah untuk membangun aplikasi dari awal:

### 1. Buat Folder Proyek & File Dasar

```bash
mkdir ekg-monitor
cd ekg-monitor
touch main.py README.md .gitignore
```

### 2. Inisialisasi Git

```bash
git init
git add .
git commit -m "initial commit"
```

### 3. Buat Virtual Environment

```bash
python -m venv env
```

### 4. Aktivasi Virtual Environment

#### Windows

```bash
env\Scripts\activate
```

#### MacOS/Linux

```bash
source env/bin/activate
```

### 5. Install Dependensi

```bash
pip install numpy pyqtgraph pyqt5 wfdb
```

### 6. Simpan ke `requirements.txt`

```bash
pip freeze > requirements.txt
```

### 7. Menonaktifkan Virtual Environment

```bash
deactivate
```

---

## 🚀 Client Guide (How to Use the App)

Langkah-langkah untuk menjalankan aplikasi ini di komputer lain:

### 1. Clone Repository

```bash
git clone <url-repository-anda>
cd ekg-monitor
```

### 2. Buat Virtual Environment

```bash
python -m venv env
```

### 3. Aktivasi Virtual Environment

#### Windows

```bash
env\Scripts\activate
```

#### MacOS/Linux

```bash
source env/bin/activate
```

### 4. Install Dependensi

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi

```bash
python main.py
```

### 6. Jika Terjadi Perubahan pada Repository (on-change update)

Untuk menarik update dari repository, gunakan:

```bash
git pull
```

Jika terjadi konflik atau error dan Anda ingin menyamakan ulang kondisi lokal dengan branch utama di remote, gunakan:

```bash
git fetch origin
git reset --hard origin/main
```

⚠️ **PERINGATAN:** Perintah di atas akan menghapus semua perubahan lokal, termasuk file yang telah dibuat namun belum di-commit. Pastikan sudah melakukan backup jika ada data penting.

---

Selamat mencoba! Jika ada masalah, silakan buat issue atau hubungi developer.
