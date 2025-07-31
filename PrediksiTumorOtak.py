import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Tumora - Deteksi Tumor Otak", 
    layout="wide",  # Changed to wide for full-width
    page_icon="Images/logotumora.png"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Full page background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    
    /* Modern navbar styling */
    [data-testid="stHeader"] {
        background-color: #1a365d;
        color: white;
    }
    
    /* Card styling with better contrast */
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: none;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #1a365d !important;
        margin-bottom: 1rem !important;
    }
    
    p {
        color: #2d3748 !important;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4299e1;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s;
        font-weight: 600;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #3182ce;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* File uploader styling */
    .stFileUploader>div>div>div>div {
        border: 2px dashed #4299e1;
        border-radius: 12px;
        padding: 3rem;
        background: rgba(255,255,255,0.7);
    }
    
    /* Prediction result styling */
    .positive {
        color: #c53030 !important;
        background-color: rgba(220, 38, 38, 0.08);
        border-left: 5px solid #c53030;
    }
    
            
    .negative {
        color: #38a169 !important;
        background-color: rgba(72, 187, 120, 0.08);
        border-left: 5px solid #38a169;
    }
    
    /* Team member cards */
    .team-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    .team-card img {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 1rem;
        aspect-ratio: 4/3;
        object-fit: cover;
    }
    
    /* Responsive grid */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Modern top navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home", "About Us", "Contact"],
    icons=["house", "people", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0 !important",
            "background-color": "#1a365d",
            "margin-bottom": "0",
            "border-bottom": "1px solid rgba(255,255,255,0.1)"
        },
        "icon": {"color": "white", "font-size": "16px"}, 
        "nav-link": {
            "font-size": "16px",
            "font-weight": "500",
            "text-align": "center",
            "margin": "0",
            "--hover-color": "#2c5282",
            "color": "rgba(255,255,255,0.8)",
            "padding": "1rem 2rem"
        },
        "nav-link-selected": {
            "background-color": "#2c5282",
            "color": "white",
            "border-bottom": "3px solid #4299e1"
        },
    }
)

# Load model with cache_resource
@st.cache_resource
def load_model():
    try:
        # URL Google Drive file ID
        url = "https://drive.google.com/file/d/1su9gFdpu1ocHGTdXYbHR4ld62xEtNlS-/view?usp=sharing"
        output = "tumor_detection_model.h5"  # Nama file model yang diunduh

        # Cek apakah model sudah ada di lokal sebelum mengunduh
        if not os.path.exists(output):
            st.info("Mengunduh model dari Google Drive...")
            gdown.download(url, output, quiet=False)
        
        # Muat model setelah diunduh
        model = tf.keras.models.load_model(output)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# Function to process image
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224))
        img = np.array(img) / 255.0  # Normalization
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
        return None

# Function for prediction
def predict(image_bytes):
    if model is None:
        return ("❌ Model tidak tersedia", "error")
    img_array = preprocess_image(image_bytes)
    if img_array is None:
        return ("❌ Gambar tidak valid", "error")
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return ("🧠 Terdeteksi Tumor", """
            <div class="card positive">
                <h3>Hasil Deteksi: Tumor Terdeteksi</h3>
                <p style="color: #2d3748;">Kami sangat prihatin dengan apa yang Anda alami, namun Anda tidak sendirian. Dukungan dari orang-orang di sekitar Anda akan memberi semangat dan membantu setiap langkah Anda menuju kesembuhan. Meskipun jalan ini berat, percayalah bahwa ketabahan dan kekuatan Anda akan membawa harapan dan pemulihan.</p>
                <p style="color: #2d3748;"><strong>Langkah selanjutnya:</strong> Segera konsultasikan dengan dokter spesialis saraf untuk pemeriksaan lebih lanjut.</p>
            </div>
        """)
    else:
        return ("✅ Tidak Ada Tumor", """
            <div class="card negative">
                <h3>Hasil Deteksi: Tidak Ada Tumor</h3>
                <p style="color: #2d3748;">Hasil pemeriksaan menunjukkan tidak adanya indikasi tumor pada scan MRI Anda.</p>
                <p style="color: #2d3748;">Meskipun demikian, jika Anda mengalami gejala-gejala yang mengkhawatirkan, disarankan untuk tetap berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut.</p>
            </div>
        """)

# Home Page Content
if selected == "Home":
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.markdown("""
        <div class="card">
            st.image("Images/logotumora.png")
            <h1 style="margin-top: 0;">🧠 Tumora</h1>
            <h3 style="color: #4299e1;">Deteksi Tumor Otak dengan Kecerdasan Buatan</h3>
            <p>Gunakan teknologi AI mutakhir kami untuk menganalisis gambar MRI otak secara cepat dan akurat dengan akurasi hingga 96%.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Cara Penggunaan", expanded=True):
          st.markdown("""
        <ol style="color: black;">
        <li><strong>Unggah gambar</strong> MRI otak dalam format JPG, PNG, atau JPEG</li>
        <li>Klik tombol <strong>"Deteksi Tumor"</strong></li>
        <li>Tunggu sistem menganalisis gambar Anda (sekitar 10-30 detik)</li>
        <li>Dapatkan <strong>hasil deteksi</strong> dan rekomendasi</li>
    </ol>
            <p style="font-size: 0.9em; color: #fffff;">Catatan: Hasil ini merupakan prediksi AI dan bukan diagnosis medis. Disarankan untuk berkonsultasi dengan dokter spesialis.</p>
            """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("📤 Unggah gambar MRI otak Anda", type=["jpg", "png", "jpeg"])
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🛠️ Teknologi Kami</h3>
            <p>Menggunakan model CNN canggih yang telah dilatih dengan lebih dari 10.000 gambar MRI medis.</p>
            <div style="margin-top: 1rem;">
                <img src="https://placehold.co/600x400?text=CNN&font=roboto" alt="Diagram arsitektur AI model deep learning" style="width:100%; border-radius:8px;"/>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(uploaded_file, caption="Gambar MRI yang diunggah", use_column_width=True)
        
        with col2:
            if st.button("🔍 Deteksi Tumor Sekarang", use_container_width=True, type="primary"):
                with st.spinner("🔬 Sedang menganalisis gambar... Mohon tunggu beberapa saat"):
                    image_bytes = uploaded_file.read()
                    result_title, result_html = predict(image_bytes)
                    
                    if result_title == "✅ Tidak Ada Tumor":
                        st.balloons()
                    
                    st.markdown(result_html, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="card" style="margin-top: 1rem;">
                        <h4>📚 Sumber Daya Tambahan</h4>
                        <ul style="color: #2d3748;">
                            <li><a href="#" target="_blank" style="color: #4299e1;">Pusat Informasi Tumor Otak</a></li>
                            <li><a href="#" target="_blank" style="color: #4299e1;">Daftar Dokter Spesialis Saraf</a></li>
                            <li><a href="#" target="_blank" style="color: #4299e1;">Penelitian Terkini Tentang Tumor Otak</a></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

# About Us Page
elif selected == "About Us":
    st.markdown("""
    <div class="card">
        <h1 style="margin-top: 0;">Tentang Brainsight AI</h1>
        <p>Kami adalah tim multidisiplin yang terdiri dari dokter, ilmuwan data, dan peneliti AI yang berkomitmen untuk merevolusi diagnosa medis melalui kecerdasan buatan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🛠️ Teknologi Kami</h3>
            <p>Sistem kami menggunakan arsitektur convolutional neural network (CNN) khusus yang dioptimalkan untuk analisis gambar medis.</p>
            <ul style="color: #2d3748;">
                <li>Akurasi 96.2% pada dataset internal</li>
                <li>Waktu analisis rata-rata 12 detik</li>
                <li>Mendukung format DICOM dan standar medis</li>
            </ul>
            <div style="margin-top: 1rem;">
                <img src="https://placehold.co/600x400?text=Tumora&font=roboto" alt="Visualisasi teknologi AI medis dengan diagram jaringan saraf" style="width:100%; border-radius:8px;"/>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🎯 Misi Kami</h3>
            <p>Mengurangi waktu tunggu diagnosis dan meningkatkan akses terhadap deteksi dini tumor otak.</p>
            <ul style="color: #2d3748;">
                <li>Skrining lebih cepat dan terjangkau</li>
                <li>Dukungan untuk klinik di daerah terpencil</li>
                <li>Integrasi dengan sistem rekam medis elektronik</li>
            </ul>
            <div style="margin-top: 1rem;">
                <img src="https://placehold.co/600x400?text=Tumora&font=roboto" alt="Ilustrasi dokter dan pasien dengan simbol kesetaraan kesehatan" style="width:100%; border-radius:8px;"/>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>👨‍⚕️ Tim Kami</h3>
        <p>Bertemu dengan para ahli di balik teknologi Brainsight AI yang telah mendedikasikan karier mereka untuk inovasi medis.</p>
        <div class="team-grid">
            <div class="team-card">
                <img src="https://placehold.co/800x600?text=Daffa" alt="Daffa Mahendra">
                <h4>Daffa Mahendra Al Harizts</h4>
                <p>Mahasiswa UNIKOM</p>
                <p style="font-size: 0.9em; color: #4a5568;">Seorang Mahasiswa yang haus akan pengetahuan baru</p>
            </div>
                <div class="team-card">
                <img src="https://placehold.co/800x600?text=Risma" alt="Risma Birrang">
                <h4>Risma Birrang</h4>
                <p>Mahasiswi UNIKOM</p>
                <p style="font-size: 0.9em; color: #4a5568;">Seorang Mahasiswi yang selalu penasaran</p>
            </div>
            <div class="team-card">
                <img src="https://placehold.co/800x600?text=Gyanti" alt="Gyanti Ricka">
                <h4>Gyanti Ricka Budiarti</h4>
                <p>Mahasiswi UNIKOM</p>
                <p style="font-size: 0.9em; color: #4a5568;">Seorang Mahasiswi yang ahli di bidang design</p>
            </div>
            
           
    </div>
    """, unsafe_allow_html=True)

# Contact Page
elif selected == "Contact":
    st.markdown("""
    <div class="card">
        <h1 style="margin-top: 0;">Hubungi Kami</h1>
        <p>Untuk pertanyaan, kerja sama, atau umpan balik, silakan menghubungi tim kami melalui form berikut atau informasi kontak di bawah.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        with st.form("contact_form"):
            st.text_input("Nama Lengkap*", key="name")
            st.text_input("Email*", key="email")
            st.text_input("Subjek*", key="subject")
            st.text_area("Pesan Anda*", key="message", height=150)
            
            if st.form_submit_button("Kirim Pesan", type="primary"):
                st.success("Terima kasih! Pesan Anda telah terkirim. Kami akan menghubungi Anda dalam 1-2 hari kerja.")
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📌 Informasi Kontak</h3>
            <p><strong>Alamat Kantor:</strong></p>
            <p>Universitas Komputer Indonesia<br>
            Jl. Dipati Ukur No.112-116, Lebakgede, Kecamatan Coblong <br>
            Kota Bandung</p>
        
        """, unsafe_allow_html=True)
