import streamlit as st
import pandas as pd
import gensim

# Load model
@st.cache_resource
def load_model():
    return gensim.models.Word2Vec.load("word2vec_model.model")

# Fungsi untuk user melakukan input dan filtering
def input_user(df1, model, harga, daerah, text1, text2, text3, fasilitas=None):
    # Memfilter data berdasarkan kriteria dengan partial matching untuk 'full_address'
    if harga < 20000:
        st.write("Harga tidak boleh kurang dari 20.000")
        return pd.DataFrame()  # Mengembalikan tabel kosong jika harga kurang dari 20.000

    data_1 = df1[
        (df1['price_from'] <= harga) &  # Format minimum harga yang akan dibelanjakan
        ((df1['price_till'] >= harga) | (harga > df1['price_till'])) &  # Memastikan harga dalam rentang yang ditentukan
        (df1['full_address'].str.contains(daerah, case=False, na=False))  # Partial match tanpa case-sensitive
    ]

    # Pemetaan fasilitas dari nama yang lebih sederhana ke kolom DataFrame
    fasilitas_mapping = {
        'smoking area': 'smoking_area_available',
        'outdoor seat': 'outdoor_seat_available',
        'full time': 'full_time_available',
        'vip room': 'vip_room_available',
        'reservation': 'reservation_available',
        'parking area': 'parking_area_available'
    }

    # Menambahkan filter untuk fasilitas jika ada
    if fasilitas is not None:
        for fasilitas_item in fasilitas:
            fasilitas_item_lower = fasilitas_item.lower()  # Menjadikan huruf kecil untuk memudahkan pencocokan
            if fasilitas_item_lower in fasilitas_mapping:
                data_1 = data_1[data_1[fasilitas_mapping[fasilitas_item_lower]] == True]

    # Jika tidak ada data yang cocok, tampilkan pesan dan kembalikan DataFrame kosong
    if data_1.empty:
        st.write("Maaf, tidak ada kafe yang cocok dengan kriteria pencarian Anda.")  # Pesan jika tidak ada data
        return pd.DataFrame()  # Mengembalikan tabel kosong jika tidak ada data yang cocok

    # List untuk menyimpan kata mirip
    all_similar_word = []

    # Mendefinisikan topn word
    topn = 5

    # Cek kata pertama dan cari similar words
    if text1 in model.wv:
        res_1 = model.wv.most_similar(text1, topn=topn)
        all_similar_word.extend([word[0] for word in res_1 if word[1] > 0.4])

    # Cek kata kedua
    if text2 in model.wv:
        res_2 = model.wv.most_similar(text2, topn=topn)
        all_similar_word.extend([word[0] for word in res_2 if word[1] > 0.4])

    # Cek kata ketiga
    if text3 in model.wv:
        res_3 = model.wv.most_similar(text3, topn=topn)
        all_similar_word.extend([word[0] for word in res_3 if word[1] > 0.4])

    # Hilangkan kata duplikat
    all_similar_word = list(set(all_similar_word))

    # List untuk menyimpan hasil score similarity
    nilai_similarity = []

    # Menghitung similarity score berdasarkan kata-kata mirip
    for review in data_1['clean stopword']:
        score = 0
        for word in all_similar_word:
            score += review.count(word)  # Menghitung kemunculan kata mirip dalam review
        nilai_similarity.append(score)

    # Menambahkan similarity score ke dalam DataFrame
    data_1 = data_1.copy()  # Buat salinan untuk menghindari SettingWithCopyWarning
    data_1['similarity_score'] = nilai_similarity

    # Memfilter hanya review dengan similarity_score > 0
    data_1 = data_1[data_1['similarity_score'] > 0]

    # Mengembalikan hasil dalam bentuk paragraf
    if data_1.empty:
        st.write("Maaf, tidak ada review yang cocok dengan kriteria pencarian Anda.")  # Pesan jika tidak ada review
        return pd.DataFrame()  # Kembalikan DataFrame kosong jika tidak ada review yang cocok

    # Mengambil review teratas
    top_reviews = data_1.sort_values(by='similarity_score', ascending=False).head(5)

    # Format output sesuai dengan permintaan
    output_list = []
    for _, row in top_reviews.iterrows():
        # Menyiapkan kolom fasilitas yang akan ditampilkan
        facilities = []
        if row['wifi_available']:
            facilities.append("WiFi")
        if row['outdoor_seat_available']:
            facilities.append("Outdoor Seat")
        if row['full_time_available']:
            facilities.append("Full Time")
        if row['vip_room_available']:
            facilities.append("VIP Room")
        if row['reservation_available']:
            facilities.append("Reservation")
        if row['parking_area_available']:
            facilities.append("Parking Area")

        st.markdown(f"<h2> {row['title']}</h2>", unsafe_allow_html=True)
        st.write(f"<h4>Rating: {row['rate']}/5</h4>", unsafe_allow_html=True)
        st.write(f"**Alamat Lengkap:** {row['full_address']}")
        st.write(f"**Jam Buka:** {row['opening_hours']}")
        st.write(f"**Range Harga per orang:** Rp. {row['price_from']} - Rp. {row['price_till']}")
        st.write(f"**Fasilitas:** {' , '.join(facilities)}")
        # Split review dan display secara terpisah
        reviews = [review.strip() for review in row['review'].split(';') if review.strip()]
        for i, review in enumerate(reviews, start=1):
            st.write(f"**Review {i}:** {review}")

        st.write("---")  # Separator antar cafés



    
# Definisikan fungsi model_page
def model_page(df1):
    st.title("HiKopi! Cafe Recommendation App")
    st.markdown(
        """
        <h3 style='text-align: left; font-size: 22px;'>Ayo, Isi Dulu Nih Buat Dapet Rekomendasi Kafe Kece! ☕</h3>
        """,
        unsafe_allow_html=True
    )

    # load model
    model = load_model()

    # User input untuk harga, daerah, fasilitas
    harga = st.number_input("Masukin kisaran harga (Minimal: Rp. 20.000) *", min_value=20000, format="%d")  # Minimum price is 20,000
    daerah = st.text_input("Masukin daerah yang pengen kamu cari: *")
    
    # User input untuk fasilitas (checkbox)
    fasilitas = st.multiselect(
        "Pilih fasilitas (Opsional):",
        options=["Smoking Area", "Outdoor Seat", "Full Time", "VIP Room", "Reservation", "Parking Area"],
        default=None
    )

    text1 = st.text_input("Kata kunci 1 (misal: 'Americano' atau 'nyaman') *")
    text2 = st.text_input("Kata kunci 2 (Opsional):")
    text3 = st.text_input("Kata kunci 3 (Opsional):")

    # Legend untuk input yang di perlukan
    st.markdown("<p style='color:red'>* Kolom ini wajib diisi</p>", unsafe_allow_html=True)

    if st.button("Cari"):
        # Cek apakah kolom yang di perlukan telah di input oleh user
        if not daerah or not text1 or not harga:  # Check apakah 'daerah', 'text1', atau 'harga' kosong
            st.warning("Mohon isi semua kolom yang diperlukan (Harga, Daerah, dan Kata Kunci 1) sebelum mencari.")
        else:
            # Panggil fungsi input_user
            input_user(df1, model, harga, daerah, text1, text2, text3, fasilitas)

# Definisikan halaman intro
def intro_page():
    # Menggunakan HTML dan CSS untuk menyimpan Logo dan Title
    st.markdown(
        """
        <div style='display: flex; align-items: center;'>
            <img src='https://raw.githubusercontent.com/Kuratchikuma/FTDS-035-RMT-group-1/refs/heads/main/hikopi_logo.png' width='150px' style='flex-shrink: 0; margin-right: 20px;'/>
            <h1 style='flex: 1; text-align: left; font-size: 36px;'>Selamat Datang di HiKopi! - Your Coffee Buddy!</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Introduction section 
    st.markdown(
        """
        <h2 style='text-align: left; font-size: 24px'>Selamat Datang di HiKopi! ☕</h2>
        <p style='text-align: left; font-size: 18px;'>
            Hej! Lagi nyari tempat ngopi yang asik dan vibes-nya pas? Nah, kamu di tempat yang tepat! HiKopi bakal jadi sahabat baru kamu 
            buat eksplor kafe-kafe kece yang tersebar di seluruh Jakarta. Mulai dari yang cozy buat nugas, sampai yang estetik buat foto-foto 
            di Instagram, semuanya ada di sini. 
        </p>
        <p style='text-align: left; font-size: 18px;'>
            HiKopi dirancang biar kamu nggak perlu bingung lagi mau ngopi di mana. Tinggal masukin aja preferensi kamu – 
            harga yang sesuai kantong, lokasi yang deket, sama fasilitas yang kamu butuhin, kayak WiFi, area smoking, atau bahkan tempat outdoor buat hangout rame-rame. 
            Dengan satu klik, kita langsung kasih rekomendasi kafe yang mantul banget buat kamu!
        </p>
        <p style='text-align: left; font-size: 18px;'>
            Gimana caranya? Gampang banget! Kamu tinggal masukin beberapa kata kunci favorit kamu. Misalnya, kalo kamu lagi ngidam kopi Americano yang murah 
            tapi rasanya juara, tinggal ketik ‘Americano,’ terus biarin HiKopi cari kafe yang paling oke buat kamu. Atau mungkin kamu nyari tempat buat nongkrong bareng temen 
            dengan suasana outdoor? Cukup pilih fasilitas yang kamu pengen, dan kita akan urutin kafe yang paling cocok!
        </p>
        <p style='text-align: left; font-size: 18px;'>
            Yang bikin HiKopi beda adalah rekomendasi kita berdasarkan ulasan dari pengunjung lain. Jadi, setiap kali kamu klik ‘Cari’, 
            hasil yang muncul udah terfilter berdasarkan review-review jujur dari orang-orang yang udah pernah ke sana. Kamu bisa yakin kalau yang kita rekomendasiin 
            emang worth it buat dicobain. 
        </p>
        <p style='text-align: left; font-size: 18px;'>
            Udah siap buat explore kafe-kafe keren di Jakarta? Yuk, langsung aja mulai cari kafe yang sesuai banget sama gaya ngopi kamu. 
            Dengan HiKopi, ngopi jadi lebih seru dan gampang! 😎☕
        </p>
        """,
        unsafe_allow_html=True
    )

    # Footer
    st.markdown(
        """
        <footer style='text-align: center; margin-top: 50px;'>
            <p>© 2024 HiKopi! | RMT - 035</p>
        </footer>
        """,
        unsafe_allow_html=True
    )

def eda():
    # title
    st.title('Exploratory Data Analysis (EDA) dari Dataset')
    st.write('##### Dibawah ini merupakan Exploratory Data Analysis (EDA) yang telah dilakukan untuk memahami bagaimana isi dataset secara general.')

    st.markdown('----')

    # show dataframe
    st.write('#### Dataset yang Digunakan')
    df = pd.read_csv('Dataset_fix.csv')
    st.dataframe(df)

    st.markdown('----')

    st.write('#### Wordcloud Kolom Cuisine')
    st.image('cuisine_cloud.png', caption='Wordcloud Kolom Cuisine')
    st.write('''
    - Terlihat kafe sudah menjadi hal yang paling sering muncul pada word cloud. Meskipun terlihat pada wordcloud masih muncul kata yang kemungkinan tidak menunjukkan kafe. Misalkan secara harfiah toko kue dan kafe itu berbeda. Namun data yang kita dapat sudah kita filter menjadi kafe pada dataset sumber.
    ''')
    
    st.markdown('----')

    st.write('#### Wordcloud Opening Hours')
    st.image('opening_hours.png', caption='Wordcloud Kolom Opening Hours')
    st.write('''
    - Rata-rata kafe buka di semua hari. Terlihat kata senin sampai minggu itu masuk kategori kata yang berukuran besar.
    ''')

    st.markdown('----')

    st.write('#### Statistik Price From dan Price Till')
    price_stats = df.groupby('kota')[['price_from', 'price_till']].describe().T
    st.dataframe(price_stats)
    st.image('price_from.png', caption='Boxplot Harga Mulai')
    st.image('price_till.png', caption='Boxplot Harga Maksimal')
    st.write('''
    - Kafe-kafe di Jakarta menunjukkan harga rata-rata price_from sebesar Rp43.233 dan price_till sebesar Rp105.466. Median untuk price_from dan price_till masing-masing adalah Rp50.000 dan Rp100.000. Ini berarti sebagian besar kafe memulai harga di sekitar Rp50.000, dengan harga tertinggi yang biasa ditemukan di sekitar Rp100.000. Harga rata-rata menunjukkan bahwa meskipun ada variasi, sebagian besar kafe cenderung menjaga harga dalam rentang yang cukup terjangkau.
    ''')
    
    st.markdown('----')

    st.write('#### Distribusi Rating Tiap Tempat')
    st.image('rating.png', caption='Distribusi Rating Tiap Tempat')
    st.write('''
    - Sebagian besar cafe di Jakarta memiliki rating antara 4.0 hingga 4.5, menunjukkan bahwa cafe umumnya mendapatkan penilaian yang baik.
    ''')

    st.markdown('----')

    st.write('#### Count Tiap Tipe Cuisine yang Ada Didalam Dataset')
    cuisine_distribution = df['cuisine'].value_counts().reset_index()
    cuisine_distribution.columns = ['Cuisine Type', 'Count']
    st.dataframe(cuisine_distribution)
    st.write('''
    - Dominasi Kafe:

    Kafe merupakan jenis tempat makan yang sangat dominan di Jakarta, dengan total 1.329 entri. Jumlah ini sangat jauh di atas jenis cuisine lain. Hal ini bisa mengindikasikan bahwa tren nongkrong atau bersosialisasi di kafe sangat kuat di Jakarta. Kafe di Jakarta bisa saja beragam, dari yang menawarkan kopi hingga makanan ringan, yang menargetkan pasar muda dan profesional.

    - Keanekaragaman Jenis Cuisine:

    Meskipun kafe mendominasi, Jakarta tetap menawarkan keragaman cuisine, dengan jenis seperti Toko Roti (35), China (20), Jepang (15), dan Italia (12). Ini menunjukkan bahwa ada variasi pilihan makanan, meskipun jumlahnya tidak sebanyak kafe. Ini juga bisa menunjukkan bahwa restoran-restoran spesifik seperti ini lebih fokus pada niche pasar tertentu.

    - Restoran Internasional di Jakarta:

    Kehadiran cuisine Jepang, Italia, India, dan China menggambarkan preferensi masyarakat urban Jakarta yang mulai mengadopsi dan menggemari makanan internasional. Restoran jenis ini mungkin berada di area dengan target pelanggan kelas menengah ke atas atau berada di pusat perbelanjaan.
    ''')

    st.markdown('----')

    st.write('#### Distribusi Kategori Harga')
    st.image('price_cat.png', caption='Distribusi Kategori Harga')
    st.write('''
    - Low Category (Rp0 - Rp25.000): Hanya sedikit cafe yang menawarkan harga di bawah Rp25.000. Ini menunjukkan bahwa cafe di Jakarta umumnya menetapkan harga yang cukup tinggi untuk layanan mereka.
    - Medium Category (Rp25.000 - Rp100.000): Mayoritas cafe di Jakarta berada dalam rentang harga ini, yang mencerminkan tren harga yang terjangkau namun tidak terlalu murah.
    - High Category (Rp100.000 - Rp400.000): Ada sejumlah cafe yang menetapkan harga minimum di atas Rp100.000, yang menunjukkan segmen pasar yang lebih premium.
    ''')

    st.markdown('----')

# Fungsi main untuk navigasi multi page
def main(df1):
    # halaman awal 
    if 'page' not in st.session_state:
        st.session_state.page = "Landing page"  # Default to the introduction page

    # Sidebar untuk memilih halaman
    page = st.sidebar.selectbox(
        "Pilih Halaman",
        ["Landing page","EDA", "Aplikasi"],
        index=0 if st.session_state.page == "Landing page" else 1
    )

    # Update sesi agar selaras dengan pilihan sidebar
    if page != st.session_state.page:
        st.session_state.page = page

    # Navigasikan ke halaman yang di pilih
    if st.session_state.page == "Landing page":
        intro_page()
    elif st.session_state.page == "Aplikasi":
        model_page(df1)
    elif st.session_state.page =="EDA":
        eda()


if __name__ == "__main__":
    # load dataframe
    df1 = pd.read_csv('data_processed.csv')

    main(df1)
