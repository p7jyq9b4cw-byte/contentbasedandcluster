import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Mô phỏng thư viện Gensim và các cấu trúc dữ liệu liên quan
# Trong môi trường thực tế, bạn sẽ cần cài đặt Gensim và load các mô hình đã lưu
try:
    from gensim.models.tfidfmodel import TfidfModel
    from gensim.corpora import Dictionary
    from gensim.similarities import SparseMatrixSimilarity
    HAS_GENSIM = True
except ImportError:
    st.warning("Không tìm thấy Gensim. Hệ thống gợi ý sẽ sử dụng Sklearn Cosine Similarity thay thế.")
    HAS_GENSIM = False

# --- Cấu hình & Dữ liệu Mô phỏng (Simulated Data & Models) ---

# Thiết lập caching để các bước nặng (tạo data, huấn luyện) chỉ chạy một lần
@st.cache_data
def load_data_and_train_models():
    # 1. MÔ PHỎNG DỮ LIỆU ĐÃ LÀM SẠCH (6546 samples)
    N_SAMPLES = 6546
    
    # Danh sách các thành phần tiêu đề mẫu
    brands = ["Honda", "Yamaha", "Piaggio"]
    models = ["Air Blade", "Vision", "SH Mode", "Vespa Sprint", "Exciter", "Grande", "Winner X", "Wave RSX"]
    conditions = ["nguyên zin, máy êm", "chính chủ, ít đi", "xe lướt, ODO thấp", "còn bảo hành, bao test hãng", "giá rẻ, xe số đời cũ"]
    years = list(range(2017, 2023))

    # Tạo DataFrame mô phỏng với Tiêu đề DUY NHẤT
    data = {
        'ID': range(1, N_SAMPLES + 1),
        'Thương hiệu': [random.choice(brands) for _ in range(N_SAMPLES)],
        'Giá (tr VNĐ)': np.round(np.random.normal(30, 15, N_SAMPLES), 1),
        'Năm ĐK': np.random.randint(2015, 2023, N_SAMPLES),
        'Km (Km)': np.random.randint(1000, 50000, N_SAMPLES),
    }
    df = pd.DataFrame(data).sort_values(by='ID').reset_index(drop=True)
    
    # Tạo Tiêu đề ngẫu nhiên và DUY NHẤT cho mỗi tin
    def generate_unique_title(row, i):
        brand = row['Thương hiệu']
        model = random.choice([m for m in models if m in ("Air Blade", "Vision", "SH Mode") or brand != "Honda"]) # Giả định mô hình phù hợp
        condition = random.choice(conditions)
        year = random.choice(years)
        return f"{brand} {model} {year} - {condition} (ID {i})"

    # Quan trọng: Gán lại Tiêu đề bằng các string duy nhất
    df['Tiêu đề'] = [generate_unique_title(df.iloc[i], i + 1) for i in range(N_SAMPLES)]
    
    # Lọc giá trị mô phỏng: đảm bảo giá > 10 triệu
    df['Giá (tr VNĐ)'] = df['Giá (tr VNĐ)'].apply(lambda x: max(10.0, x))
    
    # Tạo cột hiển thị cho selectbox
    df['Display'] = df['ID'].astype(str) + ' - ' + df['Tiêu đề'].str[:50] + '...'

    # 2. BÀI TOÁN 2: PHÂN KHÚC THỊ TRƯỜNG (SKLEARN KMEANS)
    
    # *** ĐIỀU CHỈNH K = 5 THEO YÊU CẦU ***
    N_CLUSTERS = 5
    
    # Mô phỏng ma trận đầu vào 127 features (từ Text SVD 100 + Numeric Scaled 10 + Encoded 17)
    X_clustering = np.random.rand(N_SAMPLES, 127)
    
    # Huấn luyện KMeans với K=5
    kmeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
    df['Cụm'] = kmeans_model.fit_predict(X_clustering)
    
    # Xác định hồ sơ cụm (Profiling) dựa trên mô phỏng chi tiết cho K=5
    cluster_profiles = {
        0: {"Tên": "Xe Số Phổ Thông & Đời Cũ", "Mô tả": "Tập trung vào các dòng xe số (Wave, Sirius). Giá thấp nhất, tuổi xe cao (trước 2018). Phục vụ nhu cầu đi lại cơ bản, ngân sách eo hẹp."},
        1: {"Tên": "Xe Tay Ga Phổ Thông (Đa số)", "Mô tả": "Các dòng xe tay ga tầm trung phổ biến (Vision, Air Blade). Giá và tuổi xe trung bình. Là phân khúc lớn nhất, cân bằng giữa giá và tiện ích."},
        2: {"Tên": "Xe Cao Cấp & Xe Lướt", "Mô tả": "Chủ yếu là SH, Vespa đời mới (sau 2021). Giá cao nhất, ODO cực thấp. Khách hàng tìm kiếm xe sang, chất lượng gần như mới."},
        3: {"Tên": "Xe Côn Tay/Thể Thao (Mới)", "Mô tả": "Tập trung vào Exciter, Winner X. Giá trung bình-cao. Khách hàng trẻ tuổi, đam mê tốc độ và phong cách."},
        4: {"Tên": "Xe Tay Ga Cũ & Trung Cấp", "Mô tả": "Các dòng tay ga đời sâu hơn (trước 2019) hoặc xe ít phổ biến hơn. Giá thấp-trung bình. Khách hàng ưu tiên tính năng tay ga với chi phí thấp hơn Cụm 1."},
    }

    # Tinh chỉnh nhãn cụm mô phỏng
    # (Đoạn này mô phỏng việc gán nhãn cụm dựa trên các phân tích trong thực tế)
    df.loc[(df['Giá (tr VNĐ)'] < 20) & (df['Năm ĐK'] < 2018) & (df['Thương hiệu'].isin(["Yamaha", "Honda"])), 'Cụm'] = 0
    df.loc[(df['Giá (tr VNĐ)'] > 50) & (df['Năm ĐK'] > 2021) & (df['Thương hiệu'].isin(["Honda", "Piaggio"])), 'Cụm'] = 2
    df.loc[(df['Thương hiệu'].isin(["Yamaha"]) & (df['Năm ĐK'] > 2019)) | (df['Thương hiệu'] == "Honda"), 'Cụm'] = 1 # Đại diện lớn nhất
    
    # 3. BÀI TOÁN 1: HỆ THỐNG GỢI Ý (GENSIM/SKLEARN)
    
    documents = df['Tiêu đề'].tolist()
    
    if HAS_GENSIM:
        # Sử dụng Gensim (Mô hình 2)
        texts = [doc.split() for doc in documents] 
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        index = SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))
        
        # Lưu các đối tượng cần thiết cho Gensim
        recommendation_engine = {'dictionary': dictionary, 'tfidf': tfidf, 'index': index, 'method': 'Gensim'}
        
    else:
        # Sử dụng Sklearn Cosine Similarity (Mô hình 1 - thay thế)
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        svd = TruncatedSVD(n_components=100)
        svd_matrix = svd.fit_transform(tfidf_matrix)
        cosine_sim_matrix = cosine_similarity(svd_matrix)
        
        # Lưu các đối tượng cần thiết cho Sklearn
        recommendation_engine = {
            'matrix': cosine_sim_matrix, 
            'indices': df.index, 
            'method': 'Sklearn Cosine',
            'tfidf_vectorizer': tfidf_vectorizer, 
            'svd': svd 
        }


    return df, kmeans_model, cluster_profiles, recommendation_engine

# Load data và model (chỉ chạy 1 lần nhờ @st.cache_data)
df, kmeans_model, cluster_profiles, rec_engine = load_data_and_train_models()
N_CLUSTERS = kmeans_model.n_clusters

# --- Định nghĩa các hàm chính ---

def get_recommendations_from_id(car_id, N=10):
    """Lấy N xe tương đồng nhất cho một xe dựa trên ID (sử dụng Gensim/Sklearn)"""
    
    # Lấy index của xe
    idx = df[df['ID'] == car_id].index[0]
    
    if rec_engine['method'] == 'Gensim':
        # Phương pháp Gensim
        dictionary = rec_engine['dictionary']
        tfidf = rec_engine['tfidf']
        index = rec_engine['index']
        
        # Lấy tiêu đề xe cần gợi ý và tiền xử lý
        query = df.loc[idx, 'Tiêu đề']
        query_bow = dictionary.doc2bow(query.split())
        query_tfidf = tfidf[query_bow]
        
        # Tính toán độ tương đồng
        sims = index[query_tfidf]
        
        # Kết quả từ Gensim là list tuples (doc_id, score)
        similarity_scores = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)
        
    else:
        # Phương pháp Sklearn Cosine (Fallback)
        cosine_sim_matrix = rec_engine['matrix']
        similarity_scores = list(enumerate(cosine_sim_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Lấy các xe tương đồng (bỏ qua xe đầu tiên vì là chính nó)
    sim_indices = [i[0] for i in similarity_scores[1:N+1]]
    sim_scores = [i[1] for i in similarity_scores[1:N+1]]
    
    # Lấy mảng giá trị từ DataFrame gốc
    recommended_data = df.loc[sim_indices, ['ID', 'Tiêu đề', 'Thương hiệu', 'Giá (tr VNĐ)', 'Năm ĐK']]
    
    # Tạo DataFrame kết quả mới hoàn toàn từ các cột chính xác
    result_df = pd.DataFrame({
        'ID': recommended_data['ID'].values,
        'Tiêu đề': recommended_data['Tiêu đề'].values, # Lấy tiêu đề chính xác của tin được gợi ý
        'Thương hiệu': recommended_data['Thương hiệu'].values,
        'Giá (tr VNĐ)': recommended_data['Giá (tr VNĐ)'].values,
        'Năm ĐK': recommended_data['Năm ĐK'].values,
        'Similarity Score': sim_scores
    })
    
    return result_df, df.loc[idx, 'Tiêu đề']

def get_recommendations_from_text(free_text, N=10):
    """Lấy N xe tương đồng nhất cho một văn bản tự do (sử dụng Gensim/Sklearn)"""
    
    # Lưu ý: Trong thực tế, cần tiền xử lý (PyVi, stop-words) cho free_text trước khi token/vector hóa
    
    if rec_engine['method'] == 'Gensim':
        # Phương pháp Gensim
        dictionary = rec_engine['dictionary']
        tfidf = rec_engine['tfidf']
        index = rec_engine['index']
        
        query_bow = dictionary.doc2bow(free_text.split())
        query_tfidf = tfidf[query_bow]
        
        sims = index[query_tfidf]
        
        # Kết quả từ Gensim là list tuples (doc_id, score)
        similarity_scores = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)
        
    else:
        # Phương pháp Sklearn Cosine (Fallback)
        tfidf_vectorizer = rec_engine['tfidf_vectorizer']
        svd = rec_engine['svd']
        
        # Vector hóa và giảm chiều Free Text
        query_tfidf = tfidf_vectorizer.transform([free_text])
        query_svd = svd.transform(query_tfidf)
        
        # Tính Cosine Similarity với toàn bộ ma trận dữ liệu đã nén
        cosine_sims = cosine_similarity(query_svd, svd.transform(tfidf_vectorizer.transform(df['Tiêu đề']))).flatten()
        similarity_scores = list(enumerate(cosine_sims))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)


    # Lấy các xe tương đồng (không bỏ qua xe đầu tiên vì không phải là chính nó)
    sim_indices = [i[0] for i in similarity_scores[:N]]
    sim_scores = [i[1] for i in similarity_scores[:N]]
    
    # Lấy mảng giá trị từ DataFrame gốc
    recommended_data = df.loc[sim_indices, ['ID', 'Tiêu đề', 'Thương hiệu', 'Giá (tr VNĐ)', 'Năm ĐK']]
    
    # Tạo DataFrame kết quả mới hoàn toàn từ các cột chính xác
    result_df = pd.DataFrame({
        'ID': recommended_data['ID'].values,
        'Tiêu đề': recommended_data['Tiêu đề'].values, # Lấy tiêu đề chính xác của tin được gợi ý
        'Thương hiệu': recommended_data['Thương hiệu'].values,
        'Giá (tr VNĐ)': recommended_data['Giá (tr VNĐ)'].values,
        'Năm ĐK': recommended_data['Năm ĐK'].values,
        'Similarity Score': sim_scores
    })

    return result_df

def predict_cluster(item_id):
    """Dự đoán cụm cho một xe (sử dụng Sklearn KMeans)"""
    
    # Lấy index của xe
    idx = df[df['ID'] == item_id].index[0]
    
    # Lấy cụm đã được gán nhãn
    cluster_label = df.loc[idx, 'Cụm']
    
    return cluster_label

# --- Streamlit UI ---

st.set_page_config(
    page_title="Đồ án Data Science: Phân tích Xe máy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("MENU CHÍNH")
selection = st.sidebar.radio("Chọn Bài Toán:", ["Hệ thống Gợi ý", "Phân khúc Thị trường"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Engine Gợi ý:** `{rec_engine['method']}`")
st.sidebar.markdown(f"**Engine Phân khúc:** `Sklearn KMeans (K={N_CLUSTERS})`") # Hiển thị K=5
st.sidebar.markdown(f"**Kích thước Dữ liệu:** `{len(df)} tin đăng mô phỏng`")


# --- Trang 1: Hệ thống Gợi ý (Recommendation System) ---
if selection == "Hệ thống Gợi ý":
    st.title("🛵 Hệ thống Gợi ý Tương đồng (Content-Based)")
    st.markdown("Tìm kiếm các mẫu xe tương đồng nhất dựa trên nội dung mô tả.")

    input_mode = st.radio("Chọn chế độ nhập liệu:", ("Chọn ID tin đăng có sẵn", "Nhập mô tả tìm kiếm tự do (Free Text)"))
    
    if input_mode == "Chọn ID tin đăng có sẵn":
        
        # Chọn xe đầu vào (hiển thị ID và tiêu đề)
        selected_display = st.selectbox(
            "Chọn ID tin đăng để tìm xe tương đồng:",
            df['Display'].unique(),
            index=0 # Đặt index cố định = 0 để tránh bị nhảy ID ngẫu nhiên
        )
        
        selected_id = int(selected_display.split(' - ')[0])
        
        st.markdown("---")

        if selected_id:
            st.subheader("1. Tin đăng gốc (Query)")
            query_car = df[df['ID'] == selected_id].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Thương hiệu", query_car['Thương hiệu'])
            with col2:
                st.metric("Giá", f"{query_car['Giá (tr VNĐ)']} triệu VNĐ")
            with col3:
                st.metric("Năm ĐK", query_car['Năm ĐK'])
            
            st.info(f"**Tiêu đề:** {query_car['Tiêu đề']}")
            
            st.subheader("2. Kết quả Gợi ý Tương đồng (Top 10)") # Thay đổi tiêu đề phụ
            
            # CHÚ THÍCH QUAN TRỌNG
            st.caption("Tiêu đề trong bảng kết quả là tiêu đề thực tế của tin đăng được gợi ý, không phải tiêu đề của tin đăng gốc.")
            
            # Thực hiện gợi ý từ ID
            recommended_cars, _ = get_recommendations_from_id(selected_id, N=10)
            
            # Format kết quả
            display_cols = ['ID', 'Tiêu đề', 'Thương hiệu', 'Giá (tr VNĐ)', 'Năm ĐK', 'Similarity Score']
            
            st.dataframe(
                recommended_cars[display_cols].style.format({
                    'Giá (tr VNĐ)': "{:.1f} tr",
                    'Similarity Score': "{:.4f}"
                }),
                use_container_width=True
            )
            
            st.caption(f"Kết quả được tính bằng **{rec_engine['method']}** trên ma trận đặc trưng văn bản đã nén/xử lý.")

    else:
        # Chế độ Free Text
        free_text = st.text_input(
            "Nhập từ khóa hoặc mô tả xe bạn muốn tìm (ví dụ: 'xe lướt, máy êm, còn bảo hành')",
            value="Honda Vision đời 2021, xe chính chủ, ODO thấp"
        )
        
        if st.button("Tìm kiếm Tương đồng") and free_text:
            st.subheader("1. Truy vấn Tự do")
            st.warning(f"Đang tìm kiếm xe tương đồng với: **'{free_text}'**")
            
            st.subheader("2. Kết quả Gợi ý Tương đồng (Top 10)") # Thay đổi tiêu đề phụ
            
            # CHÚ THÍCH QUAN TRỌNG
            st.caption("Tiêu đề trong bảng kết quả là tiêu đề thực tế của tin đăng được gợi ý.")

            # Thực hiện gợi ý từ Free Text
            recommended_cars = get_recommendations_from_text(free_text, N=10)
            
            # Format kết quả
            display_cols = ['ID', 'Tiêu đề', 'Thương hiệu', 'Giá (tr VNĐ)', 'Năm ĐK', 'Similarity Score']
            
            st.dataframe(
                recommended_cars[display_cols].style.format({
                    'Giá (tr VNĐ)': "{:.1f} tr",
                    'Similarity Score': "{:.4f}"
                }),
                use_container_width=True
            )
            
            st.caption(f"Kết quả được tính bằng **{rec_engine['method']}** trên ma trận đặc trưng văn bản. (Lưu ý: Tiền xử lý tiếng Việt cho Free Text cần được tích hợp PyVi trong môi trường thực tế).")


# --- Trang 2: Phân khúc Thị trường (Market Segmentation) ---
elif selection == "Phân khúc Thị trường":
    st.title(f"📈 Phân khúc Thị trường Xe máy (KMeans)") # Hiển thị K=5
    st.markdown("Phân loại tin đăng vào một trong các phân khúc thị trường chính.")
    
    st.subheader(f"1. Tổng quan các Cụm (Clusters)")
    
    # Hiển thị 5 cụm bằng expander
    for i in range(N_CLUSTERS):
        with st.expander(f"Cụm {i}: {cluster_profiles[i]['Tên']}", expanded=(i==0)):
            st.markdown(f"**Mô tả:** {cluster_profiles[i]['Mô tả']}")
        
    st.markdown("---")
    
    st.subheader("2. Kiểm tra Phân khúc cho một Tin đăng")
    
    # Chọn xe đầu vào (hiển thị ID và tiêu đề)
    selected_display_cluster = st.selectbox(
        "Chọn ID tin đăng để kiểm tra phân khúc:",
        df['Display'].unique(),
        index=0 # Đặt index cố định = 0 để tránh bị nhảy ID ngẫu nhiên
    )
    
    selected_id_cluster = int(selected_display_cluster.split(' - ')[0])
    
    if selected_id_cluster:
        car_to_predict = df[df['ID'] == selected_id_cluster].iloc[0]
        
        st.markdown(f"**Tiêu đề:** `{car_to_predict['Tiêu đề']}` | **Giá:** `{car_to_predict['Giá (tr VNĐ)']} triệu VNĐ` | **Năm:** `{car_to_predict['Năm ĐK']}`")
        
        # Dự đoán cụm
        predicted_cluster = predict_cluster(selected_id_cluster)
        
        # Hiển thị kết quả
        st.success(f"Tin đăng này thuộc về **Cụm {predicted_cluster}**: {cluster_profiles[predicted_cluster]['Tên']}")
        st.write(f"**Phân tích cụm:** {cluster_profiles[predicted_cluster]['Mô tả']}")
        
        st.caption("Việc phân cụm được thực hiện trên ma trận 127 features (bao gồm thông tin giá, tuổi xe, và đặc trưng văn bản).")