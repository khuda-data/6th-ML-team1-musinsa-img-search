from PIL import Image
import streamlit as st
import base64
import tensorflow as ts
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os 
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd 
from io import BytesIO
import random
# Streamlit Page Config (place this at the top and call only once)
st.set_page_config(page_title="무신사", page_icon=r"png/1.png", layout="wide")

def get_image_as_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
# CSV 파일 경로
file_path = r'E:\musinsa\data_file\merged_musinsa_data.csv'

# CSV 파일 읽기
df = pd.read_csv(file_path,encoding='cp949')

# Base64로 인코딩된 이미지
image_base64 = get_image_as_base64(r"png/1.png")

# CSS로 폰트 변경 및 스타일 적용
st.markdown(
    """ 
    <style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");

    body, h1, h2, h3, h4, h5, h6, .stTextInput, .stButton, .stTextarea, .stSelectbox, .stDateInput, .stNumberInput {
        font-family: 'Pretendard', sans-serif;
    }
    /* Targeting sidebar specifically */
    .stSidebar .css-1d391kg, .stSidebar .stTextInput, .stSidebar .stButton, .stSidebar .stTextarea, .stSidebar .stSelectbox, .stSidebar .stDateInput, .stSidebar .stNumberInput {
        font-family: 'Pretendard', sans-serif;
    }
    /* 헤더의 배경색을 Black으로 변경 */
    header[data-testid="stHeader"] {
        background-color: Black;
        color: white;
        display: flex;
        align-items: center;
        position: relative; /* 가상 요소 위치를 위한 상대 위치 설정 */
    }

    /* 헤더 안에 텍스트 추가 */
    header[data-testid="stHeader"]::before {
        content: 'MUSINSA';
        font-size: 24px;
        color: white;
        position: absolute;
        left: 40px; /* 헤더의 왼쪽에 텍스트 배치 */
    }

    /* "RUNNING..." 텍스트를 변경 */
    [data-testid="stAppRunningIndicator"]::before {
        content: '작동 중...';
        visibility: visible;
    }

    /* "Stop" 버튼 텍스트를 변경 */
    [data-testid="stStopButton"] {
        visibility: hidden;
    }
    [data-testid="stStopButton"]::before {
        content: '정지';
        visibility: visible;
        position: absolute;
    }

    /* "Deploy" 버튼 텍스트를 변경 */
    [data-testid="stHeader"] > div > div:nth-child(3) > div > div > button:nth-child(3) {
        visibility: hidden;
    }
    [data-testid="stHeader"] > div > div:nth-child(3) > div > div > button:nth-child(3)::before {
        content: '배포';
        visibility: visible;
        position: absolute;
    }


    /* 타이틀 스타일 */
    .stApp .css-1lcbmhc {
        font-size: 1.5em; /* 타이틀 텍스트 크기 조정 */
        color: #003366;
        font-weight: bold;
    }

    /* 파일 업로더 스타일 */
    .stFileUploader {
        background-color: Black;
        padding: 10px;
        border-radius: 10px;
    }

    .stButton > button {
        background-color: Black; /* Blue color */
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none; /* Removes default border */
        box-shadow: 4px 4px 8px rgba(0,0,0,0.3); /* Subtle 3D effect */
        transition: background-color 0.3s, box-shadow 0.1s, color 0.3s; /* Smooth transitions */
        
    }
    .stButton > button:hover {
        background-color: #00A3FF; /* Slightly darker blue on hover */
        color: white; /* Changes text color to black when hovered */
        box-shadow: 2px 2px 6px rgba(0,0,0,0.5); /* Deeper shadow on hover for 3D effect */
    }
    .stButton > button:active, .stButton > button:focus {
        background-color: Black; /* Blue color when button is clicked */
        color: white; /* Text color remains white when button is clicked */
    }    
    /* 텍스트 영역 스타일 */
    .stTextArea {
        background-color: Black;
        border-radius: 5px;
    }

    /* 이미지 크기 조절 */
    .title-image {
        width: 60px; /* 원하는 크기로 설정 */
        vertical-align: middle;
    }

    .title-text {
        display: inline;
        font-size: 1.0em; /* 텍스트 크기 조정 */
        color: Black;
        font-weight: bold;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>
    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        height: 100vh;
    }}

    .title-text {{
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .title-text img {{
        width: 200px;
        margin-right: 10px;
    }}

    .title-text h1 {{
        font-size: 2em;
        color: #2269F7;
        margin: 0;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <style>
    .title-text {{
        display: flex;
        align-items: center;
    }}
    .title-text span {{
        margin: 0;
        padding: 0;
    }}
    </style>
    <h1 class="title-text">
        <img src="data:image/png;base64,{image_base64}" class="title-image" style="margin: 0; padding: 0;"/>
        <span style="color:black; font-size:70px;">Musinsa Image Search</span>
    </h1>
    """,
    unsafe_allow_html=True
)


 #Custom styles for centering the button
st.markdown("""
<style>
div.stButton > button:first-child {
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)
    # Custom CSS to style the success message
st.markdown(
    """
    <style>
    .custom-success {
        color: #FFFFFF !important;
        background-color: Black !important;
        padding: 10px;
        border-radius: 5px;
        text-align: center; /* 텍스트를 가로 중앙에 정렬 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# CSS를 사용하여 특정 클래스를 숨김
hide_sidebar_style = """
    <style>
    .st-emotion-cache-1gv3huu.eczjsme18 {  /* 지정된 클래스에 대한 스타일 설정 */
        display: none;
    }
    </style>
    """


# Initialize session state for page management
if 'page' not in st.session_state:
    st.session_state['page'] = 'first'  



def main_page():
    hide_sidebar_style = """
    <style>
    .st-emotion-cache-1gv3huu.eczjsme18 {  /* 지정된 클래스에 대한 스타일 설정 */
        display: none;
    }
    </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)
    # 데이터 경로 및 클래스 이름 설정
    data_dir = r'E:\musinsa\data_file\train_data'
    class_names = sorted(os.listdir(data_dir))

    # 모델 로드
    model = tf.keras.models.load_model(r'E:\musinsa\checkpoints\resnet50_custom_model_V3_checkpoint_epoch_05.h5')
    
    # 이미지 파일만 업로드 가능하도록 설정
    uploaded_file = st.file_uploader("찾고 싶은 의류 사진을 올려주세요 ", type=["jpg", "jpeg", "png", "gif"])

    # 이미지 업로드 처리
    if uploaded_file is not None:
        # 업로드된 파일을 이미지로 열기
        img = Image.open(uploaded_file)
        
        # 이미지를 244x244 크기로 리사이즈
        img = img.resize((224, 244))
        
        # 이미지를 스트림릿에 표시
        st.image(img, use_column_width=False)
        
        # CSS로 이미지를 가운데 정렬
        st.markdown(
            """
            <style>
            .css-1kyxreq.ebxwdo62 {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Button to Start Incomplete Sale Check
        if st.button("Search"):
            # 이미지 전처리
            img = img.resize((224, 224))  # 모델에 맞게 이미지 크기 조정
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # ResNet50 전처리
            
            # 예측
            predictions = model.predict(img_array)
            
            # 상위 20개의 클래스 인덱스와 확률 추출
            top_20_indices = np.argsort(predictions[0])[::-1][:20]
            top_20_probabilities = predictions[0][top_20_indices]
            top_20_class_names = [class_names[i] for i in top_20_indices]

            # 결과 출력
            top_20_images = []
            top_20_urls = []
            top_20_captions = []
            for i in range(20):
                class_id = int(top_20_class_names[i])
                # 해당 클래스와 일치하는 폴더의  png 파일 로드
                folder_path = f"E:/musinsa/data_file/cropped_images_no_detection_copy/{top_20_class_names[i]}"
                image_list = os.listdir(folder_path)
                random_image = random.choice(image_list)  # 파일 리스트에서 무작위로 하나 선택
                image_path = os.path.join(folder_path, random_image)
                img = Image.open(image_path).resize((244, 244))
                top_20_images.append(img)

                # 해당 클래스에 대한 URL 가져오기
                url = df.loc[df['Class'] == class_id, '상품URL'].iloc[0]
                top_20_captions.append(f"Class {class_id} ")
                top_20_urls.append(url)
    
            # 이미지들 배치 (5x4 행렬로)
            all_images = top_20_images  
            all_urls = top_20_urls 
            all_captions = top_20_captions 
            # 5x4 행렬로 출력 (행당 4개의 이미지씩 출력)
            for i in range(0, len(all_images), 4):
                cols = st.columns(4)  # 4개의 열을 생성
                for j, col in enumerate(cols):
                    if i + j < len(all_images):
                        # 각 이미지를 Base64로 인코딩하여 HTML에 포함
                        img = all_images[i + j]
                        caption = all_captions[i + j]  # 캡션 가져오기
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()

                        # URL 설정
                        url = all_urls[i + j]  # 각 이미지에 대한 URL 가져오기
                        
                        # 이미지와 하이퍼링크를 HTML로 생성
                        html = f"""
                        <a href="{url}" target="_blank">
                            <img src="data:image/png;base64,{img_str}" width="244"/>
                        </a>
                        """
                        col.markdown(html, unsafe_allow_html=True)

# def second_page(): 

if st.session_state['page'] == 'first':
    main_page()
    
# elif st.session_state['page'] == 'main':
#     second_page()
