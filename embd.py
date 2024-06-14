import openai
import faiss
import numpy as np
import pickle
import json

secrets_file_path = './secret.json'

# 비밀 키 파일에서 API 키 읽기
with open(secrets_file_path) as f:
    secrets = json.loads(f.read())

openai.api_key = secrets["openAi-key"]

def generate_image_embedding(image_description):
    response = openai.Embedding.create(
        input=image_description,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


# Your image library
image_library = {
    './images/SoCoool.jpeg': 'SoCoool',
    './images/가지마가지마.png': '가지마가지마',
    './images/개이득.jpeg': '개이득',
    './images/결혼과죽음은끝까지미룰수록좋다.jpeg': '결혼과죽음은끝까지미룰수록좋다',
    './images/괜한말을했네내가.jpeg': '괜한말을했네내가',
    './images/그냥누워있었습니다.jpeg': '그냥누워있었습니다',
    './images/기억이나지않습니다.jpeg': '기억이나지않습니다',
    './images/너무오래기다렸네!.jpeg': '너무오래기다렸네!',
    './images/너죽을래.jpeg': '너죽을래',
    './images/네가웃으면나도좋아.jpeg': '네가웃으면나도좋아',
    './images/돈벌기참힘들다.jpeg': '돈벌기참힘들다',
    './images/뜬금x박수로시작할까요.jpeg': '뜬금x박수로시작할까요',
    './images/말좀빨리해답답해.jpeg': '말좀빨리해답답해',
    './images/먹어라!계산은내가할것이다!.png': '먹어라!계산은내가할것이다!',
    './images/멘탈붕괴.jpeg': '멘탈붕괴',
    './images/무로공격하라.jpeg': '무로공격하라',
    './images/물끄러미.jpeg': '물끄러미',
    './images/뭐,별로없습니다.jpeg': '뭐,별로없습니다',
    './images/배고픕니다.png': '배고픕니다',
    './images/밥은먹고다니니.png': '밥은먹고다니니',
    './images/방끗방끗.jpeg': '방끗방끗',
    './images/법보다주먹이앞서는거보여준다오늘.jpeg': '법보다주먹이앞서는거보여준다오늘',
    './images/분위기개똥같구먼.jpeg': '분위기개똥같구먼',
    './images/사람의생각이란바뀌는거지.jpeg': '사람의생각이란바뀌는거지',
    './images/살기내뿜는늙은살쾡이.jpeg': '살기내뿜는늙은살쾡이',
    './images/슬퍼나이너무많이먹었어.jpeg': '슬퍼나이너무많이먹었어',
    './images/싸우는것봤다고했잖아.jpeg': '싸우는것봤다고했잖아',
    './images/아저씨는누구세요.jpeg': '아저씨는누구세요',
    './images/아하하하하.jpeg': '아하하하하',
    './images/얘기하자면길어요!사연이있으신분이니까!.jpeg': '얘기하자면길어요!사연이있으신분이니까!',
    './images/언짢,분노.jpeg': '언짢,분노',
    './images/얼씨구나.jpeg': '얼씨구나',
    './images/역시배우신분.jpeg': '역시배우신분',
    './images/옥수수털어도돼나요.jpeg': '옥수수털어도돼나요',
    './images/우리가알아야할필요있을까요.jpeg': '우리가알아야할필요있을까요',
    './images/으앙거지됐다거지.jpeg': '으앙거지됐다거지',
    './images/이건또뭐야.jpeg': '이건또뭐야',
    './images/이양반이름을잘몰라,벼멸구인가.jpeg': '이양반이름을잘몰라,벼멸구인가',
    './images/이제소리내기도지쳐버린.jpeg': '이제소리내기도지쳐버린',
    './images/일이많아!.jpeg': '일이많아!',
    './images/입을가만히있으라고입을.jpeg': '입을가만히있으라고입을',
    './images/자이제모든준비는끝났다.png': '자이제모든준비는끝났다',
    './images/잠을자도피로가안풀리냐.jpeg': '잠을자도피로가안풀리냐',
    './images/제,제가그랬나요.jpeg': '제,제가그랬나요',
    './images/죽이겠다.jpeg': '죽이겠다',
    './images/지금꺼지시는것도나쁘지않을것같은데.jpeg': '지금꺼지시는것도나쁘지않을것같은데',
    './images/지켜주지못해미안해.jpeg': '지켜주지못해미안해',
    './images/친구들아고맙다.jpeg': '친구들아고맙다',
    './images/침통.jpeg': '침통',
    './images/캬아아아좋다.jpeg': '캬아아아좋다',
    './images/파이어!!.jpeg': '파이어!!',
    './images/파이팅.jpeg': '파이팅',
    './images/한계임박.jpeg': '한계임박',
    './images/한번또잘해보자.jpeg': '한번또잘해보자',
    './images/형이왜거기서나와.jpeg': '형이왜거기서나와'
}

# Generate embeddings for the images
image_embeddings = [generate_image_embedding(desc) for desc in image_library.values()]
image_filenames = list(image_library.keys())

# Convert embeddings to a numpy array
embeddings_matrix = np.array(image_embeddings).astype('float32')

# Create FAISS index and add embeddings
index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

# Save the index and filenames for later use
with open('faiss_index.pkl', 'wb') as f:
    pickle.dump((embeddings_matrix, index, image_filenames), f)