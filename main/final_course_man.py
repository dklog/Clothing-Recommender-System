from flask import Flask, request
from flask_cors import CORS, cross_origin
import base64
import json

from glob import glob
from keras.models import load_model
from matplotlib import pyplot as plt
#%matplotlib inline
from PIL import Image
from random import uniform, random, choice, sample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance

import cv2
import mahotas as mh
import numpy as np
import pandas as pd




app = Flask('CVaaS')
CORS(app)
def cv_engine(img, operation):
    if operation == 'to_grayscale':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None
def read_image(image_data):
    image_data = base64.decodebytes(image_data)
    with open('temp_image.jpg', 'wb') as f:
        f.write(image_data)
        f.close()
    img = cv2.imread('temp_image.jpg')
    return img
def encode_image(img):
    ret, data = cv2.imencode('.jpg', img)
    return base64.b64encode(data)

@app.route('/process_man_image', methods=['POST'])
def process_man_image():
    if not request.json or 'msg' not in request.json:
        return 'Server Error!', 500
    header_len = len('data:image/jpeg;base64,')
    image_data = request.json['image_data'][header_len:].encode()
    operation = request.json['operation']
    img = read_image(image_data)

    # img_out = cv_engine(img, operation)



    # 1단계 - 카테고리 맞추기
    model = load_model('./recommend/man/vgg_man.hdf5')  # 남자옷 학습시킨 모델이 있는 경로

    categories = ["man_bottoms_jeans", "man_bottoms_pants", "man_top_hoodie",
                  "man_top_hoodie_print", "man_top_knit", "man_top_knit_print",
                  "man_top_shirt", "man_top_shirt_print", "man_top_sweatshirt",
                  "man_top_sweatshirt_print", "man_top_tshirt_long", "man_top_tshirt_long_print",
                  "man_top_tshirt_short", "man_top_tshirt_short_print"]

    # 사용자가 입력한 이미지파일(여기서 한 번!)
    test = 'temp_image.jpg'

    img = Image.open(test)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float") / 256
    X = X.reshape(-1, 150, 150, 3)

    pred = model.predict(X)
    result = [np.argmax(value) for value in pred]
    print('New data category : ', categories[result[0]])

    # 2단계 - 우리가 가지고 있는 이미지 소스 중에서 유사한 옷 가져오기
    man_images = []
    # 남자 옷 이미지 소스 경로 (!! categorical_name이 파일명인 이미지파일)
    man_images = glob('./recommend/man/image_source/categorical_name/*.png')

    image_names = []
    for img in man_images:  # 경로 기준에 맞게 잘라주기
        image_names.append(img.split("name\\")[1])

    # 사용자가 입력한 이미지파일(여기서 또 한 번!)
    image = test
    man_images.append(image)

    # 유사분류기에 쓸 남자옷 배열 파일 불러오기
    features = np.load("./recommend/man/man_features.npy")
    labels = np.load("./recommend/man/man_labels.npy")

    # 유사분류기에 쓸 남자옷 이름매칭 엑셀파일 불러오기
    df_nm_match = pd.read_csv("./recommend/man/man_nm_match.csv")

    image = cv2.imread(image)
    image = image.astype(np.uint8)
    insert_data_feature = mh.features.haralick(image).ravel()
    # 1단계에서 나온 결과를 여기로 호출
    insert_data_label = categories[result[0]]

    np_insert_data_feature = np.array(insert_data_feature)
    np_insert_data_label = np.array(insert_data_label)

    new_features = np.vstack((features, np_insert_data_feature))
    new_labels = np.append([labels], [np_insert_data_label])

    clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])

    sc = StandardScaler()
    features_02 = sc.fit_transform(new_features)

    dists = distance.squareform(distance.pdist(features_02))


    similar_images = []

    def selectImage(n, m, dists, images):
        image_position = dists[n].argsort()[m]
        image = cv2.imread(man_images[image_position])
        similar_images.append(man_images[image_position])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_position(n, m, dists, images):
        image_position = dists[n].argsort()[m]
        return image_position

    def plotImage_input(n):

        plt.imshow(selectImage(n, 0, dists, man_images))
        plt.title("input")
        plt.xticks([])
        plt.yticks([])

    def plotImage_similar1(n):

        plt.imshow(selectImage(n, 1, dists, man_images))
        merge_nm_2 = df_nm_match.iloc[get_position(n, 1, dists, man_images)][0]
        non_merge_nm_2 = df_nm_match.iloc[get_position(n, 1, dists, man_images)][1]

        plt.title(merge_nm_2)
        plt.xticks([])
        plt.yticks([])

    def plotImage_similar2(n):

        plt.imshow(selectImage(n, 2, dists, man_images))
        merge_nm_3 = df_nm_match.iloc[get_position(n, 2, dists, man_images)][0]
        non_merge_nm_3 = df_nm_match.iloc[get_position(n, 2, dists, man_images)][1]

        plt.title(merge_nm_3)
        plt.xticks([])
        plt.yticks([])

    def plotImage_similar3(n):

        plt.imshow(selectImage(n, 3, dists, man_images))
        merge_nm_4 = df_nm_match.iloc[get_position(n, 3, dists, man_images)][0]
        non_merge_nm_4 = df_nm_match.iloc[get_position(n, 3, dists, man_images)][1]

        plt.title(merge_nm_4)
        plt.xticks([])
        plt.yticks([])

    similar_1_name_cat = df_nm_match.iloc[get_position(-1, 1, dists, man_images)][0]
    similar_1_name_img = df_nm_match.iloc[get_position(-1, 1, dists, man_images)][1]
    similar_2_name_cat = df_nm_match.iloc[get_position(-1, 2, dists, man_images)][0]
    similar_2_name_img = df_nm_match.iloc[get_position(-1, 2, dists, man_images)][1]
    similar_3_name_cat = df_nm_match.iloc[get_position(-1, 3, dists, man_images)][0]
    similar_3_name_img = df_nm_match.iloc[get_position(-1, 3, dists, man_images)][1]

    # 입력한 옷 보기
    plotImage_input(-1)

    # 유사한 옷 1 보기
    plotImage_similar1(-1)

    # 유사한 옷 2 보기
    plotImage_similar2(-1)

    # 유사한 옷 3 보기
    plotImage_similar3(-1)


    # 3단계 - 옷 추천하기

    # 남자 상의 전부의 특성파일 불러오기
    df_top = pd.read_csv('./recommend/man/man_top_onehot.csv')

    df_top = df_top.set_index('index')

    # 남자 하의 전부의 특성파일 불러오기
    df_bottom = pd.read_csv('./recommend/man/man_bottom_onehot.csv')

    df_bottom = df_bottom.set_index('index')

    # 남자 하의 전부의 특성파일 불러오기
    df_bottom = pd.read_csv('./recommend/man/man_bottom_onehot.csv')

    df_bottom = df_bottom.set_index('index')

    # 남자옷 전부의 속성 매트릭스파일 (어울림 정도 점수) 불러오기
    attr_mat = pd.read_excel("./recommend/man/man_attr_mat.xlsx", sheet_name='cat_mat', index_col=0)

    df_top_dic = {}
    df_bottom_dic = {}


    recommend_images = []

    def retrieve_top_N(item, df, top_N=3):
        if item[0] == 1:
            temp_df = df[item[1:]]
        else:
            df['bottom'] = df.index
            temp_df = df[df['bottom'].isin(item[1:])]
            del temp_df['bottom']
            temp_df = temp_df.transpose()

        temp_df['total'] = temp_df.sum(axis=1)

        return temp_df.nlargest(top_N, 'total')

    def top_filenames(dic_to_match, list_to_match):
        matching_files = []
        for filename, value in dic_to_match.items():
            if set(value) == set(list_to_match):
                matching_files.append(filename)
        return matching_files

    def shuffle_generator(rec):
        return (rec[idx] for idx in np.random.permutation(len(rec)))

    def yielding(ls):
        for i in ls:
            yield i

    def recommendation(file_name, top_bottom, attr_mat, show=True, top_N_rec=3):
        if top_bottom == 1:
            clothes_input = 'top'
            clothes_output = 'bottom'
            attr_df = df_top
            attr_dic = df_bottom_dic
        else:
            clothes_input = 'bottom'
            clothes_output = 'top'
            attr_df = df_bottom
            attr_dic = df_top_dic

        try:
            col = attr_df.loc[file_name]
            fn = file_name

        except:
            num = randint(0, len(attr_df))
            col = attr_df[num:num + 1]
            fn = col.index[0]

        attribute_list = [top_bottom]
        df_col = pd.DataFrame(col[1:])

        for x in df_col.index:
            if col[x].sum() == 1 and x != 'category':
                attribute_list.append(x)

        print ("Filename:", fn)

        # 이미지 소스파일이 있는 경로 (!! original_name이 파일명인 이미지파일)
        path = "./recommend/man/image_source/original_name/" + fn + '.png'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

        print ("")
        print ("입력되는 옷 : ", clothes_input)
        print ("입력되는 옷 속성 리스트: ", attribute_list)
        print ("")

        matched_df = retrieve_top_N(attribute_list, attr_mat, top_N=3)
        matched_list = list(matched_df.index)

        print ("추천해야 할 옷 종류 : ", clothes_output)
        print ("입력한 옷에 어울리는 추천 속성 리스트: ", matched_list)
        print ("")

        matching_dic = {}
        for key, value in attr_dic.items():
            intersect_val = set(matched_list).intersection(value)
            if len(intersect_val) >= 1:
                matching_dic[key] = intersect_val

        ls_to_match = sorted(matching_dic.values(), key=len)[-1]

        print ("가장 어울리는 속성 : ", ls_to_match)
        print ("")

        re_rec = []

        rec = top_filenames(matching_dic, ls_to_match)
        for i in yielding(shuffle_generator(rec)):
            re_rec.append(i)
            rec_3 = re_rec[:3]
        print ("추천 결과 : ", rec_3)

        if show == True:
            for item in rec_3:
                # 이미지 소스파일이 있는 경로 (!! original_name이 파일명인 이미지파일)
                path = "./recommend/man/image_source/original_name/" + item + '.png'
                recommend_images.append(path)
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)

        return rec

    for index, row in df_top.iterrows():
        temp_list = []
        for col, value in row.items():
            if value == 1 and col != 'category':
                temp_list.append(col)
        df_top_dic[index] = temp_list

    for index, row in df_bottom.iterrows():
        temp_list = []
        for col, value in row.items():
            if value == 1 and col != 'category':
                temp_list.append(col)
        df_bottom_dic[index] = temp_list

    df_top_nmcat = df_top[["file_name", "category"]]
    df_bottom_nmcat = df_bottom[["file_name", "category"]]
    df_total_nmcat = pd.concat([df_top_nmcat, df_bottom_nmcat], axis=0)

    # 2단계에서 나온 유사한 옷 파일 이름 (1)
    file_name_raw_1 = similar_1_name_img

    file_name_1 = file_name_raw_1.split(".")[0]

    up_low_1 = df_total_nmcat.loc[file_name_1][1]

    x = recommendation(file_name_1, up_low_1, attr_mat, show=True)

    # 2단계에서 나온 유사한 옷 파일 이름 (2)
    file_name_raw_2 = similar_2_name_img

    file_name_2 = file_name_raw_2.split(".")[0]

    up_low_2 = df_total_nmcat.loc[file_name_2][1]

    x = recommendation(file_name_2, up_low_2, attr_mat, show=True)

    # 2단계에서 나온 유사한 옷 파일 이름 (3)
    file_name_raw_3 = similar_3_name_img

    file_name_3 = file_name_raw_3.split(".")[0]

    up_low_3 = df_total_nmcat.loc[file_name_3][1]

    x = recommendation(file_name_3, up_low_3, attr_mat, show=True)



    # file_name_raw = "img_1001036230_28.png"
    # df_top = pd.read_csv('./recommender_mat/man_top_onehot.csv')
    # df_top = df_top.set_index('index')
    # df_bottom = pd.read_csv('./recommender_mat/man_bottom_onehot.csv')
    # df_bottom = df_bottom.set_index('index')
    # attr_mat = pd.read_excel("./recommender_mat/man_attr_mat.xlsx", sheet_name='cat_mat', index_col=0)

    image_data = {
        "images": [{"title": "유사 이미지1", "url": similar_images[1]}, {"title": "유사 이미지2", "url": similar_images[2]},
                   {"title": "유사 이미지3", "url": similar_images[3]}],
        "images1": [{"title": "추천 이미지1", "url": recommend_images[0]},{"title": "추천 이미지2", "url": recommend_images[1]},
                    {"title": "추천 이미지3", "url": recommend_images[2]}],
        "images2": [{"title": "추천 이미지1", "url": recommend_images[3]},{"title": "추천 이미지2", "url": recommend_images[4]},
                    {"title": "추천 이미지3", "url": recommend_images[5]}],
        "images3": [{"title": "추천 이미지1", "url": recommend_images[6]},{"title": "추천 이미지2", "url": recommend_images[7]},
                    {"title": "추천 이미지3", "url": recommend_images[8]}]
    }

    image_result = json.dumps(image_data)



    return image_result
    #return image_data, image_data2, image_data3, 200


@app.route('/process_woman_image', methods=['POST'])
def process_woman_image():
    if not request.json or 'msg' not in request.json:
        return 'Server Error!', 500
    header_len = len('data:image/jpeg;base64,')
    image_data = request.json['image_data'][header_len:].encode()
    operation = request.json['operation']
    img = read_image(image_data)

    # 1단계 - 카테고리 맞추기
    model = load_model('./recommend/woman/woman_basic_model.hdf5')  # 여자옷 학습시킨 모델이 있는 경로

    categories = ["woman_bottoms_jeans", "woman_bottoms_pants", "woman_bottoms_skirt_long", "woman_bottoms_skirt_short",
                  "woman_top_hoodie", "woman_top_hoodie_print", "woman_top_knit", "woman_top_knit_print",
                  "woman_top_shirt_long", "woman_top_shirt_long_print", "woman_top_sweatshirt",
                  "woman_top_sweatshirt_print",
                  "woman_top_tshirt_long", "woman_top_tshirt_long_print", "woman_top_tshirt_short",
                  "woman_top_tshirt_short_print",
                  "woman_top_blouse_long", "woman_top_blouse_long_print"]

    # 사용자가 입력한 이미지파일(여기서 한 번!)
    test = 'temp_image.jpg'

    img = Image.open(test)
    img = img.convert("RGB")
    img = img.resize((100, 100))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float") / 256
    X = X.reshape(-1, 100, 100, 3)

    pred = model.predict(X)
    result = [np.argmax(value) for value in pred]
    print('New data category : ', categories[result[0]])

    # 2단계 - 우리가 가지고 있는 이미지 소스 중에서 유사한 옷 가져오기
    woman_images = []
    # 여자 옷 이미지 소스 경로 (!! categorical_name이 파일명인 이미지파일)
    woman_images = glob('./recommend/woman/image_source/categorical_name/*.png')

    image_names = []
    for img in woman_images:
        image_names.append(img.split("name\\")[1])

    # 사용자가 입력한 이미지파일(여기서 또 한 번!)
    image = test
    woman_images.append(image)

    # 유사분류기에 쓸 여자옷 배열 파일 불러오기
    features = np.load("./recommend/woman/woman_features.npy")
    labels = np.load("./recommend/woman/woman_labels.npy")

    # 유사분류기에 쓸 여자옷 이름매칭 엑셀파일 불러오기
    df_nm_match = pd.read_csv("./recommend/woman/woman_nm_match.csv")

    image = cv2.imread(image)
    image = image.astype(np.uint8)
    insert_data_feature = mh.features.haralick(image).ravel()
    # 1단계에서 나온 결과를 여기로 호출
    insert_data_label = categories[result[0]]

    np_insert_data_feature = np.array(insert_data_feature)
    np_insert_data_label = np.array(insert_data_label)

    new_features = np.vstack((features, np_insert_data_feature))
    new_labels = np.append([labels], [np_insert_data_label])

    clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])

    sc = StandardScaler()
    features_02 = sc.fit_transform(new_features)

    dists = distance.squareform(distance.pdist(features_02))


    similar_images = []

    def selectImage(n, m, dists, images):
        image_position = dists[n].argsort()[m]
        image = cv2.imread(woman_images[image_position])
        similar_images.append(woman_images[image_position])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_position(n, m, dists, images):
        image_position = dists[n].argsort()[m]
        return image_position

    def plotImage_input(n):

        plt.imshow(selectImage(n, 0, dists, woman_images))
        plt.title("input")
        plt.xticks([])
        plt.yticks([])

    def plotImage_similar1(n):

        plt.imshow(selectImage(n, 1, dists, woman_images))
        merge_nm_2 = df_nm_match.iloc[get_position(n, 1, dists, woman_images)][0]
        non_merge_nm_2 = df_nm_match.iloc[get_position(n, 1, dists, woman_images)][1]

        plt.title(merge_nm_2)
        plt.xticks([])
        plt.yticks([])

    def plotImage_similar2(n):

        plt.imshow(selectImage(n, 2, dists, woman_images))
        merge_nm_3 = df_nm_match.iloc[get_position(n, 2, dists, woman_images)][0]
        non_merge_nm_3 = df_nm_match.iloc[get_position(n, 2, dists, woman_images)][1]

        plt.title(merge_nm_3)
        plt.xticks([])
        plt.yticks([])

    def plotImage_similar3(n):

        plt.imshow(selectImage(n, 3, dists, woman_images))
        merge_nm_4 = df_nm_match.iloc[get_position(n, 3, dists, woman_images)][0]
        non_merge_nm_4 = df_nm_match.iloc[get_position(n, 3, dists, woman_images)][1]

        plt.title(merge_nm_4)
        plt.xticks([])
        plt.yticks([])

    similar_1_name_cat = df_nm_match.iloc[get_position(-1, 1, dists, woman_images)][0]
    similar_1_name_img = df_nm_match.iloc[get_position(-1, 1, dists, woman_images)][1]
    similar_2_name_cat = df_nm_match.iloc[get_position(-1, 2, dists, woman_images)][0]
    similar_2_name_img = df_nm_match.iloc[get_position(-1, 2, dists, woman_images)][1]
    similar_3_name_cat = df_nm_match.iloc[get_position(-1, 3, dists, woman_images)][0]
    similar_3_name_img = df_nm_match.iloc[get_position(-1, 3, dists, woman_images)][1]

    # 입력한 옷
    plotImage_input(-1)

    # 유사한 옷 1
    plotImage_similar1(-1)

    # 유사한 옷 2
    plotImage_similar2(-1)

    # 유사한 옷 3
    plotImage_similar3(-1)


    # 3단계 - 옷 추천하기

    # 여자 상의 전부의 특성파일 불러오기
    df_top = pd.read_csv('./recommend/woman/woman_top_onehot.csv')

    df_top = df_top.set_index('index')

    # 여자 하의 전부의 특성파일 불러오기
    df_bottom = pd.read_csv('./recommend/woman/woman_bottom_onehot.csv')

    df_bottom = df_bottom.set_index('index')

    # 여자옷 전부의 속성 매트릭스파일 (어울림 정도 점수) 불러오기
    attr_mat = pd.read_excel("./recommend/woman/woman_attr_mat.xlsx", sheet_name='cat_mat', index_col=0)

    df_top_dic = {}
    df_bottom_dic = {}

    recommend_images = []

    def retrieve_top_N(item, df, top_N=3):
        if item[0] == 1:
            temp_df = df[item[1:]]
        else:
            df['bottom'] = df.index
            temp_df = df[df['bottom'].isin(item[1:])]
            del temp_df['bottom']
            temp_df = temp_df.transpose()

        temp_df['total'] = temp_df.sum(axis=1)

        return temp_df.nlargest(top_N, 'total')

    def top_filenames(dic_to_match, list_to_match):
        matching_files = []
        for filename, value in dic_to_match.items():
            if set(value) == set(list_to_match):
                matching_files.append(filename)
        return matching_files

    def shuffle_generator(rec):
        return (rec[idx] for idx in np.random.permutation(len(rec)))

    def yielding(ls):
        for i in ls:
            yield i

    def recommendation(file_name, top_bottom, attr_mat, show=True, top_N_rec=3):
        if top_bottom == 1:
            clothes_input = 'top'
            clothes_output = 'bottom'
            attr_df = df_top
            attr_dic = df_bottom_dic
        else:
            clothes_input = 'bottom'
            clothes_output = 'top'
            attr_df = df_bottom
            attr_dic = df_top_dic

        try:
            col = attr_df.loc[file_name]
            fn = file_name

        except:
            num = randint(0, len(attr_df))
            col = attr_df[num:num + 1]
            fn = col.index[0]

        attribute_list = [top_bottom]
        df_col = pd.DataFrame(col[1:])

        for x in df_col.index:
            if col[x].sum() == 1 and x != 'category':
                attribute_list.append(x)

        print ("Filename:", fn)

        # 이미지 소스파일이 있는 경로 (!! original_name이 파일명인 이미지파일)
        path = "./recommend/woman/image_source/original_name/" + fn + '.png'
        recommend_images.append(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)

        print ("")
        print ("입력되는 옷 : ", clothes_input)
        print ("입력되는 옷 속성 리스트: ", attribute_list)
        print ("")

        matched_df = retrieve_top_N(attribute_list, attr_mat, top_N=3)
        matched_list = list(matched_df.index)

        print ("추천해야 할 옷 종류 : ", clothes_output)
        print ("입력한 옷에 어울리는 추천 속성 리스트: ", matched_list)
        print ("")

        matching_dic = {}
        for key, value in attr_dic.items():
            intersect_val = set(matched_list).intersection(value)
            if len(intersect_val) >= 1:
                matching_dic[key] = intersect_val

        ls_to_match = sorted(matching_dic.values(), key=len)[-1]

        print ("가장 어울리는 속성 : ", ls_to_match)
        print ("")

        re_rec = []

        rec = top_filenames(matching_dic, ls_to_match)
        for i in yielding(shuffle_generator(rec)):
            re_rec.append(i)
            rec_3 = re_rec[:3]
        print ("추천 결과 : ", rec_3)

        if show == True:
            for item in rec_3:
                # 이미지 소스파일이 있는 경로 (!! original_name이 파일명인 이미지파일)
                path = "./recommend/woman/image_source/original_name/" + item + '.png'
                recommend_images.append(path)
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)

        return rec

    for index, row in df_top.iterrows():
        temp_list = []
        for col, value in row.items():
            if value == 1 and col != 'category':
                temp_list.append(col)
        df_top_dic[index] = temp_list

    for index, row in df_bottom.iterrows():
        temp_list = []
        for col, value in row.items():
            if value == 1 and col != 'category':
                temp_list.append(col)
        df_bottom_dic[index] = temp_list

    df_top_nmcat = df_top[["file_name", "category"]]
    df_bottom_nmcat = df_bottom[["file_name", "category"]]
    df_total_nmcat = pd.concat([df_top_nmcat, df_bottom_nmcat], axis=0)

    # 2단계에서 나온 유사한 옷 파일 이름 (1)
    file_name_raw_1 = similar_1_name_img

    file_name_1 = file_name_raw_1.split(".")[0]

    up_low_1 = df_total_nmcat.loc[file_name_1][1]

    x = recommendation(file_name_1, up_low_1, attr_mat, show=True)

    # 2단계에서 나온 유사한 옷 파일 이름 (2)
    file_name_raw_2 = similar_2_name_img

    file_name_2 = file_name_raw_2.split(".")[0]

    up_low_2 = df_total_nmcat.loc[file_name_2][1]

    x = recommendation(file_name_2, up_low_2, attr_mat, show=True)

    # 2단계에서 나온 유사한 옷 파일 이름 (3)
    file_name_raw_3 = similar_3_name_img

    file_name_3 = file_name_raw_3.split(".")[0]

    up_low_3 = df_total_nmcat.loc[file_name_3][1]

    x = recommendation(file_name_3, up_low_3, attr_mat, show=True)



    image_data = {
        "images": [{"title": "유사 이미지1", "url": similar_images[1]}, {"title": "유사 이미지2", "url": similar_images[2]},
                   {"title": "유사 이미지3", "url": similar_images[3]}],
        "images1": [{"title": "추천 이미지1", "url": recommend_images[0]},{"title": "추천 이미지2", "url": recommend_images[1]},
                    {"title": "추천 이미지3", "url": recommend_images[2]}],
        "images2": [{"title": "추천 이미지1", "url": recommend_images[3]},{"title": "추천 이미지2", "url": recommend_images[4]},
                    {"title": "추천 이미지3", "url": recommend_images[5]}],
        "images3": [{"title": "추천 이미지1", "url": recommend_images[6]},{"title": "추천 이미지2", "url": recommend_images[7]},
                    {"title": "추천 이미지3", "url": recommend_images[8]}]
    }


    image_result = json.dumps(image_data)

    return image_result

@app.route('/')
def index():
    return 'Hello World'
if __name__ == '__main__':
    app.run(debug=True)













