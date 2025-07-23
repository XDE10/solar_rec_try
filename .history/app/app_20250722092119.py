import gc
import os
import torch
from pathlib import Path
import streamlit as st
from streamlit_change_language import cst

# 将Streamlit界面汉化
cst.change(language='cn')

# PAge layout
## Page expands to full width
st.set_page_config(
    page_title='太阳能电池板检测',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/solar-energy.svg?width=500",
    layout='wide'
)

# 添加自定义CSS来优化中文显示和尝试覆盖部分英文
st.markdown("""
<style>
    /* 整体字体设置 */
    html, body, [class*="st-"] {
        font-family: "Microsoft YaHei", "Segoe UI", sans-serif !important;
    }
    
    /* 尝试覆盖上传按钮文本 */
    .stButton>button {
        font-family: "Microsoft YaHei", "Segoe UI", sans-serif !important;
    }
    
    /* 上传文件区域样式优化 */
    .upload-text {
        font-size: 0.85rem;
        color: #4a4a4a;
        margin-bottom: 8px;
        padding: 5px;
        border-radius: 4px;
        background-color: #f7f7f7;
        border-left: 3px solid #2ea1d4;
        font-weight: 500;
    }
    
    /* 使侧边栏标题更突出 */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #262730;
        margin-bottom: 10px;
    }
    
    /* 调整主标题样式 */
    h1 {
        color: #1E88E5;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

import segmentation_models_pytorch as smp
from utils import *

# 添加检查模型文件有效性的函数
@st.cache_resource
def is_valid_model_file(model_path):
    try:
        # 尝试加载模型头部信息
        with open(model_path, 'rb') as f:
            header = f.read(8)  # 只读取文件头部几个字节
        return True  # 如果能读取文件，至少文件存在
    except:
        return False

# 创建简单模型作为备选
@st.cache_resource
def create_dummy_model():
    try:
        st.warning("正在创建简单模型作为替代，性能可能不如原始模型")
        # 创建一个简单的UNet模型
        model = smp.UnetPlusPlus(
            encoder_name="resnet18",  # 使用简单的resnet18作为备选
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
        return model
    except Exception as e:
        st.error(f"创建简单模型失败: {e}")
        return None

# 加载模型的安全函数
@st.cache_resource
def load_model_safely(model_path):
    try:
        # 尝试添加模型类到安全列表
        try:
            from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
            from segmentation_models_pytorch.base.model import SegmentationModel
            from segmentation_models_pytorch.decoders.unetplusplus.model import UnetPlusPlus
            from segmentation_models_pytorch.decoders.unet.model import Unet
            import torch.serialization
            # 添加所有可能的模型类到安全列表
            torch.serialization.add_safe_globals([
                DeepLabV3Plus, 
                SegmentationModel,
                UnetPlusPlus,
                Unet
            ])
            # 加载模型 - 使用weights_only=True提高安全性
            return torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception as e:
            # 如果添加安全列表失败，尝试直接加载
            st.warning(f"安全加载模式失败: {str(e)}，尝试使用标准模式加载")
            return torch.load(model_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        st.error(f"模型加载失败: {model_path}")
        st.error(f"错误信息: {str(e)}")
        # 尝试创建备选模型
        return create_dummy_model()
    except FileNotFoundError:
        st.error(f"模型文件不存在: {model_path}")
        return create_dummy_model()
    except Exception as e:
        st.error(f"未知错误: {str(e)}")
        return create_dummy_model()

# ---------------------------------#
# Data preprocessing and Model building

@st.cache_resource
def mask_read_local(gt_mask_dir):
    gt_mask = cv2.imread(gt_mask_dir, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)[1]
    gt_mask = cv2.resize(gt_mask, (256, 256))
    return gt_mask


@st.cache_resource
def mask_read_uploaded(uploaded_mask):
    file_bytes = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
    uploaded_mask = cv2.imdecode(file_bytes, 1)
    uploaded_mask = cv2.resize(uploaded_mask, (256, 256))
    uploaded_mask = uploaded_mask[:, :, 0]
    uploaded_mask = cv2.threshold(uploaded_mask, 0, 255, cv2.THRESH_BINARY)[1]
    return uploaded_mask


@st.cache_resource
def show_detection(image, pred_mask):
    """

    :param image: original image
    :param pred_mask: predicted binary mask
    :return: original image with detected solar panels colored
    """

    pred_mask = cv2.threshold(pred_mask, 0, 255, cv2.THRESH_BINARY)[1]

    pred_mask = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)
    result = cv2.bitwise_and(image.astype('int'), pred_mask.astype('int'))

    return result


@st.cache_resource
def imgread_preprocessing(uploaded_img):  # final preprocessing function in streamlit
    # read data
    # CLASSES = ['solar_panel']
    # class_values=[CLASSES.index(cls.lower()) for cls in classes]

    # image = cv2.imread(uploaded_img)
    image = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(uploaded_mask,0)
    # mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

    # extract certain classes from mask (e.g. cars)
    # masks = [(mask!=v) for v in class_values]
    # mask = np.stack(masks, axis=-1).astype('float')

    # add background if mask is not binary
    # if mask.shape[-1] != 1:
    #    background = 1 - mask.sum(axis=-1, keepdims=True)
    #    mask = np.concatenate((mask, background), axis=-1)

    # apply augmentations
    augmentation = get_test_augmentation()
    sample = augmentation(image=image)
    image = sample['image']

    # apply preprocessing
    preprocessing = get_preprocessing(preprocess_input)
    sample = preprocessing(image=image)
    image = sample['image']
    return image

# ---------------------------------#
# Page layout
## Page expands to full width
# 已在文件顶部设置页面配置

# PAge Intro
st.write("""
# :sunny: 太阳能电池板检测
只需一键点击即可从卫星图像中检测太阳能电池板！

**您可以上传自己的图像！**
-------
""".strip())


# Formatting ---------------------------------#

hide_streamlit_style = """
        <style>
        MainMenu {visibility: hidden;}
        footer {	
            visibility: hidden;
        }
        footer:after {
            content:'Created with Streamlit';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: grey;
            #primary-color: blue;
            padding: 5px;
            top: 2px;
        }
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------#
# Sidebar - Collects user input features into dataframe

with st.sidebar.header('上传图像以检测太阳能电池板'):
    st.sidebar.markdown("""
    <style>
    .upload-text {
        font-size: 0.8rem;
        color: #808495;
        margin-bottom: 5px;
    }
    </style>
    <div class="upload-text">
    将文件拖放至下方框内或点击"Browse files"按钮选择文件<br>
    支持PNG格式，单个文件最大200MB
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("上传PNG格式的图像", type=["png"])

if uploaded_file is not None:
    with st.sidebar.header('您可以上传对应的标注掩码以计算评分指标'):
        st.sidebar.markdown("""
        <style>
        .upload-text {
            font-size: 0.8rem;
            color: #808495;
            margin-bottom: 5px;
        }
        </style>
        <div class="upload-text">
        将掩码文件拖放至下方框内或点击"Browse files"按钮选择文件<br>
        支持PNG格式，单个文件最大200MB
        </div>
        """, unsafe_allow_html=True)
        uploaded_mask = st.sidebar.file_uploader("上传二值掩码图像(PNG格式)", type=["png"])

st.sidebar.markdown("")

img_dir = 'data'
img_files = list(filter(lambda x: 'label' not in x, os.listdir(img_dir)))

model_dir = '../models'
models = {
    ' + '.join(get_model_info(model)[:2]).upper():
        {'ARCH': get_model_info(model)[0],
         'BACKBONE': get_model_info(model)[1],
         'PATH': f'{model_dir}/{model}'}
    for model in filter(lambda x: x.endswith('.pth'), os.listdir(model_dir))
}

# define network parameters
ARCHITECTURE = smp.UnetPlusPlus
BACKBONE = 'se_resnext101_32x4d'
CLASSES = ['solar_panel']
activation = 'sigmoid'
EPOCHS = 25
DEVICE = 'cpu'
n_classes = len(CLASSES)

# 骨干网络名称映射字典，处理简短名称到完整名称的映射
BACKBONE_NAME_MAP = {
    'se_resnext101': 'se_resnext101_32x4d',
}

preprocess_input = smp.encoders.get_preprocessing_fn(BACKBONE)


if uploaded_file is None:
    file_gts = {
        img.replace('.png', ''): 'Zenodo'
        for img in img_files
    }
    with st.sidebar.header('使用我们的测试集图像'):
        pre_trained_img = st.sidebar.selectbox(
            '选择一张图像',
            img_files,
            format_func=lambda x: f'{x} ({(file_gts.get(x.replace(".png", "")))})' if ".png" in x else x,
            index=1,
        )
        if pre_trained_img != "None":
            selected_img_dir = img_dir + '/' + pre_trained_img

else:
    st.sidebar.markdown("请先移除上面的文件才能使用我们的图像。")

model_options = models.keys()
with st.sidebar.subheader('选择用于预测的模型'):
    # 过滤出有效的模型文件
    valid_models = {}
    for name, model_info in models.items():
        if is_valid_model_file(model_info['PATH']):
            valid_models[name] = model_info
    
    if not valid_models:
        st.sidebar.error("没有找到有效的模型文件。请检查模型目录。")
        model_sel = None
        model = None
        model_path = None
    else:
        model_sel = st.sidebar.selectbox(
            '选择模型（架构+骨干网络）',
            valid_models.keys()
        )
        if model_sel:
            model = valid_models[model_sel]
            model_path = model['PATH']
            ARCH = model['ARCH']
            BACKBONE = model['BACKBONE']
            # 如果是简短名称，则映射到完整名称
            if BACKBONE in BACKBONE_NAME_MAP:
                BACKBONE = BACKBONE_NAME_MAP[BACKBONE]
            DEVICE = 'cpu'
            preprocess_input = smp.encoders.get_preprocessing_fn(BACKBONE)

######

st.sidebar.markdown("""
###
### 开发者:

- 浙江大学环资学院

""")

# ---------------------------------#
# Main panel


def deploy1(uploaded_file, uploaded_mask=None):
    # create model
    model = load_model_safely(model_path)
    
    if model is None:
        st.error("无法加载模型，请检查模型文件或尝试选择其他模型")
        return

    # st.write(uploaded_file)
    col1, col2, col3, col4 = st.columns((0.4, 0.4, 0.3, 0.3))

    if uploaded_mask is not None:
        gt_mask = mask_read_uploaded(uploaded_mask)

    with col1:  # visualize
        st.subheader('1.可视化图像')
        with st.spinner(text="加载图像中..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            selected_img = cv2.imdecode(file_bytes, 1)
            image = cv2.resize(selected_img, (256, 256))

        st.subheader('可视化图像')
        detec_option = st.selectbox(label='显示选项',
                                    options=['显示原始图像', '显示检测到的太阳能电池板'])

        if detec_option == '显示原始图像':

            with st.spinner(text="显示图像中..."):
                st.image(
                    image,
                    caption='已选择的图像')

        elif detec_option == '显示检测到的太阳能电池板':
            with st.spinner(text="检测太阳能电池板中..."):
                img_pre = imgread_preprocessing(image)
                image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
                pr_mask = model.predict(image_torch)
                pr_mask = (pr_mask.squeeze().numpy().round())
                boxes = compute_bboxes(image, pr_mask)
                image_bboxes = draw_bbox(image, boxes)
                st.image(image_bboxes, caption='检测到的边界框')

    with col2:  # classify
        st.subheader('模型预测掩码')
        with st.spinner(text="模型运行中..."):
            image = cv2.resize(selected_img, (256, 256))
            img_pre = imgread_preprocessing(image)
            image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(image_torch)
            pr_mask = (pr_mask.squeeze().numpy().round())

            ##################################
            # Choose what to display on mask prediction

            if uploaded_mask is not None:
                options_display_detec = ['显示预测掩码', '显示真实标注掩码', '显示两者']
                mask_option = st.selectbox(label='显示选项',
                                           options=options_display_detec)
            else:
                mask_option = '显示预测掩码'

            options_display_prediction = ['显示二值掩码', '将掩码应用到原始图像']

            display_pred = st.selectbox('掩码选项', options_display_prediction)

            if display_pred == '显示二值掩码':
                if mask_option == '显示预测掩码':
                    st.image(pr_mask, caption='预测掩码')

                elif mask_option == '显示真实标注掩码':
                    st.image(gt_mask, caption='真实标注掩码')

                elif mask_option == '显示两者':
                    st.image(pr_mask, caption='预测掩码')
                    st.image(gt_mask, caption='真实标注掩码')

            elif display_pred == '将掩码应用到原始图像':
                if mask_option == '显示预测掩码':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='检测结果应用到图像上')

                elif mask_option == '显示真实标注掩码':
                    mask = cv2.resize(gt_mask, (256, 256))
                    mask_applied = show_detection(image, mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='真实标注掩码应用到图像上')

                elif mask_option == '显示两者':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='检测结果应用到图像上')
                    mask_applied = show_detection(image, gt_mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='真实标注掩码应用到图像上')

    with col3:
        st.subheader('模型性能')
        if uploaded_mask is not None:
            iou_score = compute_iou(gt_mask, pr_mask)
            st.write(f'**IoU 分数**: {iou_score}')
            pixel_acc = compute_pixel_acc(gt_mask, pr_mask)
            st.write(f'**像素精确度**: {pixel_acc}')
            dice_coef = compute_dice_coeff(gt_mask, pr_mask)
            st.write(f'**Dice系数 (F1分数)**: {dice_coef}')
        else:
            st.write('如果您想查看评分，请上传掩码')

    with col4:
        st.subheader('其他信息')
        spatial_res = st.text_input(label='输入图像的空间分辨率：一个像素代表多长', value=0.1)
        area = np.round(float(spatial_res) ** 2 * pr_mask.sum(), 4)
        perc_area = np.round(100 * pr_mask.sum() / 256 ** 2, 2)
        total_area = np.round(256 ** 2 * float(spatial_res) ** 2, 2)
        st.write(
            f'**预测面积**: {area} 平方米，占图像的 {perc_area}%（总面积 {total_area} 平方米）')
        boxes = compute_bboxes(image, pr_mask)
        coordinates_dict = return_coordinates(boxes)
        coor_values = list(coordinates_dict.values())
        num_bboxes = len(coor_values)
        st.write(f'**坐标**:')
        mkd_pred_table = """
        | 边界框 | 左上角 | 右上角 | 左下角 | 右下角 |
        | --- | --- | --- | --- | --- |
        """ + "\n".join(
            [f"| {i + 1} | {coor_values[i][0]} | {coor_values[i][1]} | {coor_values[i][2]} | {coor_values[i][3]} |" for
             i in range(num_bboxes)])
        st.markdown(mkd_pred_table, unsafe_allow_html=True)

    del model
    gc.collect()


def deploy2(selected_img_dir):
    # Load model
    model = load_model_safely(model_path)
    
    if model is None:
        st.error("无法加载模型，请检查模型文件或尝试选择其他模型")
        return

    col1, col2, col3, col4 = st.columns((0.6, 0.6, 0.6, 0.6))

    gt_mask_dir = selected_img_dir.replace('.png', '_label.png')

    # Check if there exists GT math with "[...]_label.png"
    if os.path.isfile(gt_mask_dir):
        gt_mask = mask_read_local(gt_mask_dir)
    else:
        gt_mask = None

    selected_img = cv2.cvtColor(cv2.imread(selected_img_dir), cv2.COLOR_BGR2RGB)
    image = cv2.resize(selected_img, (256, 256))

    with col1:  # visualize
        st.subheader('可视化图像')
        detec_option = st.selectbox(label='显示选项',
                                    options=['显示原始图像', '显示检测到的太阳能电池板'])

        if detec_option == '显示原始图像':

            with st.spinner(text="加载图像中..."):
                st.image(
                    image,
                    caption='已选择的图像')

        elif detec_option == '显示检测到的太阳能电池板':
            with st.spinner(text="检测太阳能电池板中..."):
                img_pre = imgread_preprocessing(selected_img)
                image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
                pr_mask = model.predict(image_torch)
                pr_mask = (pr_mask.squeeze().numpy().round())
                boxes = compute_bboxes(image, pr_mask)
                image_bboxes = draw_bbox(image, boxes)
                st.image(image_bboxes, caption='检测到的边界框')

    with col2:  # classify
        st.subheader('模型预测掩码')
        with st.spinner(text="模型运行中..."):
            image = cv2.resize(selected_img, (256, 256))
            img_pre = imgread_preprocessing(selected_img)
            image_torch = torch.from_numpy(img_pre).to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(image_torch)
            pr_mask = (pr_mask.squeeze().numpy().round())

            if gt_mask is not None:
                options_display_detec = ['显示预测掩码', '显示真实标注掩码', '显示两者']
                mask_option = st.selectbox(label='显示选项',
                                           options=options_display_detec)
            else:
                mask_option = '显示预测掩码'

            ##################################
            # Choose what to display on mask prediction

            options_display_prediction = ['显示二值掩码', '将掩码应用到原始图像']
            display_pred = st.selectbox('掩码选项', options_display_prediction)

            if display_pred == '显示二值掩码':
                if mask_option == '显示预测掩码':
                    st.image(pr_mask, caption='预测掩码')

                elif mask_option == '显示真实标注掩码':
                    st.image(gt_mask, caption='真实标注掩码')

                elif mask_option == '显示两者':
                    st.image(pr_mask, caption='预测掩码')
                    st.image(gt_mask, caption='真实标注掩码')

            elif display_pred == '将掩码应用到原始图像':
                if mask_option == '显示预测掩码':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='检测结果应用到图像上')

                elif mask_option == '显示真实标注掩码':
                    mask = cv2.resize(gt_mask, (256, 256))
                    mask_applied = show_detection(image, mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='真实标注掩码应用到图像上')

                elif mask_option == '显示两者':
                    colored_image = show_detection(image, pr_mask)
                    colored_image = cv2.resize(colored_image, (256, 256))
                    st.image(colored_image, caption='检测结果应用到图像上')
                    mask_applied = show_detection(image, gt_mask)
                    mask_applied = cv2.resize(mask_applied, (256, 256))
                    st.image(mask_applied, caption='真实标注掩码应用到图像上')

    with col3:
        st.subheader('模型性能')
        if gt_mask is not None:
            iou_score = compute_iou(gt_mask, pr_mask)
            st.write(f'**IoU 分数**: {iou_score}')
            pixel_acc = compute_pixel_acc(gt_mask, pr_mask)
            st.write(f'**像素精确度**: {pixel_acc}')
            dice_coef = compute_dice_coeff(gt_mask, pr_mask)
            st.write(f'**Dice系数 (F1分数)**: {dice_coef}')
        else:
            st.write('该图像没有标注掩码')

    with col4:
        st.subheader('其他信息')
        spatial_res = st.text_input(label='输入图像的空间分辨率：一个像素代表多长', value=0.1)
        area = np.round(float(spatial_res) ** 2 * pr_mask.sum(), 4)
        perc_area = np.round(100 * pr_mask.sum() / 256 ** 2, 2)
        total_area = np.round(256 ** 2 * float(spatial_res) ** 2, 2)
        st.write(
            f'**预测面积**: {area} 平方米，占图像的 {perc_area}%（总面积 {total_area} 平方米）')
        boxes = compute_bboxes(image, pr_mask)
        coordinates_dict = return_coordinates(boxes)
        coor_values = list(coordinates_dict.values())
        num_bboxes = len(coor_values)
        st.write(f'**坐标**:')
        mkd_pred_table = """
        | 边界框 | 左上角 | 右上角 | 左下角 | 右下角 |
        | --- | --- | --- | --- | --- |
        """ + "\n".join(
            [f"| {i + 1} | {coor_values[i][0]} | {coor_values[i][1]} | {coor_values[i][2]} | {coor_values[i][3]} |"
             for i in range(num_bboxes)])
        st.markdown(mkd_pred_table, unsafe_allow_html=True)

    del model
    gc.collect()


if uploaded_file is not None:
    if uploaded_mask is not None:
        deploy1(uploaded_file, uploaded_mask)
    else:
        deploy1(uploaded_file)

elif pre_trained_img != 'None' and model_path is not None:
    deploy2(selected_img_dir)
else:
    st.warning("请上传图像或选择测试图像，并确保有可用的模型。")
