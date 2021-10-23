import io
import json
import streamlit as st
import torch
from torchvision import transforms
import torchvision
from PIL import Image
import pandas as pd


@st.cache
def load_model():
    # get pretrained model
    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # read imagenet class list
    index_to_label_map = json.load(open("src/imagenet_class_list.json", "r"))

    return model, transform, index_to_label_map


def main():
    # sidebar
    st.sidebar.title("Streamlit on Heroku")
    st.sidebar.write("Demo App for running streamlit on Heroku.")
    st.sidebar.write("https://github.com/tan-z-tan/streamlit-on-heroku")

    sample_img = Image.open("images/inu.jpg")
    # image in main area
    image_area = st.empty()
    image_area.image(sample_img)

    # for the default photo
    st.title("Inference Result")
    st.info("Here you can see the results of ImageNet's image classification")
    inference_df = inference(sample_img)
    inference_best_guess = st.empty()
    inference_best_guess.header(inference_df.label[0])
    inference_topk = st.empty()
    inference_topk.table(inference_df)

    uploaded_file = st.file_uploader("Choose a image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        img_bin = io.BytesIO(bytes_data)

        img = Image.open(img_bin)
        inference_df = inference(img)
        st.balloons()

        # change image
        image_area.image(uploaded_file)

        # change inference
        inference_best_guess.header(inference_df.label[0])
        inference_topk.table(inference_df)


def inference(image: Image) -> str:
    # cached
    model, transform, index_to_label_map = load_model()

    with torch.no_grad():
        inputs = transform(image).unsqueeze(0)
        res = model(inputs)
        probs = torch.nn.functional.softmax(res[0], dim=0)

        # guessed label and probs
        topk_probs, topk_indices = torch.topk(probs, 3)
        topk_labels = [index_to_label_map[str(i)] for i in topk_indices.tolist()]
        return pd.DataFrame({"label": topk_labels, "prob": topk_probs})


if __name__ == "__main__":
    main()
