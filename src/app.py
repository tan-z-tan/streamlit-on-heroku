import streamlit as st


def main():
    # initialize something if needed
    load_model()

    # sidebar
    st.sidebar.title("Streamlit on Server")

    # image in main area
    image_area = st.empty()
    image_area.image("images/sea-turtle.jpg")

    uploaded_file = st.file_uploader("Choose a image", type=["jpg", "png"])
    if uploaded_file is not None:
        # change image
        image_area.image(uploaded_file)


@st.cache
def load_model():
    return None


if __name__ == "__main__":
    main()