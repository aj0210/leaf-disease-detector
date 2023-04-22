import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

banner_path="banner.jpg"
model=tf.keras.models.load_model("leaf_disease_coloured_24_3.h5")

menu = ["Disease Detection","About"]
choice = st.sidebar.selectbox("Select Activty", menu)
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


if choice=="About":
    intro_markdown = read_markdown_file("about.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
elif choice=="Disease Detection":
    st.title("Grape Leaf Disease Diagnosis")
    file = st.file_uploader("Upload image...", type=["jpg", 'png'])
    def import_and_predict(image_data, model):
        size = (256, 256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img_reshape = img_resize[np.newaxis, ...]
        print(img_reshape)
        predict_class = np.argmax(model.predict(img_reshape), axis=1)

        return predict_class


    if file is None:
        st.write("Please upload an grape leaf image")
    else:
        image = Image.open(file)
        st.sidebar.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]

        if categories[prediction[0]] == "Healthy":
            st1 = "No Disease detected"
            st.sidebar.success(st1)
        elif categories[prediction[0]] == "Black_rot":
            stringbt = "This grape is having : Blackrot"
            st.sidebar.success(stringbt)
            st.title("Cause")
            st.markdown(
                """
                - Black rot, caused by the fungus Guignardia bidwellii, is a serious disease of cultivated and wild grapes.
                - The disease is most destructive in warm, wet seasons.
                - It attacks all green parts of the vine – leaves, shoots, leaf and fruit stems, tendrils, and fruit. The most damaging effect is to the fruit
                    
                """
                )
            st.title("Treatment")
            st.markdown(
                """
                - Planting. Space vines properly and choose a planting site where the vines will be exposed to full sun and good air circulation. Keep the vines off the ground and insure they are properly tied,limiting the amount of time the vines remain wet thus reducing infection.
                - Sanitation. Keep the fruit planting and surrounding areas free of weeds and tall grass. This practice will promote lower relative humidity and rapid drying of vines and thereby limit fungal infection.
                - Pruning. Prune the vines in early winter during dormancy. Select only a few strong, healthy canes from the previous year’s growth to produce the following season’s crop. Remove these prunings from the vineyard and burn or destroy.
                - Cultivation. Cultivate the vineyard before budbreak to bury the mummified berries. Diseased berries covered with soil do not produce spores that will reach the developing vines. For homegrown grapes, use 2–3 inches of leaf mulch or fine bark to cover infected debris.
                - Fungicides. Use protective fungicide sprays. Pesticides registered to protect the developing new growth include copper, captan, ferbam, mancozeb, maneb, triadimefon, and ziram. Important spraying times are as new shoots are 2 to 4 inches long, and again when they are 10 to 15 inches long, just before bloom, just after bloom, and when the fruit has set.Sovran 50WG,Flint 50WG, Abound Flowable (2.08F), and Pristine 38WDG.
                """
            )
        elif categories[prediction[0]] == "Esca_(Black_Measles)":
            stringbm = "This grape is having : Esca (Black_Measles)"
            st.sidebar.success(stringbm)
            st.title("Cause")
            st.write(
                "This disease is caused by Complex of fungi that includes Phaeoacremonium aleophilum, Phaeomoniella chlamydospora and Fomitiporia mediterranea")
            st.title("Treatment")
            st.markdown(
                """
                Traditional cultural methods to prevent and treat esca:
                - Avoiding large wounds and protecting any wound made with a healing varnish or a dressing containing a broad-spectrum fungicide (copper, maneb, flusilazol, carbendazim, etc.)
                - Treating grapevines with fungicides soon after a severe frost
                - Pruning healthy-looking vines before other vines
                - Curative methods like grapevine surgery, where the grapevine trunk is cut open using a chainsaw. Another smaller chainsaw is used to clean the trunk by removing dead/infected wood, leaving the trunk open to dry
                - Remove the infected berries, leaves and trunk and destroy them.

                Other method:
                - Till date there is no effective method to control this disease. However, protect the prune wounds to minimize fungal infection using wound sealant (5% boric acid in acrylic paint) or essential oil or suitable fungicides.
                - Chemical control of esca can be carried out by means of Sodium arsenite (12.5 g/liter).
                Such treatments are usually applied by spraying or painting the trunks and main branches
                with a Treatments should be applied for 2 consecutive years, at least 2 weeks after
                pruning and not later than 3 weeks before sprouting .
                In the third and fourth years, the treatment can be omitted but must then be repeated in
                the following 2 years, and so on.
                """
            )
        elif categories[prediction[0]] == "Leaf_blight_(Isariopsis_Leaf_Spot)":
            stringlb = "This grape is having : Leaf blight (Isariopsis Leaf Spot)"
            st.title("Cause")
            st.write(
                "Leaf blight is caused by the fungus Pseudocercospora vitis. The pathogen was named Isariopsis clavispora at one time and the disease is still often referred to as isariopsis leaf spot")


            st.sidebar.success(stringlb)
            st.title("Treatment")
            st.markdown(
                """
                - Practices that increase air circulation such as shoot positioning and thinning aid in management of the disease
                - Doses of 1.0 and 2.0 mL L−1 of the essential oil of lemon grass reduced isariopsis leaf spot and in two consecutive years.
                - Well-drained soils and canopy management practices to open the vine canopy to air and light to reduce the amount of trapped moisture in grape orchards can prevent the Pseudocercospora leaf spot disease

                """
            )

