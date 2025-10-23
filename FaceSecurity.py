import streamlit as st
import cv2
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

st.title("Face Security")

frame_placeholder = st.empty()



if "Mode" not in st.session_state:
    st.session_state.Mode = "menu"
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "member" not in st.session_state:
    st.session_state.member = False
if "train" not in st.session_state:
    st.session_state.train = False
if "model" not in st.session_state:
    st.session_state.model = None
if "class_names" not in st.session_state:
    st.session_state.class_names = None


# โหลดตัวตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if st.session_state.Mode == "menu":
    col1,col2,col3 = st.columns(3)
    
    SetupMode_btn = col1.button("Setup",width="stretch")
    Start_btn = col2.button("Start",width="stretch")
    Stop_btn = col3.button("Stop Camera",width="stretch")


    if SetupMode_btn:
        st.session_state.Mode = "setup"
        st.session_state.run_camera = False
        st.rerun()
    if Start_btn:
        st.session_state.run_camera = True
    if Stop_btn:
        st.session_state.run_camera = False
        
elif st.session_state.Mode == "setup":
    #Setup Mode
    

    if st.button("ย้อนกลับ", width="stretch"):
        st.session_state.Mode = "menu"
        st.rerun()

    st.markdown("<h1 style='text-align: center; color: grey;'>เพิ่มสมาชิกในบ้าน</h1>", unsafe_allow_html=True)   

    folder_name = st.text_input("ตั้งชื่อโฟลเดอร์ใหม่:", placeholder="เช่น Most")
    if st.button("สร้างโฟลเดอร์"):
        if folder_name.strip() == "":
            st.warning("กรุณากรอกชื่อโฟลเดอร์ก่อนครับ")
        else:
            new_folder_path = os.path.join("DatasetTrain", folder_name.strip())
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                st.success(f"สร้างโฟลเดอร์ {folder_name} สำเร็จแล้ว!")
            else:
                st.info(f"โฟลเดอร์ {folder_name} มีอยู่แล้ว")

    existing_folders = [f for f in os.listdir("DatasetTrain") if os.path.isdir(os.path.join("DatasetTrain", f))]
    if existing_folders:
        selected_folder = st.selectbox("เลือกสมาชิก Dataset:", existing_folders)
        st.write(f"สมาชิกที่เลือกคือ: {selected_folder}")
    else:
        st.info("กรุณาเพิ่มชื่อสมาชิก")

    if st.button("เริ่มตรวจจับใบหน้า", width="stretch",type="primary"):
        st.session_state.member = True
        st.rerun()

    st.markdown("<h1 style='text-align: center; color: grey;'>ฝึกโมเดล</h1>", unsafe_allow_html=True)
    if not os.path.exists("DataSetTrain"):
        st.warning("ยังไม่มีโฟลเดอร์")
    else:
        # ลิสต์โฟลเดอร์ทั้งหมดใน DatasetTrain
        folders = [f for f in os.listdir("DataSetTrain") if os.path.isdir(os.path.join("DataSetTrain", f))]

        if len(folders) == 0:
            st.warning("ไม่มีโฟลเดอร์")
        else:
            st.write(f"พบสมาชิกทั้งหมด {len(folders)} คน")

            for folder in folders:
                folder_path = os.path.join("DataSetTrain", folder)
                images = [f for f in os.listdir(folder_path)
                        if f.lower().endswith((".png"))]

                st.subheader(f"คุณ {folder}")
                if len(images) == 0:
                    st.error("ไม่พบรูปในโฟลเดอร์นี้")
                else:
                    random_img = random.choice(images)
                    img_path = os.path.join(folder_path, random_img)
                    st.image(Image.open(img_path), caption=f"{folder} - {random_img}", width=200)
    if st.button("เริ่มเทรนโมเดล", width="stretch",type="primary"):
        st.session_state.train = True

#Detection Mode
if st.session_state.run_camera:
    cam = cv2.VideoCapture(0)
    class_names = st.session_state.class_names
    print(class_names)
    while st.session_state.run_camera:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #ตรวจจับใบหน้าในภาพ scaleFactor
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (150, 150)) / 255.0
                face_img = np.expand_dims(face_img, axis=0)
                if st.session_state.model is None:
                    label = "Please Setup"
                    color = (0,0,255)
                else:
                    preds = st.session_state.model.predict(face_img)
                    idx = np.argmax(preds)#ค่าที่ได้มากสุดตำแหน่ง0,1,2
                    conf = np.max(preds)#ค่าความเชื่อมั่นที่มากที่สุด
                    if conf >= 0.8: #ถ้ามากกว่าค่าความเชื่อมั่นที่0.8
                        label = f"Detect:  {class_names[idx]} ({conf*100:.1f}%)" #พบเจอใบหน้าใคร มั่นใจกี่%
                        color = (0, 255, 0) #สีเขียว
                    else:
                        label = f"Unknow ({conf*100:.1f}%)" #พบเจอหน้าแต่ไม่รู้จัก
                        color = (0, 0, 255) #สีแดง
                # วาดหน้า
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # ถ้าไม่เจอใบหน้า
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_placeholder.image(frame, channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    print("ปิดกล้องเรียบร้อย")


if st.session_state.member:
    cam = cv2.VideoCapture(0)
    i=1
    dataset_dir = os.path.join(os.getcwd(), f"DatasetTrain/{selected_folder}")
    while st.session_state.member:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #ตรวจจับใบหน้าในภาพ scaleFactor
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (150, 150)) / 255.0
                face_img = np.expand_dims(face_img, axis=0)

                filename = f"{selected_folder}_{i}.png"
                filepath = os.path.join(dataset_dir, filename)

                reface_img = frame[y:y+h, x:x+w]
                reface_img = cv2.resize(reface_img, (150, 150))

                cv2.imwrite(filepath, cv2.cvtColor(reface_img, cv2.COLOR_RGB2BGR))
                i+=1
                label = f"Please Wait"
                if i>=800:
                    label = f"Look At Camera {i}"
                elif i>=700:
                    label = f"Please Smile {i}"
                elif i>=600:
                    label = f"Open Your Mouth {i}"
                elif i>= 450:
                    label = f"Blink Your Eyes {i}"
                elif i>= 300:
                    label = f"Turn Left Slowly {i}"
                elif i>= 200:
                    label = f"Turn Right Slowly {i}"
                elif i>= 100:
                    label = f"Payak Naa Cha Cha {i}"
                else:
                    label = f"Look At Camera {i}"
                color = (0,255,0)
                # วาดหน้า
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # ถ้าไม่เจอใบหน้า
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_placeholder.image(frame, channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord('q') or i > 1000:
            break
    cam.release()
    cv2.destroyAllWindows()
    frame_placeholder = st.empty()
    st.session_state.member = False
    print("ปิดกล้องเรียบร้อย")
    st.rerun()

if st.session_state.train:
    dataset_dir = os.path.join(os.getcwd(), f"DatasetTrain")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator #นำเข้าImageDataGeneratorจาก Keras ซึ่งเป็นคลาสที่ใช้สร้างออบเจกต์สำหรับ เตรียมข้อมูลภาพ
    datagen = ImageDataGenerator(
        rescale=1./255, #ปรับค่าสีของภาพจากช่วง [0, 255] ไป [0, 1]
        validation_split=0.2, #แบ่งข้อมูลสำหรับvalidation 20%
        rotation_range=20, #ซูมภาพแบบสุ่มในช่วง 20องศา ทำให้มีมุมมองโมเดลมากขึ้น แม่นยำขึ้น
        zoom_range=0.2, #ระยะซูมภาพสุ่ม20% เพื่อความแม่นยำ
        width_shift_range=0.1, #ขยับภาพในแนวนอนแบบสุ่มได้สูงสุด 10% ของความกว้าง
        height_shift_range=0.1, #ขยับภาพในแนวตั้งแบบสุ่มได้สูงสุด 10% ของความสูง
        shear_range=0.2, #ทำการ เฉือน (shear transformation) ของภาพ
        horizontal_flip=True, #กลับภาพในแนวนอน (ซ้าย↔ขวา) แบบสุ่ม
    )
        # กำหนดค่าพื้นฐาน
    BATCH_SIZE = 32
    IMG_SIZE = 150 # ขนาดรูปภาพที่ต้องการ (กว้าง x สูง)


    #datagen.flow_from_directory เพื่อโหลดภาพจากโฟลเดอร์ และเตรียมให้พร้อมสำหรับการเทรนในรูปแบบ batch
    # สร้างชุดข้อมูลสำหรับฝึก (Train)
    train_dataset = datagen.flow_from_directory(
        dataset_dir,#path folder ของข้อมูลที่จะนำไปเทรน
        target_size=(150, 150),#กำหนดขนาดภาพ resize ให้มีขนาด 150×150 pixel ก่อนนำเข้าโมเดล
        batch_size=32, #แต่ละรอบจะส่งภาพเข้าระบบทีละ 32 ภาพ เพื่อไม่ให้กินหน่วยความจำมากเกินไป
        class_mode='categorical', #ระบุว่า label ของข้อมูลเป็นแบบ “หลายคลาส” (multi-class)
        subset='training'#ใช้เฉพาะข้อมูลส่วนที่เป็น training
    )

    val_dataset = datagen.flow_from_directory(
        dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation' #ใช้เฉพาะข้อมูลส่วนที่เป็น validation
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),#กำหนดขนาดของภาพที่จะป้อนเข้าโมเดล 3 คือ RGB
        include_top=False,#→ ไม่เอาส่วน “หัวโมเดล” (fully connected layers) เพราะเราจะเพิ่ม “head” ของเราที่จำแนกเฉพาะคลาสใน dataset ของเราเอง
        weights='imagenet' #โหลดน้ำหนักที่เทรนมาจาก ImageNet มาใช้ เพื่อใช้ประโยชน์จาก feature ที่โมเดลเรียนรู้ไว้แล้ว(transfer learning)
    )

    base_model.trainable = False  # ไม่เทรนชั้น base

    # แสดงชื่อคลาสที่เจอ ('Aom','Most','New')
    class_names = list(train_dataset.class_indices.keys())
    print("คลาสที่พบ:", class_names)

        # สร้างโมเดลแบบลำดับชั้น (Sequential)
    # Sequential: คือการบอกว่าจะสร้างโมเดลโดยการนำแต่ละชั้น (layer) มาต่อกันเป็นลำดับ
    model = tf.keras.Sequential([ 
        base_model, #คือโมเดลฐาน ที่ถูกโหลดมาจากโมเดลสำเร็จรูป (pretrained model)
    tf.keras.layers.GlobalAveragePooling2D(), #ลดมิติของ Feature Map ที่ได้จาก base_model
        tf.keras.layers.Flatten(),#แปลงข้อมูลจาก2D,3D เป็น 1D
        tf.keras.layers.Dense(128, activation='relu'), #ใช้reluปรับ
        tf.keras.layers.Dropout(0.3),#ปิดการทำงานของบาง neuron แบบสุ่ม 30% (0.3) ขณะเทรน ป้องกันoverfitting
        tf.keras.layers.Dense(3, activation='softmax')  # จำแนก3class ใช้ฟังก์ชัน Softmax เพื่อให้ผลลัพธ์ออกมาเป็น ความน่าจะเป็นรวมกันได้ 1
    ])

        # optimizer (วิธีปรับปรุงโมเดล), loss (วิธีวัดความผิดพลาด) และ metrics (การวัดผล เช่น accuracy หรือความแม่นยำ)
    model.compile(optimizer='adam', #อัลกอริทึมปรับน้ำหนักอัตโนมัติ ที่ได้รับความนิยมมากที่สุดใน deep learning
                loss='categorical_crossentropy', #ใช้ในกรณีที่มี หลายคลาส
                metrics=['accuracy'])
    EPOCHS = 1 # จำนวนรอบในการฝึก
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("กำลังฝึกโมเดล กรุณารอสักครู่ . . .")
    # เริ่มฝึกโมเดล
    for epoch in range(EPOCHS):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1,  # ฝึกทีละ 1 epoch
        )
        train_acc = history.history['accuracy'][0]
        val_acc = history.history['val_accuracy'][0]
        loss = history.history['loss'][0]
        progress = int(((epoch + 1) / EPOCHS) * 100)
        progress_bar.progress(progress)
        status_text.text(f"กำลังฝึก Epoch {epoch+1}/{EPOCHS} " f"✅ ความแม่นยำ {train_acc:.2f} | Validation {val_acc:.2f} | Loss {loss:.2f}")
    st.session_state.model = model
    st.session_state.class_names = class_names
    st.success("🎉 ฝึกโมเดลเสร็จสิ้น!")
    st.session_state.train = False