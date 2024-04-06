import cv2
import numpy as np
import time

age_model = cv2.dnn.readNetFromCaffe('ageDeploy/age_deploy.prototxt', 'ageDeploy/age_net.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('genderDeploy/gender_deploy.prototxt', 'genderDeploy/gender_net.caffemodel')

cap = cv2.VideoCapture('video.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']

contador_mujeres = 0
contador_hombres_adultos = 0
contador_ancianos = 0
contador_niños = 0
contador_jovenes = 0

publicidad_mujeres = cv2.imread('publicidad_mujer.jpg')
publicidad_hombres_adultos = cv2.imread('publicidad_hombre.jpg')
publicidad_ancianos = cv2.imread('publicidad_anciano.jpg')
publicidad_niños = cv2.imread('publicidad_ninos.jpg')
publicidad_jovenes = cv2.imread('publicidad_joven.jpg')

current_advertisement = None

cv2.namedWindow('Contadores', cv2.WINDOW_NORMAL)


while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = cv2.resize(frame, (800, 600))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)

    area_pts = np.array([[240 + 50, 320], [480 + 50, 320], [620 + 50, 600], [50 + 50, 600]])

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask = fgbg.apply(image_area)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    _, fgmask = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 500 and area < 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            if x > area_pts[3][0] and x + w < area_pts[2][0] and y > area_pts[0][1] and y + h < area_pts[2][1]:
                if cv2.pointPolygonTest(area_pts, (x + w // 2, y + h // 2), False) > 0:

                    roi = frame[y:y+h, x:x+w]
                    blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))
                    age_model.setInput(blob)
                    gender_model.setInput(blob)
                    age_preds = age_model.forward()
                    gender_preds = gender_model.forward()

                    age_index = age_preds[0].argmax()
                    gender = "Male" if gender_preds[0].argmax() == 1 else "Female"

                    if age_index >= 4:
                        if gender == "Female":
                            contador_mujeres += 1
                            if contador_mujeres > contador_hombres_adultos and contador_mujeres > contador_ancianos and contador_mujeres > contador_niños and contador_mujeres > contador_jovenes:
                                current_advertisement = publicidad_mujeres
                        else:
                            contador_hombres_adultos += 1
                            if contador_hombres_adultos > contador_mujeres and contador_hombres_adultos > contador_ancianos and contador_hombres_adultos > contador_niños and contador_hombres_adultos > contador_jovenes:
                                current_advertisement = publicidad_hombres_adultos
                    if age_index >= 6:
                        contador_ancianos += 1
                        if contador_ancianos > contador_mujeres and contador_ancianos > contador_hombres_adultos and contador_ancianos > contador_niños and contador_ancianos > contador_jovenes:
                            current_advertisement = publicidad_ancianos
                    elif age_index <= 2:
                        contador_niños += 1
                        if contador_niños > contador_mujeres and contador_niños > contador_hombres_adultos and contador_niños > contador_ancianos and contador_niños > contador_jovenes:
                            current_advertisement = publicidad_niños
                    elif 15 <= age_index <= 32:
                        contador_jovenes += 1
                        if contador_jovenes > contador_mujeres and contador_jovenes > contador_hombres_adultos and contador_jovenes > contador_ancianos and contador_jovenes > contador_niños:
                            current_advertisement = publicidad_jovenes

    if current_advertisement is not None:
        cv2.imshow('Publicidad', current_advertisement)


    contador_texto = [
        f'Mujeres: {contador_mujeres}',
        f'Hombres Adultos: {contador_hombres_adultos}',
        f'Ancianos: {contador_ancianos}',
        f'Ninhos: {contador_niños}',
        f'Jovenes: {contador_jovenes}',
    ]

    text_height = 30
    text_widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0] for line in contador_texto]
    max_width = max(text_widths) + 60
    contador_frame = np.zeros((len(contador_texto) * text_height, max_width, 3), dtype=np.uint8)

    y_offset = 30
    for linea in contador_texto:
        cv2.putText(contador_frame, linea, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
    
    cv2.imshow('Contadores', contador_frame)

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 500 and area < 5000:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            age_text = f"Age: {age_list[age_index]}"
            gender_text = f"Gender: {gender}"
            
            cv2.putText(frame, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, gender_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Video Detectando', frame)

    k = cv2.waitKey(60) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

