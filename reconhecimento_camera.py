import face_recognition
import cv2
import numpy as np

# Tenta carregar a imagem de referência
try:
    # Use 'rosto1.jpeg' como a imagem de referência
    imagem_conhecida = face_recognition.load_image_file("rosto1.jpeg")

    # Extrai as características do rosto da imagem de referência
    encoding_rosto_conhecido = face_recognition.face_encodings(imagem_conhecida)[0]

    # Cria uma lista com os encodings conhecidos e seus nomes
    encodings_rostos_conhecidos = [encoding_rosto_conhecido]
    # O nome que aparecerá na tela para o rosto conhecido
    nomes_rostos_conhecidos = ["Rosto 1"]

except FileNotFoundError:
    print("Erro: O arquivo 'rosto1.jpeg' não foi encontrado. Coloque-o na mesma pasta do script.")
    exit()
except IndexError:
    print("Erro: Não foi possível detectar um rosto em 'rosto1.jpeg'.")
    exit()

# Inicializa a webcam (câmera 0 geralmente é a padrão)
video_capture = cv2.VideoCapture(0)

print("\nIniciando câmera... Olhe para a câmera e pressione 'q' para sair.")

# Loop infinito para processar cada frame da câmera
while True:
    # Captura um único frame do vídeo
    ret, frame = video_capture.read()
    if not ret:
        break

    # Converte a imagem de BGR (padrão do OpenCV) para RGB (padrão do face_recognition)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encontra todos os rostos e seus encodings no frame atual
    localizacao_rostos_frame = face_recognition.face_locations(frame_rgb)
    encodings_rostos_frame = face_recognition.face_encodings(frame_rgb, localizacao_rostos_frame)

    # Loop em cada rosto encontrado no frame
    for (top, right, bottom, left), face_encoding in zip(localizacao_rostos_frame, encodings_rostos_frame):
        # Compara o rosto encontrado com os rostos conhecidos
        matches = face_recognition.compare_faces(encodings_rostos_conhecidos, face_encoding)
        nome = "Desconhecido"

        # Encontra a melhor correspondência se houver mais de um rosto conhecido
        distancias_rosto = face_recognition.face_distance(encodings_rostos_conhecidos, face_encoding)
        melhor_indice_match = np.argmin(distancias_rosto)
        if matches[melhor_indice_match]:
            nome = nomes_rostos_conhecidos[melhor_indice_match]

        # Desenha uma caixa ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Desenha uma etiqueta com o nome abaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        fonte = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nome, (left + 6, bottom - 6), fonte, 1.0, (255, 255, 255), 1)

    # Exibe a imagem resultante em uma janela
    cv2.imshow('Video', frame)

    # Pressione a tecla 'q' para fechar a janela e sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha todas as janelas abertas
video_capture.release()
cv2.destroyAllWindows()
print("Câmera desligada.")