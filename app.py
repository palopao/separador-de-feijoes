import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import concurrent.futures
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(page_title="Separador de Feijões Interativo", layout="wide")

# --- Função para processar cada feijão ---
def processar_feijao(mask, cnt, img, gray, N_CORES, PIX_ANALISAR):
    bean_pixels = img[mask == 255].astype(np.float32)
    if len(bean_pixels) == 0:
        return None

    kmeans = MiniBatchKMeans(
        n_clusters=N_CORES,
        random_state=42,
        batch_size=1024,
        n_init=5,
    )
    if len(bean_pixels) > PIX_ANALISAR:
        idx = np.random.choice(len(bean_pixels), PIX_ANALISAR, replace=False)
        kmeans.fit(bean_pixels[idx])
    else:
        kmeans.fit(bean_pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    colors = colors[sorted_idx]
    counts = counts[sorted_idx]
    percents = counts / counts.sum()

    # Ordenar por luminosidade
    lum = 0.299 * colors[:, 2] + 0.587 * colors[:, 1] + 0.114 * colors[:, 0]
    lum_idx = np.argsort(-lum)
    colors = colors[lum_idx]
    percents = percents[lum_idx]

    area = cv2.contourArea(cnt)
    row = {"Contorno": cnt, "Area_px": int(area)}

    for j, (color, p) in enumerate(zip(colors, percents)):
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        row[f"Cor{j + 1}"] = hex_color
        row[f"Cor{j + 1}_%"] = round(float(p * 100), 2)
    return row


# --- Interface Streamlit ---
st.title("Separador de Feijões Interativo")

# Bloco de Instruções da Imagem
st.info(
    "**Requisitos Importantes para a Imagem:**\n"
    "* A fotografia **deve ter um fundo de cor uniforme** (ex: um fundo totalmente branco, azul-escuro ou preto).\n"
    "* O fundo escolhido deve ser de uma **cor nitidamente distinta da cor dos feijões** para garantir uma segmentação correta.\n"
    "* Evite sombras excessivas, texturas ou superfícies refletoras sob os feijões."
)

metodo_entrada = st.radio(
    "Como deseja adicionar as imagens dos feijões?",
    ["Carregar Ficheiros do Dispositivo", "Tirar Foto com a Câmara em Direto"],
    horizontal=True,
)

imagens_para_processar = []

if metodo_entrada == "Carregar Ficheiros do Dispositivo":
    imagens_entrada = st.file_uploader(
        "Escolher imagens (múltiplas)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if imagens_entrada:
        imagens_para_processar = imagens_entrada
else:
    foto_cam = st.camera_input("Aponte a câmara para os feijões e tire a foto")
    if foto_cam:
        foto_cam.name = "captura_camara.png"
        imagens_para_processar = [foto_cam]

with st.expander("Parâmetros de Configuração"):
    col1, col2 = st.columns(2)
    with col1:
        MIN_AREA = st.number_input("Área mínima (px)", min_value=1, value=500, step=50)

        MIN_CIRCULARIDADE = st.number_input(
            "Circularidade mínima", min_value=0.0, max_value=1.0, value=0.6, step=0.05
        )

    with col2:
        N_CORES = st.number_input(
            "Número de cores por feijão", min_value=1, value=2, step=1
        )
        PIX_ANALISAR = st.number_input(
            "Pixels a analisar (KMeans)", min_value=100, value=2000, step=100
        )

executar = st.button("Executar Processamento")

# --- INICIALIZAÇÃO DA MEMÓRIA (SESSION STATE) ---
if "resultados_imagens" not in st.session_state:
    st.session_state.resultados_imagens = []
if "todas_tabelas" not in st.session_state:
    st.session_state.todas_tabelas = []

if executar and imagens_para_processar:
    st.session_state.resultados_imagens = []
    st.session_state.todas_tabelas = []

    progresso_barra = st.progress(0)
    progresso_texto = st.empty()

    total_imagens = len(imagens_para_processar)

    for idx_img, imagem_entrada in enumerate(imagens_para_processar):
        nome_original = imagem_entrada.name
        progresso_texto.text(
            f"A processar: {nome_original} ({idx_img + 1}/{total_imagens})..."
        )

        file_bytes = np.asarray(bytearray(imagem_entrada.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error(f"Não foi possível abrir a imagem {nome_original}.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ALGORITMO AUTO-ADAPTÁVEL: Escolhe a técnica baseada na cor do fundo

        # 1. Usamos a normalização apenas para a deteção do fundo (para ser mais fiável)
        gray_norm_check = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        margens = np.concatenate(
            [
                gray_norm_check[0, :],
                gray_norm_check[-1, :],
                gray_norm_check[:, 0],
                gray_norm_check[:, -1],
            ]
        )
        mediana_fundo = np.median(margens)

        if mediana_fundo > 100:
            # ==========================================
            # CENÁRIO A: FUNDO CLARO (Usa a Lógica V1 EXATA)
            # ==========================================
            # Usa 'gray' diretamente e 'dist_factor' a 0.40, exatamente como na V1
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                51,
                5,
            )
            blur = cv2.GaussianBlur(thresh, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            binary_mask = cv2.erode(edges, kernel, iterations=2)

            dist_factor = 0.40

        else:
            # ==========================================
            # CENÁRIO B: FUNDO ESCURO (Usa a Lógica V4 EXATA)
            # ==========================================
            blur = cv2.GaussianBlur(gray, (7, 7), 0)

            v = np.median(blur)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))

            edges = cv2.Canny(blur, lower, upper)

            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges_closed = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2
            )

            binary_mask = np.zeros_like(gray)
            contornos_borda, _ = cv2.findContours(
                edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                binary_mask, contornos_borda, -1, 255, thickness=cv2.FILLED
            )

            binary_mask = cv2.morphologyEx(
                binary_mask, cv2.MORPH_OPEN, kernel_ellipse, iterations=1
            )

            dist_factor = 0.45

        # -------------------------------------------------------------

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result = img.copy()
        tasks = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue
            circularidade = 4 * np.pi * (area / (perimetro * perimetro))
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            if circularidade >= MIN_CIRCULARIDADE:
                tasks.append((mask, cnt, img, gray, N_CORES, PIX_ANALISAR))
            else:
                # Separação Watershed usando o fator escolhido
                x, y, w, h = cv2.boundingRect(cnt)
                roi = mask[y : y + h, x : x + w]
                roi_img = img[y : y + h, x : x + w]
                dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)

                # APLICAÇÃO DO FATOR DE DISTÂNCIA DINÂMICO
                _, sure_fg = cv2.threshold(dist, dist_factor * dist.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(roi, sure_fg)
                num_markers, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                markers = cv2.watershed(roi_img.copy(), markers)

                for m in range(2, num_markers + 1):
                    submask = np.zeros_like(roi)
                    submask[markers == m] = 255
                    sub_contours, _ = cv2.findContours(
                        submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for sc in sub_contours:
                        if cv2.contourArea(sc) < MIN_AREA:
                            continue
                        full_mask = np.zeros(gray.shape, np.uint8)
                        cv2.drawContours(full_mask, [sc + (x, y)], -1, 255, -1)
                        tasks.append(
                            (full_mask, sc + (x, y), img, gray, N_CORES, PIX_ANALISAR)
                        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            results = list(executor.map(lambda p: processar_feijao(*p), tasks))
        feijoes_data = [r for r in results if r is not None]

        for i, row in enumerate(feijoes_data, start=1):
            cnt = row.pop("Contorno")
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            # Cálculo do centroóide para colocar o número no meio do feijão
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cX, cY = x + w // 2, y + h // 2

            cv2.putText(
                result,
                f"{i}",
                (cX - 10, cY + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            row["Feijao"] = i
            row["Foto"] = nome_original

        if feijoes_data:
            df = pd.DataFrame(feijoes_data)
            cols = ["Foto", "Feijao"] + [
                c for c in df.columns if c not in ["Foto", "Feijao"]
            ]
            df = df[cols]
            st.session_state.todas_tabelas.append(df)

        _, img_encoded = cv2.imencode(".png", result)
        img_bytes = img_encoded.tobytes()

        st.session_state.resultados_imagens.append(
            {"nome": nome_original, "bytes": img_bytes}
        )

        progresso_barra.progress((idx_img + 1) / total_imagens)

    progresso_texto.text("Processamento concluído com sucesso!")
    progresso_barra.empty()

elif executar and not imagens_para_processar:
    st.warning("Por favor, adicione uma imagem através do método selecionado.")


# --- RENDERIZAÇÃO DOS RESULTADOS GUARDADOS ---
if st.session_state.resultados_imagens:
    for idx, item in enumerate(st.session_state.resultados_imagens):
        nome_img = item["nome"]
        img_bytes = item["bytes"]

        st.write("---")
        st.subheader(f"Resultado: {nome_img}")

        ctrl_col1, ctrl_col2 = st.columns([3, 1])

        with ctrl_col1:
            largura_img = st.slider(
                f"Ajustar escala da imagem ({nome_img})",
                min_value=200,
                max_value=1200,
                value=500,
                step=50,
                key=f"slider_{nome_img}_{idx}",
            )

        with ctrl_col2:
            st.write("")
            st.download_button(
                label="📥 Descarregar Imagem",
                data=img_bytes,
                file_name=f"processada_{nome_img}.png",
                mime="image/png",
                key=f"dl_{nome_img}_{idx}",
            )

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, width=largura_img)

    if st.session_state.todas_tabelas:
        st.write("---")
        st.subheader("📊 Dados Analisados")
        df_total = pd.concat(st.session_state.todas_tabelas, ignore_index=True)

        def aplicar_cor_fundo(valor):
            if isinstance(valor, str) and valor.startswith("#") and len(valor) == 7:
                return f"background-color: {valor}; color: {valor};"
            return ""

        colunas_cor = [
            c for c in df_total.columns if c.startswith("Cor") and not c.endswith("%")
        ]

        df_estilizado = df_total.style.map(aplicar_cor_fundo, subset=colunas_cor)
        st.dataframe(df_estilizado, width="stretch", hide_index=True)