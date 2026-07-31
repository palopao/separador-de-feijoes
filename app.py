import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import concurrent.futures
import streamlit as st
import base64
from ultralytics import FastSAM

# Configuração da página do Streamlit
st.set_page_config(page_title="Separador de Feijões", layout="wide")

# --- Função OTIMIZADA para processar cada feijão ---
def processar_feijao(
    bean_pixels, cnt, AUTO_CORES, N_CORES_CONFIG, LIMIAR_PCT
):
    if len(bean_pixels) == 0:
        return None

    # 1. Utiliza a totalidade dos pixeis (Elimina a inconsistência)
    pixels_sample = bean_pixels

    # 2. Calcula a Cor Média real e o nível de manchas (Desvio Padrão)
    cor_media = np.mean(pixels_sample, axis=0)
    std_cor = np.std(pixels_sample, axis=0)

    # 3. Executa KMeans UMA ÚNICA VEZ
    kmeans = MiniBatchKMeans(
        n_clusters=N_CORES_CONFIG,
        random_state=42,
        batch_size=1024,
        n_init=3,
    )
    kmeans.fit(pixels_sample)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    colors = colors[sorted_idx]
    counts = counts[sorted_idx]
    percents = counts / counts.sum()

    # --- DETEÇÃO AUTO RÁPIDA: Filtra cores irrelevantes/ruído ---
    if AUTO_CORES and len(colors) > 1:
        mask_relevante = percents >= (LIMIAR_PCT / 100.0)

        # Garante que pelo menos a cor dominante é mantida
        if not np.any(mask_relevante):
            mask_relevante[0] = True

        colors = colors[mask_relevante]
        counts = counts[mask_relevante]
        percents = counts / counts.sum()  # Re-normaliza as percentagens restantes

    # Ordenação por luminância
    lum = 0.299 * colors[:, 2] + 0.587 * colors[:, 1] + 0.114 * colors[:, 0]
    lum_idx = np.argsort(-lum)
    colors = colors[lum_idx]
    percents = percents[lum_idx]

    # 4. Extração de Métricas de Tamanho e Forma
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0.0

    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu1 = -np.sign(hu_moments[0]) * np.log10(abs(hu_moments[0]) + 1e-10)
    hu2 = -np.sign(hu_moments[1]) * np.log10(abs(hu_moments[1]) + 1e-10)

    # 5. Guardar dados para a Tabela Visual (Inclui Média Exata)
    row = {
        "Contorno": cnt, 
        "Area_px": int(area),
        "Media_B": float(cor_media[0]),
        "Media_G": float(cor_media[1]),
        "Media_R": float(cor_media[2])
    }

    # 6. Construir o Vetor de Características para a IA agrupar
    features = [
        float(area), float(aspect_ratio), float(solidity), float(hu1), float(hu2),
        float(cor_media[0]), float(cor_media[1]), float(cor_media[2]), # Cor Base
        float(std_cor[0]), float(std_cor[1]), float(std_cor[2])        # Variação/Manchas
    ]

    # Adiciona as cores detetadas
    for j, (color, p) in enumerate(zip(colors, percents)):
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        row[f"Cor{j + 1}"] = hex_color
        row[f"Cor{j + 1}_%"] = round(float(p * 100), 2)
        features.extend([float(color[0]), float(color[1]), float(color[2]), float(p)])

    # Padding neutro para manter dimensão constante na matriz do KMeans global
    max_pad = N_CORES_CONFIG
    dom_color = colors[0] if len(colors) > 0 else [0, 0, 0]
    for j in range(len(colors), max_pad):
        features.extend(
            [float(dom_color[0]), float(dom_color[1]), float(dom_color[2]), 0.0]
        )

    row["Features_Clustering"] = features
    return row


# --- Interface Streamlit ---
st.title("Separador de Feijões")

with st.expander("Dicas para Melhores Resultados", expanded=False):
    st.markdown("""
    **1. Como Tirar a Foto (Preparação)**
    * **Fundo:** Utilize um fundo liso e de cor neutra (uma folha branca, cartolina preta ou azul) para contrastar com os feijões.
    * **Iluminação:** Prefira luz natural ou iluminação uniforme. Evite sombras fortes e reflexos de luz diretos, pois alteram a perceção da cor da câmara.
    * **Espaçamento:** Espalhe os feijões para que não se toquem nem fiquem sobrepostos. O programa deteta contornos isolados com muito mais precisão.

    **2. Parâmetros de Configuração**
    * **Área e Circularidade:** Se o programa estiver a detetar pequenas sujidades, sombras ou feijões partidos, aumente a *Área mínima* ou a *Circularidade mínima*.
    * **Peso da Cor:** Se a separação por grupos estiver a focar-se muito no tamanho (juntando feijões de cores diferentes), aumente o *Peso da Cor* (ex: 4.0 ou 5.0).
    * **Grupos (Auto vs Manual):** Se souber exatamente quantos tipos de feijões estão na imagem, desligue a *Deteção Auto de Grupos* e insira o número exato. Isto torna o processo mais rápido e preciso.

    **3. Análise dos Resultados**
    * **Destacar Melhores:** Nos resultados, ative o filtro "Destacar Melhores por Grupo" para identificar apenas os feijões maiores e mais representativos de cada categoria.
    * **Ajustes:** Se a deteção de cores secundárias falhar ou captar manchas muito pequenas, aumente o *Limiar mínimo de área (%)* para ser mais rigoroso e clique em "Executar Processamento" novamente.
    """)

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

with st.expander("Parâmetros de Configuração", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        MIN_AREA = st.number_input("Área mínima (px)", min_value=1, value=500, step=50)
        MIN_CIRCULARIDADE = st.number_input(
            "Circularidade mínima", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )

    with col2:
        AUTO_CORES = st.toggle("Deteção Auto de Cores", value=True)
        if AUTO_CORES:
            N_CORES_CONFIG = st.number_input(
                "Máximo de cores por feijão (Auto)", min_value=1, max_value=10, value=5, step=1
            )
        else:
            N_CORES_CONFIG = st.number_input(
                "Nº exato de cores por feijão (Manual)", min_value=1, max_value=10, value=3, step=1
            )
        LIMIAR_PCT = st.number_input(
            "Limiar mínimo de área (%)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Uma cor só é mantida se ocupar pelo menos esta % da área do feijão.",
            disabled=not AUTO_CORES,
        )

    with col3:
        AUTO_GRUPOS = st.toggle("Deteção Auto de Grupos", value=True)

        if AUTO_GRUPOS:
            MAX_GRUPOS_AUTO = st.number_input(
                "Máximo de Tipos/Grupos (Auto)",
                min_value=2,
                max_value=15,
                value=5,
                step=1,
            )
            N_GRUPOS_MANUAL = 1  # Valor de fallback
        else:
            N_GRUPOS_MANUAL = st.number_input(
                "Nº exato de Grupos (Manual)",
                min_value=1,
                max_value=15,
                value=3,
                step=1,
            )
            MAX_GRUPOS_AUTO = 2  # Valor de fallback

    with col4:
        PESO_COR = st.number_input(
            "Importância da Cor (Peso)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Valores mais altos dão muito mais prioridade à cor do que à forma/tamanho ao agrupar.",
        )

executar = st.button("Executar Processamento")

if "resultados_imagens" not in st.session_state:
    st.session_state.resultados_imagens = []


@st.cache_resource
def carregar_modelo():
    return FastSAM("FastSAM-s.pt")


if executar and imagens_para_processar:
    modelo_ia = carregar_modelo()
    st.session_state.resultados_imagens = []

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

        MAX_DIMENSION = 1280
        img_h, img_w = img.shape[:2]

        if max(img_h, img_w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(img_h, img_w)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_h, img_w = img.shape[:2]

        area_total_imagem = img_h * img_w
        img_clean = img.copy()

        resultados_sam = modelo_ia(
            img, device="cpu", retina_masks=True, conf=0.5, iou=0.8, verbose=False
        )

        tasks = []
        candidatos = []

        if len(resultados_sam) > 0 and resultados_sam[0].masks is not None:
            mascaras = resultados_sam[0].masks.data.cpu().numpy()

            if mascaras.shape[1:] != (img_h, img_w):
                mascaras_resized = [
                    cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    for m in mascaras
                ]
            else:
                mascaras_resized = mascaras

            for mask_array in mascaras_resized:
                mask_uint8 = (mask_array * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:
                    area = cv2.contourArea(cnt)

                    if area < MIN_AREA or area > (area_total_imagem * 0.5):
                        continue

                    perimetro = cv2.arcLength(cnt, True)
                    if perimetro == 0:
                        continue

                    circularidade = 4 * np.pi * (area / (perimetro * perimetro))

                    if circularidade >= MIN_CIRCULARIDADE:
                        x, y, w, h = cv2.boundingRect(cnt)
                        mask_roi = np.zeros((h, w), np.uint8)

                        cnt_offset = cnt - np.array([x, y])
                        cv2.drawContours(mask_roi, [cnt_offset], -1, 255, -1)

                        candidatos.append(
                            {
                                "area": area,
                                "cnt": cnt,
                                "mask_roi": mask_roi,
                                "bbox": (x, y, w, h),
                            }
                        )

            candidatos.sort(key=lambda x: x["area"], reverse=True)
            mapa_ocupacao = np.zeros((img_h, img_w), dtype=np.uint8)

            for cand in candidatos:
                x, y, w, h = cand["bbox"]
                mask_roi = cand["mask_roi"]
                mapa_roi = mapa_ocupacao[y : y + h, x : x + w]

                intersecao = cv2.bitwise_and(mask_roi, mapa_roi)
                area_intersecao = np.count_nonzero(intersecao)
                area_mascara = np.count_nonzero(mask_roi)

                if area_mascara > 0 and (area_intersecao / area_mascara) > 0.3:
                    continue

                mapa_ocupacao[y : y + h, x : x + w] = cv2.bitwise_or(mapa_roi, mask_roi)

                img_roi = img[y : y + h, x : x + w]
                bean_pixels = img_roi[mask_roi == 255].astype(np.float32)

                tasks.append(
                    (
                        bean_pixels,
                        cand["cnt"],
                        AUTO_CORES,
                        N_CORES_CONFIG,
                        LIMIAR_PCT
                    )
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            results = list(executor.map(lambda p: processar_feijao(*p), tasks))

        feijoes_data = [r for r in results if r is not None]

        # --- LÓGICA DE AGRUPAMENTO AUTOMÁTICO OU MANUAL COM PESO NA COR ---
        if len(feijoes_data) > 1:
            features_list = [f["Features_Clustering"] for f in feijoes_data]
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_list)

            # Aplicação do Peso de Cor
            features_scaled[:, 5:] *= PESO_COR

            melhor_k = 1
            melhores_labels = np.zeros(len(feijoes_data), dtype=int)

            if AUTO_GRUPOS:
                max_k_possivel = min(MAX_GRUPOS_AUTO, len(feijoes_data) - 1)
                if max_k_possivel >= 2:
                    melhor_score = -1
                    for k in range(2, max_k_possivel + 1):
                        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels_temp = kmeans_temp.fit_predict(features_scaled)

                        score = silhouette_score(features_scaled, labels_temp)

                        if score > melhor_score:
                            melhor_score = score
                            melhor_k = k
                            melhores_labels = labels_temp
            else:
                # Agrupamento Manual direto sem testar múltiplos 'K'
                melhor_k = min(N_GRUPOS_MANUAL, len(feijoes_data))
                if melhor_k >= 2:
                    kmeans_manual = KMeans(
                        n_clusters=melhor_k, random_state=42, n_init=10
                    )
                    melhores_labels = kmeans_manual.fit_predict(features_scaled)

            num_grupos_final = melhor_k
            for row, lbl in zip(feijoes_data, melhores_labels):
                row["Grupo"] = f"Tipo {lbl + 1}"
        else:
            num_grupos_final = 1
            for row in feijoes_data:
                row["Grupo"] = "Tipo 1"

        coordenadas_feijoes = {}
        resumo_grupos = {}

        for i, row in enumerate(feijoes_data, start=1):
            cnt = row.pop("Contorno")
            row.pop("Features_Clustering")

            g_str = row["Grupo"]
            grupo_num = int(g_str.split(" ")[1])

            if g_str not in resumo_grupos:
                resumo_grupos[g_str] = {"areas": [], "r": [], "g": [], "b": []}
            resumo_grupos[g_str]["areas"].append(row["Area_px"])

            # Utiliza a Cor Média Exata gravada previamente
            if "Media_R" in row:
                resumo_grupos[g_str]["r"].append(row["Media_R"])
                resumo_grupos[g_str]["g"].append(row["Media_G"])
                resumo_grupos[g_str]["b"].append(row["Media_B"])
                # Remove do display da tabela (opcional, para ficar mais limpo)
                row.pop("Media_R")
                row.pop("Media_G")
                row.pop("Media_B")

            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = x + w // 2, y + h // 2

            raio = int(max(w, h) / 2) + 10

            coordenadas_feijoes[i] = {
                "centro": (cX, cY),
                "raio": raio,
                "cnt": cnt,
                "grupo_num": grupo_num,
            }
            row["Feijao"] = i

        df_imagem = pd.DataFrame()
        if feijoes_data:
            df = pd.DataFrame(feijoes_data)
            cols = ["Feijao", "Grupo"] + [
                c for c in df.columns if c not in ["Feijao", "Grupo"]
            ]
            df_imagem = df[cols]

        st.session_state.resultados_imagens.append(
            {
                "nome": nome_original,
                "img_clean": img_clean,
                "coords": coordenadas_feijoes,
                "tabela": df_imagem,
                "resumo_grupos": resumo_grupos if num_grupos_final > 1 else {},
            }
        )

        progresso_barra.progress((idx_img + 1) / total_imagens)

    progresso_texto.text("Processamento concluído com sucesso!")
    progresso_barra.empty()

elif executar and not imagens_para_processar:
    st.warning("Por favor, adicione uma imagem através do método selecionado.")


# --- RENDERIZAÇÃO DOS RESULTADOS GUARDADOS ---
if st.session_state.resultados_imagens:

    def aplicar_cor_fundo(valor):
        if isinstance(valor, str) and valor.startswith("#") and len(valor) == 7:
            return f"background-color: {valor}; color: {valor};"
        return ""

    cores_bgr_global = {
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 255, 255),
        4: (255, 0, 255),
        5: (0, 165, 255),
        6: (255, 255, 0),
        7: (0, 0, 255),
        8: (203, 192, 255),
        9: (255, 255, 255),
        10: (128, 0, 128),
    }

    for idx, item in enumerate(st.session_state.resultados_imagens):
        if idx > 0:
            st.divider()

        nome_img = item["nome"]
        img_clean = item["img_clean"]
        coords = item["coords"]
        df_tabela = item["tabela"]
        resumo_grupos = item.get("resumo_grupos", {})

        st.subheader(f"Imagem: {nome_img}")

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            largura_img = st.slider(
                "Escala da imagem", 100, 750, 300, 10, key=f"slider_{nome_img}_{idx}"
            )
        with ctrl_col2:
            tamanho_num = st.slider(
                "Tamanho dos números", 0.3, 4.0, 1.0, 0.1, key=f"font_{nome_img}_{idx}"
            )
        with ctrl_col3:
            max_feijao = max(coords.keys()) if coords else 0
            feijao_pesquisa = st.number_input(
                "Pesquisar Feijão Nº",
                0,
                max_feijao,
                0,
                1,
                key=f"search_{nome_img}_{idx}",
            )

        st.markdown("###### Filtro de Destaque")

        filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(
            [1.5, 0.4, 1.5, 2.5], vertical_alignment="center"
        )

        with filt_col1:
            destacar_melhores = st.toggle(
                "Destacar Melhores por Grupo", value=False, key=f"tgl_{nome_img}_{idx}"
            )

        with filt_col2:
            st.markdown(
                "<p style='margin-bottom: 0;'>Top (X):</p>",
                unsafe_allow_html=True,
            )

        with filt_col3:
            top_x = st.number_input(
                "Top (X)",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                disabled=not destacar_melhores,
                label_visibility="collapsed",
                key=f"topx_{nome_img}_{idx}",
            )

        img_display = img_clean.copy()

        feijoes_a_desenhar = set(coords.keys())

        if destacar_melhores and not df_tabela.empty:
            top_beans = (
                df_tabela.sort_values("Area_px", ascending=False)
                .groupby("Grupo")
                .head(top_x)["Feijao"]
                .tolist()
            )
            feijoes_a_desenhar = set(top_beans)

        for i, info in coords.items():
            if i in feijoes_a_desenhar:
                cor_contorno = cores_bgr_global.get(info["grupo_num"], (255, 255, 255))
                cv2.drawContours(img_display, [info["cnt"]], -1, cor_contorno, 3)

                cX, cY = info["centro"]
                if i == feijao_pesquisa:
                    cv2.circle(img_display, (cX, cY), info["raio"], (0, 0, 255), 4)
                    cv2.putText(
                        img_display,
                        f"{i}",
                        (cX - 10, cY + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        tamanho_num + 0.2,
                        (0, 0, 255),
                        max(2, int(tamanho_num * 2) + 1),
                    )
                else:
                    cv2.putText(
                        img_display,
                        f"{i}",
                        (cX - 10, cY + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        tamanho_num,
                        (0, 0, 255),
                        max(1, int(tamanho_num * 2)),
                    )

        img_download = img_display.copy()

        if resumo_grupos:
            num_grupos_reais = len(resumo_grupos)
            altura_legenda = 50 + (35 * num_grupos_reais)
            h, w, c = img_download.shape

            img_padded = np.zeros((h + altura_legenda, w, c), dtype=np.uint8)
            img_padded[:h, :w] = img_download

            y_offset = h + 30
            x_offset = 20

            cv2.putText(
                img_padded,
                "Legenda dos Grupos (Caracteristicas):",
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += 35

            for g_num in range(1, 16):
                g_str = f"Tipo {g_num}"
                if g_str in resumo_grupos and len(resumo_grupos[g_str]["areas"]) > 0:
                    avg_a = int(np.mean(resumo_grupos[g_str]["areas"]))
                    avg_r = (
                        int(np.mean(resumo_grupos[g_str]["r"]))
                        if resumo_grupos[g_str]["r"]
                        else 0
                    )
                    avg_g = (
                        int(np.mean(resumo_grupos[g_str]["g"]))
                        if resumo_grupos[g_str]["g"]
                        else 0
                    )
                    avg_b = (
                        int(np.mean(resumo_grupos[g_str]["b"]))
                        if resumo_grupos[g_str]["b"]
                        else 0
                    )

                    cor_contorno = cores_bgr_global.get(g_num, (255, 255, 255))

                    cv2.rectangle(
                        img_padded,
                        (x_offset, y_offset - 15),
                        (x_offset + 20, y_offset + 5),
                        cor_contorno,
                        -1,
                    )
                    cv2.rectangle(
                        img_padded,
                        (x_offset, y_offset - 15),
                        (x_offset + 20, y_offset + 5),
                        (255, 255, 255),
                        1,
                    )

                    texto = f"{g_str}: ~{avg_a}px | Cor Principal: "
                    cv2.putText(
                        img_padded,
                        texto,
                        (x_offset + 30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    (tw, th), _ = cv2.getTextSize(
                        texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.circle(
                        img_padded,
                        (x_offset + 30 + tw + 15, y_offset - 5),
                        10,
                        (avg_b, avg_g, avg_r),
                        -1,
                    )
                    cv2.circle(
                        img_padded,
                        (x_offset + 30 + tw + 15, y_offset - 5),
                        10,
                        (255, 255, 255),
                        1,
                    )

                    y_offset += 35

            img_download = img_padded

        _, img_encoded_display = cv2.imencode(".png", img_display)
        img_bytes_display = img_encoded_display.tobytes()

        _, img_encoded_dl = cv2.imencode(".png", img_download)
        img_bytes_dl = img_encoded_dl.tobytes()

        img_b64 = base64.b64encode(img_bytes_display).decode("utf-8")
        html_code = f"""
        <div style="max-width: 100%; max-height: 80vh; overflow: auto; border: 1px solid #444; border-radius: 5px; width: fit-content; margin: 0 auto; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_b64}" style="width: {largura_img}px; max-width: none; max-height: none; height: auto; display: block;">
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

        if resumo_grupos:
            legend_html = "<div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 15px; border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; background-color: rgba(0,0,0,0.2); margin: 0 auto 30px auto; width: fit-content;'>"

            for g_num in range(1, 16):
                g_str = f"Tipo {g_num}"
                if g_str in resumo_grupos and len(resumo_grupos[g_str]["areas"]) > 0:
                    avg_a = int(np.mean(resumo_grupos[g_str]["areas"]))
                    avg_r = (
                        int(np.mean(resumo_grupos[g_str]["r"]))
                        if resumo_grupos[g_str]["r"]
                        else 0
                    )
                    avg_g = (
                        int(np.mean(resumo_grupos[g_str]["g"]))
                        if resumo_grupos[g_str]["g"]
                        else 0
                    )
                    avg_b = (
                        int(np.mean(resumo_grupos[g_str]["b"]))
                        if resumo_grupos[g_str]["b"]
                        else 0
                    )

                    cores_rgb = {
                        1: (0, 255, 0),
                        2: (0, 0, 255),
                        3: (255, 255, 0),
                        4: (255, 0, 255),
                        5: (255, 165, 0),
                        6: (0, 255, 255),
                        7: (255, 0, 0),
                        8: (255, 192, 203),
                        9: (255, 255, 255),
                        10: (128, 0, 128),
                    }
                    cor_rgb_html = cores_rgb.get(g_num, (255, 255, 255))

                    html_contour = (
                        f"rgb({cor_rgb_html[0]}, {cor_rgb_html[1]}, {cor_rgb_html[2]})"
                    )
                    html_main = f"rgb({avg_r}, {avg_g}, {avg_b})"

                    row_html = f"<div style='display: flex; align-items: center; gap: 10px; background-color: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 5px;'><div style='width: 16px; height: 16px; background-color: {html_contour}; border: 1px solid rgba(255,255,255,0.5); border-radius: 3px;' title='Cor do Contorno'></div><span style='font-size: 15px; font-weight: 600;'>{g_str}</span><span style='font-size: 14px; color: #ccc; margin-left: 5px;'>~{avg_a}px | Cor:</span><div style='width: 18px; height: 18px; background-color: {html_main}; border: 1px solid rgba(255,255,255,0.5); border-radius: 50%;' title='Cor Principal Média'></div></div>"
                    legend_html += row_html

            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

        dl_col1, dl_col2, dl_col3 = st.columns([1, 0.7, 1])
        with dl_col2:
            st.download_button(
                label="📥 Descarregar Imagem Renderizada",
                data=img_bytes_dl,
                file_name=f"processada_{nome_img}.png",
                mime="image/png",
                key=f"dl_{nome_img}_{idx}",
                use_container_width=True,
            )

        if not df_tabela.empty:
            df_para_mostrar = (
                df_tabela
                if not destacar_melhores
                else df_tabela[df_tabela["Feijao"].isin(feijoes_a_desenhar)]
            )

            st.markdown("**Tabela de Dados:**")
            colunas_cor = [
                c
                for c in df_para_mostrar.columns
                if c.startswith("Cor") and not c.endswith("%")
            ]
            df_estilizado = df_para_mostrar.style.map(
                aplicar_cor_fundo, subset=colunas_cor
            )
            st.dataframe(df_estilizado, width="stretch", hide_index=True)