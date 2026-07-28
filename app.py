import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import streamlit as st
import base64
from ultralytics import FastSAM

# Configuração da página do Streamlit
st.set_page_config(page_title="Separador de Feijões com IA", layout="wide")


# --- Função OTIMIZADA para processar cada feijão ---
# Agora recebe apenas os pixeis já extraídos, poupando memória no Multithreading
def processar_feijao(bean_pixels, cnt, N_CORES, PIX_ANALISAR):
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

    lum = 0.299 * colors[:, 2] + 0.587 * colors[:, 1] + 0.114 * colors[:, 0]
    lum_idx = np.argsort(-lum)
    colors = colors[lum_idx]
    percents = percents[lum_idx]

    area = cv2.contourArea(cnt)
    row = {"Contorno": cnt, "Area_px": int(area)}

    features = [float(area)]

    for j, (color, p) in enumerate(zip(colors, percents)):
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        row[f"Cor{j + 1}"] = hex_color
        row[f"Cor{j + 1}_%"] = round(float(p * 100), 2)
        features.extend([float(color[0]), float(color[1]), float(color[2]), float(p)])

    row["Features_Clustering"] = features
    return row


# --- Interface Streamlit ---
st.title("Separador de Feijões Avançado")

st.info(
    "**Dica de Processamento:**\n"
    "* O modelo de IA ignora o fundo automaticamente.\n"
    "* O agrupamento considera Área, Cores (RGB) e a sua distribuição (%).\n"
    "* Os 'melhores' feijões de um grupo são avaliados com base no seu tamanho (maior área)."
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
    col1, col2, col3 = st.columns(3)
    with col1:
        MIN_AREA = st.number_input("Área mínima (px)", min_value=1, value=500, step=50)
        MIN_CIRCULARIDADE = st.number_input(
            "Circularidade mínima", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )

    with col2:
        N_CORES = st.number_input(
            "Número de cores por feijão", min_value=1, value=2, step=1
        )
        PIX_ANALISAR = st.number_input(
            "Pixels a analisar (KMeans)", min_value=100, value=2000, step=100
        )

    with col3:
        NUM_GRUPOS = st.number_input(
            "Dividir feijões em X Grupos", min_value=1, max_value=10, value=2, step=1
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
                        # OTIMIZAÇÃO: Extrair Bounding Box e criar apenas uma micro-máscara
                        x, y, w, h = cv2.boundingRect(cnt)
                        mask_roi = np.zeros((h, w), np.uint8)

                        # Deslocar o contorno para desenhar no eixo zero do ROI
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

                # OTIMIZAÇÃO: Bitwise operations apenas no recorte minúsculo
                intersecao = cv2.bitwise_and(mask_roi, mapa_roi)
                area_intersecao = np.count_nonzero(intersecao)
                area_mascara = np.count_nonzero(mask_roi)

                if area_mascara > 0 and (area_intersecao / area_mascara) > 0.3:
                    continue

                # Atualizar o mapa global usando o recorte local
                mapa_ocupacao[y : y + h, x : x + w] = cv2.bitwise_or(mapa_roi, mask_roi)

                # OTIMIZAÇÃO: Extrair pixeis imediatamente usando o recorte da imagem
                img_roi = img[y : y + h, x : x + w]
                bean_pixels = img_roi[mask_roi == 255].astype(np.float32)

                tasks.append((bean_pixels, cand["cnt"], N_CORES, PIX_ANALISAR))

        # Execução das tarefas otimizada (passando matrizes muito mais leves)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            results = list(executor.map(lambda p: processar_feijao(*p), tasks))

        feijoes_data = [r for r in results if r is not None]

        if len(feijoes_data) >= NUM_GRUPOS > 1:
            features_list = [f["Features_Clustering"] for f in feijoes_data]
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_list)

            kmeans_grupos = KMeans(n_clusters=NUM_GRUPOS, random_state=42, n_init=10)
            labels_grupos = kmeans_grupos.fit_predict(features_scaled)

            for row, lbl in zip(feijoes_data, labels_grupos):
                row["Grupo"] = f"Tipo {lbl + 1}"
        else:
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

            hex_c = row["Cor1"].lstrip("#")
            if len(hex_c) == 6:
                r_val, g_val, b_val = tuple(
                    int(hex_c[k : k + 2], 16) for k in (0, 2, 4)
                )
                resumo_grupos[g_str]["r"].append(r_val)
                resumo_grupos[g_str]["g"].append(g_val)
                resumo_grupos[g_str]["b"].append(b_val)

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
                "resumo_grupos": resumo_grupos if NUM_GRUPOS > 1 else {},
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

    # Dicionário de 10 cores BGR globais (OpenCV)
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

        # --- CONTROLOS DE VISUALIZAÇÃO ---
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

        # --- FILTROS ---
        st.markdown("###### Filtro de Destaque")

        # Estrutura plana de colunas para layout de filtros
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

        # --- DESENHO DINÂMICO DOS CONTORNOS ---
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

        # --- PREPARAÇÃO DA IMAGEM PARA DESCARREGAR (COM LEGENDA NO FUNDO) ---
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

            for g_num in range(1, 11):
                g_str = f"Tipo {g_num}"
                if g_str in resumo_grupos and len(resumo_grupos[g_str]["areas"]) > 0:
                    avg_a = int(np.mean(resumo_grupos[g_str]["areas"]))
                    avg_r = int(np.mean(resumo_grupos[g_str]["r"]))
                    avg_g = int(np.mean(resumo_grupos[g_str]["g"]))
                    avg_b = int(np.mean(resumo_grupos[g_str]["b"]))

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

        # Codificar imagens
        _, img_encoded_display = cv2.imencode(".png", img_display)
        img_bytes_display = img_encoded_display.tobytes()

        _, img_encoded_dl = cv2.imencode(".png", img_download)
        img_bytes_dl = img_encoded_dl.tobytes()

        # Mostrar imagem na UI
        img_b64 = base64.b64encode(img_bytes_display).decode("utf-8")
        html_code = f"""
        <div style="max-width: 100%; max-height: 80vh; overflow: auto; border: 1px solid #444; border-radius: 5px; width: fit-content; margin: 0 auto; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_b64}" style="width: {largura_img}px; max-width: none; max-height: none; height: auto; display: block;">
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

        # --- LEGENDA CENTRADA ---
        if resumo_grupos:
            legend_html = "<div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 15px; border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; background-color: rgba(0,0,0,0.2); margin: 0 auto 30px auto; width: fit-content;'>"

            for g_num in range(1, 11):
                g_str = f"Tipo {g_num}"
                if g_str in resumo_grupos and len(resumo_grupos[g_str]["areas"]) > 0:
                    avg_a = int(np.mean(resumo_grupos[g_str]["areas"]))
                    avg_r = int(np.mean(resumo_grupos[g_str]["r"]))
                    avg_g = int(np.mean(resumo_grupos[g_str]["g"]))
                    avg_b = int(np.mean(resumo_grupos[g_str]["b"]))

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

        # Botão de descarregar numa nova linha, centrado abaixo da legenda
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
