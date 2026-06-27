import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import concurrent.futures
import base64
import unicodedata
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Função para processar cada feijão ---
def processar_feijao(mask, cnt, img, gray, N_CORES, PIX_ANALISAR):
    bean_pixels = img[mask==255].astype(np.float32)
    if len(bean_pixels) == 0:
        return None

    kmeans = MiniBatchKMeans(n_clusters=N_CORES, random_state=42, batch_size=1024, n_init=5)
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
    lum = 0.299*colors[:,2] + 0.587*colors[:,1] + 0.114*colors[:,0]
    lum_idx = np.argsort(-lum)
    colors = colors[lum_idx]
    percents = percents[lum_idx]

    area = cv2.contourArea(cnt)
    row = {"Contorno": cnt, "Area_px": int(area)}
    for j, (color, p) in enumerate(zip(colors, percents)):
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        row[f"Cor{j+1}"] = f'<div style="width:50px;height:20px;background-color:{hex_color};border:1px solid #000;"></div>'
        row[f"Cor{j+1}_Percent"] = round(float(p*100),2)
    return row

# --- Rota principal ---
@app.route("/")
def index():
    return render_template("index.html")

# --- Rota para processar imagens ---
@app.route("/processar", methods=["POST"])
def processar():
    imagens_entrada = request.files.getlist("imagens")
    MIN_AREA = int(request.form.get("min_area", 500))
    MIN_CIRCULARIDADE = float(request.form.get("min_circularidade", 0.5))
    N_CORES = int(request.form.get("n_cores", 2))
    PIX_ANALISAR = int(request.form.get("pix_analisar", 2000))

    if not os.path.exists("static"):
        os.makedirs("static")

    todas_tabelas = []
    imagens_resultado = []

    for i, imagem_entrada in enumerate(imagens_entrada):
        # Limpa nome do arquivo (sem acentos, sem espaços)
        nome_original = imagem_entrada.filename
        nome_limpo = unicodedata.normalize("NFKD", nome_original).encode("ascii", "ignore").decode("utf-8")
        nome_limpo = re.sub(r'[^a-zA-Z0-9_.-]', '_', nome_limpo)

        # Renomeia sequencialmente
        novo_nome = f"feijoes{i}.jpg"
        img_path = os.path.join(app.root_path, "static", novo_nome)

        # Garante que a pasta static está limpa
        if i == 0:  # só na primeira imagem
            for f in os.listdir(os.path.join(app.root_path, "static")):
                os.remove(os.path.join(app.root_path, "static", f))

        imagem_entrada.save(img_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERRO] Falha ao ler imagem: {img_path}")
            return jsonify({"erro": f"Não foi possível abrir a imagem {novo_nome}."}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Pré-processamento
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        blur = cv2.GaussianBlur(thresh, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img.copy()
        tasks = []

        # Criar tarefas, incluindo watershed para blobs
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue
            circularidade = 4 * np.pi * (area / (perimetro*perimetro))
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            if circularidade >= MIN_CIRCULARIDADE:
                tasks.append((mask, cnt, img, gray, N_CORES, PIX_ANALISAR))
            else:
                # Watershed
                x, y, w, h = cv2.boundingRect(cnt)
                roi = mask[y:y+h, x:x+w]
                roi_img = img[y:y+h, x:x+w]
                dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist, 0.4*dist.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(roi, sure_fg)
                num_markers, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown==255] = 0
                markers = cv2.watershed(roi_img.copy(), markers)
                for m in range(2, num_markers+1):
                    submask = np.zeros_like(roi)
                    submask[markers==m] = 255
                    sub_contours, _ = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for sc in sub_contours:
                        if cv2.contourArea(sc) < MIN_AREA:
                            continue
                        full_mask = np.zeros(gray.shape, np.uint8)
                        cv2.drawContours(full_mask, [sc + (x,y)], -1, 255, -1)
                        tasks.append((full_mask, sc + (x,y), img, gray, N_CORES, PIX_ANALISAR))

        # Processar em paralelo
        feijoes_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORES) as executor:
            results = list(executor.map(lambda p: processar_feijao(*p), tasks))
        feijoes_data = [r for r in results if r is not None]

        # Desenhar contornos e numerar feijões
        for i, row in enumerate(feijoes_data, start=1):
            cnt = row.pop("Contorno")
            cv2.drawContours(result, [cnt], -1, (0,255,0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(result, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            row["Feijao"] = i
            row["Foto"] = imagem_entrada.filename

        if feijoes_data:
            df = pd.DataFrame(feijoes_data)
            # Reordenar colunas: Foto → Feijão → resto
            cols = ["Foto", "Feijao"] + [c for c in df.columns if c not in ["Foto","Feijao"]]
            df = df[cols]
            todas_tabelas.append(df)

            # Converter imagem para base64
            _, buffer = cv2.imencode('.jpg', result)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            imagens_resultado.append({"data": f"data:image/jpeg;base64,{img_base64}", "nome": imagem_entrada.filename})

    if todas_tabelas:
        df_total = pd.concat(todas_tabelas, ignore_index=True)
        html_table = df_total.to_html(index=False, escape=False, table_id="feijoes_table")
    else:
        html_table = "<p>Nenhum feijão detectado.</p>"

    return jsonify({"tabela": html_table, "imagens": imagens_resultado})

if __name__ == "__main__":
    app.run(debug=True)
