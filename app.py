import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect
from skimage.feature import graycomatrix, graycoprops
import joblib

app = Flask(__name__)

upload_folder = "static/uploads"
report_file = "analysis.txt"
csv_file = "form_data.csv"
model_path = "dermascan_rf_model.joblib"
app.config['upload_folder'] = upload_folder

model = joblib.load(model_path)

def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # color histogram
        hist_b = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
        color_features = np.concatenate([hist_b, hist_g, hist_r])
        color_features = color_features / np.sum(color_features)

        # texture (GLCM)
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        texture_features = [contrast, dissimilarity, homogeneity, energy, correlation]

        # edge feature
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (256*256)

        features = np.concatenate([color_features, texture_features, [edge_density]])
        return features
    except Exception as e:
        print("Error processing:", img_path, e)
        return None

def reset_uploads_folder():
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    else:
        for f in os.listdir(upload_folder):
            os.remove(os.path.join(upload_folder, f))

def personalized_analysis(data):
    lines = []
    # for age
    if int(data['age']):
        if int(data['age']) < 20:
            lines.append("- Your age indicates that hormonal changes are common, so minor skin issues are normal.")
        elif 20 <= int(data['age']) <40:
            lines.append("- Your are in an active age group where lifestyle and stress have a major impact on skin health.")
        else:
            lines.append("- At this age, natural collagen production slows down. Use hydrating and anti-aging products regularly.")
    # for gender
    if data['gender']:
        if data['gender'] == 'female':
            lines.append("- Women usually have more hormonal fluctuations affecting skin, regular care helps maintain balance.")
        elif data['gender'] == 'male':
            lines.append("- Men's skin is thicker and less sensitive, but proper cleansing and hydration are still essential.")
        else:
            lines.append("- Everyone's skin needs care - focus on balance and hydration.")
    # for diet
    if int(data['diet_score']):
        if int(data['diet_score']) < 3:
            lines.append("- Your diet seems poor. Include more fruits, vegetables, and water-rich foods for better skin.")
        elif 3 <= int(data['diet_score']) <= 4:
            lines.append("- Your diet is average, but you can improve by reducing oily and processed foods.")
        else:
            lines.append("- Your diet is Excellent! Keep maintaining a balanced diet.")
    # for stress
    if int(data['stress']):
        if int(data['stress']) < 3:
            lines.append("- Your stress levels seem manageable - great job for that!")
        elif 3 <= int(data['stress']) <= 4:
            lines.append("- Moderate stress can sometimes trigger acne or dullness. Try relaxation or mindfulness activities.")
        else:
            lines.append("- High stress affects hormones and skin barrier - focus on relaxation and healthy sleep.")
    # for water intake
    if int(data['water_intake']):
        if int(data['water_intake']) < 3:
            lines.append("- Your water intake is very low. Drink at least 6-8 glasses per day to keep skin hydration.")
        elif 3 <= int(data['water_intake']) <= 5:
            lines.append("- Try to drink more water - hydration improves skin elasticity and glow.")
        else:
            lines.append("- Your water intake is great! Keep your hydration level consistent.")

    return "\n".join(lines)
def write_report(results, form_data, form_conditions, recommendations):
    with open("analysis.txt", "w", encoding="utf-8") as f:
        f.write("HERE'S YOUR SKIN ANALYSIS:\n")
        f.write("----------------------------------\n\n")

        f.write("PERSONALIZED ANALYSIS:\n")
        f.write("-----------------------------------\n")
        f.write(personalized_analysis(form_data))
        f.write("\n\n")

        f.write("FORM-BASED OBSERVATIONS:\n")
        f.write("-----------------------------------\n")
        for c in form_conditions:
            f.write(f"- You reported: {c}\n")
        f.write("\n\n")

        f.write("IMAGE-BASED OBSERVATIONS:\n")
        f.write("------------------------------------\n")
        for idx, item in enumerate(results, start=1):
            f.write(f"Image {idx}: {item['class']}\n")
        f.write("\n\n")

        f.write("RECOMMENDED SKINCARE:\n")
        f.write("-----------------------------------\n")
        for r in recommendations:
            f.write(f"- {r}\n")
        f.write("\n\n")

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/submit", methods=["POST"])
def submit_form():
    data = {
        'age': request.form['age'],
        'gender': request.form['gender'],
        'skin_type': request.form['skin_type'],
        'sensitive': request.form['sensitive'],
        'acne': request.form['acne'],
        'pigmentation': request.form['pigmentation'],
        'wrinkles': request.form['wrinkles'],
        'dark_spots': request.form['dark_spots'],
        'whiteheads': request.form['whiteheads'],
        'blackheads': request.form['blackheads'],
        'oiliness': request.form['oiliness'],
        'dryness': request.form['dryness'],
        'redness': request.form['redness'],
        'itching': request.form['itching'],
        'diet_score': request.form['diet_score'],
        'stress': request.form['stress'],
        'water_intake': request.form['water_intake']
    }
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=data.keys()).to_csv(csv_file, index=False)
    pd.DataFrame([data]).to_csv(csv_file, mode="a", header=False, index=False)

    app.last_form_data = data
    return render_template("upload.html")

@app.route("/upload", methods = ["POST"])
def upload_images():
    reset_uploads_folder()
    files = request.files.getlist("images")
    if len(files) == 0:
        return "Please upload at least one image."
    if len(files) > 5:
        return "Maximum 5 images allowed."
    for i, file in enumerate(files):
        file.save(os.path.join(upload_folder, f"user_{i+1}.jpg"))
    return redirect("/result")
def skincare_recommendation(conditions, form_data):
    rec = []

    if "oily skin" in conditions:
        rec.append( "Your skin appears oily, which means your pores may produce excess sebum throughout the day. "
            "Use a gentle gel-based facewash twice daily to remove oil without damaging your skin barrier. "
            "Incorporate salicylic acid (1–2%) three to four times a week to control oil and prevent clogged pores. "
            "Choose oil-free, non-comedogenic moisturizers with niacinamide or hyaluronic acid to maintain balance. "
            "Avoid heavy creams or thick sunscreens and switch to a lightweight gel-based SPF 50. "
            "Using a clay mask twice a week can help clear deep impurities. "
            "Try not to overwash your face as it increases oil production, and use blotting sheets to control shine during the day.")
    if "dry skin" in conditions:
        rec.append("Your skin shows signs of dryness, which means your moisture barrier may need extra support. "
            "Use a hydrating, cream-based non-foaming cleanser to maintain hydration. "
            "Apply a thick moisturizer with ceramides, squalane, glycerin, or shea butter twice daily. "
            "Use a hydrating serum with hyaluronic acid on damp skin to improve moisture retention. "
            "Avoid using hot water on your face since it increases dryness and weakens the barrier. "
            "Daily sunscreen is essential because dry skin develops fine lines faster. "
            "Stay away from harsh scrubs and alcohol-based toners, and consider using an overnight sleeping mask to repair the skin barrier.")
    if "acne" in conditions:
        rec.append("Your skin assessment indicates acne-prone skin, which requires controlled and gentle care. "
            "Use a salicylic acid (1–2%) facewash once daily to unclog pores and treat active breakouts. "
            "Apply benzoyl peroxide (2.5%) gel on active pimples as a spot treatment. "
            "Include a niacinamide serum to reduce inflammation and balance oil production. "
            "Avoid squeezing or picking pimples as it leads to scarring and dark spots. "
            "Use a lightweight gel moisturizer to prevent dryness caused by active ingredients. "
            "Daily sunscreen is crucial to stop acne marks from darkening, and reducing sugar and dairy intake also helps control acne.")
    if "clogged pores" in conditions:
        rec.append("Your skin shows signs of clogged pores, whiteheads, or blackheads, which means your skin needs consistent exfoliation. "
            "Salicylic acid (BHA) is highly effective as it cleans deep inside the pores, so include it regularly in your routine. "
            "Use a clay mask twice a week to absorb oil and draw out impurities. "
            "Niacinamide serum can also help reduce oiliness and make pores appear smaller. "
            "Avoid using harsh physical scrubs, and instead exfoliate gently once or twice a week. "
            "Use lightweight, non-comedogenic products to avoid further pore congestion. "
            "Retinol at night can also prevent whiteheads and improve skin turnover.")
    if "pigmentation" in conditions:
        rec.append( "Your skin shows pigmentation concerns, which can be improved with consistent care. "
            "Use a Vitamin C serum every morning to brighten your complexion and fade uneven tone. "
            "At night, use Kojic Acid or Alpha Arbutin for targeted pigmentation correction. "
            "Daily sunscreen with SPF 50 is essential because sun exposure makes pigmentation worse. "
            "Avoid fragrance-heavy skincare products that may darken sensitive areas. "
            "Niacinamide can help even out skin tone, and gentle AHA exfoliation once a week promotes brighter, smoother skin. "
            "Avoid harsh scrubs as they can irritate the skin and deepen pigmentation.")
    if "wrinkles" in conditions:
        rec.append("Your skin shows early signs of fine lines or wrinkles, which can be improved with the right care. "
            "Start by using a low-strength Retinol (0.2–0.5%) twice a week to boost collagen production. "
            "A thick, deeply hydrating moisturizer will keep your skin elastic and prevent further wrinkle formation. "
            "Always use sunscreen every morning, as most signs of aging are caused by sun exposure. "
            "Adding a peptide serum at night can further support skin repair. "
            "Hyaluronic acid can keep your skin plump and hydrated. "
            "Avoid pulling or rubbing the skin and try to limit sugar intake, which accelerates skin aging.")
    if "dark spots" in conditions:
        rec.append("Your analysis shows dark spots, which can fade with the right brightening routine. "
            "Use Niacinamide (5–10%) daily to lighten spots and improve overall glow. "
            "Vitamin C serum every morning helps brighten the skin and reduce discoloration. "
            "Retinol at night accelerates cell turnover and helps fade long-term marks. "
            "Daily sunscreen is extremely important because sun exposure darkens existing spots. "
            "Avoid touching or picking pimples, as it leads to new dark marks. "
            "Ingredients like kojic acid, licorice extract, or AHA exfoliants once a week can help treat stubborn areas.")
    if "milia" in conditions:
        rec.append("Your image indicates signs of milia. These tiny white bumps form when keratin gets trapped under the skin. "
            "Use gentle exfoliation with AHA (lactic acid) 1–2 times weekly. Avoid heavy creams. "
            "Use non-comedogenic products only. Retinol at night also helps in reducing milia.")
    if "redness" in conditions:
        rec.append("Your skin shows redness which usually indicates irritation or sensitivity. "
                   "Use a fragrance-free gentle cleanser and moisturize with ceramides or centella calming cream. "
                   "Avoid scrubs, hot water, and strong actives; always use SPF to prevent flare-ups.")

    return rec


@app.route("/result")
def result():
    form_data = app.last_form_data
    images = os.listdir(upload_folder)
    feats = []
    names = []

    for img in images:
        path = os.path.join(upload_folder, img)
        ft = extract_features(path)
        if ft is not None:
            feats.append(ft)
            names.append(img)

    X = pd.DataFrame(feats)
    probs = model.predict_proba(X)

    results = []
    for name, pr in zip(names, probs):
        idx = np.argmax(pr)
        results.append({
            "image": name,
            "class": model.classes_[idx],
            "prob": round(pr[idx]* 100, 2)
        })
    form_conditions = []

    if form_data.get('acne') == 'yes':
        form_conditions.append("acne")

    if form_data.get('oiliness') == 'yes':
        form_conditions.append("oily skin")

    if form_data.get('whiteheads') == 'yes' or form_data.get('blackheads') == 'yes':
        form_conditions.append("clogged pores")

    if form_data.get('dryness') == 'yes':
        form_conditions.append("dry skin")

    if form_data.get('pigmentation') == 'yes':
        form_conditions.append("pigmentation")

    if form_data.get('wrinkles') == 'yes':
        form_conditions.append("wrinkles")

    if form_data.get('redness') == 'yes':
        form_conditions.append("redness")

    if form_data.get('dark_spots') == 'yes':
        form_conditions.append("dark spots")

    recommendations = skincare_recommendation(form_conditions, form_data)
    write_report(results, form_data, form_conditions, recommendations)

    with open(report_file, "r") as f:
        analysis_text = f.read()
    return render_template("result.html", analysis_text = analysis_text)

if __name__ == "__main__":
    app.run(debug=True)