from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load the dataset
dataset = pd.read_csv('Crop_and_fertilizer dataset.csv')

# Prepare the encoder and model if already saved
try:
    model_pipeline = joblib.load('model_pipeline.pkl')
    print("Model pipeline loaded from disk.")
except FileNotFoundError:
    # Define preprocessing for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['Nitrogen', 'Phosphorus', 'Potassium', 'pH']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Soil_color'])
        ])

    # Create and train the pipeline with the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X = dataset[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Soil_color']]
    y = dataset['Crop']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model pipeline
    model_pipeline.fit(X_train, y_train)
    
    # Save the model pipeline to disk
    joblib.dump(model_pipeline, 'model_pipeline.pkl')
    print("Model pipeline saved to disk.")


# Simple routes to serve the HTML pages from templates/
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


@app.route('/services', methods=['GET'])
def services():
    return render_template('services.html')


@app.route('/fertilizer-plan', methods=['GET'])
def fertilizer_plan():
    # template filename contains a space
    return render_template('fertilizer plan.html')


@app.route('/learn-more', methods=['GET'])
def learn_more():
    return render_template('Learn More.html')


@app.route('/power-of-data', methods=['GET'])
def power_of_data():
    return render_template('power of data.html')


# Serve the Recom input page (left in Recom/ folder)
@app.route('/recom/input', methods=['GET'])
def recom_input():
    # render the Recom input page from templates/Recom/input.html so Jinja url_for works
    try:
        return render_template('Recom/input.html')
    except Exception:
        # fallback to sending raw file if template not found
        try:
            return send_from_directory('Recom', 'input.html')
        except Exception:
            abort(404)

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        data = request.get_json()

        # Validate input data
        if not all(k in data for k in ['soil_color', 'nitrogen', 'phosphorus', 'potassium', 'pH']):
            return jsonify({'error': 'Missing required data'}), 400

        soil_color = data['soil_color']
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        pH = float(data['pH'])

        input_data = pd.DataFrame(
            [[nitrogen, phosphorus, potassium, pH, soil_color]],
            columns=['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Soil_color']
        )

        # Make predictions using the pipeline
        predicted_crop = model_pipeline.predict(input_data)

        # Find the fertilizer associated with the recommended crop
        recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop[0]]['Fertilizer'].values[0]

        cropdetails={
            'Sugarcane':{
                'desc':'Sugarcane is a tropical and subtropical crop primarily grown for its high sucrose content, used in sugar production. It thrives in warm, sunny climates with abundant rainfall or irrigation and well-drained, fertile soils. The crop requires about 10–12 months to mature and is typically harvested by cutting the stalks close to the ground. Major producers include Brazil, India, Thailand, and China. Beyond sugar, sugarcane is also used for ethanol production, animal feed, and as raw material in various industries.',
                
            },
            'Sorghum':{
                'desc':'Sorghum is a drought-tolerant cereal crop grown for food, forage, and biofuel production, thriving in arid and semi-arid regions. It is gluten-free, rich in nutrients, and a staple food in many parts of Africa and Asia. Sorghum is also used in livestock feed, ethanol production, and traditional beverages.',
            
            },
            'Cotton':{
                'desc':'Cotton is a fiber crop grown in warm climates, primarily for its soft, fluffy fibers used in textile production. It thrives in well-drained, fertile soils with a long frost-free growing season and moderate rainfall. Cotton plants produce bolls containing seeds surrounded by fibers, which are harvested mechanically or by hand. Major producers include India, China, the United States, and Pakistan. In addition to textiles, cottonseed is used for oil production and as livestock feed.',
                
            },
            'Paddy': {
                'desc':'The paddy crop, or rice, is a staple food crop grown primarily in flooded fields called paddy fields. It thrives in warm, humid climates and requires ample water and fertile soil. Paddy is cultivated in regions with monsoon rains or artificial irrigation systems. It is a major cereal crop in countries like India, China, and Southeast Asia. The harvested grains are processed into edible rice, a vital food source for billions worldwide.',
                
            },
            'Wheat':{
                'desc':'Wheat is a versatile cereal grain and a staple food globally, rich in carbohydrates and nutrients. It is primarily grown in temperate regions and used to produce flour for bread, pasta, and other baked goods. Wheat has numerous varieties, including hard and soft types, suited for different culinary uses. Its cultivation dates back thousands of years, making it a cornerstone of human agriculture and diet.',
               
            },
            'Maize':{
                'desc':'Maize, also known as corn, is a versatile cereal crop widely grown for food, feed, and industrial uses. It thrives in warm climates with well-drained, fertile soils and requires moderate rainfall or irrigation. The plant produces ears with kernels, which are used for human consumption, animal feed, and biofuel production. Major producers include the United States, China, Brazil, and India. Maize is a staple food in many cultures and is also a key ingredient in processed foods and beverages.',
                
            },
            'Soybean':{
                'desc':'Soybean is a versatile legume known for its high protein content and numerous health benefits. It is widely used in food products like tofu, soy milk, and soy sauce, as well as in animal feed and industrial applications. Rich in nutrients, soybeans also play a significant role in promoting heart health and reducing cholesterol levels.',
                
            },
            'Ginger':{
                'desc':'Ginger is a flowering plant whose rhizome, commonly called ginger root, is widely used as a spice and traditional medicine. Known for its pungent and aromatic flavor, it is a key ingredient in cuisines worldwide. Rich in bioactive compounds like gingerol, it is valued for its anti-inflammatory and antioxidant properties. Ginger is used to alleviate nausea, improve digestion, and boost immunity. Cultivated mainly in tropical and subtropical regions, it has been a vital part of herbal remedies for centuries.',
                
            },
            'Grapes':{
                'desc':'Grapes are small, sweet fruits that grow in clusters on vines and are eaten fresh or used to make wine, juice, and raisins. They come in various colors and are rich in vitamins and antioxidants. Thriving in temperate climates, grapes hold global culinary and economic importance.',
                
            }
        }
        crop_info = cropdetails.get(predicted_crop[0], {'desc': 'No description available.'})

        response = {
            'recommended_crop': predicted_crop[0],
            'recommended_fertilizer': recommended_fertilizer,
            'crop_description':crop_info['desc'],
            
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

