from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask import session
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from utils import load_and_preprocess_image
from models import load_model, predict_disease, detector
from methods import method1
from image_metrics import ImageMetrics
from feature_visualization import FeatureVisualization
import base64
import io
from PIL import Image
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from database import db, User
from forms import LoginForm, RegistrationForm

import logging
import traceback

# simple file logger for upload debugging
logger = logging.getLogger('predict_debug')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('predict_debug.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tomato-disease-detection-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def image_to_base64(image_array):
    """Convert numpy array image to base64 string."""
    # Convert to PIL Image
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)

    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/')
def landing():
    """Public landing page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Home page with upload form."""
    return render_template('index.html')

@app.route('/analysis_report')
@login_required
def analysis_report():
    """Display algorithm analysis report."""
    from models import DISEASE_CLASSES
    selected = session.get('selected_method', 'method1')
    report_context = {}
    # attempt to import a method-specific report provider
    try:
        if selected == 'method1':
            from methods.method1 import get_report_context
            report_context = get_report_context()
        elif selected == 'method2':
            from methods.method2 import get_report_context
            report_context = get_report_context()
        elif selected == 'method3':
            from methods.method3 import get_report_context
            report_context = get_report_context()
        # future: support method2, method3 similarly
    except Exception:
        report_context = {}

    return render_template('analysis_report.html', disease_classes=DISEASE_CLASSES, method_selected=selected, method_report=report_context)


@app.route('/select_method', methods=['POST'])
@login_required
def select_method():
    """AJAX endpoint to set the selected method in session."""
    try:
        data = request.get_json(force=True)
        method = data.get('method')
        if method:
            session['selected_method'] = method
            return jsonify({'status': 'ok', 'selected': method})
    except Exception:
        pass
    return jsonify({'status': 'error'}), 400


@app.route('/about')
@login_required
def about():
    """About page with algorithm comparison charts."""
    return render_template('about.html')


@app.route('/disease_types')
@login_required
def disease_types():
    """Display all disease types that can be detected by the system."""
    from models import DISEASE_CLASSES
    return render_template('disease_types.html', disease_classes=DISEASE_CLASSES)


@app.route('/about/data')
@login_required
def about_data():
    """Return comparison metrics for the About page charts.

    Currently returns example values. Replace with real metrics retrieval
    (e.g., load from training results JSON, database, or model metadata).
    """
    # Support returning per-model comparison across classifiers (RandomForest, XGBoost, LightGBM)
    model = request.args.get('model', 'glcm')

    methods = ['RandomForest', 'XGBoost', 'LightGBM']

    # Placeholder example values; replace with real metrics retrieval
    if model == 'glcm':
        accuracy = [0.92, 0.90, 0.91]
        f1 = [0.915, 0.89, 0.905]
        precision = [0.92, 0.88, 0.90]
        recall = [0.91, 0.90, 0.91]
        auc = [0.96, 0.95, 0.955]
        inference_ms = [50, 55, 52]
        train_time = [60, 120, 80]
    elif model == 'mobilenet':
        accuracy = [0.94, 0.945, 0.943]
        f1 = [0.939, 0.94, 0.941]
        precision = [0.94, 0.945, 0.942]
        recall = [0.94, 0.935, 0.94]
        auc = [0.97, 0.971, 0.969]
        inference_ms = [30, 35, 32]
        train_time = [600, 650, 620]
    elif model == 'fusion':
        accuracy = [0.95, 0.948, 0.949]
        f1 = [0.947, 0.945, 0.946]
        precision = [0.95, 0.947, 0.948]
        recall = [0.945, 0.943, 0.945]
        auc = [0.975, 0.974, 0.976]
        inference_ms = [70, 75, 72]
        train_time = [700, 720, 710]
    else:
        # default safe fallback
        accuracy = [0.93, 0.925, 0.927]
        f1 = [0.925, 0.92, 0.923]
        precision = [0.93, 0.92, 0.925]
        recall = [0.92, 0.915, 0.92]
        auc = [0.965, 0.962, 0.964]
        inference_ms = [60, 65, 62]
        train_time = [200, 300, 250]

    metrics = {
        'methods': methods,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'inference_ms': inference_ms,
        'train_time': train_time
    }
    return jsonify(metrics)

@app.route('/image_metrics', methods=['GET', 'POST'])
@login_required
def image_metrics_page():
    """Display image metrics analysis page."""
    metrics = None
    original_b64 = None
    processed_b64 = None
    grayscale_b64 = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('image_metrics_page'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('image_metrics_page'))
        
        if file and allowed_file(file.filename):
            try:
                # Read file bytes
                file_bytes = file.read()
                if not file_bytes:
                    raise ValueError("Uploaded file is empty.")
                
                # Load original image
                original_pil = Image.open(io.BytesIO(file_bytes))
                original_image = np.array(original_pil)
                
                # Convert to RGB if grayscale
                if len(original_image.shape) == 2:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                elif original_image.shape[2] == 4:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
                
                # Preprocess image
                processed_image, grayscale_image, clahe_image = load_and_preprocess_image(io.BytesIO(file_bytes))
                
                # Convert processed image to uint8 for metrics calculation
                if processed_image.max() <= 1.0:
                    processed_for_metrics = (processed_image * 255).astype(np.uint8)
                else:
                    processed_for_metrics = processed_image.astype(np.uint8)
                
                # Ensure images are uint8
                if grayscale_image.max() <= 1.0:
                    grayscale_uint8 = (grayscale_image * 255).astype(np.uint8)
                else:
                    grayscale_uint8 = grayscale_image.astype(np.uint8)
                
                if clahe_image.max() <= 1.0:
                    clahe_uint8 = (clahe_image * 255).astype(np.uint8)
                else:
                    clahe_uint8 = clahe_image.astype(np.uint8)
                
                # Convert all images to grayscale for PSNR calculations
                original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
                grayscale_gray = cv2.cvtColor(grayscale_uint8, cv2.COLOR_RGB2GRAY)
                clahe_gray = cv2.cvtColor(clahe_uint8, cv2.COLOR_RGB2GRAY)
                
                # Calculate all metrics
                metrics = ImageMetrics.calculate_all_metrics(original_image, clahe_image)
                
                # Calculate PSNR between Original and Grayscale
                psnr_original_grayscale = ImageMetrics.calculate_psnr(original_gray, grayscale_gray)
                
                # Calculate PSNR between Grayscale and Processed (CLAHE)
                psnr_grayscale_clahe = ImageMetrics.calculate_psnr(grayscale_gray, clahe_gray)
                
                # Add PSNR comparisons to metrics
                metrics['psnr_original_grayscale'] = psnr_original_grayscale
                metrics['psnr_grayscale_clahe'] = psnr_grayscale_clahe
                
                # Convert images to base64
                original_b64 = image_to_base64(original_image)
                processed_b64 = image_to_base64(clahe_image)
                grayscale_b64 = image_to_base64(grayscale_image)
                
                flash('Metrics calculated successfully!', 'success')
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(url_for('image_metrics_page'))
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF).', 'danger')
            return redirect(url_for('image_metrics_page'))
    
    return render_template('image_metrics.html', metrics=metrics, original_image=original_b64, processed_image=processed_b64, grayscale_image=grayscale_b64)

@app.route('/feature_maps', methods=['GET', 'POST'])
@login_required
def feature_maps_page():
    """Display feature maps and histogram analysis page."""
    feature_maps_viz = None
    histogram_viz = None
    histogram_rgb_viz = None
    comparison_viz = None
    histogram_stats = None
    original_b64 = None
    processed_b64 = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('feature_maps_page'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('feature_maps_page'))
        
        if file and allowed_file(file.filename):
            try:
                # Read file bytes
                file_bytes = file.read()
                if not file_bytes:
                    raise ValueError("Uploaded file is empty.")
                
                # Load original image
                original_pil = Image.open(io.BytesIO(file_bytes))
                original_image = np.array(original_pil)
                
                # Convert to RGB if grayscale
                if len(original_image.shape) == 2:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                elif original_image.shape[2] == 4:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
                
                # Normalize for display
                if original_image.max() > 1.0:
                    original_image_norm = original_image.astype(np.float32) / 255.0
                else:
                    original_image_norm = original_image.astype(np.float32)
                
                # Preprocess image
                processed_image, _, clahe_image = load_and_preprocess_image(io.BytesIO(file_bytes))
                
                if clahe_image.max() > 1.0:
                    clahe_image_norm = clahe_image.astype(np.float32) / 255.0
                else:
                    clahe_image_norm = clahe_image.astype(np.float32)
                
                # Load model for feature extraction
                load_model()
                
                # Extract feature maps using the detector's feature extractor (or simple features if CNN not available)
                try:
                    feature_maps = FeatureVisualization.extract_feature_maps(detector.feature_extractor, processed_image, layer_index=-2)
                    
                    # Generate visualizations
                    feature_maps_viz = FeatureVisualization.visualize_feature_maps(feature_maps, 
                                                                               title="CNN Feature Maps Visualization",
                                                                               num_maps=12)
                except Exception as fe:
                    flash(f'Could not extract feature maps: {str(fe)}', 'warning')
                    feature_maps_viz = None
                
                histogram_viz = FeatureVisualization.visualize_histogram(clahe_image_norm,
                                                                        title="Grayscale Histogram Analysis")
                
                histogram_rgb_viz = FeatureVisualization.visualize_histogram_rgb(original_image_norm,
                                                                                 title="RGB Channel Histogram")
                
                comparison_viz = FeatureVisualization.visualize_image_comparison(original_image_norm,
                                                                                 clahe_image_norm,
                                                                                 title="Image Comparison: Original vs Processed")
                
                # Generate histogram statistics
                histogram_stats = FeatureVisualization.generate_histogram_statistics(clahe_image_norm)
                
                # Convert images to base64
                original_b64 = image_to_base64(original_image)
                processed_b64 = image_to_base64(clahe_image)
                
                flash('Feature maps and histograms generated successfully!', 'success')
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(url_for('feature_maps_page'))
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF).', 'danger')
            return redirect(url_for('feature_maps_page'))
    
    return render_template('feature_maps.html', 
                         feature_maps=feature_maps_viz,
                         histogram=histogram_viz,
                         histogram_rgb=histogram_rgb_viz,
                         comparison=comparison_viz,
                         histogram_stats=histogram_stats,
                         original_image=original_b64,
                         processed_image=processed_b64)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and prediction."""
    logger.debug('Entered predict route')
    if 'file' not in request.files:
        logger.warning('No file in request.files')
        flash('No file part')
        return redirect(url_for('dashboard'))

    file = request.files['file']

    if file.filename == '':
        logger.warning('Empty filename in uploaded file')
        flash('No selected file')
        return redirect(url_for('dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        try:
            logger.debug(f'Received file: {filename}')
            # Read file bytes once so we can reuse the stream safely
            file_bytes = file.read()
            logger.debug(f'Read {len(file_bytes) if file_bytes else 0} bytes')
            if not file_bytes:
                logger.error('Uploaded file bytes empty')
                raise ValueError("Uploaded file is empty.")

            # Load original image for display and metrics
            original_pil_image = Image.open(io.BytesIO(file_bytes))
            original_image_array = np.array(original_pil_image)

            # Preprocess the uploaded image
            processed_image, grayscale_image, clahe_image = load_and_preprocess_image(io.BytesIO(file_bytes))

            # Determine selected method (default to method1)
            selected_method = request.form.get('method', 'method1')
            logger.debug(f'selected_method: {selected_method}')

            if selected_method == 'method1':
                # Use method1 pipeline which now integrates the image_pipeline features
                try:
                    method_output = method1.run_method1(file_bytes)
                except Exception as e:
                    logger.error('method1.run_method1 raised exception: %s', str(e))
                    logger.error(traceback.format_exc())
                    raise

                results = method_output.get('results', {})

                # Convert images to base64 for display
                original_b64 = image_to_base64(original_image_array)
                grayscale_b64 = image_to_base64(method_output.get('grayscale', grayscale_image))
                clahe_b64 = image_to_base64(method_output.get('processed', clahe_image))

                # Prepare prediction data including feature vector
                feature_vector = method_output.get('feature_vector')
                feature_names = method_output.get('feature_names')

                prediction_data = {
                    'selected_method': selected_method,
                    'predicted_class': results.get('predicted_class', ''),
                    'confidence': results.get('confidence', 0.0) * 100,
                    'original_image': original_b64,
                    'grayscale_image': grayscale_b64,
                    'clahe_image': clahe_b64,
                    'processed_image': clahe_b64,
                    'probabilities': results.get('probabilities'),
                    'feature_vector': feature_vector.tolist() if hasattr(feature_vector, 'tolist') else feature_vector,
                    'feature_names': feature_names,
                }

                logger.debug('Rendering result template (method1)')
                return render_template('result.html', **prediction_data)
            elif selected_method == 'method2':
                # Use method2 pipeline
                try:
                    from methods import method2
                    method_output = method2.run_method2(file_bytes)
                except Exception as e:
                    logger.error('method2.run_method2 raised exception: %s', str(e))
                    logger.error(traceback.format_exc())
                    raise

                results = method_output.get('results', {})

                # Convert images to base64 for display
                original_b64 = image_to_base64(original_image_array)
                grayscale_b64 = image_to_base64(method_output.get('grayscale', grayscale_image))
                clahe_b64 = image_to_base64(method_output.get('processed', clahe_image))

                # Prepare feature names: use glcm names if available, then deep feature indices
                glcm_names = method_output.get('glcm_names') or []
                deep_vec = method_output.get('deep_vector')
                deep_len = int(np.prod(deep_vec.shape)) if deep_vec is not None else 0
                deep_names = [f'deep_{i}' for i in range(deep_len)]
                feature_names = list(glcm_names) + deep_names

                combined_vec = method_output.get('combined_vector')

                prediction_data = {
                    'selected_method': selected_method,
                    'predicted_class': str(results.get('predicted_class', '')),
                    'confidence': float(results.get('confidence', 0.0)) * 100,
                    'original_image': original_b64,
                    'grayscale_image': grayscale_b64,
                    'clahe_image': clahe_b64,
                    'processed_image': clahe_b64,
                    'probabilities': results.get('probabilities', {}),
                    'feature_vector': combined_vec.tolist() if hasattr(combined_vec, 'tolist') else combined_vec,
                    'feature_names': feature_names,
                }

                logger.debug('Rendering result template (method2)')
                return render_template('result.html', **prediction_data)
            elif selected_method == 'method3':
                # Use method3 pipeline
                try:
                    from methods import method3
                    method_output = method3.run_method3(file_bytes)
                except Exception as e:
                    logger.error('method3.run_method3 raised exception: %s', str(e))
                    logger.error(traceback.format_exc())
                    raise

                results = method_output.get('results', {})

                # Convert images to base64 for display
                original_b64 = image_to_base64(original_image_array)
                grayscale_b64 = image_to_base64(method_output.get('grayscale', grayscale_image))
                clahe_b64 = image_to_base64(method_output.get('processed', clahe_image))

                # Prepare feature names: use glcm names if available, then deep feature indices
                glcm_names = method_output.get('glcm_names') or []
                deep_vec = method_output.get('deep_vector')
                deep_len = int(np.prod(deep_vec.shape)) if deep_vec is not None else 0
                deep_names = [f'deep_{i}' for i in range(deep_len)]
                feature_names = list(glcm_names) + deep_names

                combined_vec = method_output.get('combined_vector')

                prediction_data = {
                    'selected_method': selected_method,
                    'predicted_class': str(results.get('predicted_class', '')).replace('Tomato___', '').replace('_', ' '),
                    'confidence': float(results.get('confidence', 0.0)) * 100,
                    'original_image': original_b64,
                    'grayscale_image': grayscale_b64,
                    'clahe_image': clahe_b64,
                    'processed_image': clahe_b64,
                    'probabilities': results.get('probabilities', {}),
                    'feature_vector': combined_vec.tolist() if hasattr(combined_vec, 'tolist') else combined_vec,
                    'feature_names': feature_names,
                    'classifier_used': method_output.get('classifier_used')
                }

                logger.debug('Rendering result template (method3)')
                return render_template('result.html', **prediction_data)
            else:
                logger.debug('Using default CNN prediction path')
                # Default behavior: use existing CNN-based prediction pipeline
                results = predict_disease(processed_image)

                # Convert images to base64 for display
                original_b64 = image_to_base64(original_image_array)
                grayscale_b64 = image_to_base64(grayscale_image)
                clahe_b64 = image_to_base64(clahe_image)

                prediction_data = {
                    'selected_method': selected_method,
                    'predicted_class': results['predicted_class'],
                    'confidence': results['confidence'] * 100,
                    'original_image': original_b64,
                    'grayscale_image': grayscale_b64,
                    'clahe_image': clahe_b64,
                    'processed_image': clahe_b64,  # Use CLAHE as processed image
                    'probabilities': results['probabilities']
                }

                logger.debug('Rendering result template (default)')
                return render_template('result.html', **prediction_data)

        except Exception as e:
            logger.error('Exception in predict: %s', str(e))
            logger.error(traceback.format_exc())
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('dashboard'))

    else:
        flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF).')
        return redirect(url_for('dashboard'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Please upload an image smaller than 16MB.')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

    # Load the model on startup
    print("Loading disease detection model...")
    load_model()
    print("Model loaded successfully!")

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
