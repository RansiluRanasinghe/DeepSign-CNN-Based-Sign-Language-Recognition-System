# 📡 API Documentation — DeepSign Inference Service

This project exposes a **RESTful inference API** that performs real-time **American Sign Language (ASL)** alphabet recognition using a trained CNN model.

The API is designed for production-style usage, enabling easy integration with frontend applications, mobile apps, or other backend services.

---

## 🔗 Base URL (Local)

```
http://localhost:8000
```

---

## 🩺 Health Check Endpoint

### `GET /health`

Verifies that the model is loaded and the service is ready to accept inference requests.

**Response:**

```json
{
  "status": "ready",
  "model": "deepsign_model.keras"
}
```

**Status Codes:**
- `200 OK` - Service is healthy and ready

---

## 🔮 Prediction Endpoint

### `POST /predict`

Performs sign language classification on an uploaded image.

### 📥 Input Requirements

| Property | Details |
|----------|---------|
| **Request Type** | `multipart/form-data` |
| **Field Name** | `file` |
| **File Type** | Image (`.png`, `.jpg`, `.jpeg`) |

**Image Constraints:**
- Single hand gesture
- Converted internally to grayscale
- Resized to 28 × 28
- Pixel values normalized to `[0, 1]`

### 📤 Output Format

The API returns:
- Predicted ASL letter
- Model confidence score
- Inference metadata

**Response:**

```json
{
  "prediction": "B",
  "confidence": 0.1095,
  "metadata": {
    "input_shape": [1, 28, 28, 1],
    "model_name": "DeeSign_V1"
  }
}
```

### 📌 Response Fields Explained

| Field | Description |
|-------|-------------|
| `prediction` | Predicted ASL alphabet letter |
| `confidence` | Softmax probability of the predicted class |
| `input_shape` | Final tensor shape fed into the CNN |
| `model_name` | Versioned model identifier |

**Status Codes:**
- `200 OK` - Successful prediction
- `400 Bad Request` - Invalid file format or missing file
- `500 Internal Server Error` - Model inference error

---

## 🧪 Example Requests

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_asl_image.png"
```

### Using Python (requests)

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("sample_asl_image.png", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => console.log(data));
```

---

## 📘 Interactive API Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: 👉 [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: 👉 [http://localhost:8000/redoc](http://localhost:8000/redoc)

These interfaces allow you to:
- ✅ Upload images directly
- ✅ Test predictions in real-time
- ✅ Inspect request/response schemas
- ✅ View all available endpoints

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
uvicorn main:app --reload
```

### 3. Test the Health Endpoint

```bash
curl http://localhost:8000/health
```

### 4. Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/your/image.png"
```

---

## 🧠 Engineering Notes

- ✓ The API strictly mirrors the **training-time preprocessing pipeline**
- ✓ Uses the `.keras` saved model format for safe, versioned inference
- ✓ Designed to be **deployment-ready** (Docker / cloud compatible)
- ✓ **Stateless inference** — suitable for horizontal scaling
- ✓ Production-grade error handling and validation
- ✓ CORS-enabled for frontend integration

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t deepsign-api .
```

### Run Container

```bash
docker run -p 8000:8000 deepsign-api
```

### Access API

```
http://localhost:8000/docs
```

---

## 🔒 Security Considerations

- **File Size Limit**: Maximum upload size is 5MB
- **File Type Validation**: Only `.png`, `.jpg`, `.jpeg` files are accepted
- **Rate Limiting**: Consider implementing rate limiting for production
- **Authentication**: Add API key authentication for production deployment

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **API Framework** | FastAPI |
| **ML Framework** | TensorFlow / Keras |
| **Server** | Uvicorn |
| **Image Processing** | Pillow, NumPy |
| **Model Format** | `.keras` |

---

## 📊 API Performance

| Metric | Value |
|--------|-------|
| **Average Response Time** | ~100ms |
| **Model Load Time** | ~2s on startup |
| **Supported Formats** | PNG, JPG, JPEG |
| **Max File Size** | 5MB |

---

## 🤝 Support

For issues, questions, or contributions:

- **GitHub Issues**: [Report a bug](https://github.com/RansiluRanasinghe/DeepSign-CNN-Based-Sign-Language-Recognition-System/issues)
- **Email**: [dinisthar@gmail.com](mailto:dinisthar@gmail.com)
- **LinkedIn**: [Ransilu Ranasinghe](https://www.linkedin.com/in/ransilu-ranasinghe-a596792ba)

---

<div align="center">

**Built with ❤️ using FastAPI and TensorFlow**

⭐ **Star this repo if you find it useful!**

</div>