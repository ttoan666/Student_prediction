import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Thiếu cấu hình SUPABASE_URL hoặc SUPABASE_SECRET_KEY trong file .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="API Dự đoán kết quả học tập")

# 1. Cấu hình CORS (Cho phép giao diện web gọi API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mở khóa cho mọi tên miền. Khi đưa lên mạng thật sẽ cấu hình lại.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Tải mô hình học máy (.pkl)
try:
    model = joblib.load("model.pkl")
    print("Đã tải thành công mô hình AI.")
except Exception as e:
    model = None
    print("Cảnh báo: Chưa tìm thấy file model.pkl. API sẽ trả về kết quả giả lập.")

# 3. Hướng đối tượng: Định nghĩa khuôn đúc dữ liệu đầu vào
class PredictionInput(BaseModel):
    study_hours: int
    absence_days: int

@app.get("/")
def read_root():
    return {"message": "Máy chủ Backend đang hoạt động!"}

@app.get("/test-db")
def test_database_connection():
    try:
        response = supabase.table("predictions").select("*").limit(1).execute()
        return {"status": "Kết nối Database thành công", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi kết nối CSDL: {str(e)}")

# 4. API Dự đoán và Lưu trữ
@app.post("/predict")
def predict_score(data: PredictionInput):
    try:
        # Xử lý bằng mô hình (nếu có), ngược lại dùng công thức giả lập
        if model:
            # AI nhận vào mảng 2 chiều theo chuẩn scikit-learn
            prediction = model.predict([[data.study_hours, data.absence_days]])
            predicted_score = round(float(prediction[0]), 2)
        else:
            # Giả lập: Điểm gốc là 50 + (0.5 * giờ học) - (2 * số ngày nghỉ)
            predicted_score = 50 + (data.study_hours * 0.5) - (data.absence_days * 2)
            predicted_score = max(0, min(100, predicted_score)) # Ép kết quả vào thang 0-100

        # Lệnh lưu dữ liệu vào Supabase
        db_response = supabase.table("predictions").insert({
            "study_hours": data.study_hours,
            "absence_days": data.absence_days,
            "predicted_score": predicted_score
        }).execute()

        # Trả kết quả về cho người dùng
        return {
            "message": "Dự đoán và lưu lịch sử thành công",
            "predicted_score": predicted_score,
            "db_record": db_response.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý dự đoán: {str(e)}")