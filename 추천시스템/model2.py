import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# ============================================
# 1. 데이터 로드 및 전처리
# ============================================

# CSV 파일 예시:
# A_눈,A_코,A_턱,A_얼굴형,B_눈,B_코,B_턱,B_얼굴형,성공
df = pd.read_csv("matching_data.csv")

# 모든 범주형 컬럼
A_cols = ["A_눈","A_코","A_턱","A_얼굴형"]
B_cols = ["B_눈","B_코","B_턱","B_얼굴형"]

# OneHot Encoding
encoder_A = OneHotEncoder(sparse_output=False)
encoder_B = OneHotEncoder(sparse_output=False)


A_encoded = encoder_A.fit_transform(df[A_cols])
B_encoded = encoder_B.fit_transform(df[B_cols])

# 학습 라벨 (성공 여부)
y = df["성공"].values

# Train/Test 분할
A_train, A_test, B_train, B_test, y_train, y_test = train_test_split(
    A_encoded, B_encoded, y, test_size=0.2, random_state=42
)

# Tensor 변환
A_train_tensor = torch.tensor(A_train, dtype=torch.float32)
B_train_tensor = torch.tensor(B_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

A_test_tensor = torch.tensor(A_test, dtype=torch.float32)
B_test_tensor = torch.tensor(B_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ============================================
# 2. Two Tower 모델 정의
# ============================================

# Encoder 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Two Tower Model
class TwoTowerModel(nn.Module):
    def __init__(self, input_dim_A, input_dim_B, embed_dim=64):
        super().__init__()
        self.encoder_A = Encoder(input_dim_A, embed_dim)
        self.encoder_B = Encoder(input_dim_B, embed_dim)

    def forward(self, A, B):
        # A,B 임베딩 생성
        embed_A = self.encoder_A(A)
        embed_B = self.encoder_B(B)
        # Dot Product로 점수 계산
        score = (embed_A * embed_B).sum(dim=1, keepdim=True)
        return score

# 모델 인스턴스 생성
model = TwoTowerModel(
    input_dim_A=A_train.shape[1],
    input_dim_B=B_train.shape[1],
    embed_dim=32
)

# ============================================
# 3. 학습 Loop
# ============================================

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 50

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(A_train_tensor, B_train_tensor)
    loss = criterion(preds, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# ============================================
# 4. 추천 수행
# ============================================

# 새 조합 A 입력 (예시)
new_A_df = pd.DataFrame([{
    "A_눈": "긴",
    "A_코": "낮음",
    "A_턱": "각짐",
    "A_얼굴형": "둥근"
}])

# 조합 A를 OneHot Encoding
new_A_encoded = encoder_A.transform(new_A_df)
new_A_tensor = torch.tensor(new_A_encoded, dtype=torch.float32)

# 추천 후보군 B: 기존 데이터의 모든 B 조합
b_candidates_df = df[B_cols].drop_duplicates().reset_index(drop=True)
b_candidates_encoded = encoder_B.transform(b_candidates_df)
b_candidates_tensor = torch.tensor(b_candidates_encoded, dtype=torch.float32)

# A 임베딩 계산
model.eval()
with torch.no_grad():
    a_embed = model.encoder_A(new_A_tensor)          # shape: [1, embed_dim]
    b_embeds = model.encoder_B(b_candidates_tensor)  # shape: [N, embed_dim]

    # cosine similarity 계산
    cosine_scores = F.cosine_similarity(b_embeds, a_embed)  # shape: [N]

    # -1~+1 범위를 0~1로 변환
    normalized_scores = (cosine_scores + 1.0) / 2.0

    # 0~100% 변환
    percent_scores = normalized_scores * 100

# Top-K 추출
k = min(5, percent_scores.shape[0])
topk = torch.topk(percent_scores, k=k)

# 출력
print("\n=== 추천 결과 ===")
for rank, (idx, score) in enumerate(zip(topk.indices, topk.values), start=1):
    b_features = b_candidates_df.iloc[idx.item()]
    print(f"[{rank}] 예상 매칭률: {score:.1f}%, 추천 조합: {list(b_features.values)}")