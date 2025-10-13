# genDOPA

**genDOPA**는 DOPA-driven adhesive molecule 을 design하는 생성형 AI 모델들을 구현한 프로젝트입니다. 이 프로젝트는 PyTorch 및 PyTorch Lightning 프레임워크를 기반으로 하며, Semi-supervised (C)VAE 모델을 통해 Graphite에 대한 특정 adhesive energy 영역대의 DOPA-derived adehsive molecule을 생성함.
## 📁 프로젝트 구조

```
genDOPA/
├── config/               # YAML 설정 파일 (모델, 데이터셋, 학습 등)
├── data/                 # 데이터셋 저장 디렉토리
├── gendopa/              # genDOPA source code
|   ├── unitcell.py           # UnitCell object
|   ├── dataset.py            # Dataset obejct 및 전처리 모듈
|   ├── reader.py             # 데이터 로드 및 DataLoader
|   ├── utils.py              # 시각화 및 기타 유틸 함수         (작성중)
|   └── nn/                   # Neural Network 모델 정의
|       ├── ANN.py                   # ANN 모델 정의
|       ├── CNN.py                   # CNN 모델 정의
|       └── SteerableCNN.py          # Steerable CNN 모델 정의 (작성중)
├── main.py               # 실행 스크립트
├── requirements.txt      # 필요한 패키지 목록
└── README.md             # 프로젝트 안내 문서
```
