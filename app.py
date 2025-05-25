from flask import Flask, request, jsonify
import openai
import base64
import os
from dotenv import load_dotenv  # ← 이거 추가

load_dotenv()  # ← 이거 추가

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    image1 = data.get("image1")  # base64 string
    image2 = data.get("image2")

    messages = [
        {"role": "system", "content": "You are a fingerprint and emotional expert."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """너는 손끝의 무늬와 흐름을 예술적 상징으로 바라보는 감성 심리 해석가야.  
지금부터 제공할 손가락 이미지 2장을 보고, 이 안에 담긴 상징을 분석해줘.

1. 🔍 이미지 속 선의 패턴과 가지 뻗음 방향을 설명해줘.  
2. 🧠 이 이미지가 상징하는 감정·성향을 유추해줘 (직접 추정 말고 상징적으로 표현).  
3. 🌳 이 이미지에서 연상되는 나무의 형태와 뿌리 구조를 Why / What / How 구조로 설명해줘.  
4. 💬 사용자에게 감성적으로 건네는 한 문장 메시지를 적어줘.  
5. ✨ 마지막으로 ‘심리 예술가의 노트’라는 이름으로 한 줄 정리를 해줘.
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image1}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image2}"
                    }
                }
            ]
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1500
    )

    return jsonify({"result": response.choices[0].message.content})
