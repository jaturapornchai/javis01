"""One-time Google consent (read-only scopes). Run: python google_login.py"""
import os
import sys

import google_tools as G

STEPS = """ยังไม่พบไฟล์ client_secret.json ทำตามขั้นตอนนี้:
1. เข้า https://console.cloud.google.com แล้วสร้างโปรเจกต์ใหม่
2. APIs & Services > Library เปิดใช้ Gmail API, Google Calendar API, Google Drive API
3. OAuth consent screen: เลือก External แล้วเพิ่มอีเมลของตัวเองเป็น Test user
4. Credentials > Create credentials > OAuth client ID > ชนิด Desktop app
5. ดาวน์โหลด JSON แล้วบันทึกเป็น D:\\bctest01\\client_secret.json
จากนั้นรัน python google_login.py อีกครั้ง"""

if __name__ == "__main__":
    if not os.path.exists(G.CLIENT_SECRET_PATH):
        print(STEPS)
        sys.exit(1)
    from google_auth_oauthlib.flow import InstalledAppFlow
    creds = InstalledAppFlow.from_client_secrets_file(G.CLIENT_SECRET_PATH, G.SCOPES).run_local_server(port=0)
    os.makedirs(os.path.dirname(G.TOKEN_PATH), exist_ok=True)
    with open(G.TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    print("เชื่อม Google สำเร็จ")
