import requests
import time

print("Starting training via OpenEnv API...")
r = requests.post("http://localhost:8000/train/start", json={"max_steps": 50, "phase": 4})
print(r.json())

while True:
    try:
        status = requests.get("http://localhost:8000/train/status").json()
        print(f"Step: {status.get('step')}/{status.get('max_steps')} | Status: {status.get('status')} | Reward: {status.get('current_reward')} | GPU: {status.get('gpu_name')}")
        if status.get('status') in ['complete', 'error']:
            if status.get('status') == 'error':
                print(f"ERROR: {status.get('error')}")
            break
    except Exception as e:
        print(f"API Error: {e}")
    time.sleep(10)
