import os
import subprocess
import pyautogui
import time
import psutil
from PIL import Image

# Program 실행
od = os.curdir
os.chdir(r'D:\MnS\Pinokio.V2\Pinokio.VTE\Pinokio.VTE\bin\Debug')
p1 = subprocess.Popen('Pinokio.exe',
                      shell=True, stdin=None, stdout=None, stderr=None,
                      close_fds=True)
time.sleep(3)
png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-026.png")
rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
pyautogui.moveTo(rtn)
pyautogui.click()
pyautogui.click()

SW_HIDE = 0
info = subprocess.STARTUPINFO()
info.dwFlags = subprocess.STARTF_USESHOWWINDOW
info.wShowWindow = SW_HIDE

os.chdir(r'D:\MnS\Pinokio.V2\Pinokio.ACS\Pinokio.ACS\bin\Debug')
p2 = subprocess.Popen('Pinokio.ACS.exe',
                      stdin=None, stdout=None, stderr=None,
                      close_fds=True, startupinfo=info)
time.sleep(5)

png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-025.png")
rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
pyautogui.moveTo(rtn)
pyautogui.click()

time.sleep(1)

png_file = Image.open(r"C:\Users\Simon Anderson\Desktop\스크린샷\K-028.png")
rtn = pyautogui.locateCenterOnScreen(png_file, confidence=0.8)
pyautogui.moveTo(rtn)
pyautogui.click()

time.sleep(40)
for proc in psutil.process_iter():
    try:
        # 프로세스 이름, PID값 가져오기
        processName = proc.name()
        processID = proc.pid
        print(processName, ' - ', processID)

        if processName == "Pinokio.exe" or processName == "Pinokio.ACS.exe":
            parent_pid = processID  # PID
            parent = psutil.Process(parent_pid)  # PID 찾기
            for child in parent.children(recursive=True):  # 자식-부모 종료
                child.kill()
            parent.kill()

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):  # 예외처리
        pass
